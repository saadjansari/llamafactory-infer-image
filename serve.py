import os, json, re, time
from pathlib import Path
from urllib.parse import urlparse

import boto3
import torch
from fastapi import FastAPI, Request, Response
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

app = FastAPI()
S3 = boto3.client("s3")

MODEL_DIR = Path("/opt/ml/model")  # SageMaker extracts model.tar.gz here
MEDIA_S3_PREFIX = os.environ.get("MEDIA_S3_PREFIX", "").strip()
VIDEO_FPS = float(os.environ.get("VIDEO_FPS", "1.0"))
VIDEO_MAX_PIXELS = int(os.environ.get("VIDEO_MAX_PIXELS", str(256 * 28 * 28)))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))

_cache_dir = Path("/tmp/media_cache")
_cache_dir.mkdir(parents=True, exist_ok=True)

processor = None
model = None
primary_device = None


def parse_s3_uri(uri: str):
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def s3_download(uri: str, dst: Path):
    b, k = parse_s3_uri(uri)
    dst.parent.mkdir(parents=True, exist_ok=True)
    S3.download_file(b, k, str(dst))


def resolve_video_uri(video_ref: str) -> str:
    if video_ref.startswith("s3://"):
        return video_ref
    if not MEDIA_S3_PREFIX:
        raise ValueError("Got non-s3 video ref but MEDIA_S3_PREFIX is empty.")
    return MEDIA_S3_PREFIX.rstrip("/") + "/" + video_ref.lstrip("/")


def strip_video_audio_tags(text: str) -> str:
    return (
        text.replace("<video><audio>", "")
            .replace("<video>", "")
            .replace("<audio>", "")
            .lstrip()
    )


def best_effort_json(s: str):
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def ensure_loaded():
    global processor, model, primary_device
    if model is not None:
        return

    print(f"Loading processor/model from {MODEL_DIR}", flush=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if hasattr(processor, "max_pixels"):
        processor.max_pixels = VIDEO_MAX_PIXELS

    try:
        primary_device = next(model.parameters()).device
    except StopIteration:
        primary_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.get("/ping")
@app.post("/ping")
def ping():
    ensure_loaded()
    return {"status": "ok"}


@app.post("/invocations")
async def invocations(req: Request):
    """
    Expected JSON payload (flexible):
      - either {"messages": [...], "videos":[...], "audios":[...]}
      - or {"record": <same-structure>}
    videos[0] can be an s3://... uri OR basename (if MEDIA_S3_PREFIX is set).
    """
    ensure_loaded()
    t0 = time.time()

    body = await req.body()
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as e:
        return Response(
            content=json.dumps({"status": "failed", "error": f"Invalid JSON: {e}"}),
            status_code=400,
            media_type="application/json",
        )

    rec = payload.get("record", payload)

    try:
        messages = rec["messages"]
        video_ref = rec["videos"][0]
        _ = rec.get("audios", [None])[0]  # we load audio from video

        video_uri = resolve_video_uri(video_ref)
        _, key = parse_s3_uri(video_uri)
        local_path = _cache_dir / Path(key).name
        if not local_path.exists():
            s3_download(video_uri, local_path)

        convo = []
        for m in messages:
            role = m["role"]
            text = strip_video_audio_tags(m["content"])
            if role == "system":
                convo.append({"role": "system", "content": [{"type": "text", "text": text}]})
            elif role == "user":
                convo.append({"role": "user", "content": [
                    {"type": "video", "video": str(local_path)},
                    {"type": "text", "text": text},
                ]})
            else:
                convo.append({"role": role, "content": [{"type": "text", "text": text}]})

        inputs = processor.apply_chat_template(
            convo,
            load_audio_from_video=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            fps=float(rec.get("video_fps", VIDEO_FPS)),
            padding=True,
            use_audio_in_video=True,
        )

        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(primary_device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=int(rec.get("max_new_tokens", MAX_NEW_TOKENS)),
            )

        prompt_len = inputs["input_ids"].shape[-1]
        out_ids = gen[:, prompt_len:]
        raw = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

        out = {
            "status": "ok",
            "video_s3_uri": video_uri,
            "raw_text": raw,
            "parsed_json": best_effort_json(raw),
            "elapsed_sec": round(time.time() - t0, 3),
        }
        return Response(content=json.dumps(out), media_type="application/json")

    except Exception as e:
        out = {
            "status": "failed",
            "error": str(e),
            "elapsed_sec": round(time.time() - t0, 3),
        }
        return Response(content=json.dumps(out), status_code=500, media_type="application/json")


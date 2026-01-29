import os, json, re, time, hashlib, uuid, asyncio, threading
from pathlib import Path
from urllib.parse import urlparse

import boto3
from fastapi import FastAPI, Request, Response

app = FastAPI()
S3 = boto3.client("s3")

MODEL_DIR = Path("/opt/ml/model")  # SageMaker extracts model.tar.gz here
MEDIA_S3_PREFIX = os.environ.get("MEDIA_S3_PREFIX", "").strip()

VIDEO_FPS = float(os.environ.get("VIDEO_FPS", "1.0"))
VIDEO_MAX_PIXELS = int(os.environ.get("VIDEO_MAX_PIXELS", str(256 * 28 * 28)))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))

# Worker pool sizing
NUM_GPU_WORKERS = int(os.environ.get("NUM_GPU_WORKERS", "8"))
GPU_IDS = os.environ.get("GPU_IDS", "").strip()  # e.g. "0,1,2,3,4,5,6,7"
REQUEST_TIMEOUT_SEC = int(os.environ.get("REQUEST_TIMEOUT_SEC", "3600"))

_cache_dir = Path("/tmp/media_cache")
_cache_dir.mkdir(parents=True, exist_ok=True)

# multiprocessing objects (initialized on startup)
_mp_ctx = None
_job_q = None
_res_q = None
_workers = []
_pending = {}  # job_id -> asyncio.Future


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


def _local_cache_path_for(uri: str) -> Path:
    # stable unique filename (avoid collisions)
    _, key = parse_s3_uri(uri)
    ext = Path(key).suffix or ".mp4"
    h = hashlib.sha1(uri.encode("utf-8")).hexdigest()
    return _cache_dir / f"{h}{ext}"


def _worker_main(gpu_id: str, model_dir: str, job_q, res_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import torch
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, trust_remote_code=True)

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map=None,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda:0")
    model.eval()

    # Apply processor settings (if supported)
    if hasattr(processor, "max_pixels"):
        try:
            processor.max_pixels = int(os.environ.get("VIDEO_MAX_PIXELS", "0")) or processor.max_pixels
        except Exception:
            pass

    while True:
        job = job_q.get()
        if job is None:
            break

        t0 = time.time()
        job_id = job["job_id"]
        try:
            messages = job["messages"]
            local_video_path = job["local_video_path"]
            fps = float(job.get("video_fps", os.environ.get("VIDEO_FPS", "1.0")))
            max_new_tokens = int(job.get("max_new_tokens", os.environ.get("MAX_NEW_TOKENS", "128")))

            convo = []
            for m in messages:
                role = m["role"]
                text = strip_video_audio_tags(m["content"])
                if role == "system":
                    convo.append({"role": "system", "content": [{"type": "text", "text": text}]})
                elif role == "user":
                    convo.append({"role": "user", "content": [
                        {"type": "video", "video": local_video_path},
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
                fps=fps,
                padding=True,
                use_audio_in_video=True,
            )

            if torch.cuda.is_available():
                for k, v in list(inputs.items()):
                    if torch.is_tensor(v):
                        inputs[k] = v.to("cuda:0")

            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=max_new_tokens)

            prompt_len = inputs["input_ids"].shape[-1]
            out_ids = gen[:, prompt_len:]
            raw = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

            res_q.put({
                "job_id": job_id,
                "ok": True,
                "raw_text": raw,
                "parsed_json": best_effort_json(raw),
                "elapsed_sec": round(time.time() - t0, 3),
            })
        except Exception as e:
            res_q.put({
                "job_id": job_id,
                "ok": False,
                "error": str(e),
                "elapsed_sec": round(time.time() - t0, 3),
            })


def _result_pump(loop):
    """Background thread: move results from multiprocessing queue -> asyncio futures."""
    while True:
        res = _res_q.get()
        if res is None:
            return
        job_id = res.get("job_id")
        fut = _pending.pop(job_id, None)
        if fut and not fut.done():
            loop.call_soon_threadsafe(fut.set_result, res)


@app.on_event("startup")
def _startup():
    global _mp_ctx, _job_q, _res_q, _workers

    import multiprocessing as mp
    _mp_ctx = mp.get_context("spawn")

    _job_q = _mp_ctx.Queue(maxsize=NUM_GPU_WORKERS * 4)
    _res_q = _mp_ctx.Queue()

    gpu_ids = [x.strip() for x in GPU_IDS.split(",") if x.strip()] if GPU_IDS else [str(i) for i in range(NUM_GPU_WORKERS)]

    for gpu_id in gpu_ids:
        p = _mp_ctx.Process(
            target=_worker_main,
            args=(gpu_id, str(MODEL_DIR), _job_q, _res_q),
            daemon=True,
        )
        p.start()
        _workers.append(p)

    loop = asyncio.get_event_loop()
    t = threading.Thread(target=_result_pump, args=(loop,), daemon=True)
    t.start()


@app.on_event("shutdown")
def _shutdown():
    # stop workers
    try:
        for _ in _workers:
            _job_q.put(None)
    except Exception:
        pass


@app.get("/ping")
@app.post("/ping")
def ping():
    # Basic health: workers alive
    alive = sum(1 for p in _workers if p.is_alive())
    return {"status": "ok", "gpu_workers_alive": alive, "gpu_workers_total": len(_workers)}


@app.post("/invocations")
async def invocations(req: Request):
    """
    Expected JSON payload (flexible):
      - either {"messages": [...], "videos":[...], "audios":[...]}
      - or {"record": <same-structure>}
    videos[0] can be an s3://... uri OR basename (if MEDIA_S3_PREFIX is set).
    """
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

        video_uri = resolve_video_uri(video_ref)
        local_path = _local_cache_path_for(video_uri)
        if not local_path.exists():
            s3_download(video_uri, local_path)

        job_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        _pending[job_id] = fut

        _job_q.put({
            "job_id": job_id,
            "messages": messages,
            "local_video_path": str(local_path),
            "video_fps": rec.get("video_fps", VIDEO_FPS),
            "max_new_tokens": rec.get("max_new_tokens", MAX_NEW_TOKENS),
        })

        res = await asyncio.wait_for(fut, timeout=REQUEST_TIMEOUT_SEC)

        if res["ok"]:
            out = {
                "status": "ok",
                "video_s3_uri": video_uri,
                "raw_text": res["raw_text"],
                "parsed_json": res["parsed_json"],
                "elapsed_sec": round(time.time() - t0, 3),
                "worker_elapsed_sec": res["elapsed_sec"],
            }
            return Response(content=json.dumps(out), media_type="application/json")
        else:
            out = {
                "status": "failed",
                "video_s3_uri": video_uri,
                "error": res.get("error", "unknown"),
                "elapsed_sec": round(time.time() - t0, 3),
                "worker_elapsed_sec": res["elapsed_sec"],
            }
            return Response(content=json.dumps(out), status_code=500, media_type="application/json")

    except Exception as e:
        out = {
            "status": "failed",
            "error": str(e),
            "elapsed_sec": round(time.time() - t0, 3),
        }
        return Response(content=json.dumps(out), status_code=500, media_type="application/json")


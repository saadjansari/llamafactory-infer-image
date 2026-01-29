#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
    a = torch.randn((512,512), device="cuda")
    b = torch.randn((512,512), device="cuda")
    _ = a @ b
    torch.cuda.synchronize()
    print("cuda_smoke_test: OK")
PY

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
exec uvicorn serve:app --host 0.0.0.0 --port 8080 --workers 1


#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   bash benchmark_single_image.sh
#   GPU_ID=0 CHANNEL=awgn M=25 bash benchmark_single_image.sh
#   USE_DENOISE=0 CHECKPOINT=/path/to/checkpoint.pth bash benchmark_single_image.sh

cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES="${GPU_ID:-3}"
export BENCH_USE_DENOISE="${USE_DENOISE:-1}"
export BENCH_CHANNEL="${CHANNEL:-rayleigh}"
export BENCH_CR="${CR:-3}"
export BENCH_SNR="${SNR:-18}"
export BENCH_M="${M:-0}"
export BENCH_WARMUP="${WARMUP:-10}"
export BENCH_TRIALS="${TRIALS:-50}"
export BENCH_CHECKPOINT="${CHECKPOINT:-}"

PYTHON_BIN="${PYTHON_BIN:-/4T/zhangguangyi/anaconda3/envs/torch/bin/python}"

"$PYTHON_BIN" - <<'PY'
import os
import statistics
import time

import torch

from app_common import build_model


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; check GPU_ID and the selected Python environment")

channel = os.environ["BENCH_CHANNEL"]
cr = float(os.environ["BENCH_CR"])
snr = float(os.environ["BENCH_SNR"])
quant_levels = float(os.environ["BENCH_M"])
warmup = int(os.environ["BENCH_WARMUP"])
trials = int(os.environ["BENCH_TRIALS"])
use_denoise = bool(int(os.environ["BENCH_USE_DENOISE"]))
checkpoint_path = os.environ["BENCH_CHECKPOINT"]

if warmup < 0 or trials < 1:
    raise ValueError("WARMUP must be >= 0 and TRIALS must be >= 1")

model = build_model("Inverse", cr, channel, "cuda", use_denoise=use_denoise)
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    state = checkpoint.get("net", checkpoint)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)

model.eval()
x = torch.randn(1, 3, 32, 32, device="cuda")

with torch.inference_mode():
    for _ in range(warmup):
        model(x, snr=snr, M=quant_levels)
    torch.cuda.synchronize()

    samples_ms = []
    for _ in range(trials):
        start = time.perf_counter()
        model(x, snr=snr, M=quant_levels)
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Model: {'InverseNetDenoise' if use_denoise else 'InverseNet'}, channel={channel}, CR={cr:g}, SNR={snr:g}, M={quant_levels:g}")
print(f"Single image (batch=1, 3x32x32), warmup={warmup}, trials={trials}")
print(f"Latency: mean={statistics.mean(samples_ms):.3f} ms, median={statistics.median(samples_ms):.3f} ms, min={min(samples_ms):.3f} ms")
PY

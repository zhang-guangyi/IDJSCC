#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-1}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-/4T/zhangguangyi/anaconda3/envs/torch/bin/python}"
LOG_DIR="${LOG_DIR:-outputs/latency}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/idjscc_$(date +%Y%m%d_%H%M%S)_$$.log"

# Same timing scope as sim_main.py and the DFD/MHPSC benchmark_latency.sh:
# random weights before checkpoint loading, one hop, no_grad, loop-level sync.
"$PYTHON" -u - "$@" <<'PY' 2>&1 | tee "$LOG_FILE"
import argparse
import json
import os
import time

import torch
from app_common import build_model, seed_initial

p = argparse.ArgumentParser(description='Reference-compatible IDJSCC dummy latency')
p.add_argument('--snr', type=float, default=18.)
p.add_argument('--warmup', type=int, default=10)
p.add_argument('--trials', type=int, default=50)
args = p.parse_args()
if args.warmup < 0 or args.trials < 1:
    p.error('warmup must be nonnegative and trials must be positive')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is required; check GPU_ID and the Python environment')
seed_initial(13890)
net = build_model('Inverse', 3, 'awgn', 'cuda', use_denoise=True,
                  x_min=-0.8, x_max=0.8).eval()
dummy = torch.randn(1, 3, 32, 32, device='cuda')
print(json.dumps(dict(model='InverseNetDenoise', gpu=torch.cuda.get_device_name(0),
    cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], torch_version=torch.__version__,
    cpu_threads=torch.get_num_threads(), checkpoint='none (matches reference pre-load timing)',
    input_shape=list(dummy.shape), input='randn float32', hops=1, channel='awgn',
    cbr=1/6, quantization_levels=0, seed=13890,
    parameters=sum(p.numel() for p in net.parameters()), arguments=vars(args)), indent=2), flush=True)
with torch.no_grad():
    for _ in range(args.warmup):
        output = net(dummy, snr=args.snr)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.trials):
        output = net(dummy, snr=args.snr)
    torch.cuda.synchronize()
    t1 = time.time()
reconstruction = output[0] if isinstance(output, tuple) else output
if reconstruction.shape != dummy.shape or not torch.isfinite(reconstruction).all():
    raise RuntimeError('Invalid model output')
print('Dummy input latency (batch=1): avg %.3f ms over %d runs' %
      ((t1-t0)/args.trials*1000., args.trials), flush=True)
PY
printf 'Log: %s\n' "$LOG_FILE"

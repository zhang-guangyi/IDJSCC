#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-/4T/zhangguangyi/anaconda3/envs/torch/bin/python}"
GPU_ID="${GPU_ID:-3}"
LOG_DIR="${LOG_DIR:-outputs/test_ofdm}"
CHECKPOINT="${CHECKPOINT:-checkpoints/ckpt-t-cr3-idjscc-denoise-t10db-awgn-rt-M10.pth}"
if [[ ! -s "$CHECKPOINT" ]]; then
    echo "Checkpoint not found or incomplete: $CHECKPOINT" >&2
    exit 1
fi
mkdir -p "$LOG_DIR"
RUN_DIR="$LOG_DIR/ofdm_$(date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -u test_ofdm/run_ofdm.py \
    --resume "$CHECKPOINT" --output-dir "$RUN_DIR" \
    --cr 3 --snr "${SNR:-24}" --hops 30 \
    --test_batch_size "${TEST_BATCH_SIZE:-50}" \
    --group_num "${GROUP_NUM:-1}" \
    --M "${QUANT_LEVELS:-18}" --x_min "${X_MIN:--1.5}" --x_max "${X_MAX:-1.5}" \
    --nfft "${NFFT:-64}" --cp-len "${CP_LEN:-16}" \
    --taps "${TAPS:-8}" --decay "${DECAY:-2.0}" \
    "$@"

#!/usr/bin/env bash
set -euo pipefail

# Experiment_1 in sim_engine.py performs 30 recursive transmission hops.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-/4T/zhangguangyi/anaconda3/envs/torch/bin/python}"
GPU_ID="${GPU_ID:-3}"
LOG_DIR="${LOG_DIR:-outputs/test_30hops_quantize}"
CHECKPOINT="${CHECKPOINT:-checkpoints/ckpt-t-cr3-idjscc-denoise-t18db-awgn-analog.pth}"

if [[ ! -s "$CHECKPOINT" ]]; then
    echo "Checkpoint not found or incomplete: $CHECKPOINT" >&2
    exit 1
fi
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/test_idjscc_no_rt_cbr_1_6_snr18_30hops_quantize_$(date +%Y%m%d_%H%M%S)_$$.log"
printf 'Checkpoint: %s\nLog: %s\n' "$CHECKPOINT" "$LOG_FILE"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -u sim_main.py \
    --cr 3 \
    --test_batch_size "${TEST_BATCH_SIZE:-50}" \
    --test_number Experiment_1 \
    --group_num "${GROUP_NUM:-1}" \
    --test \
    --snr 18 \
    --M "${QUANT_LEVELS:-24}" \
    --x_min "${X_MIN:--3.0}" \
    --x_max "${X_MAX:-3.0}" \
    --model Inverse \
    --use_denoise \
    --channel awgn \
    --resume "$CHECKPOINT" \
    "$@" 2>&1 | tee "$LOG_FILE"

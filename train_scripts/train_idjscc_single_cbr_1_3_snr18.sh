#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-/4T/zhangguangyi/anaconda3/envs/torch/bin/python}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/idjscc_single_cbr1_3_snr18}"
RESUME="${RESUME-outputs/idjscc_single_cbr1_3_snr18/checkpoint-best.pth}"
resume_args=()
if [[ -n "$RESUME" ]]; then
    if [[ ! -f "$RESUME" ]]; then
        echo "Checkpoint not found: $RESUME" >&2
        exit 1
    fi
    resume_args=(--resume "$RESUME")
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" main.py \
    --model Inverse --use_denoise --channel awgn \
    --cr 1.5 --snr 18 --M 0 \
    "${resume_args[@]}" \
    --denoise_lambda 0.12 \
    --batch_size "${BATCH_SIZE:-50}" --test_batch_size "${TEST_BATCH_SIZE:-50}" \
    --num_epochs "${EPOCHS:-500}" --lr "${LR:-0.00002}" \
    --save_freq 5 --output_dir "$OUTPUT_DIR" \
    "$@"

#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CUDA_VISIBLE_DEVICES=0 python deepsc_main.py \
	--cr 3 \
	--batch_size 64 \
	--test_batch_size 50 \
	--save_freq 10 \
	--lr 3e-5 \
	--loop \
	--output_dir outputs/train_inverse_denoise_SNR18dB_M200_withloop_rayleigh \
	--resume /4T/zhangguangyi/Idempotent_DJSCC/Idempotent_DJSCC_Curated/Small_model/outputs/train_inverse_denoise_SNR18dB_M0_withloop_rayleigh/checkpoint-best.pth \
	--model Inverse \
	--use_denoise \
	--num_epochs 600 \
	--channel rayleigh \
	--snr 18 \
	--M 200
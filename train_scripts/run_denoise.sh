#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# CUDA_VISIBLE_DEVICES=1 python deepsc_main.py \
# 	--cr 3 \
# 	--batch_size 50 \
# 	--test_batch_size 50 \
# 	--save_freq 3 \
# 	--lr 3e-6 \
# 	--loop \
# 	--output_dir  outputs/train_inverse_denoise_SNR10dB_M10_withloop_newquan_v5_purequan \
# 	--resume  /4T/zhangguangyi/Idempotent_DJSCC/Idempotent_DJSCC_Curated/IDJSCC_CIFAR10/checkpoints/ckpt-t-cr3-idjscc-denoise-t18db-awgn-rt-M25.pth \
# 	--model Inverse \
# 	--use_denoise \
# 	--num_epochs 600 \
# 	--channel awgn \
# 	--snr 10 \
# 	--test
# 	--M 12

# CUDA_VISIBLE_DEVICES=0 python deepsc_main.py \
# 	--cr 3 \
# 	--batch_size 50 \
# 	--test_batch_size 50 \
# 	--save_freq 2 \
# 	--lr 5e-6 \
# 	--loop \
# 	--output_dir  outputs/train_inverse_denoise_SNR12dB_M40_withloop_rayleigh_v3_purequan \
# 	--resume  /4T/zhangguangyi/Idempotent_DJSCC/Idempotent_DJSCC_Curated/IDJSCC_CIFAR10/outputs/train_inverse_denoise_SNR12dB_M10_withloop_rayleigh_withloop/checkpoint-best.pth \
# 	--model Inverse \
# 	--use_denoise \
# 	--num_epochs 600 \
# 	--channel rayleigh \
# 	--snr 12 \
# 	--M 40

CUDA_VISIBLE_DEVICES=1 python deepsc_main.py \
	--cr 3 \
	--batch_size 50 \
	--test_batch_size 50 \
	--save_freq 2 \
	--lr 1e-4 \
	--output_dir  outputs/train_inverse_denoise_SNR12dB_M10_withloop_rayleigh_withloop \
	--model Inverse \
	--lr 2e-5 \
	--use_denoise \
	--num_epochs 600 \
	--channel awgn \
	--snr 18 \
	--M 0
	
# CUDA_VISIBLE_DEVICES=1 python deepsc_main.py \
# 	--cr 3 \
# 	--batch_size 64 \
# 	--test_batch_size 50 \
# 	--save_freq 10 \
# 	--lr 2e-5 \
# 	--loop \
# 	--output_dir outputs/train_inverse_denoise_SNR18dB_M10_withloop_rayleigh_v1 \
# 	--resume  /4T/zhangguangyi/Idempotent_DJSCC/Idempotent_DJSCC_Curated/IDJSCC_CIFAR10/outputs/train_inverse_denoise_SNR12dB_M20_withloop_rayleigh_v3/checkpoint-best.pth \
# 	--model Inverse \
# 	--use_denoise \
# 	--num_epochs 600 \
# 	--channel rayleigh \
# 	--snr 18 \
# 	--M 10
#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python deepsc_main.py \
	--cr 3 \
	--batch_size 64 \
	--test_batch_size 50 \
	--lr 1e-5 \
	--output_dir outputs/train_deepsc \
	--model DeepSC

 

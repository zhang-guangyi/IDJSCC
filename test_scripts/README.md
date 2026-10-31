# IDJSCC CIFAR10 30 跳测试

共 27 个脚本，覆盖 `checkpoints/` 中全部 15 个权重。权重路径已填入，脚本可从任意工作目录运行。

不带 `_quantize` 后缀的脚本使用 `--M 0`；带后缀的脚本启用潜变量量化。同一配置的普通版和量化版默认使用同一个 analog 权重。文件名中的 `m10/m25/m80` 表示专用权重名称，不代表当前测试的量化级数。

CBR 1/3、1/4、1/6、1/12 分别使用 `--cr 1.5/2/3/6`。文件名未注明信道的脚本使用 AWGN，包含 `rayleigh` 的脚本使用 Rayleigh。仅提供有对应权重的配置。

所有脚本使用 `Inverse + use_denoise`、`Experiment_1`，执行 30 跳。每跳重建图像量化到 8 bit 后再作为下一跳输入；默认评估完整 CIFAR10 测试集，输出各跳平均 PSNR。

## 运行

```bash
GPU_ID=3 bash test_scripts/test_idjscc_rt_cbr_1_6_snr18_30hops.sh
GPU_ID=3 bash test_scripts/test_idjscc_rt_cbr_1_6_snr18_rayleigh_30hops_quantize.sh
GPU_ID=3 bash test_scripts/test_idjscc_rt_cbr_1_6_snr18_m25_30hops_quantize.sh

# 顺序运行全部普通版，失败即停止
for script in test_scripts/test_*_30hops.sh; do
    bash "$script" || break
done
```

默认 Python 为 `/4T/zhangguangyi/anaconda3/envs/torch/bin/python`，GPU 为 3，batch size 为 50，独立重复次数为 1。
可用 `PYTHON`、`GPU_ID`、`TEST_BATCH_SIZE`、`GROUP_NUM`、`CHECKPOINT`、`LOG_DIR` 覆盖默认值。量化脚本还支持 `QUANT_LEVELS`、`X_MIN`、`X_MAX`；普通版的 M 固定为 0。额外命令行参数透传给 `sim_main.py`。

普通版日志保存到 `outputs/test_30hops/`，量化版保存到 `outputs/test_30hops_quantize/`，文件名包含脚本名称、时间和进程号。

## 量化默认值

保留原有 CBR 1/3、1/4 脚本参数。量化默认配置如下：AWGN 18 dB 使用 [-3, 3]；Rayleigh 18 dB 使用说明中推荐的 M=24、[-1, 1]。专用 M10/M25 权重沿用原有测试参数；M80 使用上述 Rayleigh 推荐参数。以下为测试默认值，可通过环境变量调整。

| 量化脚本 | M | 范围 |
| --- | --- | --- |
| `test_idjscc_no_rt_cbr_1_12_snr18_30hops_quantize.sh` | 120 | [-3.0, 3.0] |
| `test_idjscc_no_rt_cbr_1_3_snr18_30hops_quantize.sh` | 12 | [-0.8, 0.8] |
| `test_idjscc_no_rt_cbr_1_4_snr18_30hops_quantize.sh` | 12 | [-0.8, 0.8] |
| `test_idjscc_no_rt_cbr_1_6_snr18_30hops_quantize.sh` | 24 | [-3.0, 3.0] |
| `test_idjscc_no_rt_cbr_1_6_snr18_rayleigh_30hops_quantize.sh` | 24 | [-1.0, 1.0] |
| `test_idjscc_rt_cbr_1_12_snr18_30hops_quantize.sh` | 120 | [-3.0, 3.0] |
| `test_idjscc_rt_cbr_1_3_snr18_30hops_quantize.sh` | 24 | [-0.3, 0.3] |
| `test_idjscc_rt_cbr_1_4_snr18_30hops_quantize.sh` | 24 | [-0.3, 0.3] |
| `test_idjscc_rt_cbr_1_6_snr10_30hops_quantize.sh` | 24 | [-1.5, 1.5] |
| `test_idjscc_rt_cbr_1_6_snr10_m10_30hops_quantize.sh` | 24 | [-1.5, 1.5] |
| `test_idjscc_rt_cbr_1_6_snr12_rayleigh_30hops_quantize.sh` | 50 | [-0.3, 0.3] |
| `test_idjscc_rt_cbr_1_6_snr18_30hops_quantize.sh` | 24 | [-3.0, 3.0] |
| `test_idjscc_rt_cbr_1_6_snr18_m25_30hops_quantize.sh` | 18 | [-3.0, 3.0] |
| `test_idjscc_rt_cbr_1_6_snr18_rayleigh_30hops_quantize.sh` | 24 | [-1.0, 1.0] |
| `test_idjscc_rt_cbr_1_6_snr18_rayleigh_m80_30hops_quantize.sh` | 24 | [-1.0, 1.0] |

迁入 `checkpoints/` 的四个 analog 模型来源、MSE 和 SHA256 见 `checkpoints/best_models_manifest.csv`；其余脚本直接引用 `checkpoints/` 原文件。已有普通版 RT CBR 1/6、10 dB 和 CBR 1/12、18 dB 的日志及参数透传已补齐，RT CBR 1/3 的量化版亦已补齐。

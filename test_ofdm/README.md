# IDJSCC 多跳 OFDM 测试（完美 CSI）

在 `IDJSCC_CIFAR10` 下运行：

```bash
GPU_ID=3 bash test_ofdm/test_idjscc_rt_cbr_1_6_snr10_m10_30hops_quantize.sh
```

继承 `test_scripts/test_idjscc_rt_cbr_1_6_snr10_m10_30hops_quantize.sh` 的配置：
`InverseNetDenoise`、`cr=3`（复符号 CBR=1/6）、24 dB、30 跳、batch=50、group=1、
量化级数 `M=24`、量化范围 `[-1.5,1.5]`。使用同一份
`checkpoints/ckpt-t-cr3-idjscc-denoise-t10db-awgn-rt-M10.pth`，严格加载全部参数。
文件名里的 M10 是权重标识；推理量化级数沿用原脚本的 24。
与 `inversemodel.py` 一致，`M <= 1` 时不量化，进行模拟传输；
可通过脚本环境变量 `QUANT_LEVELS=0` 或命令行参数 `--M 0` 启用。

## 信道与 SNR

- 将归一化 latent 的相邻两个实数作为一个复符号，顺序映射到全部 64 个子载波。
  最后一帧不满时补零；单位正交 IFFT 后加 16 点 CP。
- 默认 8 抽头复高斯瑞利多径，功率时延谱
  `p[l]=exp(-l/2)/sum(exp(-l/2))`，因此 `E[sum |h[l]|²]=1`。
  不对每次信道实现重新归一化。每张图每跳独立抽样，一张图的所有 OFDM 符号共享该信道。
- 对整个串行时域包执行因果线性卷积并加入复高斯噪声，移除 CP，单位正交 FFT。
  CP 必须至少为 `taps-1`。假设完美同步，无 CFO、无多普勒。
- 接收端直接获得真实 `H=FFT(h)`，使用 `Y/H` 的 ZF 均衡。没有导频或信道估计。
  深衰落会放大噪声，完美 CSI 不会消除这一影响。
- SNR 沿用现有信道的 **Es/N0**：有效复符号平均能量 `Es=1`，
  `N0=10^(-snr/10)`；时域噪声实部、虚部各为 `N0/2`。
  单位正交变换保持噪声功率。24 dB 指衰落和均衡前的标称 SNR，
  不按接收瞬时功率调整噪声；不是均衡后 SNR，也不是 Eb/N0。
  `--fading identity` 将 h 固定为单位冲激，可核对 OFDM 与原 AWGN 的噪声功率一致。
- CP 与补零增加传输开销，但不更改 Es/N0。标称 CBR=1/6 不含这些开销。
  每图含 K 个有效复符号时，实际时域复采样数为
  `ceil(K/nfft)*(nfft+cp_len)`，可据此计算含开销的 CBR；没有额外的发射功率放缩。

为了复用原网络的完整前后处理，runner 将网络的信道对象替换为 OFDM 适配器，
内部沿用 `awgn` 分支入口；实际传输路径由 `PerfectCSIOFDM.AWGN` 全部接管。
原有模型、AWGN/Rayleigh 测试代码不需要修改。

## 结果和快速检查

默认评测完整 CIFAR10 测试集（10000 张）。每跳与原图比较 uint8 舍入后的逐图 PSNR，
下一跳输入为 `round(clamp(output,0,1)*255)/255`，与原脚本相同。
结果位于 `outputs/test_ofdm/ofdm_<时间>_<PID>/`：
`config.json`（含权重哈希和信道/SNR 定义）、`per_hop.csv`、
`psnr_per_group.npy`、`psnr.txt`，旁边保存日志。发现非有限输出时直接报错。

```bash
# 两张图、两跳的 CPU 冒烟检查
bash test_ofdm/test_idjscc_rt_cbr_1_6_snr10_m10_30hops_quantize.sh \
  --device cpu --sample_num 2 --hops 2 --test_batch_size 2

# 更改 OFDM 参数；CP 必须覆盖最大离散时延
NFFT=128 CP_LEN=16 TAPS=8 GPU_ID=3 \
  bash test_ofdm/test_idjscc_rt_cbr_1_6_snr10_m10_30hops_quantize.sh

# 信道物理一致性测试
/4T/zhangguangyi/anaconda3/envs/torch/bin/python -m unittest discover -s test_ofdm -v
```

也可直接调用 `python test_ofdm/run_ofdm.py --output-dir <新目录>`；
`--help` 列出参数，`--sample_num 0` 表示全量。已有结果目录不会被覆盖。

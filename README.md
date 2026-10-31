# IDJSCC CIFAR10

Training and evaluation code for IDJSCC on CIFAR10, including single-hop
transmission, recursive training, and multi-hop simulation over AWGN and
Rayleigh channels. The repository includes training scripts, test scripts,
and pretrained checkpoints.

## Project Structure
- `main.py`: training entry.
- `sim_main.py`: simulation/testing entry.
- `app_common.py`: shared initialization, CIFAR loader creation, and IDJSCC model factory.
- `engine.py`: training/testing loops.
- `sim_engine.py`: simulation experiment loops.

The duplicated model/data setup logic from entry scripts has been merged into `app_common.py` to keep behavior consistent and code maintenance simpler.

Only `InverseNet` and `InverseNetDenoise` are supported.

## Run
1. Train Inverse model: `bash train_scripts/run.sh`

## AWGN training scripts

The ten `train_scripts/train_idjscc_{single,rt}_cbr_*_snr*.sh` scripts train the local
InverseNetDenoise model. `single` uses one transmission; `rt` trains over
multiple transmissions and starts from the matching `single` checkpoint by
default. Run the matching `single` script before its `rt` script.

| SNR | CBR | Single transmission | Recursive training |
| --- | --- | --- | --- |
| 18 dB | 1/6 | `train_scripts/train_idjscc_single_cbr_1_6_snr18.sh` | `train_scripts/train_idjscc_rt_cbr_1_6_snr18.sh` |
| 18 dB | 1/12 | `train_scripts/train_idjscc_single_cbr_1_12_snr18.sh` | `train_scripts/train_idjscc_rt_cbr_1_12_snr18.sh` |
| 10 dB | 1/6 | `train_scripts/train_idjscc_single_cbr_1_6_snr10.sh` | `train_scripts/train_idjscc_rt_cbr_1_6_snr10.sh` |
| 18 dB | 1/3 | `train_scripts/train_idjscc_single_cbr_1_3_snr18.sh` | `train_scripts/train_idjscc_rt_cbr_1_3_snr18.sh` |
| 18 dB | 1/4 | `train_scripts/train_idjscc_single_cbr_1_4_snr18.sh` | `train_scripts/train_idjscc_rt_cbr_1_4_snr18.sh` |

CBR 1/3, 1/4, 1/6, and 1/12 correspond to `--cr 1.5`, `2`, `3`, and `6`.
The existing local checkpoints initialize the first three single scripts;
the 1/3 and 1/4 single scripts resume their local best training checkpoints. Set `RESUME=` to start
any script from scratch, or `RESUME=/path/to/checkpoint.pth` to choose weights.
Other overrides: `GPU_ID`, `PYTHON`, `OUTPUT_DIR`, `BATCH_SIZE`,
`TEST_BATCH_SIZE`, `EPOCHS`, and `LR`. RT scripts also accept `TRAIN_HOPS`,
`TEST_HOPS`, and `GAMMA`.

## Simulation
Run simulation: `bash test_scripts/test_idjscc_rt_cbr_1_6_snr18_30hops.sh`.
See `test_scripts/README.md` for all test configurations and checkpoint overrides.

## Checkpoints

All 15 weights are stored in `checkpoints/`. Analog weights use the
`ckpt-t-cr{cr}-idjscc-denoise-t{snr}db-{channel}[-rt]-analog.pth` naming
convention; `rt` denotes recursive training. CBR 1/3 and 1/4 use `cr1.5`
and `cr2`. The four migrated models retain their source, epoch, MSE, and
SHA256 in `checkpoints/best_models_manifest.csv`.

Training launchers in `train_scripts/` change into the project root, so
they can also be invoked by absolute path from another working directory.

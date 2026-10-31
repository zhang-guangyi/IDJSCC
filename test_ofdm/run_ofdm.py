"""Evaluate the existing quantized IDJSCC checkpoint over perfect-CSI OFDM."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app_common import build_cifar10_loaders, build_model, seed_initial
from ofdm_channel import PerfectCSIOFDM


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, default=ROOT / "checkpoints/ckpt-t-cr3-idjscc-denoise-t10db-awgn-rt-M10.pth")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snr", type=float, default=24)
    parser.add_argument("--cr", type=float, default=3)
    parser.add_argument("--M", type=int, default=24, help="M > 1: quantization levels; M <= 1: analog transmission (same as inversemodel.py)")
    parser.add_argument("--x_min", type=float, default=-1.5)
    parser.add_argument("--x_max", type=float, default=1.5)
    parser.add_argument("--hops", type=int, default=30)
    parser.add_argument("--test_batch_size", type=int, default=50)
    parser.add_argument("--group_num", type=int, default=1)
    parser.add_argument("--sample_num", type=int, default=0, help="0: all 10000 test images; positive: limit for quick runs")
    parser.add_argument("--nfft", type=int, default=64)
    parser.add_argument("--cp-len", type=int, default=16)
    parser.add_argument("--taps", type=int, default=8)
    parser.add_argument("--decay", type=float, default=2.0)
    parser.add_argument("--fading", choices=("rayleigh", "identity"), default="rayleigh")
    parser.add_argument("--seed", type=int, default=13890)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def main(args):
    if min(args.hops, args.test_batch_size, args.group_num, args.cpu_threads) < 1 or args.sample_num < 0:
        raise ValueError("Counts must be positive; sample_num may be zero")
    if not args.x_min < args.x_max or not math.isfinite(args.snr):
        raise ValueError("Require x_min < x_max and finite SNR")
    channel = PerfectCSIOFDM(args.nfft, args.cp_len, args.taps, args.decay, args.fading)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(args.cpu_threads)
    seed_initial(args.seed)
    net = build_model("Inverse", args.cr, "awgn", device, use_denoise=True,
                      x_min=args.x_min, x_max=args.x_max).eval()
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    state = checkpoint.get("net", checkpoint)
    net.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    net.channel = channel
    _, loader = build_cifar10_loaders(test_batch_size=args.test_batch_size, test_shuffle=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device=device, torch_version=torch.__version__, channel="ofdm",
                  checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                  snr_definition="Es/N0 before fading and equalization; Es=1 per active complex latent symbol; N0=10^(-snr/10)",
                  fading_coherence="independent per image and hop; constant across all OFDM symbols of one image",
                  pdp="exp(-l/decay), normalized in expectation; no instantaneous channel normalization",
                  equalizer="ZF using exact FFT(h); no pilots, channel estimation, CFO or timing error",
                  fft_normalization="unitary data FFT/IFFT; ordinary FFT for channel response",
                  overhead="CP and final zero padding consume extra channel uses; excluded from Es/N0 and nominal CBR",
                  nominal_cbr=1 / (2 * args.cr),
                  hop_input="round(clamp(reconstruction,0,1)*255)/255",
                  psnr="mean per-image RGB uint8-rounded PSNR against original at every hop; exact match=100 dB")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    sums = np.zeros((args.group_num, args.hops), dtype=np.float64)
    counts = np.zeros(args.group_num, dtype=np.int64)
    with torch.inference_mode():
        for group in range(args.group_num):
            seed_initial(args.seed + group)
            for batch, (reference, _) in enumerate(loader, 1):
                if args.sample_num:
                    remaining = args.sample_num - int(counts[group])
                    if remaining <= 0:
                        break
                    reference = reference[:remaining]
                reference = reference.to(device)
                current = reference
                for hop in range(args.hops):
                    output, _ = net(current, is_train=False, snr=args.snr, M=args.M)
                    if not torch.isfinite(output).all():
                        raise RuntimeError(f"Non-finite reconstruction: group={group + 1}, batch={batch}, hop={hop + 1}")
                    output = output.clamp(0, 1)
                    mse = (torch.round(output * 255) - torch.round(reference * 255)).square().flatten(1).mean(1)
                    psnr = torch.where(mse == 0, 100., 10 * torch.log10(255 ** 2 / mse))
                    sums[group, hop] += psnr.double().sum().item()
                    current = torch.round(output * 255) / 255
                counts[group] += len(reference)
                print(f"Group {group + 1}/{args.group_num}, batch {batch}: {counts[group]} images, {args.hops} hops complete", flush=True)
    per_group = sums / counts[:, None]
    means = sums.sum(0) / counts.sum()
    with (args.output_dir / "per_hop.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("hop", "samples", "psnr_db"))
        writer.writerows((hop + 1, int(counts.sum()), float(value)) for hop, value in enumerate(means))
    np.save(args.output_dir / "psnr_per_group.npy", per_group)
    config["samples_per_group"] = counts.tolist()
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    vector = "PSNR = [" + ", ".join(f"{value:.2f}" for value in means) + "]\n"
    (args.output_dir / "psnr.txt").write_text(vector)
    print(vector, end="")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main(get_args())

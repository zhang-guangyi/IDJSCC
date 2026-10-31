import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from utils import *
from sim_engine_changing_snr import *
from app_common import seed_initial, build_cifar10_loaders, build_model


def safe_load_checkpoint(model, checkpoint_state):
    model_state = model.state_dict()
    filtered_state = {}
    skipped_keys = []

    for key, value in checkpoint_state.items():
        normalized_key = key[7:] if key.startswith('module.') else key
        if normalized_key in model_state and model_state[normalized_key].shape == value.shape:
            filtered_state[normalized_key] = value
        else:
            skipped_keys.append(key)

    incompatible = model.load_state_dict(filtered_state, strict=False)

    print(f'Loaded params: {len(filtered_state)}')
    print(f'Missing params: {len(incompatible.missing_keys)}')
    print(f'Skipped params: {len(skipped_keys)}')
    if incompatible.missing_keys:
        print('Missing (first 8):', incompatible.missing_keys[:8])
    if skipped_keys:
        print('Skipped (first 8):', skipped_keys[:8])

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
    parser.add_argument('--test_batch_size', default=100, type=int, help='Testing batchsize')
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    parser.add_argument('--model', default='Inverse', choices=['Inverse'], type=str, help='model name')
    parser.add_argument('--channel', default='awgn', type=str, help='model name')
    parser.add_argument('--use_denoise', action='store_true', help='Use InverseNetDenoise when model is Inverse')

    parser.add_argument('--output_dir', default='output', help='Path where to save, empty for no saving')

    parser.add_argument('--cr', default=4, type=float, help='Compression ratio for the image')

    parser.add_argument('--test', action='store_true', help='Whether only test the model')
    parser.add_argument('--M', default=0, type=float, help='Compression ratio for the image')
    parser.add_argument('--snr_start', required=True, type=float, help='First-hop SNR in dB for a linear 30-hop schedule (Experiment_1)')
    parser.add_argument('--snr_end', required=True, type=float, help='Last-hop SNR in dB for a linear 30-hop schedule (Experiment_1)')
    parser.add_argument('--x_min', default=-3.0, type=float, help='Lower clipping bound for uniform quantization')
    parser.add_argument('--x_max', default=3.0, type=float, help='Upper clipping bound for uniform quantization')
    
    parser.add_argument('--test_number', default='', choices=['Experiment_1'],
                        type=str, help='Eval number of the experiment')
    parser.add_argument('--group_num', default=1, type=int, help='How many independent test turns to run')
    parser.add_argument('--sample_num', default=1000, type=int, help='Number of samples used by simulation functions')
    
    args = parser.parse_args()
    return args

def main(args):
    if not np.isfinite([args.snr_start, args.snr_end]).all():
        raise ValueError('SNR endpoints must be finite')
    hop_snrs = np.linspace(args.snr_start, args.snr_end, 30).tolist()
    if args.x_min >= args.x_max:
        raise ValueError('--x_min must be smaller than --x_max')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_initial(13890)

    print('==> Preparing data..')
    _, testloader = build_cifar10_loaders(
        train_batch_size=None,
        test_batch_size=args.test_batch_size,
        test_shuffle=True,
    )

    net = build_model(
        args.model,
        args.cr,
        args.channel,
        device,
        use_denoise=args.use_denoise,
        x_min=args.x_min,
        x_max=args.x_max,
    )
    net.to(device)
    try:
        net.eval()
        LATENCY_TRIALS = 50
        dummy = torch.randn(1, 3, 32, 32, device=device)
        with torch.no_grad():
            for _ in range(10):
                _ = net(dummy, snr=18)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(LATENCY_TRIALS):
                _ = net(dummy, snr=18)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            t1 = time.time()
        print('Dummy input latency (batch=1): avg %.3f ms over %d runs' % ((t1 - t0) / LATENCY_TRIALS * 1000.0, LATENCY_TRIALS))
    except Exception:
        pass
    
    if args.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume, map_location=device)
        checkpoint_state = checkpoint.get('net', checkpoint)
        safe_load_checkpoint(net, checkpoint_state)

    criterion = nn.MSELoss()

    if args.group_num < 1:
        raise ValueError('--group_num must be >= 1')

    Group_Num = args.group_num
    Sample_Num = args.sample_num
    print('group_num', Group_Num)
    if args.test:
        if   args.test_number == 'Experiment_1':
            Experiment_1(net, device, testloader, criterion, Group_Num, Sample_Num, args.M, hop_snrs)
            
if __name__ == '__main__':
    opts = get_args()
    disp_args(opts)
    main(opts)

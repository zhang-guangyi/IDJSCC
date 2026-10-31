"""Main entry for training/testing the Small_model semantic communication pipeline."""
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import sys

from utils import *
from engine import *
from app_common import seed_initial, build_cifar10_loaders, build_model

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=200, type=int, help='Training batchsize')
    parser.add_argument('--test_batch_size', default=100, type=int, help='Testing batchsize')
    parser.add_argument('--num_epochs', default=900, type=int, help='Number of taining epochs')
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    parser.add_argument('--model', default='Inverse', choices=['Inverse'], type=str, help='model name')
    parser.add_argument('--channel', default='awgn', type=str, help='model name')
    parser.add_argument('--use_denoise', action='store_true', help='Use InverseNetDenoise when model is Inverse')

    parser.add_argument('--output_dir', default='outputs', help='Path where checkpoints and logs are saved')
    parser.add_argument('--save_ckpt', default=True, help='Wether save the checpoint')
    parser.add_argument('--loop', action='store_true', help='Wether save the checpoint')
    parser.add_argument('--save_freq', default=4, type=int, help='Frequency to save tyhe model')

    parser.add_argument('--cr', default=4, type=float, help='Compression ratio for the image')
    parser.add_argument('--snr', default=10, type=float, help='Compression ratio for the image')
    parser.add_argument('--M', default=0, type=float, help='Compression ratio for the image')
    parser.add_argument('--denoise_lambda', default=0.12, type=float, help='Weight for denoising loss when model returns it')
    parser.add_argument('--train_hops', default=15, type=int)
    parser.add_argument('--test_hops', default=20, type=int)
    parser.add_argument('--gamma', default=1.0, type=float)

    parser.add_argument('--test', action='store_true', help='Whether only test the model')
    
    parser.add_argument('--gendata', action='store_true', help='Whether only generate the data of GAN model')
    args = parser.parse_args()
    return args

class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


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

def main(args):
    if args.train_hops < 1 or args.test_hops < 1 or args.gamma <= 0:
        raise ValueError('Hop counts and gamma must be positive')
    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(f"./{args.output_dir}", 'log.txt'), sys.stdout)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_initial(1010)
    best_mse = 1000
    start_epoch = 0

    print('==> Preparing data..')
    trainloader, testloader = build_cifar10_loaders(
        train_batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        test_shuffle=False,
    )

    net = build_model(args.model, args.cr, args.channel, device, use_denoise=args.use_denoise)


    if args.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume, map_location=device)
        checkpoint_state = checkpoint.get('net', checkpoint)
        safe_load_checkpoint(net, checkpoint_state)
        best_mse = np.inf
        start_epoch = checkpoint.get('epoch', 0)

    criterion = nn.MSELoss()              
    optimizer =  optim.AdamW(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if args.test:
        if args.loop:
            mse, psnr = test_idjscc_loop(net, device, testloader, criterion, args.snr, args.M,
                                        denoise_lambda=args.denoise_lambda, tx_time=args.test_hops)
        else:
            mse, psnr = test_idjscc(net, device, testloader, criterion, args.snr, args.M,
                                   denoise_lambda=args.denoise_lambda)
        print('Performance on current checkpoint is MSE: %.3f , PSNR: %.3f dB' % (mse, psnr))
        exit(0)

    for epoch in range(start_epoch, start_epoch+args.num_epochs):
        if args.loop:
            train_idjscc_loop(epoch, net, device, trainloader, criterion, optimizer, args.snr, args.M,
                              denoise_lambda=args.denoise_lambda, tx_time=args.train_hops, gamma=args.gamma)
        else:
            train_idjscc(epoch, net, device, trainloader, criterion, optimizer, args.snr, args.M, denoise_lambda=args.denoise_lambda)
        scheduler.step()

        if testloader is not None:
            if args.loop:
                mse,psnr = test_idjscc_loop(net, device, testloader, criterion, args.snr, args.M,
                                            denoise_lambda=args.denoise_lambda, tx_time=args.test_hops)
            else:
                mse,psnr = test_idjscc(net, device, testloader, criterion, args.snr, args.M, denoise_lambda=args.denoise_lambda)
            if args.output_dir and args.save_ckpt:
                if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.num_epochs:
                    print('Saving model..')
                    save_checkpont(args, net, mse, epoch, best_model=False)

            if mse < best_mse:
                print('Saving best model..')
                save_checkpont(args, net, mse, epoch, best_model=True)
                best_mse = mse
            
if __name__ == '__main__':
    opts = get_args()
    disp_args(opts)
    main(opts)

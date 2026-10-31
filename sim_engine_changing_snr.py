import torch
import torch.nn as nn
import numpy as np
import math

from utils import *

from model_utils import *


def _split_model_output(outputs):
    if isinstance(outputs, tuple):
        if len(outputs) == 2:
            return outputs[0], outputs[1]
        return outputs[0], None
    return outputs, None


def _stabilize_recursive_output(outputs):
    finite_mask = torch.isfinite(outputs)
    sanitized = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)
    return sanitized.clamp(0.0, 1.0), bool(finite_mask.all())


def _batch_image_mse(outputs, targets, criterion):
    transmitted_image = torch.round(targets.clamp(0.0, 1.0) * 255.0)
    received_image = torch.round(outputs.clamp(0.0, 1.0) * 255.0)
    return criterion(received_image, transmitted_image).item()

  
def test_idjscc_tx(net, device, dataloader, criterion, sample_num, M, hop_snrs):
    net.eval()

    tx_time = 30
    if len(hop_snrs) != tx_time:
        raise ValueError('hop_snrs must contain exactly 30 SNR values')
    print('Per-hop SNR (dB):', ', '.join(f'{value:.6f}' for value in hop_snrs))
    
    PSNR_Group = np.zeros(tx_time)
    with torch.no_grad():
        warned_nonfinite = False
        MSE_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), inputs.to(device)
            for tx in range(tx_time):
                hop_snr = hop_snrs[tx]
                outputs = net(inputs, snr=hop_snr, M=M)
                outputs, _ = _split_model_output(outputs)
                outputs, outputs_were_finite = _stabilize_recursive_output(outputs)
                if (not outputs_were_finite) and (not warned_nonfinite):
                    print('Warning: non-finite outputs detected during recursive simulation; values were clamped to [0, 1].')
                    warned_nonfinite = True

                loss = criterion(outputs, targets)
                test_loss = loss.item()
                MSE_loss += _batch_image_mse(outputs, targets, criterion)
                predictions = torch.chunk(outputs, chunks=outputs.size(0), dim=0)
                target = torch.chunk(targets, chunks=inputs.size(0), dim=0)

                psnr_vals = calc_psnr(predictions, target)
                PSNR = torch.mean(torch.tensor(psnr_vals)).item()

                # inputs = outputs
                image = outputs.clamp(0, 1) * 255.0
                inputs = torch.round(image) / 255.0
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Mse: %.3f | PSNR: %.2f'
                            % (test_loss, MSE_loss/(batch_idx+1), PSNR))
                PSNR_Group[tx] += PSNR
        PSNR_Group /= (batch_idx+1) 
    return PSNR_Group


def Experiment_1(net, device, testloader, criterion, Group_Num, Sample_Num, M, hop_snrs):
    PSNR = 0.
    np.set_printoptions(precision=2)
    torch.set_printoptions(precision=2)
    for i in range(Group_Num):
        print("Current test turn: %d" %(i))
        PSNR += test_idjscc_tx(net, device, testloader, criterion, Sample_Num, M, hop_snrs)
    print('PSNR : ', torch.tensor(PSNR/Group_Num))
    exit(0)


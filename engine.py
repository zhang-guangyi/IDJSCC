import torch
import numpy as np
import os
import math
from utils import *


def _split_model_output(outputs):
    if isinstance(outputs, tuple):
        if len(outputs) == 2:
            return outputs[0], outputs[1]
        return outputs[0], None
    return outputs, None
  
def train_idjscc(epoch, net, device, dataloader, criterion, optimizer, snr, M, denoise_lambda=0.0):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    recon_loss_total = 0
    denoise_loss_total = 0
    MSE_loss = 0
    print('training snr:', snr)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), inputs.to(device)
        optimizer.zero_grad()
        outputs = net(inputs,snr=snr, M=M)
        outputs, denoise_loss = _split_model_output(outputs)

        recon_loss = criterion(outputs, targets)
        loss = recon_loss
        if denoise_loss is not None:
            loss = loss + denoise_lambda * denoise_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        recon_loss_total += recon_loss.item()
        if denoise_loss is not None:
            denoise_loss_total += denoise_loss.item()
        
        transmitted_image = torch.tensor((inputs*255).detach().cpu().numpy().astype(int).clip(0,255))+0.
        received_image = torch.tensor((outputs*255).detach().cpu().numpy().astype(int).clip(0,255))+0.
        MSE = criterion(transmitted_image, received_image)
        MSE_loss += MSE.item()

        PSNR = 10 * math.log10(255.0**2/(MSE_loss/(batch_idx+1)))
        if denoise_loss is not None:
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Recon: %.3f | Denoise: %.3f | SNR: %.1f | Mse: %.3f | PSNR: %.2f'
                         % (train_loss/(batch_idx+1), recon_loss_total/(batch_idx+1), denoise_loss_total/(batch_idx+1), snr, MSE_loss/(batch_idx+1), PSNR))
        else:
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | SNR: %.1f | Mse: %.3f | PSNR: %.2f'
                         % (train_loss/(batch_idx+1), snr, MSE_loss/(batch_idx+1), PSNR))

def test_idjscc(net, device, dataloader, criterion, snr, M, denoise_lambda=0.0):
    net.eval()
    test_loss = 0
    recon_loss_total = 0
    denoise_loss_total = 0
    MSE_loss = 0
    psnr_list = []
    psnr = 0
    print('test snr:', snr)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), inputs.to(device)
            outputs = net(inputs, snr=snr, M=M)
            outputs, denoise_loss = _split_model_output(outputs)
            recon_loss = criterion(outputs, targets)
            loss = recon_loss
            if denoise_loss is not None:
                loss = loss + denoise_lambda * denoise_loss
            test_loss += loss.item()
            recon_loss_total += recon_loss.item()
            if denoise_loss is not None:
                denoise_loss_total += denoise_loss.item()
            MSE_loss += test_loss
            predictions = torch.chunk(outputs, chunks=outputs.size(0), dim=0)
            target = torch.chunk(inputs, chunks=inputs.size(0), dim=0)

            psnr_vals = calc_psnr(predictions, target)
            psnr_list.extend(psnr_vals)
            psnr += torch.mean(torch.tensor(psnr_vals)).item()
            if denoise_loss is not None:
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Recon: %.3f | Denoise: %.3f | SNR: %.1f | Mse: %.3f | PSNR: %.2f ' 
                            % (test_loss/(batch_idx+1), recon_loss_total/(batch_idx+1), denoise_loss_total/(batch_idx+1), snr, MSE_loss/(batch_idx+1), psnr/(batch_idx+1)))
            else:
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | SNR: %.1f | Mse: %.3f | PSNR: %.2f ' 
                            % (test_loss/(batch_idx+1), snr, MSE_loss/(batch_idx+1), psnr/(batch_idx+1)))
    MSE = MSE_loss/(batch_idx+1)
    psnr_list = torch.tensor(psnr_list)
    PSNR = torch.mean(psnr_list).item()
    return MSE, PSNR


def train_idjscc_loop(epoch, net, device, dataloader, criterion, optimizer, snr, M,
                      denoise_lambda=0.0, tx_time=15, gamma=1.0):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    recon_loss_total = 0
    denoise_loss_total = 0
    MSE_loss = 0
    print('train_snr:', snr)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), inputs.to(device)
        loss = 0
        optimizer.zero_grad()
        for tx in range(tx_time):
            outputs = net(inputs,snr=snr, M=M)
            outputs, denoise_loss = _split_model_output(outputs)
            recon_loss = criterion(outputs, targets)
            total_step_loss = recon_loss
            if denoise_loss is not None:
                total_step_loss = total_step_loss + denoise_lambda * denoise_loss
                denoise_loss_total += denoise_loss.item()
            recon_loss_total += recon_loss.item()
            loss += total_step_loss * gamma**(tx_time-tx)
            if tx<(tx_time-1):
                inputs = outputs[:] 
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        transmitted_image = torch.tensor((targets*255).detach().cpu().numpy().astype(int).clip(0,255))+0.
        received_image = torch.tensor((outputs*255).detach().cpu().numpy().astype(int).clip(0,255))+0.
        
        MSE = criterion(transmitted_image, received_image)
        MSE_loss += MSE.item()

        PSNR = 10 * math.log10(255.0**2/(MSE_loss/(batch_idx+1)))
        if denoise_loss_total > 0:
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Recon: %.3f | Denoise: %.3f | SNR: %.1f | Final Mse: %.3e | Final PSNR: %.2f'
                        % (train_loss/(batch_idx+1), recon_loss_total/((batch_idx+1)*tx_time), denoise_loss_total/((batch_idx+1)*tx_time), snr, MSE_loss/(batch_idx+1), PSNR))
        else:
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | SNR: %.1f | Final Mse: %.3e | Final PSNR: %.2f'
                        % (train_loss/(batch_idx+1), snr, MSE_loss/(batch_idx+1), PSNR))



def test_idjscc_loop(net, device, dataloader, criterion, snr, M,
                     denoise_lambda=0.0, tx_time=20):
    net.eval()
    test_loss = 0
    recon_loss_total = 0
    denoise_loss_total = 0
    MSE_loss = 0
    psnr_list = []
    psnr = 0
    print('test_snr:', snr)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), inputs.to(device)
            for tx in range(tx_time):
                outputs = net(inputs, snr=snr, M=M)
                outputs, denoise_loss = _split_model_output(outputs)
                recon_loss = criterion(outputs, targets)
                loss = recon_loss
                if denoise_loss is not None:
                    loss = loss + denoise_lambda * denoise_loss
                    denoise_loss_total += denoise_loss.item()
                recon_loss_total += recon_loss.item()
                test_loss += loss.item()
                MSE_loss += test_loss
                predictions = torch.chunk(outputs, chunks=outputs.size(0), dim=0)
                target = torch.chunk(targets, chunks=inputs.size(0), dim=0)
                inputs = outputs[:]
            psnr_vals = calc_psnr(predictions, target)
            psnr_list.extend(psnr_vals)
            psnr += torch.mean(torch.tensor(psnr_vals)).item()
            if denoise_loss_total > 0:
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Recon: %.3f | Denoise: %.3f | SNR: %.1f | Mse: %.3f | PSNR: %.2f ' 
                            % (test_loss/(batch_idx+1), recon_loss_total/((batch_idx+1)*tx_time), denoise_loss_total/((batch_idx+1)*tx_time), snr, MSE_loss/(batch_idx+1), psnr/(batch_idx+1)))
            else:
                progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | SNR: %.1f | Mse: %.3f | PSNR: %.2f ' 
                            % (test_loss/(batch_idx+1), snr, MSE_loss/(batch_idx+1), psnr/(batch_idx+1)))
    MSE = MSE_loss/(batch_idx+1)
    psnr_list = torch.tensor(psnr_list)
    PSNR = torch.mean(psnr_list).item()
    return MSE, PSNR



def save_img(source, target, decoded, batch_size=4, epoch=10):
    source = (source*255).detach().cpu().numpy().astype(int).clip(0,255)
    target = (target*255).detach().cpu().numpy().astype(int).clip(0,255)
    decoded= (decoded*255).detach().cpu().numpy().astype(int).clip(0,255)
    import imageio
    def merge_images_two(sources, targets, decoded):
        _, _, h, w = sources.shape
        row = int(np.sqrt(batch_size))

        merged = np.zeros([3, row * h, row * w * 2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
            merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
        return merged.transpose(1, 2, 0)
    
    def merge_images_thr(sources, targets, decoded):
        _, _, h, w = sources.shape
        row = int(np.sqrt(batch_size))
        merged = np.zeros([3, row * h, row * w * 3])
        for idx, (s, t, de) in enumerate(zip(sources, targets, decoded)):
            i = idx // row
            j = idx % row
            merged[:, i * h:(i + 1) * h, (j * 3) * h:(j * 3 + 1) * h] = s
            merged[:, i * h:(i + 1) * h, (j * 3 + 1) * h:(j * 3 + 2) * h] = t
            merged[:, i * h:(i + 1) * h, (j * 3 + 2) * h:(j * 3 + 3) * h] = de
        return merged.transpose(1, 2, 0)
    
    if not os.path.exists('imgs_compare'):
        os.makedirs('imgs_compare')
    path = os.path.join('imgs_compare', f'epoch-{epoch:03d}.jpg')
    merged = merge_images_thr(source, target, decoded)
    imageio.imwrite(path, merged)
      

import torch
import itertools
import torchvision
from modules import *
from channel import *
import torch.nn as nn
from vit_utils import *
from model_utils import *
from einops import rearrange
import torch.nn.functional as F
from torchvision import transforms


def uniform_quantize(x, M, training=True, x_min=-3.0, x_max=3.0, eps=1e-8):
    x_min = torch.tensor(x_min, device=x.device, dtype=x.dtype)
    x_max = torch.tensor(x_max, device=x.device, dtype=x.dtype)
    scale = (x_max - x_min) / (M - 1)
    scale = torch.clamp(scale, min=eps)
    x_clipped = torch.clamp(x, x_min, x_max)
    x_q = torch.round((x_clipped - x_min) / scale)
    x_q = torch.clamp(x_q, 0, M - 1)
    x_dequant = x_q * scale + x_min
    if training:
        x_dequant = x + (x_dequant - x).detach()
    return x_dequant


class ViTSCNet(nn.Module):
    def __init__(self,
                 compression_ratio=3, img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, 
                 encoder_embed_dim=768, encoder_depth=12,encoder_num_heads=12, decoder_num_classes=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4., 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, init_values=0.,use_learnable_pos_emb=False,num_classes=0, 
                 channel='awgn',
                 ):
        super().__init__()
        embedding_len = (patch_size) ** 2 * encoder_in_chans
        self.encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, 
                                num_classes=encoder_num_classes, embed_dim=encoder_embed_dim,depth=encoder_depth,
                                num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, 
                                drop_path_rate=drop_path_rate,norm_layer=norm_layer, init_values=init_values,
                                use_learnable_pos_emb=use_learnable_pos_emb)
        self.head = nn.Linear(decoder_embed_dim, embedding_len)
        self.encoder_to_channel = nn.Linear(encoder_embed_dim, int(embedding_len//compression_ratio))
        self.channel_to_decoder = nn.Linear(int(embedding_len//compression_ratio), decoder_embed_dim)
        self.decoder  = ViTDecoder(patch_size=patch_size, num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, embed_dim=decoder_embed_dim, depth=decoder_depth,num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,  norm_layer=norm_layer, init_values=init_values)
        
        self.channel = Channels()
        self.channle_type = channel
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x, is_train=True, snr=18, M=0):
        x = self.encoder(x)
        x = self.encoder_to_channel(x)
        x, _ = power_norm_batchwise(x)
        noise_std = 10 ** (-snr / 20)
        if self.channle_type == 'awgn':
            x = self.channel.AWGN(x, noise_std)
        elif self.channle_type == 'rayleigh':
            x = self.channel.Rayleigh(x, noise_std)
        else:
            raise ValueError(f'Unsupported channel type: {self.channle_type}')
        
        x = self.channel_to_decoder(x)

        if M > 1:
            x = uniform_quantize(x, M=M, training=is_train)
        x = self.decoder(x)
        x = self.head(x)
        x = rearrange(x, 'b n (p c) -> b n p c', c=3)
        x = rearrange(x, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=4, p2=4, h=8, w=8)
        return x


def qam_mapper(signal, M,d):
    device=signal.device
    signal_r = signal/(2*d)+(math.sqrt(M)-1)/2
    signal_r=torch.round(signal_r).to(device)
    signal_r=torch.clamp(signal_r,0,math.sqrt(M)-1).to(device)
    signal_r=(2*signal_r-math.sqrt(M)+1)*d
    return signal_r


class DeepSCNet(nn.Module):
    def __init__(self, filters, middle_kernel=64, channel='awgn'):
        super().__init__()
        # Encoder
        self.filters = filters
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, middle_kernel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(middle_kernel)
        self.conv3 = nn.Conv2d(middle_kernel, middle_kernel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(middle_kernel)
        self.conv4 = nn.Conv2d(middle_kernel, middle_kernel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(middle_kernel)
        self.conv5 = nn.Conv2d(middle_kernel, self.filters, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.filters)

        # Decoder
        self.tconv1 = nn.ConvTranspose2d(self.filters, middle_kernel, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.tbn1 = nn.BatchNorm2d(middle_kernel)
        self.tconv2 = nn.ConvTranspose2d(middle_kernel, middle_kernel, kernel_size=3, stride=1, padding=1, bias=False)
        self.tbn2 = nn.BatchNorm2d(middle_kernel)
        self.tconv3 = nn.ConvTranspose2d(middle_kernel, middle_kernel, kernel_size=3, stride=1, padding=1, bias=False)
        self.tbn3 = nn.BatchNorm2d(middle_kernel)
        self.tconv4 = nn.ConvTranspose2d(middle_kernel, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.tbn4 = nn.BatchNorm2d(16)
        self.tconv5 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        
        self.channel = Channels()
        self.channle_type = channel

    def encoder(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        return x

    def decoder(self, z):
        z = self.relu(self.tbn1(self.tconv1(z)))
        z = self.relu(self.tbn2(self.tconv2(z)))
        z = self.relu(self.tbn3(self.tconv3(z)))
        z = self.relu(self.tbn4(self.tconv4(z)))
        z = self.tconv5(z).view(-1, 3, 32, 32) 
        return z

    def forward(self, x, is_train=True, snr=18):
        x = self.encoder(x)
        x,_ = power_norm_batchwise(x, power=1)
        noise_std = 10**(-snr/20)
        if self.channle_type == 'awgn':
            x = self.channel.AWGN(x, noise_std)
        elif self.channle_type == 'rayleigh':
            x = self.channel.Rayleigh(x, noise_std)
        else:
            raise ValueError(f'Unsupported channel type: {self.channle_type}')
        x = self.decoder(x)
        return x


class Cycle_GAN(nn.Module):
    def __init__(self, args, middle_kernel=32):
        super().__init__()
        # Initial criterion
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Initial network
        self.netG_A2B = Generator(middle_kernel=middle_kernel)
        self.netG_B2A = Generator(middle_kernel=middle_kernel)
        self.netD_A = Discriminator(middle_kernel=middle_kernel)
        self.netD_B = Discriminator(middle_kernel=middle_kernel)
        
        # Initial optimizer
        self.optimizer_G =   torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=args.lr, betas=(0.5, 0.999))
    

        # Initial scheduler
        self.lr_scheduler_G =   torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=200)
        self.lr_scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=200)
       
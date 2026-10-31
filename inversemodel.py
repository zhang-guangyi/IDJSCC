from compressai.models.utils import conv, deconv
import torch.nn as nn
from channel import *
from model_utils import *
from compressai.layers import conv3x3, subpel_conv3x3, GDN, ResidualBlock, MaskedConv2d
from torch.nn.parameter import Parameter
from torch.autograd.profiler import record_function
import geotorch
try:
  import geotorch.parametrize as geotorch_parametrize
except ModuleNotFoundError:
  # Recent geotorch versions use PyTorch's parametrization registry.
  import torch.nn.utils.parametrize as geotorch_parametrize
from model import qam_mapper

iGDN = lambda N: GDN(N, inverse=True)
cat = lambda *x_list: torch.cat(x_list, dim=1)


import torch

def uniform_quantize(x, M, training=True, x_min=-3., x_max=3, eps=1e-8):
    x_min = torch.tensor(x_min, device=x.device, dtype=x.dtype)
    x_max = torch.tensor(x_max, device=x.device, dtype=x.dtype)
    # print(x.min().item(), x.max().item())
    if M <= 1:
        return x
    scale = (x_max - x_min) / (M - 1)
    scale = torch.clamp(scale, min=eps)
    x_clipped = torch.clamp(x, x_min, x_max)
    x_q = torch.round((x_clipped - x_min) / scale)
    x_q = torch.clamp(x_q, 0, M - 1)
    x_dequant = x_q * scale + x_min
    if training:
      # Use clipped STE: gradient flows only inside clipping range, reducing low-bit instability.
      x_dequant = x_clipped + (x_dequant - x_clipped).detach()
    return x_dequant


class single(nn.Module):
    def __init__( self, N=128, M=192, channel = 'awgn', **kwargs ):
        super().__init__()
        C = [N, N, N]
        self.quant_x_min = kwargs.get('x_min', -3.0)
        self.quant_x_max = kwargs.get('x_max', 3.0)
        
        # Encoder
        self.g_a = nn.Sequential(
                 conv(3,    C[0]), 
        GDN(C[0]), conv(C[0], C[1]),
        GDN(C[1]), conv(C[1], C[2]),
        GDN(C[2]), conv(C[2], M),
        )
        # Decoder
        self.g_s = nn.Sequential(
                    deconv(M,    C[2]),
        iGDN(C[2]), deconv(C[2], C[1]),
        iGDN(C[1]), deconv(C[1], C[0]),
        iGDN(C[0]), deconv(C[0], 3),
        )
        self.relu = nn.PReLU()
        self.channel = Channels()
        self.channle_type = channel

    def forward(self, x, is_train=True, snr=18, M=0):
        x = self.g_a(x)
        y = x
        # # # AWGN noise
        y,power = power_norm_batchwise(y, power=1)
        noise_std = 10**(-snr/20)
        if self.channle_type=='awgn':
          y = self.channel.AWGN(y, noise_std) 
        elif self.channle_type=='rayleigh':
          y = self.channel.Rayleigh(y, noise_std)
        y = y*torch.sqrt(power.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        if M > 1:
            x = uniform_quantize(
                x,
                M=M,
                training=is_train,
                x_min=self.quant_x_min,
                x_max=self.quant_x_max,
            )
        x_pred = self.g_s(y)
        return x_pred


class DenoiseResBlock(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.body = nn.Sequential(
      conv3x3(channels, channels),
            nn.LeakyReLU(inplace=True),
      conv3x3(channels, channels),
        )
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + x)


class DenoiseResNet(nn.Module):
    def __init__(self, channels, num_blocks=1, kernel_size=3):
        super().__init__()
        self.stem = nn.Sequential(
      conv3x3(channels, channels),
            nn.LeakyReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[DenoiseResBlock(channels, kernel_size=kernel_size) for _ in range(num_blocks)])

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x

def qam_mapper(signal, M,d):
    device=signal.device
    signal_r = signal/(2*d)+(math.sqrt(M)-1)/2
    signal_r=torch.round(signal_r).to(device)
    signal_r=torch.clamp(signal_r,0,math.sqrt(M)-1).to(device)
    signal_r=(2*signal_r-math.sqrt(M)+1)*d
    return signal_r

class single_denoise(single):
    def __init__(self, N=128, M=192, channel='awgn', **kwargs):
        super().__init__(N, M, channel=channel, **kwargs)

        num_blocks = kwargs.get('denoise_blocks', 1)
        denoise_kernel = kwargs.get('cnnk', 5)
        self.en = DenoiseResNet(M, num_blocks=num_blocks, kernel_size=denoise_kernel)
        self.de = DenoiseResNet(M, num_blocks=num_blocks, kernel_size=denoise_kernel)
        self.denoise_criterion = nn.MSELoss()

    def forward(self, x, is_train=True, snr=18, M=0):
        x = self.g_a(x)
        x_in = x
        if M > 1:
            x_in = uniform_quantize(
                x_in,
                M=M,
                training=is_train,
                x_min=self.quant_x_min,
                x_max=self.quant_x_max,
            )
        y = self.en(x_in) + x_in
        y, power = power_norm_batchwise(y, power=1)
        noise_std = 10**(-snr/20)
        if self.channle_type == 'awgn':
          y = self.channel.AWGN(y, noise_std)
        elif self.channle_type == 'rayleigh':
          y = self.channel.Rayleigh(y, noise_std)
        y = y * torch.sqrt(power.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        y = self.de(y) + y
        denoise_loss = self.denoise_criterion(y, x_in)
        
        # if M > 1:
        #     y = uniform_quantize(y, M=M, training=is_train)
        x_pred = self.g_s(y)
        return x_pred, denoise_loss

# class single_denoise(single):
#     def __init__(self, N=128, M=192, channel='awgn', **kwargs):
#         super().__init__(N, M, channel=channel, **kwargs)

#         num_blocks = kwargs.get('denoise_blocks', 1)
#         denoise_kernel = kwargs.get('cnnk', 5)
#         self.en = DenoiseResNet(M, num_blocks=num_blocks, kernel_size=denoise_kernel)
#         self.de = DenoiseResNet(M, num_blocks=num_blocks, kernel_size=denoise_kernel)
#         self.denoise_criterion = nn.MSELoss()

#     def forward(self, x, is_train=True, snr=18, M=0):
#         x = self.g_a(x)
#         x_in = x
#         y = self.en(x_in) + x_in
#         y, power = power_norm_batchwise(y, power=1)
#         noise_std = 10**(-snr/20)
#         if self.channle_type == 'awgn':
#           y = self.channel.AWGN(y, noise_std)
#         elif self.channle_type == 'rayleigh':
#           y = self.channel.Rayleigh(y, noise_std)
#         y = y * torch.sqrt(power.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
#         y = self.de(y) + y
#         if M > 1:
#             d=math.sqrt(3*1/(2*M-2))
#             y=torch.clamp(y,(1-math.sqrt(M))*d,(math.sqrt(M)-1)*d).to(x.device)    
#             y = y+(qam_mapper(y,M=M,d=d)-y).detach()
#         denoise_loss = self.denoise_criterion(y, x_in)
#         x_pred = self.g_s(y)
#         return x_pred, denoise_loss


          

class InverseNet(single):
  def __init__(self, N=128, M=16, **kwargs):
    super().__init__(N, M, **kwargs)
    self.g_a = None
    self.g_s = None
    C = [int(_c) for _c in kwargs.get('C', [150])]
    P = kwargs.get('pk', 1)
    cnnk = kwargs.get('cnnk', 5)
    c_nn = lambda c: nn.Conv2d(c, c, cnnk, padding=cnnk//2, stride=1)
    g = [
      [ offset() ],

      [ pconv2d(3,    2*P, C[0], 1*P, **kwargs) ],
      [ channel_coup(C[0], c_nn), channel_coup(C[0], c_nn, shuffle=True) ],
      [ channel_coup(C[0], GDN),  channel_coup(C[0], GDN,  shuffle=True) ],
      [ pconv2d(C[0], 2*P, M,    1*P, **kwargs) ],
      [ channel_coup(M,    c_nn), channel_coup(M,    c_nn, shuffle=True) ],
    ]
    self.g = nn.ModuleList([__g for _g in g if _g for __g in _g])
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data = m.weight.data * 0.1
        m.bias  .data = m.bias  .data * 0
  
  def g_a(self, x):
    for m in self.g:
      x = m(x)
    return x
  
  def g_s(self, x):
    for m in self.g[::-1]:
      x = m.forward_p(x)
    return x

  def forward(self, x, is_train=True, snr=18, M=0):
    # The same orthogonal factors are read by the forward, inverse, and
    # zero-space paths. Cache their parametrizations for this call only.
    with geotorch_parametrize.cached():
      return super().forward(x, is_train=is_train, snr=snr, M=M)


class InverseNetDenoise(single_denoise):
  def __init__(self, N=128, M=16, **kwargs):
    super().__init__(N, M, **kwargs)
    self.g_a = None
    self.g_s = None
    C = [int(_c) for _c in kwargs.get('C', [150])]
    P = kwargs.get('pk', 1)
    cnnk = kwargs.get('cnnk', 5)
    c_nn = lambda c: nn.Conv2d(c, c, cnnk, padding=cnnk//2, stride=1)
    g = [
      [ offset() ],

      [ pconv2d(3,    2*P, C[0], 1*P, **kwargs) ],
      [ channel_coup(C[0], c_nn), channel_coup(C[0], c_nn, shuffle=True) ],
      [ channel_coup(C[0], GDN),  channel_coup(C[0], GDN,  shuffle=True) ],
      [ pconv2d(C[0], 2*P, M,    1*P, **kwargs) ],
      [ channel_coup(M,    c_nn), channel_coup(M,    c_nn, shuffle=True) ],
    ]
    self.g = nn.ModuleList([__g for _g in g if _g for __g in _g])
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data = m.weight.data * 0.1
        m.bias  .data = m.bias  .data * 0
  
  def g_a(self, x):
    for m in self.g:
      x = m(x)
    return x
  
  def g_s(self, x):
    for m in self.g[::-1]:
      x = m.forward_p(x)
    return x

  def forward(self, x, is_train=True, snr=18, M=0):
    with geotorch_parametrize.cached():
      return super().forward(x, is_train=is_train, snr=snr, M=M)

class pconv2d(nn.Module):
  # (ci, ki, ki) <-> (co, ko, ko)
  # stride === kernel size
  def __init__(self, ci, ki, co, ko, **kwargs):
    super().__init__()
    self.ci = ci
    self.ki = ki
    self.co = co
    self.ko = ko
    self.A = A_lazy(co * ko**2, ci * ki**2)
    self.zero_path = kwargs.get('zero_path', True)
    if self.zero_path:
      zn = kwargs.get('ZN', 0)
      if co * ko**2 < ci * ki**2: # surjective A
        self.zp_func = ResidualBlockUpsample(co, ci) if zn == 0 else nn.Sequential(ResidualBlockUpsample(co, zn), ResidualBlock(zn, ci))
      elif co * ko**2 > ci * ki**2: # surjective A+
        self.zp_func = ResidualBlockWithStride(ci, co) if zn == 0 else nn.Sequential(ResidualBlockWithStride(ci, zn), ResidualBlock(zn, co))
    self.bias = Parameter(torch.randn(co)) if kwargs.get('bias', True) else None
  
  def cal_y(self, x):
    with record_function('### range_path'):
      y = fold(self.A.w @ unfold(x, self.ki), self.ko)
      if self.bias is not None:
        y = y + self.bias.view(1, -1, *[1 for _ in range(y.ndim - 2)])
    return y

  def cal_x(self, y):
    with record_function('### range_path'):
      if self.bias is not None:
        y = y - self.bias.view(1, -1, *[1 for _ in range(y.ndim - 2)])
      x = fold(self.A.w_pinv @ unfold(y, self.ko), self.ki)
    return x

  def cal_y_zp(self, x):
    with record_function('### zero_path'):
      if (self.zero_path) and (self.co * self.ko**2 > self.ci * self.ki**2):
        y_zp = self.zp_func(x)
        y_zp = y_zp - fold(self.A.project_output(unfold(y_zp, self.ko)), self.ko)
        # print('a')
      else:
        y_zp = 0
        # print('b')
    return y_zp
  
  def cal_x_zp(self, y):
    with record_function('### zero_path'):
      if (self.zero_path) and (self.co * self.ko**2 < self.ci * self.ki**2):
        x_zp = self.zp_func(y)
        x_zp = x_zp - fold(self.A.project_input(unfold(x_zp, self.ki)), self.ki)
        # print('a')
      else:
        x_zp = 0
        # print('b')
    return x_zp

  def forward(self, x):
    # print(self.cal_y_zp(x).shape)
    return self.cal_y(x) + self.cal_y_zp(x)
  
  def forward_p(self, y):
    return self.cal_x(y) + self.cal_x_zp(y)

class A_lazy(nn.Module):
  def __init__(self, O, I, **kwargs):
    super().__init__()
    self.rank = min(O, I)
    self.U = Q_lazy(O, **kwargs)
    self.V = Q_lazy(I, **kwargs)
    geotorch.orthogonal(self.U, '_Q')
    geotorch.orthogonal(self.V, '_Q')
    self._S = Parameter(torch.zeros(self.rank))
  
  @property
  def S(self):
    return (self._S.sigmoid() * 2 - 1).exp()
  
  @property
  def w(self):
    return self.U.Q[:, :self.rank] @ torch.diag(self.S)     @ self.V.Q[:, :self.rank].T
  
  @property
  def w_pinv(self):
    return self.V.Q[:, :self.rank] @ torch.diag(1.0/self.S) @ self.U.Q[:, :self.rank].T

  @property
  def w_wp(self):
    return self.U.Q[:, :self.rank] @ self.U.Q[:, :self.rank].T

  @property
  def wp_w(self):
    return self.V.Q[:, :self.rank] @ self.V.Q[:, :self.rank].T

  def project_output(self, x):
    q = self.U.Q[:, :self.rank]
    return q @ (q.T @ x)

  def project_input(self, x):
    q = self.V.Q[:, :self.rank]
    return q @ (q.T @ x)

class Q_lazy(nn.Module):
  def __init__(self, n, **kwargs):
    super().__init__()
    self._Q = Parameter(torch.randn(n, n))
  
  @property
  def Q(self):
    return self._Q

class channel_coup(nn.Module):
  def __init__(self, C, fclass, shuffle=False):
    super().__init__()
    assert C % 2 == 0
    self._fs = fclass(C // 2)
    self.ft = fclass(C // 2)
    self.fs = lambda x: (self._fs(x).sigmoid() * 2 - 1).exp() # [1/e^2, e^2]
    self.shuffle = channel_shuffle if shuffle else lambda x: x
  
  def forward(self, x):
    x1, x2 = self.shuffle(x).chunk(2, 1)
    y2 = x2 * self.fs(x1) + self.ft(x1)
    return cat(x1, y2)
  
  def forward_p(self, y):
    y1, y2 = y.chunk(2, 1)
    x2 = (y2 - self.ft(y1)) / self.fs(y1)
    return self.shuffle(cat(y1, x2))

def channel_shuffle(x):
  x1, x2 = x.chunk(2, 1)
  return cat(x2, x1)

class offset(nn.Module):
  def __init__(self, o=0.5):
    super().__init__()
    self.o = o
  
  def forward(self, x):
    return x - self.o
  
  def forward_p(self, y):
    return y + self.o
  
class ResidualBlockUpsample(nn.Module):
    
  def __init__(self, in_ch: int, out_ch: int, upsample: int = 2):
    super().__init__()
    self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
    self.leaky_relu = nn.LeakyReLU(inplace=False)
    self.conv = conv3x3(out_ch, out_ch)
    self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.subpel_conv(x)
    out = self.leaky_relu(out)
    out = self.conv(out)
    out = self.leaky_relu(out)
    identity = self.upsample(x)
    out += identity
    return out

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

class ResidualBlockWithStride(nn.Module):

  def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
    super().__init__()
    self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
    self.leaky_relu = nn.LeakyReLU(inplace=False)
    self.conv2 = conv3x3(out_ch, out_ch)
    self.skip = conv1x1(in_ch, out_ch, stride=stride)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.conv1(x)
    out = self.leaky_relu(out)
    out = self.conv2(out)
    out = self.leaky_relu(out)
    identity = self.skip(x)
    out += identity
    return out
  
def unfold(x, K):
  N, C, H, W = x.size()
  return F.unfold(x, [K, K], stride=[K, K]).reshape(N, C * K**2, H//K, W//K).permute(0, 2, 3, 1)[..., None]

def fold(x, K):
  N, nh, nw, ckk = x.squeeze(-1).size()
  C = ckk // K**2
  return x.reshape(N, nh, nw, C, K, K).permute(0, 3, 1, 4, 2, 5).reshape(N, C, nh * K, nw * K)

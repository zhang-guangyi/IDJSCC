"""CP-OFDM transport for the model's adjacent real/imaginary latent pairs."""

import math

import torch
import torch.nn.functional as F


class PerfectCSIOFDM:
    def __init__(self, nfft=64, cp_len=16, taps=8, decay=2.0, fading="rayleigh"):
        if nfft < 1 or not 0 <= cp_len <= nfft:
            raise ValueError("Require nfft >= 1 and 0 <= cp_len <= nfft")
        if not 1 <= taps <= nfft or cp_len < taps - 1:
            raise ValueError("Require 1 <= taps <= nfft and cp_len >= taps - 1")
        if not math.isfinite(decay) or decay <= 0 or fading not in ("rayleigh", "identity"):
            raise ValueError("Require positive finite decay and rayleigh/identity fading")
        self.nfft, self.cp_len, self.taps = nfft, cp_len, taps
        self.decay, self.fading = decay, fading

    def sample_impulse(self, batch, device, dtype):
        if self.fading == "identity":
            impulse = torch.zeros(batch, self.taps, device=device, dtype=dtype)
            impulse[:, 0] = 1
            return impulse
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        pdp = torch.exp(-torch.arange(self.taps, device=device, dtype=real_dtype) / self.decay)
        pdp /= pdp.sum()
        # E[sum_l |h_l|^2] = 1. Do not normalize each fading realization.
        return torch.randn(batch, self.taps, device=device, dtype=dtype) * pdp.sqrt()

    def transmit(self, signal, n_std, impulse=None):
        if signal.dtype not in (torch.float32, torch.float64) or signal.ndim < 2:
            raise ValueError("Expected batched float32/float64 latent tensor")
        if not math.isfinite(float(n_std)) or n_std < 0:
            raise ValueError("Noise standard deviation must be finite and nonnegative")
        batch = signal.shape[0]
        flat = signal.reshape(batch, -1)
        if flat.shape[1] % 2:
            raise ValueError("Each sample must contain an even number of real coordinates")
        symbols = torch.view_as_complex(flat.reshape(batch, -1, 2).contiguous())
        count = symbols.shape[1]
        padding = (-count) % self.nfft
        carriers = F.pad(symbols, (0, padding)).reshape(batch, -1, self.nfft)
        waveform = torch.fft.ifft(carriers, norm="ortho")
        if self.cp_len:
            waveform = torch.cat((waveform[..., -self.cp_len:], waveform), dim=-1)
        tx = waveform.flatten(1)
        if impulse is None:
            impulse = self.sample_impulse(batch, signal.device, symbols.dtype)
        if impulse.shape != (batch, self.taps):
            raise ValueError("Impulse must have shape (batch, taps)")
        impulse = impulse.to(device=signal.device, dtype=symbols.dtype)
        # Actual causal, linear time-domain convolution of the serialized packet.
        rx = torch.zeros_like(tx)
        for delay in range(self.taps):
            if delay == 0:
                rx += impulse[:, delay:delay + 1] * tx
            else:
                rx[:, delay:] += impulse[:, delay:delay + 1] * tx[:, :-delay]
        # torch complex randn has E|z|^2=1: each real component has variance 1/2.
        rx = rx + torch.randn_like(rx) * n_std
        useful = rx.reshape_as(waveform)[..., self.cp_len:]
        received = torch.fft.fft(useful, norm="ortho")
        # Ordinary FFT for H: unitary FFTs apply only to the data waveform.
        response = torch.fft.fft(impulse, n=self.nfft).unsqueeze(1)
        if torch.any(response.abs() == 0):
            raise RuntimeError("Exact channel null: ZF equalization is undefined")
        estimated = received / response  # perfect CSI, no pilots/estimation
        restored = estimated.flatten(1)[:, :count].contiguous()
        return torch.view_as_real(restored).reshape_as(signal)

    def AWGN(self, signal, n_std):
        """Adapter for InverseNetDenoise's existing channel transport hook.

        The model keeps its 'awgn' dispatch label; this method replaces the
        entire transport with OFDM, including fading, noise and equalization.
        """
        return self.transmit(signal, n_std)

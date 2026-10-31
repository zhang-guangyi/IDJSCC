"""Physical transport checks; run with python -m unittest discover -s test_ofdm."""

import unittest

import torch

from ofdm_channel import PerfectCSIOFDM


class OFDMChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        torch.set_num_threads(2)

    def test_noiseless_multipath_and_padding(self):
        # Non-multiple of NFFT exercises padding; several symbols exercise ISI/CP.
        signal = torch.randn(3, 2, 7, 19, dtype=torch.float64)
        channel = PerfectCSIOFDM()
        restored = channel.transmit(signal, 0)
        torch.testing.assert_close(restored, signal, atol=1e-11, rtol=1e-11)

    def test_identity_matches_original_awgn_noise_power(self):
        signal = torch.randn(1000, 128)
        symbols = torch.view_as_complex(signal.reshape(1000, -1, 2))
        signal /= symbols.abs().square().mean(1).sqrt()[:, None]
        channel = PerfectCSIOFDM(fading="identity")
        error = channel.transmit(signal, 10 ** (-10 / 20)) - signal
        # Original AWGN variance = N0/2 per real coordinate; complex SNR = 10 dB.
        self.assertAlmostEqual(error.square().mean().item(), 0.05, delta=0.001)
        self.assertLess(abs(error.mean().item()), 0.003)

    def test_known_flat_fade_zf_noise_enhancement(self):
        signal = torch.randn(1000, 128)
        channel = PerfectCSIOFDM(taps=1, cp_len=0)
        impulse = torch.full((1000, 1), 0.5j, dtype=torch.complex64)
        error = channel.transmit(signal, 0.1, impulse) - signal
        self.assertAlmostEqual(error.square().mean().item(), 0.02, delta=0.0005)

    def test_expected_channel_energy(self):
        channel = PerfectCSIOFDM()
        impulse = channel.sample_impulse(20000, "cpu", torch.complex64)
        self.assertAlmostEqual(impulse.abs().square().sum(1).mean().item(), 1., delta=0.02)
        self.assertGreater(impulse.abs().square().sum(1).std().item(), 0.1)

    def test_insufficient_cp_rejected(self):
        with self.assertRaises(ValueError):
            PerfectCSIOFDM(cp_len=6, taps=8)


if __name__ == "__main__":
    unittest.main()

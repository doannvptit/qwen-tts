import numpy as np
import torch
from torch import nn
import torchaudio


def log(t, eps=1e-5):
    return t.clamp(min=eps).log()


class WavEncoder(nn.Module):
    def __init__(
        self,
        mel_bins=100,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=24_000,
        normalize=False,
        power=1,
        norm=None,
        center=True,
        dtype=torch.bfloat16,
    ):
        super(WavEncoder, self).__init__()
        self.mel_bins = mel_bins
        self.hop_length = hop_length
        self.dtype = dtype

        self.mel_stft = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=filter_length,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=mel_bins,
            power=power,
            center=center,
            normalized=normalize,
            norm=norm,
        )

    def encode(self, audios: list[np.ndarray], device, return_audio=False):
        lens = [len(audio) for audio in audios]
        len_max = max(lens)
        audios = np.array(
            [np.pad(audio, (0, len_max - len(audio))) for audio in audios]
        )
        audios = torch.tensor(audios).to(device)
        lens = torch.tensor(lens).to(device)
        return self.forward(audios, lens, return_audio)

    def forward(self, audios: torch.Tensor, lens: torch.Tensor, return_audio=False):
        """
        audios: Tensor<B, Wav>
        lens: Tensor<B>
        return: Tensor<B, S, D>, Tensor<B>, Tensor<B, Wav>, Tensor<B>
        """
        mel = self.mel_stft(audios)
        mel = log(mel).transpose(1, 2).to(self.dtype)

        mel_lens = lens // self.hop_length + 1

        if return_audio:
            return mel, mel_lens, audios, lens
        return mel, mel_lens

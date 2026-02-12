import torch
from torch import nn
from vocos import Vocos


class WavDecoder(nn.Module):
    def __init__(self, vocos_model_id: str):
        super(WavDecoder, self).__init__()
        self.vocos = Vocos.from_pretrained(vocos_model_id)

    def forward(self, mel: torch.Tensor, lens: torch.Tensor):
        """
        mel: Tensor<B, S, D>
        lens: Tensor<B>
        return: Tensor<B, Wav>, Tensor<B>
        """
        mel = mel.transpose(1, 2)
        audios = self.vocos.decode(mel)
        rate = audios.shape[-1] // mel.shape[-1]
        lens = lens * rate
        return audios, lens

    def decode(self, mel: torch.Tensor, lens: torch.Tensor):
        """
        mel: Tensor<B, S, D>
        lens: Tensor<B>
        return: Tensor<B, Wav>, Tensor<B>
        """
        return self.forward(mel, lens)

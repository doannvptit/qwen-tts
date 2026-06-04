import torch
import torch.nn as nn
import torch.nn.functional as F


class MelSubDiscriminator(nn.Module):
    def __init__(self, in_channels=100, use_spectral_norm=False):
        super().__init__()
        norm_f = (
            nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        )

        self.conv_layers = nn.ModuleList(
            [
                norm_f(
                    nn.Conv1d(in_channels, 128, kernel_size=5, stride=1, padding=2)
                ),
                norm_f(
                    nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
                ),
                norm_f(
                    nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2)
                ),
                norm_f(
                    nn.Conv1d(512, 1024, kernel_size=5, stride=2, padding=2)
                ),
                norm_f(
                    nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)
                ),
            ]
        )

        self.out_layer = norm_f(
            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        fmap = []
        for layer in self.conv_layers:
            x = F.leaky_relu(layer(x), 0.2)
            fmap.append(x)
        out = self.out_layer(x)
        return out, fmap


class MultiScaleMelDiscriminator(nn.Module):
    def __init__(self, n_mels=100):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                MelSubDiscriminator(in_channels=n_mels, use_spectral_norm=True),
                MelSubDiscriminator(in_channels=n_mels, use_spectral_norm=True),
                MelSubDiscriminator(in_channels=n_mels, use_spectral_norm=True),
            ]
        )

        self.meanpools = nn.ModuleList(
            [
                nn.AvgPool1d(kernel_size=4, stride=2, padding=1),
                nn.AvgPool1d(kernel_size=4, stride=2, padding=1),
            ]
        )

    def forward(self, mel):
        x = mel.transpose(1, 2)

        ret_outs = []
        ret_fmaps = []

        for i, disc in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i - 1](x)
            out, fmap = disc(x)
            ret_outs.append(out)
            ret_fmaps.append(fmap)

        return ret_outs, ret_fmaps


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()

        real_loss = torch.mean((dr - 1.0) ** 2)
        fake_loss = torch.mean(dg ** 2)
        loss += real_loss + fake_loss

    return loss


def generator_adversarial_loss(disc_generated_outputs):
    loss = 0
    for dg in disc_generated_outputs:
        dg = dg.float()
        loss += torch.mean((dg - 1.0) ** 2)

    return loss


def feature_matching_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2.0

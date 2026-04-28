"""
Pix2Pix GAN Architecture for Virtual H&E Staining.

Generator:  U-Net with ResNet-34 encoder (ImageNet pre-trained via torchvision).
            Skip connections at every encoder stage feed into the symmetric decoder.
Discriminator: 70×70 PatchGAN that scores local texture realism.

Input/output convention: all tensors in [-1, 1] range (Tanh output, matching the
dataset normalization used during training).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ────────────────────────────────────────────────────────────────────

def _conv_norm_relu(in_ch, out_ch, kernel=3, stride=1, pad=1, norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _up_conv_norm_relu(in_ch, out_ch):
    """Upsample ×2 then conv, matching U-Net decoder style."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True),
    )


# ── U-Net Generator ────────────────────────────────────────────────────────────

class UNetGenerator(nn.Module):
    """
    U-Net generator with a ResNet-34 encoder (torchvision, ImageNet weights).

    Architecture:
      Encoder stages (ResNet-34):
        e0: initial 7×7 conv  → 64 ch  (stride 2, H/2)
        e1: layer1            → 64 ch  (H/4)
        e2: layer2            → 128 ch (H/8)
        e3: layer3            → 256 ch (H/16)
        e4: layer4            → 512 ch (H/32)

      Bottleneck: 1024 ch (H/32 — unchanged spatial size, deepens capacity)

      Decoder (symmetric, with skip connections):
        d4: (512+512) → 256
        d3: (256+256) → 128
        d2: (128+128) → 64
        d1: (64+64)   → 64
        d0: (64+64)   → 32
        out: 32 → 3, Tanh

    Falls back to a scratch encoder when torchvision is unavailable
    (useful for local import checks without downloading weights).
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # ── Encoder (ResNet-34) ────────────────────────────────────────────────
        try:
            from torchvision import models
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            self._using_pretrained = True
        except ImportError:
            resnet = None
            self._using_pretrained = False

        if resnet is not None:
            self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64,  H/2
            self.pool  = resnet.maxpool                                         #      H/4
            self.enc1  = resnet.layer1   # 64,  H/4
            self.enc2  = resnet.layer2   # 128, H/8
            self.enc3  = resnet.layer3   # 256, H/16
            self.enc4  = resnet.layer4   # 512, H/32
        else:
            # Fallback scratch encoder (same channel sizes as ResNet-34)
            self.enc0 = _conv_norm_relu(3,   64,  kernel=7, stride=2, pad=3)
            self.pool  = nn.MaxPool2d(2, 2)
            self.enc1  = _conv_norm_relu(64,  64)
            self.enc2  = _conv_norm_relu(64,  128, stride=2)
            self.enc3  = _conv_norm_relu(128, 256, stride=2)
            self.enc4  = _conv_norm_relu(256, 512, stride=2)

        # ── Bottleneck ─────────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            _conv_norm_relu(512, 1024),
            _conv_norm_relu(1024, 1024),
        )

        # ── Decoder ────────────────────────────────────────────────────────────
        # Input channels = (bottleneck or lower decoder) + skip connection
        self.up4 = _up_conv_norm_relu(1024, 512)
        self.dec4 = _conv_norm_relu(512 + 512, 256)

        self.up3 = _up_conv_norm_relu(256, 256)
        self.dec3 = _conv_norm_relu(256 + 256, 128)

        self.up2 = _up_conv_norm_relu(128, 128)
        self.dec2 = _conv_norm_relu(128 + 128, 64)

        self.up1 = _up_conv_norm_relu(64, 64)
        self.dec1 = _conv_norm_relu(64 + 64, 64)

        self.up0 = _up_conv_norm_relu(64, 64)
        self.dec0 = _conv_norm_relu(64 + 64, 32)

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)          # 64,  H/2
        ep = self.pool(e0)          # 64,  H/4
        e1 = self.enc1(ep)          # 64,  H/4
        e2 = self.enc2(e1)          # 128, H/8
        e3 = self.enc3(e2)          # 256, H/16
        e4 = self.enc4(e3)          # 512, H/32

        # Bottleneck
        b = self.bottleneck(e4)     # 1024, H/32

        # Decoder with skip connections
        d = self.up4(b)             # 512, H/16
        d = self._cat(d, e4)
        d = self.dec4(d)            # 256, H/16

        d = self.up3(d)             # 256, H/8
        d = self._cat(d, e3)
        d = self.dec3(d)            # 128, H/8

        d = self.up2(d)             # 128, H/4
        d = self._cat(d, e2)
        d = self.dec2(d)            # 64,  H/4

        d = self.up1(d)             # 64,  H/4 → H/2
        d = self._cat(d, e1)
        d = self.dec1(d)            # 64,  H/2

        d = self.up0(d)             # 64,  H
        d = self._cat(d, e0)
        d = self.dec0(d)            # 32,  H

        return self.out_conv(d)     # 3,   H  in [-1,1]

    @staticmethod
    def _cat(a, b):
        """Center-crop b to match a's spatial size, then concatenate."""
        if a.shape[2:] != b.shape[2:]:
            b = F.interpolate(b, size=a.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([a, b], dim=1)


# ── PatchGAN Discriminator ─────────────────────────────────────────────────────

class PatchGANDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator.

    Input: concatenated [unstained | stained] pair → 6 channels.
    Output: grid of validity scores (real=1, fake=0) for local patches.

    Architecture (standard Pix2Pix):
      C64 → C128 → C256 → C512 → Conv(1, no norm)
    """

    def __init__(self, in_channels=6):
        super().__init__()

        def block(in_ch, out_ch, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64,  norm=False),
            *block(64,  128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1),
        )

    def forward(self, unstained, stained):
        """
        Args:
            unstained: [B, 3, H, W] input (condition)
            stained:   [B, 3, H, W] target or generated image

        Returns:
            [B, 1, H', W'] patch validity map
        """
        x = torch.cat([unstained, stained], dim=1)
        return self.model(x)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.disc_local = PatchGANDiscriminator(in_channels=6) # 3+3=6
        self.disc_global = PatchGANDiscriminator(in_channels=6)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, unstained, stained):
        out_local = self.disc_local(unstained, stained)
        u_down = self.downsample(unstained)
        s_down = self.downsample(stained)
        out_global = self.disc_global(u_down, s_down)
        return out_local, out_global

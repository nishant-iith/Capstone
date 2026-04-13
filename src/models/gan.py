import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class UNetGenerator(nn.Module):
    """
    U-Net Generator with a pre-trained ResNet-34 encoder for transfer learning.
    Implements skip connections to preserve spatial morphology.
    """
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()
        
        # Encoder: Pre-trained ResNet-34
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        # ResNet layers for downsampling
        self.layer1 = resnet.layer1 # 64
        self.layer2 = resnet.layer2 # 128
        self.layer3 = resnet.layer3 # 256
        self.layer4 = resnet.layer4 # 512
        
        # Decoder: Transposed Convolutions (Up-sampling)
        self.up4 = self._block(512, 256, upsample=True)
        self.up3 = self._block(512, 128, upsample=True) # 256+256
        self.up2 = self._block(256, 64, upsample=True)  # 128+128
        self.up1 = self._block(128, 64, upsample=True)  # 64+64
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=1),
            nn.Tanh() # Output in range [-1, 1]
        )

    def _block(self, in_ch, out_ch, upsample=False):
        layers = []
        if upsample:
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
        
        layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x0 = self.encoder(x)       # 1/4 size
        x1 = self.layer1(x0)       # 1/4
        x2 = self.layer2(x1)       # 1/8
        x3 = self.layer3(x2)       # 1/16
        x4 = self.layer4(x3)       # 1/32
        
        # Decoder with skip connections
        d4 = self.up4(x4)          # 1/16
        d3 = self.up3(torch.cat([d4, x3], dim=1)) # 1/8
        d2 = self.up2(torch.cat([d3, x2], dim=1)) # 1/4
        d1 = self.up1(torch.cat([d2, x1], dim=1)) # 1/4
        
        # Final upsampling to original size
        out = F.interpolate(d1, scale_factor=4, mode='bilinear', align_corners=False)
        return self.final_conv(out)

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Analyzes 70x70 local patches to determine if they are real or fake.
    """
    def __init__(self, in_channels=3):
        super(PatchGANDiscriminator, self).__init__()
        
        def discriminator_block(in_f, out_f, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 4, stride, 1, bias=False),
                nn.InstanceNorm2d(out_f),
                nn.LeakyReLU(0.2)
            )

        self.model = nn.Sequential(
            discriminator_block(in_channels, 64), # 512 -> 256
            discriminator_block(64, 128),        # 256 -> 128
            discriminator_block(128, 256),       # 128 -> 64
            discriminator_block(256, 512, stride=1), # 64 -> 64
            nn.Conv2d(512, 1, 4, padding=1)      # 64 -> 63x63 patch
        )

    def forward(self, x):
        return self.model(x)

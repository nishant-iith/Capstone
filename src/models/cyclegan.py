import torch
import torch.nn as nn
from src.models.gan import PatchGANDiscriminator

class CycleGANDiscriminator(nn.Module):
    """
    Standard PatchGAN discriminator for a single image (not concatenated pair).
    Used in CycleGAN for Domain A and Domain B.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        # We can reuse the same logic but with in_channels=3
        self.patchgan = PatchGANDiscriminator(in_channels=in_channels)
        
    def forward(self, x):
        # PatchGANDiscriminator expects (unstained, stained) so we'll 
        # just pass x as both, but we need to bypass the cat in its forward.
        # Actually, it's better to just implement a clean one here.
        return self.patchgan.model(x)

# Re-exporting UNetGenerator as well for clarity in CycleGAN training
from src.models.gan import UNetGenerator

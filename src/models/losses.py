import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PerceptualLoss(nn.Module):
    """
    VGG-19 Feature Matching Loss for Weakly Supervised Histology.
    Computes the L1 distance between intermediate feature maps of real and generated images.
    """
    def __init__(self):
        super().__init__()
        # Load VGG-19 and extract the 'features' portion
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # We slice the network to extract features at different depths.
        # Shallow layers = Fine textures (edges, grain)
        # Deep layers = Global structures (cell clusters, tissue architecture)
        self.slices = nn.ModuleList([
            vgg[:4],    # relu1_2
            vgg[4:9],   # relu2_2
            vgg[9:18],  # relu3_4
            vgg[18:27], # relu4_4
            vgg[27:36]  # relu5_4
        ])
        
        # Freeze VGG-19 parameters (we only use it to evaluate, not train it)
        for param in self.parameters():
            param.requires_grad = False
            
        # ImageNet normalization constants required by VGG
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, generated, target):
        # 1. Convert images from [-1, 1] (GAN range) to [0, 1]
        gen_norm = (generated + 1.0) / 2.0
        tgt_norm = (target + 1.0) / 2.0
        
        # 2. Apply ImageNet normalization
        gen_norm = (gen_norm - self.mean) / self.std
        tgt_norm = (tgt_norm - self.mean) / self.std
        
        loss = 0.0
        feat_gen = gen_norm
        feat_tgt = tgt_norm
        
        # 3. Pass through slices and accumulate the L1 difference
        for slice_layer in self.slices:
            feat_gen = slice_layer(feat_gen)
            feat_tgt = slice_layer(feat_tgt)
            loss += torch.nn.functional.l1_loss(feat_gen, feat_tgt)
            
        return loss


class HEDStainLoss(nn.Module):
    """
    Stain decomposition loss in Hematoxylin-Eosin-DAB (HED) color space.
    Forces the generator to match H&E stain channels (purple nuclei, pink cytoplasm)
    rather than just minimizing RGB pixel distance.

    Macenko stain matrix for H&E (standard values from literature).
    Input images in [-1, 1] range.
    """

    def __init__(self):
        super().__init__()
        # Macenko H&E stain matrix — rows = [Hematoxylin, Eosin]
        # Columns = [R, G, B] optical density contributions
        M = torch.tensor([
            [0.6500286, 0.7044536, 0.2860126],   # Hematoxylin
            [0.0704894, 0.9927070, 0.0997278],   # Eosin
        ], dtype=torch.float32)  # [2, 3]
        self.register_buffer("M", M)

    def _rgb_to_hed(self, img_01):
        """Convert [B,3,H,W] in [0,1] → HED concentration map [B,2,H,W]."""
        # force float32 — torch.inverse does not support Half
        img_01 = img_01.float()
        od = -torch.log(img_01.clamp(min=1e-6))  # [B,3,H,W]
        B, C, H, W = od.shape
        od_flat = od.view(B, C, -1)              # [B,3,N]
        M = self.M.float()
        Mt = M.t()                               # [3,2]
        MtM = M @ Mt                             # [2,2]
        MtM_inv = torch.inverse(MtM)             # [2,2]
        proj = MtM_inv @ M                       # [2,3]
        conc = proj @ od_flat                    # [B,2,N]
        return conc.view(B, 2, H, W)

    def forward(self, generated, target):
        # disable AMP — torch.inverse does not support Half precision
        with torch.autocast(device_type="cuda", enabled=False):
            gen_01 = (generated.float() + 1.0) / 2.0
            tgt_01 = (target.float()    + 1.0) / 2.0
            gen_hed = self._rgb_to_hed(gen_01)
            tgt_hed = self._rgb_to_hed(tgt_01)
            return F.l1_loss(gen_hed, tgt_hed)

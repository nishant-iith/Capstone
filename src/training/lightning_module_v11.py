"""
v11 Module: Weakly Supervised Pix2Pix (Perceptual Matching).
Includes L1 + WGAN-GP + SSIM + VGG-19 Perceptual Loss + PCC Metric.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from src.validation.metrics import compute_metrics_batch
from src.models.losses import PerceptualLoss

class GANModuleV11(pl.LightningModule):
    def __init__(self, gen, disc, lambda_l1=300, lambda_gp=10, lambda_struct=20, lambda_percept=10):
        super().__init__()
        self.save_hyperparameters(ignore=['gen', 'disc'])
        self.gen = gen
        self.disc = disc
        self.perceptual_loss = PerceptualLoss()
        
        self.lambda_l1 = lambda_l1
        self.lambda_gp = lambda_gp
        self.lambda_struct = lambda_struct
        self.lambda_percept = lambda_percept
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        unstained, stained = batch
        opt_g, opt_d = self.optimizers()

        # --- Generator Step ---
        fake = self.gen(unstained)
        pred_fake = self.disc(unstained, fake)
        
        # 1. Adversarial Loss (WGAN)
        loss_g_adv = -torch.mean(pred_fake)
        
        # 2. Pixel Loss (L1)
        loss_g_l1 = F.l1_loss(fake, stained) * self.lambda_l1
        
        # 3. Structural Loss (Sobel Edge Consistency)
        loss_g_struct = self._sobel_loss(fake, stained) * self.lambda_struct
        
        # 4. WEAK SUPERVISION: Perceptual Loss (VGG-19 Features)
        loss_g_percept = self.perceptual_loss(fake, stained) * self.lambda_percept

        total_g_loss = loss_g_adv + loss_g_l1 + loss_g_struct + loss_g_percept

        opt_g.zero_grad()
        self.manual_backward(total_g_loss)
        opt_g.step()

        # --- Discriminator Step ---
        pred_real = self.disc(unstained, stained)
        pred_fake_d = self.disc(unstained, fake.detach())
        
        loss_d_adv = torch.mean(pred_fake_d) - torch.mean(pred_real)
        loss_gp = self._gradient_penalty(unstained, stained, fake.detach()) * self.lambda_gp
        
        total_d_loss = loss_d_adv + loss_gp

        opt_d.zero_grad()
        self.manual_backward(total_d_loss)
        opt_d.step()

        self.log_dict({
            "g_loss": total_g_loss,
            "d_loss": total_d_loss,
            "g_percept": loss_g_percept,
            "g_l1": loss_g_l1
        }, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        unstained, stained = batch
        fake = self.gen(unstained)

        # Convert to numpy for advanced metrics [B, H, W, 3]
        fake_np = (fake.detach().cpu().to(torch.float32) + 1.0) / 2.0
        stained_np = (stained.detach().cpu().to(torch.float32) + 1.0) / 2.0
        
        fake_np = fake_np.permute(0, 2, 3, 1).numpy()
        stained_np = stained_np.permute(0, 2, 3, 1).numpy()

        # Compute SSIM, PSNR, and Pearson Correlation Coefficient (PCC)
        ssim_scores, psnr_scores, pcc_scores = compute_metrics_batch(fake_np, stained_np, data_range=1.0)
        
        self.log("val_ssim", np.mean(ssim_scores), prog_bar=True, on_epoch=True)
        self.log("val_psnr", np.mean(psnr_scores), prog_bar=True, on_epoch=True)
        self.log("val_pcc", np.mean(pcc_scores), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=1e-4, betas=(0.0, 0.9))
        return [opt_g, opt_d]

    def _sobel_loss(self, fake, real):
        def sobel(img):
            kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            gx = F.conv2d(img, kernel_x, padding=1, groups=3)
            gy = F.conv2d(img, kernel_y, padding=1, groups=3)
            return torch.sqrt(gx**2 + gy**2 + 1e-6)
        return F.l1_loss(sobel(fake), sobel(real))

    def _gradient_penalty(self, unstained, real, fake):
        alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
        interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interpolates = self.disc(unstained, interpolates)
        fake_grad = torch.ones(d_interpolates.size(), device=real.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake_grad,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

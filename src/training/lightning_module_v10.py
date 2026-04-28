"""
v10 training: WGAN-GP + Macenko normalization (v7 architecture, no perceptual/elastic).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from src.validation.metrics import compute_metrics_batch, normalize_images_to_01


class GANModuleV10(pl.LightningModule):
    def __init__(self, gen, disc, lambda_l1=10, lambda_gp=10, lambda_struct=10):
        super().__init__()
        self.gen = gen
        self.disc = disc
        self.lambda_l1 = lambda_l1
        self.lambda_gp = lambda_gp
        self.lambda_struct = lambda_struct
        self.automatic_optimization = False

    def compute_gradient_penalty(self, unstained, real, fake):
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
        interp = alpha * real + (1 - alpha) * fake
        interp.requires_grad = True

        with torch.cuda.amp.autocast(enabled=False):
            d_interp = self.disc(unstained, interp)

        fake_out = torch.ones(d_interp.size(), device=real.device, requires_grad=True)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_out,
            create_graph=True, retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def training_step(self, batch, batch_idx):
        unstained, stained = batch
        opt_g, opt_d = self.optimizers()

        # D step
        for _ in range(2):
            with torch.cuda.amp.autocast():
                fake = self.gen(unstained)
                real_pred = self.disc(unstained, stained)
                fake_pred = self.disc(unstained, fake.detach())
                w_dist = fake_pred.mean() - real_pred.mean()

            with torch.cuda.amp.autocast(enabled=False):
                gp = self.compute_gradient_penalty(unstained, stained, fake.detach())

            d_loss = w_dist + self.lambda_gp * gp
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()

        # G step
        with torch.cuda.amp.autocast():
            fake = self.gen(unstained)
            fake_pred = self.disc(unstained, fake)
            adv_loss = -fake_pred.mean()
            l1_loss = F.l1_loss(fake, stained)

            # Sobel
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=fake.device).unsqueeze(0).unsqueeze(0) / 8
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=fake.device).unsqueeze(0).unsqueeze(0) / 8

            fake_gray = fake.mean(dim=1, keepdim=True)
            stained_gray = stained.mean(dim=1, keepdim=True)
            fake_edge_x = F.conv2d(fake_gray, sobel_x, padding=1)
            fake_edge_y = F.conv2d(fake_gray, sobel_y, padding=1)
            stained_edge_x = F.conv2d(stained_gray, sobel_x, padding=1)
            stained_edge_y = F.conv2d(stained_gray, sobel_y, padding=1)
            struct_loss = F.l1_loss(fake_edge_x, stained_edge_x) + F.l1_loss(fake_edge_y, stained_edge_y)

            g_loss = adv_loss + self.lambda_l1 * l1_loss + self.lambda_struct * struct_loss

        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log("d_loss", d_loss, prog_bar=True)
        self.log("w_dist", w_dist, prog_bar=True)
        self.log("g_loss", g_loss, prog_bar=True)
        self.log("g_struct", struct_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        unstained, stained = batch
        fake = self.gen(unstained)

        # Convert to numpy [B, H, W, 3] for SSIM computation
        # Cast to float32 first to avoid BFloat16 error on CPU
        fake_np = fake.detach().cpu().to(torch.float32).permute(0, 2, 3, 1).numpy()
        stained_np = stained.detach().cpu().to(torch.float32).permute(0, 2, 3, 1).numpy()

        # Normalize to [0, 1]
        fake_np = (fake_np + 1.0) / 2.0
        stained_np = (stained_np + 1.0) / 2.0

        # Compute SSIM, PSNR, and PCC batch
        ssim_scores, psnr_scores, pcc_scores = compute_metrics_batch(fake_np, stained_np, data_range=1.0)
        val_ssim = np.mean(ssim_scores)
        val_psnr = np.mean(psnr_scores)
        val_pcc = np.mean(pcc_scores)

        # Log with prog_bar=True to make them visible in the terminal
        self.log("val_ssim", val_ssim, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_psnr", val_psnr, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log("val_pcc", val_pcc, prog_bar=True, sync_dist=True, on_epoch=True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=1e-4, betas=(0.0, 0.9))
        return [opt_g, opt_d]

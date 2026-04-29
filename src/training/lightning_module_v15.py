"""
v15: Attention U-Net + MultiScale Disc + HED Loss + Cosine LR.
Trained on high-quality 512×512 patches (mean SSIM 0.625).
Validates on patches + full-size held-out images.
Based on v13, optimized for patch-level training.
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from src.validation.metrics import compute_metrics_batch
from src.models.losses import PerceptualLoss, HEDStainLoss


class GANModuleV15(pl.LightningModule):
    def __init__(
        self,
        gen,
        disc,
        lambda_l1=100,
        lambda_gp=10,
        lambda_struct=20,
        lambda_percept=10,
        lambda_hed=5,
        lr=1e-4,
        max_epochs=200,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["gen", "disc"])
        self.gen = gen
        self.disc = disc
        self.perceptual_loss = PerceptualLoss()
        self.hed_loss = HEDStainLoss()

        self.lambda_l1      = lambda_l1
        self.lambda_gp      = lambda_gp
        self.lambda_struct  = lambda_struct
        self.lambda_percept = lambda_percept
        self.lambda_hed     = lambda_hed
        self.lr             = lr
        self.max_epochs     = max_epochs
        self.automatic_optimization = False

    # ── Training ───────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        unstained, stained = batch
        opt_g, opt_d = self.optimizers()

        # ─── Generator step ───────────────────────────────────────────────────
        fake = self.gen(unstained)
        pred_fake_local, pred_fake_global = self.disc(unstained, fake)

        loss_g_adv = (-torch.mean(pred_fake_local) - torch.mean(pred_fake_global)) * 0.5
        loss_g_l1      = F.l1_loss(fake, stained) * self.lambda_l1
        loss_g_struct  = self._sobel_loss(fake, stained) * self.lambda_struct
        loss_g_percept = self.perceptual_loss(fake, stained) * self.lambda_percept
        loss_g_hed     = self.hed_loss(fake, stained) * self.lambda_hed

        total_g_loss = loss_g_adv + loss_g_l1 + loss_g_struct + loss_g_percept + loss_g_hed

        opt_g.zero_grad()
        self.manual_backward(total_g_loss)
        opt_g.step()

        # ─── Discriminator step ───────────────────────────────────────────────
        pred_real_local,  pred_real_global  = self.disc(unstained, stained)
        pred_fake_local_d, pred_fake_global_d = self.disc(unstained, fake.detach())

        loss_d_local  = torch.mean(pred_fake_local_d)  - torch.mean(pred_real_local)
        loss_d_global = torch.mean(pred_fake_global_d) - torch.mean(pred_real_global)

        # GP on local scale
        loss_gp = self._gradient_penalty(unstained, stained, fake.detach()) * self.lambda_gp

        total_d_loss = (loss_d_local + loss_d_global) * 0.5 + loss_gp

        opt_d.zero_grad()
        self.manual_backward(total_d_loss)
        opt_d.step()

        self.log_dict({
            "g_loss":    total_g_loss,
            "d_loss":    total_d_loss,
            "g_l1":      loss_g_l1,
            "g_percept": loss_g_percept,
            "g_hed":     loss_g_hed,
        }, prog_bar=True)

    # ── Validation ─────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        unstained, stained = batch
        fake = self.gen(unstained)

        fake_np    = (fake.detach().cpu().to(torch.float32)    + 1.0) / 2.0
        stained_np = (stained.detach().cpu().to(torch.float32) + 1.0) / 2.0

        fake_np    = fake_np.permute(0, 2, 3, 1).numpy()
        stained_np = stained_np.permute(0, 2, 3, 1).numpy()

        ssim_scores, psnr_scores, pcc_scores = compute_metrics_batch(
            fake_np, stained_np, data_range=1.0
        )

        self.log("val_ssim", float(np.mean(ssim_scores)), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_psnr", float(np.mean(psnr_scores)), prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_pcc",  float(np.mean(pcc_scores)),  prog_bar=True, on_epoch=True, sync_dist=True)

    # ── Optimizers + LR Schedule ───────────────────────────────────────────────

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.gen.parameters(),  lr=self.lr, betas=(0.0, 0.9))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=self.lr, betas=(0.0, 0.9))

        sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_g, T_max=self.max_epochs, eta_min=1e-6
        )
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_d, T_max=self.max_epochs, eta_min=1e-6
        )

        return (
            [opt_g, opt_d],
            [
                {"scheduler": sched_g, "interval": "epoch"},
                {"scheduler": sched_d, "interval": "epoch"},
            ],
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _sobel_loss(self, fake, real):
        def sobel(img):
            kx = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                dtype=img.dtype, device=img.device
            ).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            ky = kx.transpose(2, 3)
            gx = F.conv2d(img, kx, padding=1, groups=3)
            gy = F.conv2d(img, ky, padding=1, groups=3)
            return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return F.l1_loss(sobel(fake), sobel(real))

    def _gradient_penalty(self, unstained, real, fake):
        alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interp, _ = self.disc(unstained, interp)
        grads = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
        )[0]
        grads = grads.view(grads.size(0), -1)
        return ((grads.norm(2, dim=1) - 1) ** 2).mean()

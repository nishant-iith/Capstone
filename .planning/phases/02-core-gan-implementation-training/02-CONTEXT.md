# Phase 2: Core GAN Implementation & Training - Context

**Gathered:** 2026-04-13
**Status:** Ready for planning

<domain>
## Phase Boundary
This phase delivers a functional Pix2Pix conditional GAN capable of translating unstained tissue patches into H&E stained images, including a trained model checkpoint and a basic inference script.
</domain>

<decisions>
## Implementation Decisions

### Generator Architecture (U-Net)
- Use a pre-trained ResNet-34 encoder for transfer learning to ensure convergence on limited data.
- Use strided convolutions for downsampling to allow the model to learn the optimal compression.
- Use Instance Normalization to maintain consistency across individual images.
- Use LeakyReLU activation to prevent dead neurons.

### Discriminator & Loss
- Use a PatchGAN discriminator (70x70) to prioritize local textural fidelity (nuclei and cell membranes).
- Implement Least Squares GAN (LSGAN) loss using MSE for improved stability.
- Use L1 loss as the structural constraint to maintain spatial alignment with the target.
- Set the L1 weighting factor to $\lambda=100$.

### Kaggle Training Strategy
- Use a small batch size (4-8) to fit within VRAM constraints and improve stability.
- Use Adam optimizer with $\beta_1=0.5$ and a learning rate of $2e-4$.
- Enable Mixed Precision (FP16) to optimize training speed on NVIDIA T4/P100 GPUs.

### the agent's Discretion
- Specific PyTorch Lightning logger configuration and checkpointing frequency are at the agent's discretion.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 1 data pipeline (`pairing.py`, `splitting.py`, `augmentation.py`) will be used to feed the model.

### Established Patterns
- Local development for architecture, cloud execution for training.
- Use of MONAI for medical-grade U-Net implementations.

### Integration Points
- The output of the Phase 1 registration pipeline feeds directly into the Generator.
- The trained model weights will be the primary input for Phase 3.
</code_context>

<specifics>
## Specific Ideas
- No specific requirements — following standard SOTA Pix2Pix implementation.
</specifics>

<deferred>
## Deferred Ideas
- Whole-slide reconstruction (deferred to v2/WSI phase).
</deferred>

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sobel_edges(image_tensor):
    """
    Compute edge magnitude using Sobel filters.

    Args:
        image_tensor: torch.Tensor of shape [batch, channels, height, width]
                     in range [-1, 1] (from generator output)

    Returns:
        edge_tensor: torch.Tensor of shape [batch, 1, height, width]
                    with edge magnitudes
    """
    # Convert to grayscale by averaging across channels
    if image_tensor.shape[1] == 3:
        gray = image_tensor.mean(dim=1, keepdim=True)
    else:
        gray = image_tensor

    # Get device and dtype
    device = image_tensor.device
    dtype = image_tensor.dtype

    # Define Sobel kernels
    sobel_h = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype
    ).view(1, 1, 3, 3)

    sobel_v = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype
    ).view(1, 1, 3, 3)

    # Apply Sobel filters via convolution (padding=1 to preserve size)
    edges_h = F.conv2d(gray, sobel_h, padding=1)
    edges_v = F.conv2d(gray, sobel_v, padding=1)

    # Compute edge magnitude
    edge_magnitude = torch.sqrt(edges_h ** 2 + edges_v ** 2 + 1e-8)

    return edge_magnitude


class StructuralConsistencyLoss(nn.Module):
    """
    Structural Consistency Loss based on Sobel edge matching.

    Penalizes divergence in edge structure between generated and target images.
    This helps suppress morphological hallucinations by ensuring the generator
    produces outputs with similar structural patterns to the ground truth.
    """

    def __init__(self):
        super(StructuralConsistencyLoss, self).__init__()

    def forward(self, fake_images, real_images):
        """
        Compute structural consistency loss between fake and real images.

        Args:
            fake_images: torch.Tensor of shape [batch, channels, height, width]
                        Generator output
            real_images: torch.Tensor of shape [batch, channels, height, width]
                        Ground truth target

        Returns:
            loss: torch.Tensor (scalar) L1 distance between edge maps
        """
        # Compute edges for both images
        fake_edges = compute_sobel_edges(fake_images)
        real_edges = compute_sobel_edges(real_images)

        # Compute L1 loss between edge maps
        loss = F.l1_loss(fake_edges, real_edges)

        return loss

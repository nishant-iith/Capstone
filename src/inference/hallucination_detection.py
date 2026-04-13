"""
Hallucination Detection Module

Detects morphological hallucinations using high-frequency artifact analysis.
Implements artifact scoring and comparative analysis between baseline and guarded models.
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_high_pass_filter(image_tensor, kernel_type='laplacian'):
    """
    Compute high-pass filter response on image tensor.

    Applies Laplacian kernel to isolate high-frequency components (edges and noise).

    Parameters
    ----------
    image_tensor : torch.Tensor
        Input image tensor [batch, channels, height, width] in range [0, 1] or [-1, 1].
    kernel_type : str, optional
        Type of kernel to use. Currently only 'laplacian' is supported (default).

    Returns
    -------
    torch.Tensor
        High-pass filtered tensor [batch, 1, height, width] containing edge/artifact responses.
    """
    # Normalize to [0, 1] if needed
    if image_tensor.min() < 0:
        image_tensor = (image_tensor + 1.0) / 2.0

    # Convert to grayscale if multi-channel
    if image_tensor.shape[1] > 1:
        # Simple mean-based grayscale conversion
        gray = image_tensor.mean(dim=1, keepdim=True)
    else:
        gray = image_tensor

    # Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    # Normalized for stable output
    laplacian_kernel = torch.tensor(
        [[0., 1., 0.],
         [1., -4., 1.],
         [0., 1., 0.]],
        dtype=image_tensor.dtype,
        device=image_tensor.device
    ).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

    # Apply convolution with padding to preserve size
    high_pass = F.conv2d(gray, laplacian_kernel, padding=1)

    return high_pass


def compute_artifact_score(generated_image, threshold_percentile=90):
    """
    Compute artifact score from generated image.

    Score represents the magnitude of high-frequency components (edges and noise).
    Higher score indicates more artifacts/hallucinations.

    Parameters
    ----------
    generated_image : torch.Tensor
        Generated image tensor, single or batch.
    threshold_percentile : int, optional
        Percentile for alternative scoring method (default 90).

    Returns
    -------
    float or torch.Tensor
        Artifact score(s). Scalar if single image, tensor if batch.
    """
    # Ensure input is batched
    if generated_image.dim() == 3:
        generated_image = generated_image.unsqueeze(0)

    # Compute high-pass response
    high_pass = compute_high_pass_filter(generated_image)

    # Compute artifact score as mean absolute value of high-pass output
    artifact_score = torch.mean(torch.abs(high_pass), dim=[2, 3])  # [batch, 1]
    artifact_score = artifact_score.squeeze(1)  # [batch]

    # Return scalar if single image, otherwise return tensor
    if artifact_score.shape[0] == 1:
        return artifact_score.item()
    else:
        return artifact_score.detach().cpu().numpy()


def detect_artifacts(image_batch, artifact_threshold=0.05):
    """
    Detect artifacts in a batch of images.

    Flags images where artifact score exceeds threshold as containing hallucinations.

    Parameters
    ----------
    image_batch : torch.Tensor
        Batch of generated images [batch, channels, height, width].
    artifact_threshold : float, optional
        Threshold for flagging artifacts (default 0.05).

    Returns
    -------
    tuple
        (artifact_flags, artifact_scores)
        - artifact_flags: numpy array of bool [batch_size]
        - artifact_scores: numpy array of float [batch_size]
    """
    # Compute scores for all images in batch
    batch_size = image_batch.shape[0]
    scores = []

    for i in range(batch_size):
        score = compute_artifact_score(image_batch[i:i+1])
        if isinstance(score, (float, int)):
            scores.append(score)
        else:
            scores.append(float(score[0]))

    scores = np.array(scores)
    flags = scores > artifact_threshold

    return flags, scores


def compute_artifact_reduction(baseline_scores, guarded_scores):
    """
    Compute artifact reduction statistics.

    Calculates percentage reduction and count reduction between baseline and guarded models.

    Parameters
    ----------
    baseline_scores : array-like
        Artifact scores from baseline model.
    guarded_scores : array-like
        Artifact scores from guarded model.

    Returns
    -------
    dict
        Dictionary containing:
        - reduction_pct: percentage reduction in mean artifact score
        - count_reduction: number of patches with reduced artifact counts
        - baseline_mean: mean artifact score of baseline
        - baseline_std: standard deviation of baseline scores
        - guarded_mean: mean artifact score of guarded model
        - guarded_std: standard deviation of guarded scores
        - baseline_high: count of baseline patches with high artifacts (score > 0.05)
        - guarded_high: count of guarded patches with high artifacts
    """
    baseline_scores = np.array(baseline_scores)
    guarded_scores = np.array(guarded_scores)

    baseline_mean = np.mean(baseline_scores)
    guarded_mean = np.mean(guarded_scores)

    # Avoid division by zero
    if baseline_mean > 0:
        reduction_pct = ((baseline_mean - guarded_mean) / baseline_mean) * 100.0
    else:
        reduction_pct = 0.0

    # Count high-artifact patches
    threshold = 0.05
    baseline_high = np.sum(baseline_scores > threshold)
    guarded_high = np.sum(guarded_scores > threshold)
    count_reduction = baseline_high - guarded_high

    return {
        'reduction_pct': float(reduction_pct),
        'count_reduction': int(count_reduction),
        'baseline_mean': float(baseline_mean),
        'baseline_std': float(np.std(baseline_scores)),
        'guarded_mean': float(guarded_mean),
        'guarded_std': float(np.std(guarded_scores)),
        'baseline_high': int(baseline_high),
        'guarded_high': int(guarded_high),
    }

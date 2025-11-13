# src/attacks_project/utils.py

import torch
import torch.nn as nn
from typing import Tuple, Optional

def evaluate_attack_success(
    model: nn.Module,
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    true_labels: torch.Tensor,
    target_labels: Optional[torch.Tensor] = None,
    targeted: bool = False
) -> Tuple[float, float]:
    """
    Evaluate the success rate of an adversarial attack.
    
    Args:
        model: The neural network model.
        original_images: Original clean images.
        adversarial_images: Adversarial images.
        true_labels: True labels of the images.
        target_labels: Target labels for targeted attacks.
        targeted: Whether this is a targeted attack.
    
    Returns:
        Tuple of (success_rate, avg_confidence)
    """
    model.eval()
    device = next(model.parameters()).device
    
    original_images = original_images.to(device)
    adversarial_images = adversarial_images.to(device)
    true_labels = true_labels.to(device)
    
    with torch.no_grad():
        # Get predictions
        outputs = model(adversarial_images)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, 1)
        
        if targeted:
            # For targeted attacks, success means predicting the target class
            if target_labels is None:
                target_labels = true_labels
            target_labels = target_labels.to(device)
            success = (predictions == target_labels).float()
        else:
            # For untargeted attacks, success means misclassifying
            success = (predictions != true_labels).float()
        
        success_rate = success.mean().item()
        avg_confidence = confidences.mean().item()
    
    return success_rate, avg_confidence


def compute_perturbation_metrics(
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor
) -> dict:
    """
    Compute various perturbation metrics.
    
    Args:
        original_images: Original clean images.
        adversarial_images: Adversarial images.
    
    Returns:
        Dictionary containing L0, L2, and Linf norms.
    """
    diff = (adversarial_images - original_images)
    batch_size = diff.shape[0]
    
    # Flatten for easier computation
    diff_flat = diff.view(batch_size, -1)
    
    # L0 norm (number of changed pixels)
    l0 = (diff_flat != 0).sum(dim=1).float()
    
    # L2 norm
    l2 = diff_flat.pow(2).sum(dim=1).sqrt()
    
    # Linf norm
    linf = diff_flat.abs().max(dim=1)[0]
    
    return {
        'l0_mean': l0.mean().item(),
        'l0_max': l0.max().item(),
        'l2_mean': l2.mean().item(),
        'l2_max': l2.max().item(),
        'linf_mean': linf.mean().item(),
        'linf_max': linf.max().item(),
    }


def clip_by_tensor(tensor: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> torch.Tensor:
    """
    Clip tensor values element-wise between min_val and max_val tensors.
    
    Args:
        tensor: Input tensor.
        min_val: Minimum values (can be tensor or scalar).
        max_val: Maximum values (can be tensor or scalar).
    
    Returns:
        Clipped tensor.
    """
    result = torch.max(tensor, min_val)
    result = torch.min(result, max_val)
    return result


def normalize_tensor(tensor: torch.Tensor, mean: Tuple[float, float, float], 
                     std: Tuple[float, float, float]) -> torch.Tensor:
    """
    Normalize a tensor with given mean and std.
    
    Args:
        tensor: Input tensor of shape (B, C, H, W).
        mean: Mean for each channel.
        std: Standard deviation for each channel.
    
    Returns:
        Normalized tensor.
    """
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    return (tensor - mean) / std


def denormalize_tensor(tensor: torch.Tensor, mean: Tuple[float, float, float],
                       std: Tuple[float, float, float]) -> torch.Tensor:
    """
    Denormalize a tensor with given mean and std.
    
    Args:
        tensor: Normalized tensor of shape (B, C, H, W).
        mean: Mean for each channel.
        std: Standard deviation for each channel.
    
    Returns:
        Denormalized tensor.
    """
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean

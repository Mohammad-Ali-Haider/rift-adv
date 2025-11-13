# src/attacks_project/attacks/bim.py

import torch
import torch.nn as nn
from typing import Optional
from .base import Attack

class BIM(Attack):
    """
    Implements the Basic Iterative Method (BIM) attack, also known as I-FGSM.
    
    BIM is similar to PGD but without random initialization and typically uses
    a simpler clipping strategy.
    
    Reference:
        Kurakin et al., "Adversarial examples in the physical world", ICLR 2017 Workshop.
    
    Args:
        model: The neural network model to attack.
        eps: Maximum perturbation size (L-infinity norm).
        alpha: Step size for each iteration.
        steps: Number of attack iterations.
        targeted: If True, performs a targeted attack. Default is False.
    
    Example:
        >>> attack = BIM(model, eps=8/255, alpha=1/255, steps=10)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model: nn.Module, eps: float = 0.03, alpha: float = 0.007, 
                 steps: int = 10, targeted: bool = False):
        """
        Args:
            model: The neural network model to attack.
            eps: Maximum perturbation size.
            alpha: Step size for each iteration.
            steps: Number of attack iterations.
            targeted: If True, performs a targeted attack.
        """
        super().__init__(model, targeted)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

    def forward(self, images: torch.Tensor, labels: torch.Tensor, target_labels: Optional[torch.Tensor] = None):
        """
        Generates adversarial examples using BIM.
        
        Args:
            images: Input images to perturb.
            labels: True labels (for untargeted) or target labels (for targeted if target_labels not provided).
            target_labels: Explicit target labels for targeted attacks (optional).
        
        Returns:
            Adversarial examples.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        
        original_images = images.clone().detach()
        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            loss = self._get_loss(outputs, labels, target_labels)
            self.model.zero_grad()
            loss.backward()

            grad = adv_images.grad.data
            
            # Note: _get_loss already handles the sign (negative for targeted)
            # So we always add the gradient of the loss
            adv_images = adv_images.detach() + self.alpha * grad.sign()

            # Clip to epsilon ball and valid pixel range
            adv_images = torch.max(torch.min(adv_images, original_images + self.eps), original_images - self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images

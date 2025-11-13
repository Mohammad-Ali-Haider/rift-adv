# src/attacks_project/attacks/fgsm.py

import torch
import torch.nn as nn
from typing import Optional
from .base import Attack

class FGSM(Attack):
    """
    Implements the Fast Gradient Sign Method (FGSM) attack.
    
    Reference:
        Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015.
    
    Args:
        model: The neural network model to attack.
        eps: The maximum perturbation size (L-infinity norm).
        targeted: If True, performs a targeted attack. Default is False.
    
    Example:
        >>> # Untargeted attack
        >>> attack = FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)
        
        >>> # Targeted attack
        >>> attack = FGSM(model, eps=8/255, targeted=True)
        >>> adv_images = attack(images, labels, target_labels)
    """
    def __init__(self, model: nn.Module, eps: float = 0.007, targeted: bool = False):
        """
        Args:
            model: The neural network model to attack.
            eps: The maximum perturbation size (L-infinity norm).
            targeted: If True, performs a targeted attack.
        """
        super().__init__(model, targeted)
        self.eps = eps

    def forward(self, images: torch.Tensor, labels: torch.Tensor, target_labels: Optional[torch.Tensor] = None):
        """
        Generates adversarial examples using FGSM.
        
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
        
        images.requires_grad = True

        outputs = self.model(images)
        loss = self._get_loss(outputs, labels, target_labels)

        self.model.zero_grad()
        loss.backward()

        grad_sign = images.grad.data.sign()
        
        adv_images = images + self.eps * grad_sign
        
        adv_images = torch.clamp(adv_images, min=0, max=1)

        return adv_images.detach()
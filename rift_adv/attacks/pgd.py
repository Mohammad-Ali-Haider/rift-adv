# src/attacks_project/attacks/pgd.py

import torch
import torch.nn as nn
from typing import Optional
from .base import Attack

class PGD(Attack):
    """
    Implements the Projected Gradient Descent (PGD) attack.
    
    PGD is an iterative version of FGSM that applies the attack multiple times
    with smaller step sizes, projecting back into the epsilon ball after each step.
    
    Reference:
        Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018.
    
    Args:
        model: The neural network model to attack.
        eps: Maximum perturbation size (L-infinity norm).
        alpha: Step size for each iteration.
        steps: Number of attack iterations.
        targeted: If True, performs a targeted attack. Default is False.
        random_start: If True, starts from a random point in the epsilon ball.
    
    Example:
        >>> # Untargeted attack
        >>> attack = PGD(model, eps=8/255, alpha=2/255, steps=10)
        >>> adv_images = attack(images, labels)
        
        >>> # Targeted attack
        >>> attack = PGD(model, eps=8/255, alpha=2/255, steps=40, targeted=True)
        >>> adv_images = attack(images, labels, target_labels)
    """
    def __init__(self, model: nn.Module, eps: float = 0.03, alpha: float = 0.007, 
                 steps: int = 10, targeted: bool = False, random_start: bool = True):
        """
        Args:
            model: The neural network model to attack.
            eps: Maximum perturbation size.
            alpha: Step size for each iteration.
            steps: Number of attack iterations.
            targeted: If True, performs a targeted attack.
            random_start: If True, starts from a random point in the epsilon ball.
        """
        super().__init__(model, targeted)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def forward(self, images: torch.Tensor, labels: torch.Tensor, target_labels: Optional[torch.Tensor] = None):
        """
        Generates adversarial examples using PGD.
        
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

        # Random start: initialize adversarial images with random perturbation
        if self.random_start:
            adv_images = images + torch.empty_like(images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
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

            # Project the perturbation back into the L-infinity epsilon ball
            delta = torch.clamp(adv_images - original_images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(original_images + delta, min=0, max=1).detach()

        return adv_images
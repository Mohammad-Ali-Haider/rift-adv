# src/attacks_project/attacks/base.py

import torch
import torch.nn as nn
from typing import Optional

class Attack:
    """
    Abstract base class for all adversarial attacks.

    Attributes:
        model (nn.Module): The model to be attacked.
        device (torch.device): The device the model is on (CPU or CUDA).
        targeted (bool): Whether to perform a targeted attack.
    """
    def __init__(self, model: nn.Module, targeted: bool = False):
        """
        Initializes the Attack class.

        Args:
            model (nn.Module): The neural network model to attack.
            targeted (bool): If True, performs targeted attack. Default is False (untargeted).
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.targeted = targeted

    def forward(self, images: torch.Tensor, labels: torch.Tensor, target_labels: Optional[torch.Tensor] = None):
        """
        The main method to generate adversarial examples.
        This must be implemented by all subclasses.

        Args:
            images: Input images to perturb.
            labels: True labels of the images (for untargeted attacks).
            target_labels: Target labels for targeted attacks (optional).

        Returns:
            Adversarial examples.
        """
        raise NotImplementedError

    def __call__(self, images: torch.Tensor, labels: torch.Tensor, target_labels: Optional[torch.Tensor] = None):
        """
        Makes the class callable, e.g., attack(images, labels).
        
        Args:
            images: Input images to perturb.
            labels: True labels (for untargeted) or target labels (for targeted if target_labels not provided).
            target_labels: Explicit target labels for targeted attacks (optional).
        
        Returns:
            Adversarial examples.
        """
        return self.forward(images, labels, target_labels)

    def _get_loss(self, outputs: torch.Tensor, labels: torch.Tensor, target_labels: Optional[torch.Tensor] = None):
        """
        Computes the loss for the attack.
        
        For untargeted attacks: maximize loss with respect to true labels.
        For targeted attacks: minimize loss with respect to target labels.
        
        Args:
            outputs: Model outputs (logits).
            labels: True labels.
            target_labels: Target labels for targeted attacks.
        
        Returns:
            Loss value (scalar tensor).
        """
        if self.targeted:
            # For targeted attacks, use target_labels if provided, otherwise use labels as targets
            targets = target_labels if target_labels is not None else labels
            # Minimize loss to make the model predict the target class
            loss = -nn.CrossEntropyLoss()(outputs, targets)
        else:
            # For untargeted attacks, maximize loss to misclassify
            loss = nn.CrossEntropyLoss()(outputs, labels)
        
        return loss
# src/attacks_project/attacks/cw.py

import torch
import torch.nn as nn
from typing import Optional
from .base import Attack

class CW(Attack):
    """
    Implements the Carlini & Wagner (C&W) L2 attack.
    
    C&W is an optimization-based attack that finds minimal L2 perturbations
    by solving an optimization problem with a specially designed loss function.
    
    Reference:
        Carlini and Wagner, "Towards Evaluating the Robustness of Neural Networks", 
        IEEE S&P 2017.
    
    Args:
        model: The neural network model to attack.
        c: Confidence parameter (trade-off between perturbation size and attack success).
        kappa: Confidence gap parameter.
        steps: Number of optimization steps.
        lr: Learning rate for the Adam optimizer.
        targeted: If True, performs a targeted attack. Default is False.
    
    Example:
        >>> attack = CW(model, c=1.0, steps=1000, targeted=True)
        >>> adv_images = attack(images, labels, target_labels)
    """
    def __init__(self, model: nn.Module, c: float = 1.0, kappa: float = 0, 
                 steps: int = 1000, lr: float = 0.01, targeted: bool = False):
        """
        Args:
            model: The neural network model to attack.
            c: Confidence parameter.
            kappa: Confidence gap parameter.
            steps: Number of optimization steps.
            lr: Learning rate.
            targeted: If True, performs a targeted attack.
        """
        super().__init__(model, targeted)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

    def _arctanh(self, x):
        """Inverse hyperbolic tangent (for optimization in tanh space)."""
        # Clamp to avoid numerical issues at the boundaries (x -> ±1)
        x = torch.clamp(x, -0.999999, 0.999999)
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _f(self, outputs, labels, target_labels=None):
        """
        C&W loss function.
        
        For targeted: max(max{Z_i : i != t} - Z_t, -kappa)
        For untargeted: max(Z_y - max{Z_i : i != y}, -kappa)
        """
        batch_size = outputs.shape[0]
        
        if self.targeted:
            # Targeted attack
            targets = target_labels if target_labels is not None else labels
            one_hot = torch.zeros_like(outputs)
            one_hot.scatter_(1, targets.unsqueeze(1), 1)
            
            # Get target class logits
            target_logits = (outputs * one_hot).sum(1)
            
            # Get max non-target logits
            other_logits = (outputs * (1 - one_hot) - one_hot * 10000).max(1)[0]
            
            # Loss: make other classes less confident than target
            loss = torch.clamp(other_logits - target_logits + self.kappa, min=0)
        else:
            # Untargeted attack
            one_hot = torch.zeros_like(outputs)
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            
            # Get true class logits
            true_logits = (outputs * one_hot).sum(1)
            
            # Get max other logits
            other_logits = (outputs * (1 - one_hot) - one_hot * 10000).max(1)[0]
            
            # Loss: make true class less confident than other classes
            loss = torch.clamp(true_logits - other_logits + self.kappa, min=0)
        
        return loss

    def forward(self, images: torch.Tensor, labels: torch.Tensor, target_labels: Optional[torch.Tensor] = None):
        """
        Generates adversarial examples using C&W attack.
        
        Args:
            images: Input images to perturb (values in [0, 1]).
            labels: True labels (for untargeted) or target labels (for targeted if target_labels not provided).
            target_labels: Explicit target labels for targeted attacks (optional).
        
        Returns:
            Adversarial examples.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
        
        # Convert images to tanh space for unbounded optimization
        # tanh(w) maps to [0, 1] when we use: (tanh(w) + 1) / 2
        # Use a clamped input to _arctanh to avoid infinities at exact 0/1
        tanh_input = (images - 0.5) / 0.5
        tanh_input = torch.clamp(tanh_input, -0.999999, 0.999999)
        w = self._arctanh(tanh_input)
        w = w.detach()
        w.requires_grad = True

        optimizer = torch.optim.Adam([w], lr=self.lr)

        best_adv = images.clone()
        best_l2 = float('inf') * torch.ones(images.shape[0]).to(self.device)

        for step in range(self.steps):
            # Convert from tanh space to [0, 1]
            adv_images = (torch.tanh(w) + 1) / 2
            
            # Get model outputs
            outputs = self.model(adv_images)
            
            # Compute C&W loss
            f_loss = self._f(outputs, labels, target_labels)
            
            # Compute squared L2 distance (more stable for optimization)
            l2_squared = ((adv_images - images) ** 2).sum(dim=[1, 2, 3])

            # Total loss: squared L2 plus weighted attack loss
            loss = l2_squared + self.c * f_loss
            loss = loss.sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update best adversarial examples
            with torch.no_grad():
                # Check which examples are successful
                pred = outputs.argmax(1)
                if self.targeted:
                    targets = target_labels if target_labels is not None else labels
                    successful = (pred == targets)
                else:
                    successful = (pred != labels)
                
                # Update best examples if better L2 and successful
                improved = successful & (l2_squared < best_l2)
                best_l2[improved] = l2_squared[improved]
                best_adv[improved] = adv_images[improved]
        
        return best_adv.detach()

# src/attacks_project/attacks/deepfool.py

import torch
import torch.nn as nn
from typing import Optional
from .base import Attack

class DeepFool(Attack):
    """
    Implements the DeepFool attack.
    
    DeepFool is an untargeted attack that finds the minimal perturbation needed
    to change the classification by iteratively linearizing the decision boundary.
    
    Reference:
        Moosavi-Dezfooli et al., "DeepFool: a simple and accurate method to fool 
        deep neural networks", CVPR 2016.
    
    Args:
        model: The neural network model to attack.
        steps: Maximum number of iterations.
        overshoot: Overshoot parameter to ensure the perturbation crosses the boundary.
        num_classes: Number of classes to consider. If None, uses all classes.
    
    Example:
        >>> attack = DeepFool(model, steps=50, overshoot=0.02)
        >>> adv_images = attack(images, labels)
    
    Note:
        DeepFool is inherently an untargeted attack. The 'targeted' parameter is
        not supported for this attack.
    """
    def __init__(self, model: nn.Module, steps: int = 50, overshoot: float = 0.02, 
                 num_classes: Optional[int] = None):
        """
        Args:
            model: The neural network model to attack.
            steps: Maximum number of iterations.
            overshoot: Overshoot parameter.
            num_classes: Number of classes to consider (None = all classes).
        """
        super().__init__(model, targeted=False)  # DeepFool is untargeted
        self.steps = steps
        self.overshoot = overshoot
        self.num_classes = num_classes

    def forward(self, images: torch.Tensor, labels: torch.Tensor, target_labels: Optional[torch.Tensor] = None):
        """
        Generates adversarial examples using DeepFool.
        
        Args:
            images: Input images to perturb.
            labels: True labels (used for reference only).
            target_labels: Not used (DeepFool is untargeted).
        
        Returns:
            Adversarial examples.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        batch_size = images.shape[0]
        adv_images = images.clone().detach()

        with torch.no_grad():
            outputs = self.model(adv_images)
            _, original_labels = torch.max(outputs, 1)
        
        # Process each image in the batch
        for i in range(batch_size):
            image = adv_images[i:i+1].clone()
            original_label = original_labels[i].item()
            
            perturbed_image = image.clone()
            
            for _ in range(self.steps):
                perturbed_image.requires_grad = True
                outputs = self.model(perturbed_image)
                
                # Get current prediction
                _, current_label = torch.max(outputs, 1)
                
                # If misclassified, stop
                if current_label.item() != original_label:
                    break
                
                # Compute gradients for all classes
                num_classes = outputs.shape[1] if self.num_classes is None else min(self.num_classes, outputs.shape[1])
                
                # Get top classes
                _, top_classes = torch.topk(outputs, k=num_classes, dim=1)
                
                # Find the minimal perturbation
                min_dist = float('inf')
                min_w = None
                
                for k in range(num_classes):
                    target_class = top_classes[0, k].item()
                    if target_class == original_label:
                        continue
                    
                    # Zero gradients
                    self.model.zero_grad()
                    if perturbed_image.grad is not None:
                        perturbed_image.grad.zero_()
                    
                    # Compute gradient of (f_k - f_original)
                    diff = outputs[0, target_class] - outputs[0, original_label]
                    diff.backward(retain_graph=True)
                    
                    grad = perturbed_image.grad.data.clone()
                    w = grad.view(-1)
                    f = diff.item()
                    
                    # Compute distance
                    dist = abs(f) / (torch.norm(w) + 1e-8)
                    
                    if dist < min_dist:
                        min_dist = dist
                        min_w = w / (torch.norm(w) + 1e-8)
                
                # Apply perturbation
                if min_w is not None:
                    r = (min_dist + 1e-4) * min_w
                    perturbed_image = perturbed_image.detach() + (1 + self.overshoot) * r.view_as(perturbed_image)
                    perturbed_image = torch.clamp(perturbed_image, min=0, max=1)
                else:
                    break
            
            adv_images[i:i+1] = perturbed_image.detach()
        
        return adv_images

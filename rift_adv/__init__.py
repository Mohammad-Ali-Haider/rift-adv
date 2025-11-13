"""
Attacks Project - A library for adversarial attacks on neural networks.

This library provides easy-to-use implementations of various adversarial attack
algorithms. Simply pass your model, specify the attack parameters, and generate
adversarial examples.

Example:
    >>> from attacks_project.attacks import FGSM, PGD
    >>> attack = FGSM(model, eps=8/255)
    >>> adv_images = attack(images, labels)
"""

__version__ = "0.1.0"

from .attacks import (
    Attack,
    FGSM,
    PGD,
    BIM,
    DeepFool,
    CW,
)

from . import utils

__all__ = [
    'Attack',
    'FGSM',
    'PGD',
    'BIM',
    'DeepFool',
    'CW',
    'utils',
]

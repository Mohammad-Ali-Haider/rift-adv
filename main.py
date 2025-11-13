# demo.py - Minimal Demo of Adversarial Attacks Library

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from rift_adv.attacks import FGSM, PGD, BIM, DeepFool, CW


# Simple model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load MNIST
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(test_data, batch_size=6, shuffle=True)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    
    # Load model
    model = SimpleCNN().to(device).eval()
    try:
        model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
    except:
        pass
    
    # Original predictions
    with torch.no_grad():
        outputs = model(images)
        original_preds = outputs.argmax(dim=1)

    print("Original Predictions:", original_preds.cpu().numpy())

    # Targeted attack: force model to predict next digit (0→1, 1→2, ..., 9→0)
    # target_labels = (labels + 1) % 10
    # print("Target Predictions:", target_labels.cpu().numpy())

    # Untargeted Attack
    # attacked_image = PGD(model, eps=0.3, steps=50, targeted=True)(images, labels, target_labels)
    # attacked = model(attacked_image)
    # attacked_preds = attacked.argmax(dim=1)
    # print("Attacked Predictions:", attacked_preds.cpu().numpy())

    attacked_image = PGD(model, eps=0.3, steps=50)(images, labels)
    attacked = model(attacked_image)
    attacked_preds = attacked.argmax(dim=1)
    print("Attacked Predictions:", attacked_preds.cpu().numpy())

    plt.figure(figsize=(10,4))
    for i in range(len(images)):
        plt.subplot(2, 6, i+1)
        plt.title(f"Orig: {original_preds[i].item()}")
        plt.imshow(images[i].detach().cpu().squeeze(), cmap='gray')
        plt.axis('off')

        plt.subplot(2, 6, i+7)
        plt.title(f"Adv: {attacked_preds[i].item()}")
        plt.imshow(attacked_image[i].detach().cpu().squeeze(), cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

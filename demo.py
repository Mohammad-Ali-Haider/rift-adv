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
    
    # Load MNIST
    test_data = datasets.MNIST('./data', train=False, download=True,
                               transform=transforms.ToTensor())
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
    
    # Run untargeted attacks
    attacks = {
        'FGSM': FGSM(model, eps=0.3),
        'PGD': PGD(model, eps=0.3, alpha=0.01, steps=40),
        'BIM': BIM(model, eps=0.3, alpha=0.01, steps=40),
        'DeepFool': DeepFool(model, steps=50, overshoot=0.02),
        'CW': CW(model, c=0.9, steps=1000, lr=0.01)
    }
    
    results = {}
    for name, attack in attacks.items():
        adv_images = attack(images, labels)
        with torch.no_grad():
            preds = model(adv_images).argmax(dim=1)
        results[name] = {'adv': adv_images, 'preds': preds}
        success = (preds != labels).sum().item()
        print(f"{name}: {success}/6 successful attacks")
    
    # Run targeted attacks (make model predict next digit: 0→1, 1→2, ..., 9→0)
    target_labels = (labels + 1) % 10
    print(f"\nTargeted attacks (forcing predictions to {target_labels.tolist()}):")
    
    targeted_attacks = {
        'T-FGSM': FGSM(model, eps=0.3, targeted=True),
        'T-PGD': PGD(model, eps=0.3, alpha=0.01, steps=40, targeted=True),
    }
    
    targeted_results = {}
    for name, attack in targeted_attacks.items():
        adv_images = attack(images, labels, target_labels)
        with torch.no_grad():
            preds = model(adv_images).argmax(dim=1)
        targeted_results[name] = {'adv': adv_images, 'preds': preds}
        success = (preds == target_labels).sum().item()
        print(f"{name}: {success}/6 successful targeted attacks")
    
    # Combine all results for visualization
    results.update(targeted_results)
    
    # Visualize adversarial examples
    n_attacks = len(results)  # Use total results (untargeted + targeted)
    fig, axes = plt.subplots(6, n_attacks + 1, figsize=(3 * (n_attacks + 1), 12))
    
    for i in range(6):
        # Original
        axes[i, 0].imshow(images[i].squeeze().detach().cpu().numpy(), cmap='gray')
        axes[i, 0].set_title(f'Original\n{labels[i].item()}→{original_preds[i].item()}', fontsize=9)
        axes[i, 0].axis('off')
        
        # All attacks
        for j, (name, result) in enumerate(results.items()):
            axes[i, j + 1].imshow(result['adv'][i].squeeze().detach().cpu().numpy(), cmap='gray')
            color = 'green' if result['preds'][i] != labels[i] else 'red'
            axes[i, j + 1].set_title(f'{name}\n→{result["preds"][i].item()}', fontsize=9, color=color)
            axes[i, j + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_results.png")
    
    # Visualize perturbation heatmaps
    fig2, axes2 = plt.subplots(6, n_attacks, figsize=(3 * n_attacks, 12))  # n_attacks already includes targeted
    
    for i in range(6):
        for j, (name, result) in enumerate(results.items()):
            # Compute perturbation
            perturbation = (result['adv'][i] - images[i]).detach().cpu().numpy().squeeze()
            
            # Normalize for visualization
            vmax = max(abs(perturbation.min()), abs(perturbation.max()))
            if vmax == 0:
                vmax = 1
            
            # Plot heatmap (red = increased pixels, blue = decreased pixels)
            im = axes2[i, j].imshow(perturbation, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes2[i, j].set_title(f'{name}', fontsize=9)
            axes2[i, j].axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=axes2[i, j], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('demo_heatmaps.png', dpi=150, bbox_inches='tight')
    print("Saved: demo_heatmaps.png")
    
    plt.show()


if __name__ == '__main__':
    main()

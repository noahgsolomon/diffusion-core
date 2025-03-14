import os
import torchvision
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

os.makedirs('data', exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

subset_size = 50000
indices = torch.randperm(len(dataset))[:subset_size].tolist()
subset_dataset = Subset(dataset, indices)

batch_size = 64
dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def show_samples(dataloader, num_samples=25):
    images, labels = next(iter(dataloader))
    images = images[:num_samples]
    
    images = images * 0.5 + 0.5
    
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].permute(1, 2, 0).cpu())
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    plt.close()
    print(f"Sample images saved to 'sample_images.png'")
    
    return images

if __name__ == "__main__":
    print(f"Dataset size: {len(subset_dataset)} images")
    samples = show_samples(dataloader)

    print("Dataset ready for training!")
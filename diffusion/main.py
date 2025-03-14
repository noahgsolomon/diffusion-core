import torch
import matplotlib.pyplot as plt
from ddpm import DDPM, UNet
from data import dataloader
import os

os.makedirs('models', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ddpm = DDPM(noise_steps=1000, img_size=32, device=device)
model = UNet(in_channels=3, time_dim=256, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

ddpm.train(model, dataloader, optimizer, num_epochs=50)

samples = ddpm.sample(model, n=16)

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(samples[i].cpu().permute(1, 2, 0))
    plt.axis('off')
plt.tight_layout()
plt.savefig('generated_samples.png')
plt.show()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Set up device and hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 20
latent_dim = 20  # Size of the latent space

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# VAE Model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), 
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256)
        self.decoder = nn.Sequential(
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()  # Output values between 0 and 1
        )
        
    def encode(self, x):
        # Encode input to get mean and log variance of latent distribution
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # Reparameterization trick: z = mu + std * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        # Decode latent vector to reconstruct input
        z = self.decoder_input(z)
        reconstruction = self.decoder(z)
        return reconstruction
    
    def forward(self, x):
        # Full forward pass
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss (binary cross entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

def train_vae():
    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    model = VAE(latent_dim=latent_dim).to(device)
    print(model)
    
    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    test_losses = []
    
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch}')
        for batch_idx, (data, _) in progress_bar:
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item() / len(data)})
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        print(f'====> Epoch: {epoch} Average training loss: {avg_train_loss:.4f}')
        
        # Testing
        model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                test_loss += vae_loss(recon_batch, data, mu, logvar).item()
        
        avg_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
        print(f'====> Test set loss: {avg_test_loss:.4f}')
        
        # Save a checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'vae_epoch_{epoch}.pth')
    
    # Visualize training progress
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('vae_training_loss.png')
    plt.show()
    
    return model, train_loader, test_loader

def visualize_reconstructions(model, data_loader):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(data_loader))
        data = data.to(device)
        recon_batch, _, _ = model(data)
        
        # Move tensors to CPU for visualization
        data = data.cpu()
        recon_batch = recon_batch.cpu()
        
        # Plot original and reconstructed images
        plt.figure(figsize=(12, 6))
        for i in range(10):
            # Original images
            plt.subplot(2, 10, i + 1)
            plt.imshow(data[i].squeeze().numpy(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Original')
                
            # Reconstructed images
            plt.subplot(2, 10, i + 11)
            plt.imshow(recon_batch[i].squeeze().numpy(), cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Reconstructed')
                
        plt.tight_layout()
        plt.savefig('vae_reconstructions.png')
        plt.show()

def generate_digits(model, num_samples=10):
    model.eval()
    with torch.no_grad():
        # Sample from standard normal distribution
        z = torch.randn(num_samples, latent_dim).to(device)
        # Decode the latent vectors
        samples = model.decode(z).cpu()
        
        # Plot generated digits
        plt.figure(figsize=(12, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(samples[i].squeeze().numpy(), cmap='gray')
            plt.axis('off')
        plt.suptitle('Generated Digits')
        plt.tight_layout()
        plt.savefig('vae_generated_digits.png')
        plt.show()

if __name__ == "__main__":
    # Train the VAE
    model, train_loader, test_loader = train_vae()
    
    # Visualize reconstructions
    visualize_reconstructions(model, test_loader)
    
    # Generate new digits
    generate_digits(model) 
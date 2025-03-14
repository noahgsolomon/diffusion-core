import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
    
    def forward(self, t):
        # Create embedding of timestep
        t = t.unsqueeze(-1).float()
        t_embed = self.time_embed(t)
        return t_embed

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        
    def forward(self, x, t_emb):
        # First convolution
        h = F.silu(self.norm1(self.conv1(x)))
        
        # Time embedding
        time_emb = self.time_mlp(t_emb)
        time_emb = time_emb.view(-1, time_emb.shape[1], 1, 1)
        h = h + time_emb
        
        # Second convolution
        h = F.silu(self.norm2(self.conv2(h)))
        return h

class UNet(nn.Module):
    def __init__(self, in_channels=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_dim)
        
        # Downsampling path
        self.down1 = ConvBlock(in_channels, 64, time_dim)
        self.down2 = ConvBlock(64, 128, time_dim)
        self.down3 = ConvBlock(128, 256, time_dim)
        self.down4 = ConvBlock(256, 512, time_dim)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024, time_dim)
        
        # Upsampling path
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv_up4 = ConvBlock(1024, 512, time_dim)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up3 = ConvBlock(512, 256, time_dim)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up2 = ConvBlock(256, 128, time_dim)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up1 = ConvBlock(128, 64, time_dim)
        
        # Output layer
        self.output = nn.Conv2d(64, in_channels, 1)
        
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embedding(t)
        
        # Downsampling
        d1 = self.down1(x, t_emb)
        x = self.pool(d1)
        
        d2 = self.down2(x, t_emb)
        x = self.pool(d2)
        
        d3 = self.down3(x, t_emb)
        x = self.pool(d3)
        
        d4 = self.down4(x, t_emb)
        x = self.pool(d4)
        
        # Bottleneck
        x = self.bottleneck(x, t_emb)
        
        # Upsampling with skip connections
        x = self.up4(x)
        x = torch.cat([x, d4], dim=1)
        x = self.conv_up4(x, t_emb)
        
        x = self.up3(x)
        x = torch.cat([x, d3], dim=1)
        x = self.conv_up3(x, t_emb)
        
        x = self.up2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.conv_up2(x, t_emb)
        
        x = self.up1(x)
        x = torch.cat([x, d1], dim=1)
        x = self.conv_up1(x, t_emb)
        
        # Output layer - predict the noise
        x = self.output(x)
        
        return x

class DDPM:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.noise_variance = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        
        self.signal_rate = 1. - self.noise_variance
        
        self.cumulative_signal_rate = torch.cumprod(self.signal_rate, dim=0)
        
        self.sqrt_cumulative_signal = torch.sqrt(self.cumulative_signal_rate)
        self.sqrt_cumulative_noise = torch.sqrt(1. - self.cumulative_signal_rate)
    
    
    def noise_images(self, x, t):

        random_noise = torch.randn_like(x).to(self.device)
        
        signal_scaling = self.sqrt_cumulative_signal[t].view(-1, 1, 1, 1)
        noise_scaling = self.sqrt_cumulative_noise[t].view(-1, 1, 1, 1)
        
        noised_image = signal_scaling * x + noise_scaling * random_noise

        return noised_image, random_noise
    
    def sample(self, model, n):

        model.eval()
        
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            
            # Iterate from the highest noise level to the lowest
            for i in reversed(range(self.noise_steps)):
                # Create a batch of the same timestep
                t = torch.full((n,), i, device=self.device, dtype=torch.long)
                
                # Get the noise predicted by the model
                predicted_noise = model(x, t)
                
                # Get the denoising coefficients for this timestep
                noise_level = self.noise_variance[i]
                signal_keep_rate = self.signal_rate[i]
                
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = (1 / torch.sqrt(signal_keep_rate)) * (
                    x - (noise_level / torch.sqrt(1 - self.cumulative_signal_rate[i])) * predicted_noise
                ) + torch.sqrt(noise_level) * noise
            
            x = (x.clamp(-1, 1) + 1) / 2
            
            return x
    
    def p_losses(self, denoise_model, x_0):
        batch_size = x_0.shape[0]
        
        random_timesteps = torch.randint(0, self.noise_steps, (batch_size,), device=self.device).long()
        
        noisy_images, original_noise = self.noise_images(x_0, random_timesteps)
        
        predicted_noise = denoise_model(noisy_images, random_timesteps)
        
        
        loss = torch.nn.functional.mse_loss(original_noise, predicted_noise)
        
        return loss

    def train(self, denoise_model, dataloader, optimizer, num_epochs, save_path="models/ddpm_model.pth"):

        denoise_model.train()
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx, (images, _) in enumerate(dataloader):
                images = images.to(self.device)
                
                optimizer.zero_grad()
                
                loss = self.p_losses(denoise_model, images)
                
                loss.backward()
                
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_epoch_loss:.4f}")
            
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(denoise_model.state_dict(), save_path)
                print(f"Model saved at {save_path}")
                
        print("Training completed!")
        return denoise_model
    
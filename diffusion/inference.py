import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
from ddpm import DDPM, UNet

def generate_samples(num_samples=16, model_path="models/ddpm_model.pth", save_dir="generated_images"):
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    img_size = 32
    model = UNet(in_channels=3, time_dim=256, device=device).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model not found at {model_path}. Please provide a valid model path.")
        return
    
    ddpm = DDPM(noise_steps=1000, img_size=img_size, device=device)
    
    print(f"Generating {num_samples} samples...")
    model.eval()
    samples = ddpm.sample(model, num_samples)
    
    for i, sample in enumerate(samples):
        img = sample.cpu().permute(1, 2, 0).numpy()
        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(f"{save_dir}/sample_{i+1}.png", bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    grid = make_grid(samples, nrow=int(num_samples**0.5))
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.cpu().permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.savefig(f"{save_dir}/sample_grid.png")
    plt.close()
    
    print(f"Samples saved to {save_dir}")
    return samples

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate samples from trained DDPM model")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--model_path", type=str, default="models/ddpm_model.pth", help="Path to trained model")
    parser.add_argument("--save_dir", type=str, default="generated_images", help="Directory to save generated images")
    
    args = parser.parse_args()
    
    generate_samples(
        num_samples=args.num_samples,
        model_path=args.model_path,
        save_dir=args.save_dir
    )
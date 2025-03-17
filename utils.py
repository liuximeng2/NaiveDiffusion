import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
import os

def denormalize(images):
    """
    Denormalize images from [-1, 1] to [0, 1] range
    """
    return (images + 1) / 2

def show_image(image, title=None):
    """
    Display a single image
    
    Args:
        image (torch.Tensor): Image tensor of shape [C, H, W]
        title (str, optional): Title for the plot
    """
    if image.dim() == 4:
        image = image[0]  # Take first image if it's a batch
    
    # Denormalize and convert to numpy for display
    image = denormalize(image)
    image = image.permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

def show_images(images, nrow=8, title=None):
    """
    Display a grid of images
    
    Args:
        images (torch.Tensor): Image tensor of shape [B, C, H, W]
        nrow (int): Number of images per row
        title (str, optional): Title for the plot
    """
    # Denormalize and create grid
    images = denormalize(images)
    grid = make_grid(images, nrow=nrow)
    
    # Convert to numpy for display
    grid = grid.permute(1, 2, 0).cpu().numpy()
    grid = np.clip(grid, 0, 1)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

def save_images(images, path, filename, nrow=8):
    """
    Save a grid of images to disk
    
    Args:
        images (torch.Tensor): Image tensor of shape [B, C, H, W]
        path (str): Directory to save the image
        filename (str): Filename for the saved image
        nrow (int): Number of images per row
    """
    os.makedirs(path, exist_ok=True)
    
    # Denormalize and create grid
    images = denormalize(images)
    grid = make_grid(images, nrow=nrow)
    
    # Convert to PIL image and save
    grid = grid.permute(1, 2, 0).cpu().numpy()
    grid = np.clip(grid * 255, 0, 255).astype(np.uint8)
    
    # Handle single channel images
    if grid.shape[2] == 1:
        grid = grid[:, :, 0]
    
    Image.fromarray(grid).save(os.path.join(path, filename))
    print(f"Image saved at {os.path.join(path, filename)}")

def visualize_denoising_process(model, diffusion, n_steps=10, size=(3, 32, 32), device="cuda"):
    """
    Visualize the denoising process from random noise to image
    
    Args:
        model: The UNet model
        diffusion: The diffusion model
        n_steps (int): Number of steps to visualize
        size (tuple): Size of the image (C, H, W)
        device (str): Device to run on
    """
    model.eval()
    
    # Start with random noise
    img = torch.randn(1, *size).to(device)
    plt.figure(figsize=(15, 3))
    
    # Visualize original noise
    plt.subplot(1, n_steps+1, 1)
    img_np = img[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    plt.imshow(img_np)
    plt.title("Noise")
    plt.axis('off')
    
    # Show denoising steps
    steps = list(range(diffusion.noise_steps-1, 0, -diffusion.noise_steps//n_steps))[:n_steps]
    for i, step in enumerate(steps):
        t = torch.tensor([step]).to(device)
        with torch.no_grad():
            img = diffusion.sample_timestep(model, img, t)
        
        plt.subplot(1, n_steps+1, i+2)
        img_np = denormalize(img[0]).permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        plt.imshow(img_np)
        plt.title(f"Step {step}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

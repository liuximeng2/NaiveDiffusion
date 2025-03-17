import os
import torch
import torch.nn as nn
from torch import optim
import torchvision
import torchvision.transforms as transforms

from model.module import UNet
from model.diffusion import Diffusion
from utils import save_images

def train():
    # Define the transform (convert to tensor, normalize, etc.)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    """
    Inside trainset, there are 50000 sample tuples. Each tuple contains an image and a label.
    The image is a 3x32x32 tensor (3 channels, 32x32 pixels).
    The label is an integer from 0 to 9.
    Classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    """
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    device = "cuda:2"
    epochs = 5

    model = UNet(in_channels=3, out_channels=32, time_dim=256, device=device).to(device)
    diffusion = Diffusion(noise_steps=1000, img_size=32, device=device)
    mse = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        total_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            t = diffusion.sample_timesteps(inputs.shape[0])
            x_t, noise = diffusion.noise_images(inputs, t)

            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 200 == 199:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {(total_loss / 200):.3f}")
                total_loss = 0.0

    # Save the model
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/diffusion.pth")

def generate():
    device = "cuda:2"
    model = UNet(in_channels=3, out_channels=32, time_dim=256, device=device).to(device)
    model.load_state_dict(torch.load("saved_models/diffusion.pth"))
    model.eval()

    diffusion = Diffusion(noise_steps=1000, img_size=32, device=device)

    n = 16 # must be exponent of 2
    images = diffusion.sample(model, n)
    save_images(images, "images", "generated_image.png")

if __name__ == "__main__":
    train()
    generate()
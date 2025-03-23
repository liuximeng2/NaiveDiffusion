import os
import time
import torch
import torch.nn as nn
from torch import optim
import torchvision
import torchvision.transforms as transforms

from model.module import UNet
from model.diffusion import Diffusion
from utils import save_images

def train(args, generate=True):
    # Define the transform (convert to tensor, normalize, etc.)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    """
    Inside trainset, there are 50000 sample tuples. Each tuple contains an image and a label.
    The image is a 3x32x32 tensor (3 channels, 32x32 pixels).
    The label is an integer from 0 to 9.
    Classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    """
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    device = args.device
    print(f"Device: {device}")
    model = UNet(in_channels=3, out_channels=64, time_dim=args.time_dim, device=args.device).to(args.device)
    diffusion = Diffusion(noise_steps=args.noise_steps, img_size=args.img_size, device=args.device)
    mse = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            t = diffusion.sample_timesteps(inputs.shape[0]) # sample timesteps for each batch
            x_t, noise = diffusion.noise_images(inputs, t)

            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            if i % 200 == 199:
                print(f"Epoch: {epoch}, Batch: {i + 1}, Loss: {(total_loss / 200):.3f}")
                total_loss = 0.0

        # Save the model for every 10 epochs
        if epoch % 10 == 9:
            torch.save(model.state_dict(), f"saved_models/checkpoint@epoch{epoch + 1}.pth")
            if generate:
                model.eval()
                images = diffusion.sample(model, args.sample_num)
                save_images(images, "images", f"checkpoint@epoch{epoch + 1}_generated.png")


def generate(args, device="cuda"):
    model_name = "checkpoint@epoch49"
    device = device
    model = UNet(in_channels=3, out_channels=64, time_dim=256, device=device).to(device)
    model.load_state_dict(torch.load(f"saved_models/{model_name}.pth"))
    model.eval()

    diffusion = Diffusion(noise_steps=args.noise_steps, img_size=64, device=device)

    n = 16 # must be exponent of 2
    images = diffusion.sample(model, n)
    save_images(images, "images", f"{model_name}_generated.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.time_dim = 256
    args.img_size = 64
    args.noise_steps = 1000
    args.device = "cuda:2"
    args.sample_num = 16

    args.lr = 3e-4
    args.batch_size = 16
    args.epochs = 100
    train(args, True)
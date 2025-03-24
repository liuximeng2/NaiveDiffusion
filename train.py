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
    """
    Inside trainset, there are 50000 sample tuples. Each tuple contains an image and a label.
    The image is a 3x32x32 tensor (3 channels, 32x32 pixels).
    The label is an integer from 0 to 9.
    Classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    """
    if args.dataset == "cifar10":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(80),
            torchvision.transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        in_channels, decoder_out = 3, 3
    elif args.dataset == "mnist":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Pad(2),
            ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        in_channels, decoder_out = 1, 1
    else:
        raise ValueError("Invalid dataset")
    
    image_save_path = os.path.join("images", args.dataset)
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    model_save_path = os.path.join("saved_models", args.dataset)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    #select the subset with the label 'airplane'
    trainset_partial = torch.utils.data.Subset(trainset, [i for i in range(len(trainset)) if trainset[i][1] == 0])
    print(f"Number of samples: {len(trainset_partial)}")
    trainloader = torch.utils.data.DataLoader(trainset_partial, batch_size=args.batch_size, shuffle=True)

    device = args.device
    print(f"Device: {device}")
    model = UNet(in_channels=in_channels, out_channels=64, decoder_out=decoder_out, time_dim=args.time_dim, device=args.device).to(args.device)
    diffusion = Diffusion(noise_steps=args.noise_steps, img_size=args.img_size, decoder_channel=decoder_out, device=args.device)
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
            optimizer.step()

            total_loss += loss.item()
            if i % 200 == 199:
                print(f"Epoch: {epoch}, Batch: {i + 1}, Loss: {(total_loss / 200):.3f}")
                total_loss = 0.0

        if epoch % 10 == 9:
            if generate:
                model.eval()
                images = diffusion.sample(model, args.sample_num)
                save_images(images, f"images/{args.dataset}", f"checkpoint@epoch{epoch + 1}_generated.png")
                torch.save(model.state_dict(), f"saved_models/{args.dataset}/checkpoint@epoch{epoch + 1}.pth")


def generate(args):
    model_name = "unconditional_ckpt.pt"
    model = UNet(in_channels=3, out_channels=64, time_dim=args.time_dim, device=args.device).to(args.device)
    model.load_state_dict(torch.load(f"saved_models/{model_name}"))
    model.eval()

    diffusion = Diffusion(noise_steps=args.noise_steps, img_size=64, device=args.device)

    n = 16 # must be exponent of 2
    images = diffusion.sample(model, n)
    save_images(images, "images", f"{model_name}_generated.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.dataset = "mnist"

    args.time_dim = 256
    args.img_size = 64
    args.noise_steps = 1000
    args.device = "cuda:2"
    args.sample_num = 16

    args.lr = 3e-4
    args.batch_size = 16
    args.epochs = 500
    train(args, True)
    # generate(args)
# NaiveDiffusion

NaiveDiffusion is a self-learning project focused on implementing a diffusion model for image generation. This implementation closely follows the original algorithms presented in the [DDPM paper](https://arxiv.org/pdf/2006.11239) and adopts the U-Net architecture from the [U-Net paper](https://arxiv.org/pdf/1505.04597), enhanced with a multi-head attention block inserted between convolution and downsampling/upsampling operations.

## 📘 Overview

The diffusion model minimizes the KL divergence between the forward noise process \( q(x_t \mid x_{t-1}) \) and the reverse process \( p(x_{t-1} \mid x_t) \). Practically, this is done by minimizing the mean squared error (MSE) between the generated noise and the predicted noise.

The predicted noise is output by a parameterized U-Net, which takes in a noised image along with its corresponding timestep. The forward noise process is computed using formulas defined in the DDPM paper, and is encapsulated in the `Diffusion` class.

## 📂 Repository Structure

The repo consists of the following key Python files:

- `train.py`:  
  Implements the training loop for the diffusion model using selected datasets such as CIFAR-10 or MNIST.

- `model/diffusion.py`:  
  Defines the `Diffusion` class, which handles the forward noise process and denoising during image generation.

- `model/module.py`:  
  Implements the `UNet` class, which learns the mapping function from Gaussian noise space to the training data distribution. This version includes a multi-head attention block for improved feature learning.

## 🛠️ Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/liuximeng09/NaiveDiffusion.git
   cd NaiveDiffusion
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   ```bash
   python train.py  # remember to select your dataset, autodownload
   ```

## 📄 References

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/pdf/2006.11239)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597)
- [Diffusion-Models-pytorch](https://github.com/dome272/Diffusion-Models-pytorch)
- [Diffusion Models | PyTorch Implementation](https://www.youtube.com/watch?v=TBCRlnwJtZU&t=547s)

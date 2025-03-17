import torch
from tqdm import tqdm

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device) # linear beta schedule
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) # Cumulative product

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        x: Tensor of shape (n, 3, img_size, img_size)
        t: Tensor of shape (n,)
        The forward nosie process
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # [:, None, None, None] to add three extra dimensions.
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        # corresponds to the one-step diffusion process formula in the paper
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)

    def sample(self, model, n): # this is the inferece step
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device) # (n, 3, img_size, img_size)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): # for T = T - 1, denoise for each step
                t = (torch.ones(n) * i).long().to(self.device) # (n,)
                predicted_noise = model(x, t) # given an image and a noise step, the model predicts the noise, predicted noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None] # [:, None, None, None] to add three extra dimensions.
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x) # added random noise
                else:
                    noise = torch.zeros_like(x) # do not add noise at step 0
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8) # cast to valid image pixels
        return x
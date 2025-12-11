import torch
import math

class DDPMScheduler:
    """
    Denoising Diffusion Probabilistic Models (DDPM) Scheduler.
    Linear Beta Schedule.
    """
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute SQRT terms
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.posterior_variance = self.betas * (1. - torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])) / (1. - self.alphas_cumprod)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def add_noise(self, original_samples, noise, timesteps):
        """
        Forward process: q(x_t | x_0)
        """
        # Make sure timesteps are on correct device and shape
        # original_samples: (B, C, L)
        # timesteps: (B,)
        
        # Gather
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting (B, 1, 1)
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()[:, None, None]
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()[:, None, None]
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output, timestep, sample):
        """
        Predict x_{t-1} from x_t and model_output (epsilon).
        """
        t = timestep
        
        # 1. Compute Coeffs
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_prod_t = self.alphas_cumprod[t]
        
        # 2. Predict x0 (optional but part of derivation) or mean
        # mean = 1/sqrt(alpha) * (x_t - beta / sqrt(1-alpha_prod) * eps)
        
        target_shape = sample.shape
        coeff1 = self.sqrt_recip_alphas[t]
        coeff2 = beta_t / self.sqrt_one_minus_alphas_cumprod[t]
        
        pred_mean = coeff1 * (sample - coeff2 * model_output)
        
        # 3. Add noise (if t > 0)
        variance = 0
        if t > 0:
            noise = torch.randn_like(sample)
            # variance = self.betas[t] # Simple version
            variance = self.posterior_variance[t] # Correct variance
            std = torch.sqrt(variance)
            pred_sample = pred_mean + std * noise
        else:
            pred_sample = pred_mean
            
        return pred_sample

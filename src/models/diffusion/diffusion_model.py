import torch
import torch.nn as nn
from src.models.base_model import BaseTrajectoryModel
from src.models.diffusion.unet1d import UNet1D
from src.models.diffusion.scheduler import DDPMScheduler

class DiffusionTrajectoryModel(BaseTrajectoryModel):
    """
    Data-only Trajectory Diffusion Model.
    Architecture: 1D UNet + DDPM.
    Conditioning: Encoded History + Global Cond.
    """
    def __init__(self, 
                 obs_dim: int = 4, 
                 act_dim: int = 2, 
                 cond_dim: int = 6, 
                 obs_len: int = 8,
                 pred_len: int = 12,
                 hidden_dim: int = 64,
                 diffusion_steps: int = 100):
        super().__init__()
        
        self.pred_len = pred_len
        self.act_dim = act_dim
        
        # Condition Encoder: History(Flattened) + Cond -> Emb
        hist_flat_dim = obs_len * obs_dim
        input_cond_dim = hist_flat_dim + cond_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(input_cond_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4) # Matches UNet emb_dim usually?
        )
        
        # UNet
        # Input channels = act_dim (Velocity sequences)
        # Condition dim = hidden_dim * 4
        self.unet = UNet1D(
            in_dim=act_dim,
            model_dim=hidden_dim,
            emb_dim=hidden_dim * 4,
            dim_mults=(1, 2, 4),
            cond_dim=hidden_dim * 4
        )
        
        # Scheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=diffusion_steps)

    def to(self, device):
        super().to(device)
        self.scheduler.to(device)
        return self

    def get_global_cond(self, obs, cond):
        """Flatten obs and concat with cond."""
        B = obs.shape[0]
        # obs: (B, H, 4) -> (B, H*4)
        obs_flat = obs.reshape(B, -1)
        # cond: (B, 6)
        x = torch.cat([obs_flat, cond], dim=-1)
        return self.cond_encoder(x)

    def forward(self, obs, cond, target=None):
        """
        Target is FUTURE velocities (B, F, 2).
        Returns diffusion loss.
        """
        if target is None:
            # Cannot train without target
            return torch.tensor(0.0, device=obs.device)
            
        B = obs.shape[0]
        device = obs.device
        
        # 1. Prepare Inputs
        x_0 = target.permute(0, 2, 1) # (B, 2, F) for Conv1d
        
        # 2. Sample Noise
        noise = torch.randn_like(x_0)
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=device).long()
        
        # 3. Add Noise
        self.scheduler.to(device) # Ensure scheduler on device
        x_t = self.scheduler.add_noise(x_0, noise, timesteps)
        
        # 4. Predict Noise
        global_cond = self.get_global_cond(obs, cond) # (B, emb_dim)
        
        noise_pred = self.unet(x_t, timesteps, cond=global_cond)
        
        # 5. Loss
        return nn.functional.mse_loss(noise_pred, noise)

    def sample_trajectory(self, obs, cond, horizon, **kwargs):
        """
        Reverse diffusion sampling.
        """
        B = obs.shape[0]
        device = obs.device
        self.scheduler.to(device)
        
        # 1. Prepare Condition
        global_cond = self.get_global_cond(obs, cond)
        
        # 2. Random Noise
        # Shape: (B, Act_Dim, Horizon)
        shape = (B, self.act_dim, horizon)
        x_t = torch.randn(shape, device=device)
        
        # 3. Denoise Loop
        for t in reversed(range(self.scheduler.num_train_timesteps)):
            # Broadcast timestep
            ts = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.unet(x_t, ts, cond=global_cond)
            
            # Step
            x_t = self.scheduler.step(noise_pred, t, x_t)
            
        # 4. Return (B, F, 2)
        return x_t.permute(0, 2, 1)

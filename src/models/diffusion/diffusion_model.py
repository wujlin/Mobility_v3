import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
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
                 diffusion_steps: int = 100,
                 prediction_type: str = "eps"):
        super().__init__()
        
        self.pred_len = pred_len
        self.act_dim = act_dim
        prediction_type = str(prediction_type)
        if prediction_type not in ("eps", "v"):
            raise ValueError(f"Unknown prediction_type: {prediction_type} (expected: eps|v)")
        self.prediction_type = prediction_type
        
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

    def compute_loss(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor,
        target: torch.Tensor,
        *,
        sample_weight: Optional[torch.Tensor] = None,
        return_x0_pred: bool = False,
        return_timesteps: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Compute diffusion training loss. Optionally return the predicted clean sample x0_pred and/or the sampled timesteps.

        Args:
            obs: (B, H, obs_dim)
            cond: (B, cond_dim)
            target: (B, F, act_dim) future velocities (normalized, step displacement)
            return_x0_pred: if True, also return x0_pred (B, act_dim, F)
            return_timesteps: if True, also return timesteps (B,)

        Returns:
            loss if return_x0_pred=False and return_timesteps=False
            (loss, x0_pred) if return_x0_pred=True and return_timesteps=False
            (loss, timesteps) if return_x0_pred=False and return_timesteps=True
            (loss, x0_pred, timesteps) if return_x0_pred=True and return_timesteps=True
        """
        B = obs.shape[0]
        device = obs.device

        x_0 = target.permute(0, 2, 1)  # (B, act_dim, F) for Conv1d

        noise = torch.randn_like(x_0)
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=device).long()

        self.scheduler.to(device)
        x_t = self.scheduler.add_noise(x_0, noise, timesteps)

        global_cond = self.get_global_cond(obs, cond)  # (B, emb_dim)
        model_out = self.unet(x_t, timesteps, cond=global_cond)

        sqrt_alpha_prod = self.scheduler.sqrt_alphas_cumprod[timesteps].flatten()[:, None, None]
        sqrt_one_minus_alpha_prod = self.scheduler.sqrt_one_minus_alphas_cumprod[timesteps].flatten()[:, None, None]

        # Train target depends on parameterization:
        # - eps: target is Gaussian noise epsilon
        # - v:   target is v = alpha*epsilon - sigma*x0
        if self.prediction_type == "v":
            target_out = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * x_0
        else:
            target_out = noise

        # Per-sample MSE for optional weighting (mitigate low-displacement dominance).
        per = (model_out - target_out) ** 2  # (B, C, F)
        per = per.mean(dim=(1, 2))       # (B,)
        if sample_weight is not None:
            w = sample_weight.to(dtype=per.dtype, device=per.device).flatten()
            w = torch.clamp_min(w, 1e-6)
            diff_loss = (per * w).sum() / w.sum()
        else:
            diff_loss = per.mean()

        if not return_x0_pred and not return_timesteps:
            return diff_loss

        # x0_pred reconstruction:
        # - eps: x0 = (x_t - sigma*eps) / alpha
        # - v:   x0 = alpha*x_t - sigma*v
        if self.prediction_type == "v":
            x0_pred = sqrt_alpha_prod * x_t - sqrt_one_minus_alpha_prod * model_out
        else:
            x0_pred = (x_t - sqrt_one_minus_alpha_prod * model_out) / (sqrt_alpha_prod + 1e-8)

        if return_x0_pred and return_timesteps:
            return diff_loss, x0_pred, timesteps
        if return_x0_pred:
            return diff_loss, x0_pred
        return diff_loss, timesteps

    def forward(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Target is FUTURE velocities (B, F, 2).
        Returns diffusion loss.
        """
        if target is None:
            # Cannot train without target
            return torch.tensor(0.0, device=obs.device)

        return self.compute_loss(obs, cond, target, sample_weight=sample_weight, return_x0_pred=False)

    def sample_trajectory(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor,
        horizon: int,
        *,
        cond_uncond: Optional[torch.Tensor] = None,
        cfg_scale: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Reverse diffusion sampling.

        Args:
            obs: (B, H, 4)
            cond: (B, cond_dim)
            horizon: future length F
            cond_uncond: (B, cond_dim), unconditional condition for CFG (optional)
            cfg_scale: CFG guidance scale; 0 disables CFG
        """
        B = obs.shape[0]
        device = obs.device
        self.scheduler.to(device)

        # 1. Prepare Condition
        global_cond = self.get_global_cond(obs, cond)
        use_cfg = (cond_uncond is not None) and (float(cfg_scale) != 0.0)
        global_cond_uncond = None
        if use_cfg:
            global_cond_uncond = self.get_global_cond(obs, cond_uncond.to(device=device, dtype=cond.dtype))

        # 2. Random Noise
        # Shape: (B, Act_Dim, Horizon)
        shape = (B, self.act_dim, horizon)
        x_t = torch.randn(shape, device=device)

        # 3. Denoise Loop
        for t in reversed(range(self.scheduler.num_train_timesteps)):
            # Broadcast timestep
            ts = torch.full((B,), t, device=device, dtype=torch.long)

            if use_cfg:
                # out = out_u + s*(out_c - out_u)
                out_u = self.unet(x_t, ts, cond=global_cond_uncond)
                out_c = self.unet(x_t, ts, cond=global_cond)
                model_out = out_u + float(cfg_scale) * (out_c - out_u)
            else:
                model_out = self.unet(x_t, ts, cond=global_cond)

            # Convert v-prediction to epsilon for DDPM step (scheduler expects epsilon).
            if self.prediction_type == "v":
                alpha = self.scheduler.sqrt_alphas_cumprod[t].to(device=device, dtype=x_t.dtype)
                sigma = self.scheduler.sqrt_one_minus_alphas_cumprod[t].to(device=device, dtype=x_t.dtype)
                # epsilon = sigma*x_t + alpha*v
                eps_pred = sigma * x_t + alpha * model_out
            else:
                eps_pred = model_out
            
            # Step
            x_t = self.scheduler.step(eps_pred, t, x_t)
            
        # 4. Return (B, F, 2)
        return x_t.permute(0, 2, 1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from src.models.base_model import BaseTrajectoryModel


class SeqCVAE(BaseTrajectoryModel):
    """
    Conditional VAE baseline for multi-modal future velocity generation.

    - Condition: (hour, weekday, trip_origin, trip_destination) => cond_dim=6
    - Input: observed sequence obs (B, H, 4) where 4=[pos(2), vel(2)]
    - Output: future vel (B, F, 2)
    """

    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 2,
        cond_dim: int = 6,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        num_layers: int = 1,
    ):
        super().__init__()
        if int(num_layers) != 1:
            raise ValueError("SeqCVAE 目前仅支持 num_layers=1（保持实现简洁）")

        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)

        # Encode observed history.
        self.obs_encoder = nn.LSTM(
            input_size=obs_dim + cond_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # Encode future (teacher-forced) velocities for posterior.
        self.future_encoder = nn.LSTM(
            input_size=act_dim + cond_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # Conditional prior p(z | obs, cond) and posterior q(z | obs, cond, future).
        self.prior_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.prior_logvar = nn.Linear(self.hidden_dim, self.latent_dim)
        self.post_mu = nn.Linear(self.hidden_dim * 2, self.latent_dim)
        self.post_logvar = nn.Linear(self.hidden_dim * 2, self.latent_dim)

        # Autoregressive decoder in velocity space: v_t -> v_{t+1}
        self.decoder_cell = nn.LSTMCell(act_dim + cond_dim + self.latent_dim, self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, act_dim)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _kl_diag_gaussians(
        mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor
    ) -> torch.Tensor:
        logvar_q = torch.clamp(logvar_q, min=-10.0, max=10.0)
        logvar_p = torch.clamp(logvar_p, min=-10.0, max=10.0)
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl_per_sample = 0.5 * torch.sum(
            logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0, dim=1
        )
        return kl_per_sample.mean()

    def _encode_obs(self, obs: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, _ = obs.shape
        cond_expanded = cond.unsqueeze(1).repeat(1, H, 1)
        enc_input = torch.cat([obs, cond_expanded], dim=-1)
        _, (h_n, c_n) = self.obs_encoder(enc_input)
        return h_n[-1], c_n[-1]  # (B, hidden_dim)

    def _encode_future(
        self, target_vel: torch.Tensor, cond: torch.Tensor, init_hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        B, F_steps, _ = target_vel.shape
        cond_expanded = cond.unsqueeze(1).repeat(1, F_steps, 1)
        enc_input = torch.cat([target_vel, cond_expanded], dim=-1)
        h0, c0 = init_hidden
        h0 = h0.unsqueeze(0)  # (1, B, H)
        c0 = c0.unsqueeze(0)
        _, (h_n, _) = self.future_encoder(enc_input, (h0, c0))
        return h_n[-1]  # (B, hidden_dim)

    def _decode_teacher_forcing(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor,
        z: torch.Tensor,
        init_hidden: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        B, _, _ = obs.shape
        F_steps = int(target.shape[1])
        hidden = init_hidden

        # Initial input: last observed velocity
        curr_vel = obs[:, -1, 2:4]

        recon = 0.0
        for t in range(F_steps):
            dec_input = torch.cat([curr_vel, cond, z], dim=-1)
            h_t, c_t = self.decoder_cell(dec_input, hidden)
            hidden = (h_t, c_t)

            pred_vel = self.head(h_t)
            recon = recon + F.mse_loss(pred_vel, target[:, t], reduction="mean")

            # Teacher forcing
            curr_vel = target[:, t]

        return recon / float(F_steps)

    def forward(self, obs: torch.Tensor, cond: torch.Tensor, target: Optional[torch.Tensor] = None, kl_weight: float = 1.0):
        if target is None:
            return torch.tensor(0.0, device=obs.device)

        h_obs, c_obs = self._encode_obs(obs, cond)

        prior_mu = self.prior_mu(h_obs)
        prior_logvar = self.prior_logvar(h_obs)

        h_future = self._encode_future(target, cond, (h_obs, c_obs))
        post_in = torch.cat([h_obs, h_future], dim=-1)
        post_mu = self.post_mu(post_in)
        post_logvar = self.post_logvar(post_in)

        z = self._reparameterize(post_mu, post_logvar)

        recon = self._decode_teacher_forcing(obs, cond, z, (h_obs, c_obs), target)
        kl = self._kl_diag_gaussians(post_mu, post_logvar, prior_mu, prior_logvar)

        loss = recon + float(kl_weight) * kl
        return loss, recon.detach(), kl.detach()

    def sample_trajectory(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor,
        horizon: int,
        z_temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        h_obs, c_obs = self._encode_obs(obs, cond)
        prior_mu = self.prior_mu(h_obs)
        prior_logvar = torch.clamp(self.prior_logvar(h_obs), min=-10.0, max=10.0)
        std = torch.exp(0.5 * prior_logvar) * float(z_temperature)
        z = prior_mu + torch.randn_like(std) * std

        hidden = (h_obs, c_obs)
        curr_vel = obs[:, -1, 2:4]
        preds = []
        for _ in range(int(horizon)):
            dec_input = torch.cat([curr_vel, cond, z], dim=-1)
            h_t, c_t = self.decoder_cell(dec_input, hidden)
            hidden = (h_t, c_t)

            pred_vel = self.head(h_t)
            preds.append(pred_vel)
            curr_vel = pred_vel

        return torch.stack(preds, dim=1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseTrajectoryModel

class SeqBaseline(BaseTrajectoryModel):
    """
    LSTM-based Sequence Prediction Baseline.
    Encoder-Decoder architecture.
    """
    def __init__(self, 
                 obs_dim: int = 4, 
                 act_dim: int = 2, 
                 cond_dim: int = 6, 
                 hidden_dim: int = 128, 
                 num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Encoder: [pos, vel, cond]
        self.encoder = nn.LSTM(
            input_size=obs_dim + cond_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )
        
        # Decoder: [vel, cond]
        # We use LSTMCell for flexibility in autoregressive loop
        # Note: If num_layers > 1, we need multiple cells or a StackedLSTMCell.
        # For baseline simplicity, let's stick to 1 layer for now or use nn.LSTM with loop.
        # If num_layers=1:
        self.decoder_cell = nn.LSTMCell(act_dim + cond_dim, hidden_dim)
        
        self.head = nn.Linear(hidden_dim, act_dim)
        
    def forward(self, obs, cond, target=None):
        """
        Args:
            obs: (B, H, 4)
            cond: (B, 6)
            target: (B, F, 2) [future_vel]
        """
        B, H, _ = obs.shape
        device = obs.device
        
        # 1. Encode
        cond_expanded = cond.unsqueeze(1).repeat(1, H, 1)
        enc_input = torch.cat([obs, cond_expanded], dim=-1)
        _, (h_n, c_n) = self.encoder(enc_input)
        
        # LSTM output hidden is (num_layers, B, H). 
        # LSTMCell expects (B, H). We take the last layer.
        hidden = (h_n[-1], c_n[-1])
        
        if target is not None:
            # Training with Teacher Forcing
            F_steps = target.shape[1]
            loss = 0.0
            
            # Initial input: last observed velocity
            curr_vel = obs[:, -1, 2:4] # indices 2,3 are vel
            
            for t in range(F_steps):
                # Prepare input: [vel, cond]
                dec_input = torch.cat([curr_vel, cond], dim=-1)
                
                # Step
                h_t, c_t = self.decoder_cell(dec_input, hidden)
                hidden = (h_t, c_t)
                
                # Predict
                pred_vel = self.head(h_t)
                
                # Accumulate Loss
                loss += F.mse_loss(pred_vel, target[:, t])
                
                # Next input: Ground Truth (Teacher Forcing)
                curr_vel = target[:, t]
                
            return loss / F_steps
        else:
            return torch.tensor(0.0, device=device)

    def sample_trajectory(self, obs, cond, horizon, **kwargs):
        """
        Inference with Autoregressive Rollout.
        """
        B, H, _ = obs.shape
        device = obs.device
        
        # 1. Encode
        cond_expanded = cond.unsqueeze(1).repeat(1, H, 1)
        enc_input = torch.cat([obs, cond_expanded], dim=-1)
        _, (h_n, c_n) = self.encoder(enc_input)
        
        hidden = (h_n[-1], c_n[-1])
        
        # 2. Decode
        preds = []
        curr_vel = obs[:, -1, 2:4]
        
        for t in range(horizon):
            dec_input = torch.cat([curr_vel, cond], dim=-1)
            h_t, c_t = self.decoder_cell(dec_input, hidden)
            hidden = (h_t, c_t)
            
            pred_vel = self.head(h_t)
            preds.append(pred_vel)
            
            # Autoregressive input
            curr_vel = pred_vel
            
        return torch.stack(preds, dim=1) # (B, F, 2)

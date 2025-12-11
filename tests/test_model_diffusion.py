import torch
from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel

def test_diffusion_model():
    print("Testing DiffusionTrajectoryModel...")
    
    B, H, F = 4, 8, 16
    obs_dim = 4
    act_dim = 2
    cond_dim = 6
    
    model = DiffusionTrajectoryModel(
        obs_dim=obs_dim, 
        act_dim=act_dim, 
        cond_dim=cond_dim,
        obs_len=H,
        pred_len=F,
        diffusion_steps=10 # Short steps for testing
    )
    
    # Dummy data
    obs = torch.randn(B, H, obs_dim)
    cond = torch.randn(B, cond_dim)
    target = torch.randn(B, F, act_dim)
    
    # 1. Test Forward (Training)
    loss = model(obs, cond, target)
    print(f"Diffusion Loss: {loss.item()}")
    assert loss.shape == ()
    assert not torch.isnan(loss)
    
    # 2. Test Sample (Inference)
    horizon = F
    samples = model.sample_trajectory(obs, cond, horizon=horizon)
    print(f"Sample Shape: {samples.shape}")
    
    assert samples.shape == (B, horizon, act_dim)
    assert not torch.isnan(samples).any()
    
    print("DiffusionModel Test Passed.")

if __name__ == "__main__":
    test_diffusion_model()

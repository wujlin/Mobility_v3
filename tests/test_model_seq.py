import torch
from src.models.seq.seq_baseline import SeqBaseline

def test_seq_baseline():
    print("Testing SeqBaseline...")
    
    B, H, F = 4, 8, 12
    obs_dim = 4
    act_dim = 2
    cond_dim = 6
    
    model = SeqBaseline(obs_dim=obs_dim, act_dim=act_dim, cond_dim=cond_dim)
    
    # Dummy data
    obs = torch.randn(B, H, obs_dim) # [pos, vel]
    cond = torch.randn(B, cond_dim)
    target = torch.randn(B, F, act_dim)
    
    # 1. Test Forward (Training)
    loss = model(obs, cond, target)
    print(f"Forward Loss: {loss.item()}")
    assert loss.shape == ()
    assert not torch.isnan(loss)
    
    # 2. Test Sample (Inference)
    horizon = F
    samples = model.sample_trajectory(obs, cond, horizon=horizon)
    print(f"Sample Shape: {samples.shape}")
    
    assert samples.shape == (B, horizon, act_dim)
    
    print("SeqBaseline Test Passed.")

if __name__ == "__main__":
    test_seq_baseline()

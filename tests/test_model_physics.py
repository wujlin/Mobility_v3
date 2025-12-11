import torch
from src.models.physics.physics_condition_diffusion import PhysicsConditionDiffusion

def test_physics_model():
    print("Testing PhysicsConditionDiffusion...")
    
    B, H, F = 4, 8, 16
    obs_dim = 4
    act_dim = 2
    cond_dim = 6
    nav_patch_size = 32
    
    model = PhysicsConditionDiffusion(
        obs_dim=obs_dim, 
        act_dim=act_dim, 
        cond_dim=cond_dim,
        nav_patch_size=nav_patch_size,
        obs_len=H,
        pred_len=F,
        diffusion_steps=10
    )
    
    # Dummy data
    obs = torch.randn(B, H, obs_dim)
    cond = torch.randn(B, cond_dim)
    target = torch.randn(B, F, act_dim)
    nav_patch = torch.randn(B, 3, nav_patch_size, nav_patch_size)
    
    # 1. Test Forward
    loss = model(obs, cond, target=target, nav_patch=nav_patch)
    print(f"Physics Loss: {loss.item()}")
    assert loss.shape == ()
    
    # 2. Test Sample
    horizon = F
    samples = model.sample_trajectory(obs, cond, horizon=horizon, nav_patch=nav_patch)
    print(f"Sample Shape: {samples.shape}")
    
    assert samples.shape == (B, horizon, act_dim)
    
    print("PhysicsModel Test Passed.")

if __name__ == "__main__":
    test_physics_model()

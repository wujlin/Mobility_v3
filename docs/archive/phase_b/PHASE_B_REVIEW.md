# Phase B Results Review & Recommendations

## 1. Summary of Observation
I have reviewed `docs/archive/phase_b/PHASE_B_RESULTS.md` and compared it with `docs/PHASE_A_RESULTS.md`. The key observations are:
1.  **Baseline Rebound**: The Baseline model performance significantly **improved** in Phase B (ADE 6.42 -> 5.47). This is expected as `dt_fixed=30s` removes time-interval noise, making the "conditional mean" easier to learn for a deterministic LSTM.
2.  **Diffusion Regression**: The Diffusion/Physics models **degraded** in Phase B (ADE 6.02 -> 6.74), leading to a large performance gap where Baseline >> Diffusion.
3.  **Capacity Mismatch**: There is a critical discrepancy in model capacity:
    *   Baseline: `hidden_dim=128` (Inferred)
    *   Diffusion: `hidden_dim=64` (Configured)

## 2. Root Cause Analysis
Why did Diffusion lose its advantage?

### A. The "Strong Baseline" Effect
In Phase A (variable dt), the "mean future" was likely hard to estimate due to irregular time steps. Diffusion, capable of modeling multimodal distributions, handled the noise better.
In Phase B (fixed dt), the underlying motion physics is smoother. The Baseline (optimizing MSE) can very accurately predict the *mean* trajectory. Since `ADE_mean` (of samples) is mathematically lower-bounded by the error of the mean prediction plus variance, the Diffusion model suffers in this metric unless it has extremely high fidelity.

### B. Capacity Bottleneck (Primary Suspect)
Diffusion models approximate the score function (gradient of log-density), which is a much more complex surface than the conditional mean function learned by Baseline.
Using **half the hidden dimension (64 vs 128)** of the Baseline for a harder task (Diffusion) is the most likely cause of underfitting.
*   **Symptom**: "Shrinkage" (Low MSD/Rog). An under-capacitated diffusion model often converges to predicting "mean" (zero noise) or small perturbations, failing to generate the full variance of the data.

### C. Optimization Dynamics
*   **Batch Size 2048**: With `hidden_dim=64`, this is a very large batch size. It provides stable gradients but might trap the small model in suboptimal minima early on.
*   **Epochs=50**: Might be insufficient for the Diffusion model to learn fine-grained details after grasping the global mean.

## 3. Actionable Recommendations
To align Phase B with the success of Phase A, we must conduct an "Apples-to-Apples" comparison.

### Priority 1: Scale Up (Immediate Fix)
Retrain the Diffusion and Physics models with increased capacity to match or exceed the Baseline.
*   **Action**: Set `hidden_dim=128` (or `256` if possible).
*   **Action**: Set `model_type` parameters consistent with `evaluate.py` expectations.

### Priority 2: Training Tune-up
*   **Action**: Increase `epochs` to 100 to ensure convergence.
*   **Action**: Decrease `batch_size` to 512 or 1024 to introduce slightly more stochasticity during training, which can help generalization, or linearly scale LR if keeping 2048.

### Priority 3: Normalization sanity check
*   **Verification**: Ensure `data_stats.json` for Phase B (dt30) is actually being used. The `vel_std` in dt30 (approx 2.0) is smaller than Phase A default (approx 4.0). If the model uses the wrong default stats, it will see input values that are 2x larger than expected (if normalizing with small std) or 0.5x (if normalizing with large std).
    *   *Current code check*: `src/training/train_diffusion.py` loads the dataset which loads the json. **Crucial**: Ensure the `args.data_path` passed to training pointed specifically to `data/processed_dt30`.

## 4. Conclusion
The "bad" results are likely an implementation artifact (small model size) rather than a method failure. The fact that `Physics (Best-of-K)` still improves over `Diffusion (Best-of-K)` (ADE 2.51 vs 2.84, Fréchet 4.03 vs 4.43) confirms the **Nav Field is still working** effectively to improve the *upper bound* of generation quality. The *mean* performance is simply dragging behind due to general underfitting.

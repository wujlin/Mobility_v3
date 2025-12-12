import unittest
import torch
import numpy as np
from src.evaluation.micro_metrics import compute_frechet, compute_dtw, compute_micro_metrics

class TestAdvancedMetrics(unittest.TestCase):
    def setUp(self):
        # Create dummy trajectories
        # Batch size 2, Length 5, Dim 2
        self.pred = torch.tensor([
            [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
            [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
        ], dtype=torch.float32)
        
        self.target = torch.tensor([
            [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
            [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1]] # Shifted by y=1
        ], dtype=torch.float32)

    def test_frechet_identical(self):
        # Distance between identical trajectories should be 0
        dist = compute_frechet(self.pred[:1], self.pred[:1])
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_frechet_offset(self):
        # Second trajectory is shifted by 1 unit in y everywhere
        # Frechet distance should be exactly 1.0
        dist = compute_frechet(self.pred[1:], self.target[1:])
        self.assertAlmostEqual(dist, 1.0, places=5)

    def test_dtw_identical(self):
        dist = compute_dtw(self.pred[:1], self.pred[:1])
        self.assertAlmostEqual(dist, 0.0, places=5)
        
    def test_dtw_offset(self):
        # Second trajectory is shifted by 1 unit
        # DTW sums up costs. Length 5. Each step dist is 1.
        # Cost matrix diagonal path is optimal.
        # Sum of 1s ideally? 
        # Standard DTW usually sums distances along warp path.
        # Path length is 5 (diagonal). Total dist should be 5.0?
        # Let's check logic: D[k, l] = cost + min(prev)
        # Diag: D[1,1] = 1 + 0 = 1
        # D[2,2] = 1 + 1 = 2
        # ... D[5,5] = 5
        dist = compute_dtw(self.pred[1:], self.target[1:])
        self.assertAlmostEqual(dist, 5.0, places=5)

    def test_integration(self):
        metrics = compute_micro_metrics(self.pred, self.target)
        self.assertIn('Frechet', metrics)
        self.assertIn('DTW', metrics)
        print("\ncomputed metrics:", metrics)

if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import sys
import os

# Add package to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sfa.core import FractalDimensionEstimator
from sfa.bayesian_d3 import bayesian_d3_inference

class TestSFA(unittest.TestCase):
    
    def setUp(self):
        # Create synthetic 2D plane in 3D
        np.random.seed(42)
        self.coords_2d = np.column_stack([
            np.random.rand(500),
            np.random.rand(500),
            np.zeros(500)
        ])
        
        # Create synthetic 3D volume
        self.coords_3d = np.random.rand(500, 3)
        
    def test_gp_dimension_2d(self):
        estimator = FractalDimensionEstimator()
        d2, err = estimator.compute_gp_dimension(self.coords_2d, bootstrap_iterations=5)
        print(f"2D Plane D2: {d2:.3f}")
        self.assertTrue(1.8 < d2 < 2.2, f"D2 should be ~2.0, got {d2}")
        
    def test_gp_dimension_3d(self):
        estimator = FractalDimensionEstimator()
        d2, err = estimator.compute_gp_dimension(self.coords_3d, bootstrap_iterations=5)
        print(f"3D Volume D2: {d2:.3f}")
        self.assertTrue(2.7 < d2 < 3.3, f"D2 should be ~3.0, got {d2}")
        
    def test_bayesian_inference_smoke(self):
        # Quick run with minimal steps
        res = bayesian_d3_inference(
            self.coords_3d, 
            n_steps=50, 
            n_burnin=10, 
            n_walkers=10, 
            verbose=False,
            # Fallback to MH if emcee is missing/fails or just force for speed
            use_emcee=False 
        )
        self.assertIn('d3_mean', res)
        self.assertIn('samples', res)

if __name__ == '__main__':
    unittest.main()

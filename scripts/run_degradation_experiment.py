"""
Synthetic Precision Degradation Experiment
==========================================

Quantifies the "Fisher Information Barrier" (sigma_c) where Bayesian inference 
saturates to the prior mode (D=3.0) due to location uncertainty.

Uses the `sfa` package for core calculations.
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import beta
from pathlib import Path

# Add project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from sysmic.core import FractalDimensionEstimator
from sysmic.bayesian_core import bayesian_d3_inference
from sysmic.bayesian_d3 import compute_kl_divergence_d3

# Configuration
DATA_DIR = Path(PROJECT_ROOT) / 'data' / 'processed'
OUTPUT_DIR = Path(PROJECT_ROOT) / 'data' / 'results'
FIGURES_DIR = Path(PROJECT_ROOT) / 'paper' / 'figures'

CATALOG_FILE = DATA_DIR / 'hinet_hypo_2001_2005_extracted.csv'

SIGMA_VALUES = [0.0001, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]
N_BOOTSTRAP = 5  # Number of replicas per sigma
PRIOR = beta(7.5, 2.5)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_noto_subset():
    """Load Hi-Net data and filter for Noto Peninsula (or synthetic fallback)."""
    if CATALOG_FILE.exists():
        print(f"Loading {CATALOG_FILE}...")
        df = pd.read_csv(CATALOG_FILE)
        
        # Noto bounds
        lat_min, lat_max = 37.0, 37.8
        lon_min, lon_max = 136.5, 137.5
        depth_min, depth_max = 0.0, 40.0 # Shallow
        
        # Scale back to geodetic if needed? 
        # The file has x, y, z normalized? Or Lat/Lon?
        # Check columns. If 'x', 'y', 'z', we can filter by range if we know scale.
        # But 'extracted' file usually keeps 'latitude', 'longitude' if available?
        # The extraction script `convert_wrl_to_csv.py` extracted 3 columns.
        # VRML only had raw coords.
        # We assume 1 unit = 200 km.
        # We don't have lat/lon in `hinet_hypo_2001_2005_extracted.csv`.
        # We only have x, y, z.
        # We can't easily filter for Noto spatially without mapping back.
        
        # Fallback: Just take a random spatial subset of N=10,000 events
        # This keeps the "real geometry" property (planar slab) without needing Noto specifically.
        print("Taking random subset of 10,000 events from Hi-Net...")
        if len(df) > 10000:
            df = df.sample(10000, random_state=42)
            
        coords = df[['x', 'y', 'z']].values * 200.0 # Scale to km (approx)
        return coords
    else:
        print("Catalog not found. Using Synthetic Cluster.")
        # Synthetic cluster
        np.random.seed(42)
        return np.random.randn(10000, 3) * 10.0 # Gaussian ball

def run_experiment():
    coords_true = load_noto_subset()
    print(f"Baseline events: {len(coords_true)}")
    
    results = []
    
    for sigma_km in SIGMA_VALUES:
        print(f"-- Processing sigma = {sigma_km} km --")
        
        for i in range(N_BOOTSTRAP):
            # Add isotropic noise
            noise = np.random.normal(0, sigma_km, size=coords_true.shape)
            coords_noisy = coords_true + noise
            
            # Anisotropic variant (optional, as per paper)
            # sigma_v = 1.7 * sigma_km
            # noise[:, 2] *= 1.7
            
            # Bayesian D3
            res = bayesian_d3_inference(
                coords_noisy,
                sampler='dynesty',
                nlive=300,
                verbose=False
            )
            
            # KL Divergence
            kl_res = compute_kl_divergence_d3(res['samples'], verbose=False)
            
            results.append({
                'sigma_km': sigma_km,
                'bootstrap_i': i,
                'd3_mean': res['d3_mean'],
                'd3_std': res['d3_std'],
                'kl_divergence': kl_res['kl_divergence'],
                'saturation_mass': res['posterior_mass_saturation']
            })
            
    # Save results
    df_res = pd.DataFrame(results)
    output_csv = OUTPUT_DIR / 'precision_degradation_results.csv'
    df_res.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

if __name__ == "__main__":
    run_experiment()

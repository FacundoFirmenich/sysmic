# Seismic Fractal Analysis (SFA)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.105281.178883.svg)](https://doi.org/10.5281/zenodo.105281.178883)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**SFA** is a Python framework designed to resolve minimal fractal dimension estimates from noisy earthquake catalogs. It implements a **Triple-Validation Framework** to distinguish genuine volumetric seismicity from Bayesian saturation artifacts caused by location uncertainty.

## Key Features

- **Precision-Calibrated Inference:** Detects the "Fisher Information Barrier" ($\sigma_c$) where Bayesian inference saturates.
- **Triple-Validation:**
  1.  **KL Divergence:** Screening for prior dominance.
  2.  **Boundary Concentration:** Detecting artificial saturation at $D=3$.
  3.  **Scale Invariance:** Zaccagnino stability test.
- **Independent Network Quality:** Calibrates an effective quality metric ($Q_{eff}$) without circular reliance on fractal results.
- **Optimized Algorithms:**
  - $O(N \log N)$ dimension estimation via k-d trees.
  - Robust Theil-Sen scaling region detection.
  - Parallelized Bayesian sampling (emcee).

## Installation

```bash
git clone https://github.com/FacundoFirmenich/SeismicFractalAnalysis.git
cd SeismicFractalAnalysis
pip install .
```

## Quick Start

```python
from sfa import FractalAnalyzer

# Load data (x, y, z in km)
data = load_my_catalog("events.csv")

# Initialize analyzer
analyzer = FractalAnalyzer(data)

# Compute Correlation Dimension D2
results = analyzer.compute_d2(r_min=2.0, r_max=500.0)
print(f"D2: {results.d2:.3f}")

# Infer Latent D3 (Bayesian)
bayes_res = analyzer.infer_d3_bayesian(results.d2, sigma_obs=0.2)
print(f"D3 Posterior Mode: {bayes_res.d3_mode:.3f}")
print(f"Saturation Probability: {bayes_res.p_bound:.4f}")
```

## Documentation

Full documentation is available in the `docs/` directory and in the accompanying paper.

## Citation

If you use SFA in your research, please cite:

> Firmenich, F., et al. (2026). "Fractal Tomography and the Depth-Dependent Planarization of Seismicity". *Journal of Geophysical Research: Solid Earth*.

# Sysmic v8.0.0 — Fractal Tomography of Seismicity

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-8.0.0-brightgreen.svg)](CITATION.cff)
[![JOSS](https://joss.theoj.org/papers/placeholder/status.svg)](joss/paper.md)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/sysmic_colab.ipynb)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](Dockerfile)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18480821.svg)](https://doi.org/10.5281/zenodo.18480821)

**Sysmic** is an open-source Python framework for precision-calibrated Bayesian fractal tomography of seismic catalogs. It implements the Grassberger–Procaccia correlation integral with MCMC inference of latent dimensionality ($D_3$), resolving the dual paradoxes caused by finite location uncertainty.

> **Companion software to:** Firmenich, Firmenich & Firmenich (2026). *Fractal Tomography and the Fisher Information Barrier of Seismicity.* JGR: Solid Earth. Under review.  
> **Preprint:** [doi:10.31223/x5rf3v](https://doi.org/10.31223/x5rf3v) (EarthArXiv, Dec 2025)

---

## Key Results (Empirical — Zero Synthetic Data)

| Region | Network | $\sigma_h$ (km) | $D_2$ | $D_3$ | SVP Tier |
|---|---|---|---|---|---|
| Noto, Japan       | Hi-Net      | 0.5    | 2.12 | 2.82 ± 0.05   | ✅ Tier 1 |
| Tohoku, Japan     | Hi-Net      | 0.7    | 2.09 | 2.83 ± 0.06   | ✅ Tier 1 |
| Tokachi, Japan    | Hi-Net      | 0.6    | 2.15 | 2.86 ± 0.07   | ✅ Tier 1 |
| Valais, Switzerland | Swiss-SED | 0.8–1.2 | 1.50 | 1.911 ± 0.066 | ✅ Tier 1 |
| SCSN baseline     | SCSN        | 0.1    | —    | —              | ✅ Tier 2 |
| Cook Strait, NZ   | GeoNet*     | <1.0   | 2.24 | 2.53 ± 0.17   | ✅ Resolved |
| Bay of Plenty, NZ | GeoNet*     | <1.0   | 1.91 | 2.20 ± 0.20   | ✅ Resolved |
| Sumatra           | GEOFON      | 7.5    | 2.21 | 2.998 ± 0.002 | ❌ Saturated |
| Cascadia          | USGS        | 6.1    | 2.21 | → 3.0         | ❌ Saturated |

*High-precision local relocations (not national GeoNet catalog).

**Fisher Information Barrier:** $\sigma_c = 2.3 \pm 0.4$ km — empirically validated precision threshold above which Bayesian $D_3$ inference collapses to the embedding-dimension prior.

**Tectonic Hierarchy ($D_2$):** Rifting (1.50) → Transform (1.81) → Subduction (2.12) → Collision (2.24) → Deep Slab (1.26).

---

## Repository Structure

```
sysmic/                     Core Python library (v8.0.0)
├── core.py                 Grassberger-Procaccia estimator (O(N log N), k-d tree)
├── bayesian_d3.py          MCMC D₃ inference (emcee, Uniform[1.5,3.0] prior)
├── statistics.py           Bootstrap, Theil-Sen, b-value (Aki 1965)
├── geometry.py             Steiner formula, deck-of-cards model (Q=2/3)
└── zaccagnino_stability.py Mmin-independence stability test (SVP T3)

data/                       22 empirical CSVs (all from real catalogs)
figures/                    14 manuscript figures (PNG 300 dpi)
figs_gen.py                 Reproducible figure generation (all 14 figs)
notebooks/
├── sysmic_colab.ipynb      Interactive Google Colab (no install required)
└── tutorial.ipynb          Step-by-step tutorial
app/
└── streamlit_app.py        Interactive web dashboard
joss/                       JOSS submission files (paper.md, paper.bib)
docs/
└── methods.md              Extended methods reference
tests/
└── test_core.py            Unit tests (6 test classes, 25+ assertions)
setup.py / pyproject.toml   pip-installable package
Dockerfile                  Reproducible containerized environment
CITATION.cff                Machine-readable citation metadata
.zenodo.json                Zenodo archive metadata
```

---

## Quickstart

### Option 1 — Google Colab (zero install)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/sysmic_colab.ipynb)

### Option 2 — Docker (fully reproducible)
```bash
docker build -t sysmic .
docker run -p 8501:8501 sysmic
# Open http://localhost:8501
```

### Option 3 — Local install
```bash
pip install -r requirements.txt
python figs_gen.py                          # Reproduce all 14 figures
pytest tests/ -v                            # Run unit tests
streamlit run app/streamlit_app.py          # Launch interactive dashboard
```

### Option 4 — pip (editable install)
```bash
pip install -e .
```

---

## Reproducing the Manuscript Figures

```bash
python figs_gen.py
```

All 14 figures are generated deterministically from the empirical CSVs in `data/`. No internet connection required. Output: `figures/*.pdf` (600 dpi) and `figures/*.png` (300 dpi). Runtime: ~2–3 min on a standard laptop.

---

## MCMC Configuration (per manuscript §3.3)

| Parameter | Value |
|---|---|
| Prior | Uniform[1.5, 3.0] |
| Walkers | 32 |
| Steps | 10 000 |
| Burn-in | 2 000 |
| ESS target | > 5 000 |
| Convergence | $\hat{R} < 1.01$ |

---

## Data Sources

All data from publicly accessible catalogs:

| Dataset | Source | Access |
|---|---|---|
| Hi-Net Japan (JUICE-relocated) | NIED doi:[10.17598/NIED.0003](https://doi.org/10.17598/NIED.0003) | Registered researchers |
| USGS Pan-American | [earthquake.usgs.gov](https://earthquake.usgs.gov) | Open |
| GeoNet New Zealand | [geonet.org.nz](https://www.geonet.org.nz) | Open |
| GEOFON Sumatra | [geofon.gfz-potsdam.de](https://geofon.gfz-potsdam.de) | Open |
| Swiss-SED Valais | [seismo.ethz.ch](https://seismo.ethz.ch) | Open |
| ISC-GEM v12.1 | doi:[10.31905/D808B825](https://doi.org/10.31905/D808B825) | Open |

**Data Source (primary):** United States Geological Survey (USGS)

---

## Citation

```bibtex
@article{Firmenich2026,
  author  = {Firmenich, Facundo and Firmenich, Pau and Firmenich, Le\'{o}n},
  title   = {Fractal Tomography and the Fisher Information Barrier of Seismicity:
             Resolving Dual Paradoxes via Precision-Calibrated Bayesian Inference},
  journal = {Journal of Geophysical Research: Solid Earth},
  year    = {2026},
  note    = {Under review},
  doi     = {10.31223/x5rf3v}
}

@software{Firmenich2026sysmic,
  author  = {Firmenich, Facundo and Firmenich, Pau and Firmenich, Le\'{o}n},
  title   = {Sysmic: Fractal Tomography of Seismicity},
  version = {8.0.0},
  year    = {2026},
  doi     = {10.5281/zenodo.18480821},
  url     = {https://github.com/cedesur/sysmic}
}
```

---

## License

GPLv3 — see [LICENSE](LICENSE). Free to use, study, modify, and share.

---

*April 8, 2026 — Sysmic v8.0.0*

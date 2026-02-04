# 🌋 SYSMIC: Geophysical Analysis Framework

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-green.svg)](https://hub.docker.com/)
[![Open Science](https://badges.frapsoft.com/os/v1/open-science.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

> **"Unveiling the hidden architecture of tectonic systems."**

**Sysmic** (formerly FractalSystemPro) is a state-of-the-art computational framework for advanced seismic analysis. It integrates fractal geometry, multifractal spectra, temporal dynamics, and Bayesian inference to provide a holistic understanding of earthquake physics.

---

## 🌟 Key Features

### 🔬 Scientific Core
- **Fractal Dimension (D₂)**: Robust Grassberger-Procaccia estimation with bootstrap uncertainty.
- **Multifractal Spectrum (Dq)**: Full $f(\alpha)$ singularity spectrum analysis.
- **Temporal Dynamics**: $D(t)$ evolution and critical slowing down detection.
- **Bayesian Validation**: Probabilistic assessment of dimensional estimates.
- **Topological Analysis**: Graph-based community detection in seismic networks.

### 💻 Modern Interfaces
- **Interactive CLI**: Professional terminal interface with ASCII art visualization.
- **Web Dashboard**: Streamlit-powered GUI for easy data exploration.
- **Cloud Studio**: Unified Google Colab notebook for instant execution.
- **Containerized**: Full Docker support for reproducible deployment.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/facundofirmenich/sysmic.git
cd sysmic
pip install -r requirements.txt
```

### 2. Interactive Mode (CLI)

Access the professional command-line interface:

```bash
python -m sysmic.interactive
```

### 3. Web Dashboard (Streamlit)

Launch the local web server:

```bash
streamlit run streamlit_app.py
```

### 4. Docker Deployment

Run the containerized environment:

```bash
docker build -t sysmic .
docker run -it sysmic
```

---

## 📊 Scientific Methodology

Sysmic implements a rigorous **Triple Validation Protocol**:

1.  **Geometric Consistency**: $D_q$ spectrum width ($\Delta\alpha$) must exceed noise floor.
2.  **Temporal Stability**: $\Delta D_2 / \Delta t$ must remain bounded during quiescence.
3.  **Statistical Significance**: Surrogate data testing (random shuffling) with $p < 0.05$.

For detailed formulations, see [MATHEMATICAL_APPENDIX.md](MATHEMATICAL_APPENDIX.md).

---

## 🌍 Open Science Manifesto

We believe that knowledge locked away is not knowledge—it is a secret. 
Sysmic is built on three pillars:

1.  **Transparency**: Every algorithm is open for inspection.
2.  **Reproducibility**: Containerized environments ensure consistent results.
3.  **Accessibility**: Tools designed for everyone, from students to top-tier researchers.

---

## 🤝 Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and submission process.

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0**. 
See [LICENSE](LICENSE) for details.

**Data Sources:** 
1. **United States Geological Survey (USGS)**
2. **National Research Institute for Earth Science and Disaster Resilience (NIED)**, Japan (Hi-net/F-net High-Sensitivity Seismograph Network). Access provided under registered research protocols.
3. **International Seismological Centre (ISC)**: ISC-GEM Global Instrumental Earthquake Catalogue (Ver. 11.0).

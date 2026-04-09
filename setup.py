"""
Sysmic v8.0.0 — Setup
======================
Fractal Tomography and the Fisher Information Barrier of Seismicity.
JGR Solid Earth companion software (Firmenich et al., 2026).
License: GPLv3
"""

from setuptools import setup, find_packages

try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Sysmic: Fractal Tomography of Seismicity"

setup(
    name="sysmic",
    version="8.0.0",
    author="Facundo Firmenich, Pau Firmenich, León Firmenich",
    author_email="f.firmenich@cedesur.org",
    description=(
        "Precision-calibrated Bayesian fractal tomography of seismic catalogs. "
        "Implements the Fisher Information Barrier (σ_c = 2.3 ± 0.4 km) and "
        "Grassberger-Procaccia correlation dimension via MCMC (emcee)."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FacundoFirmenich/sysmic",
    project_urls={
        "Bug Tracker":    "https://github.com/FacundoFirmenich/sysmic/issues",
        "Documentation":  "https://github.com/FacundoFirmenich/sysmic/tree/main/docs",
        "JOSS Paper":     "https://joss.theoj.org/papers/placeholder",
        "Zenodo Archive": "https://doi.org/10.5281/zenodo.18480821",
    },
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "emcee>=3.1.0",
        "joblib>=1.2.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "app": ["streamlit>=1.28.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sysmic-analyze=sysmic.core:main",
        ],
    },
)

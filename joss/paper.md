title: 'Sysmic: An Open-Source Intelligence Framework for High-Precision Fractal Tomography and Real-Time Monitoring'
tags:
  - Python
  - seismology
  - fractal dimension
  - bayesian inference
  - real-time monitoring
  - machine learning
  - topological data analysis
  - hi-net
authors:
  - name: Facundo Firmenich
    orcid: 0009-0002-6578-3811
    affiliation: 1
  - name: Pau Firmenich
    affiliation: 1
  - name: León Firmenich
    affiliation: 1
affiliations:
  - name: Centro de Estudios del Sur (CEDESUR)
    index: 1
date: 04 February 2026
bibliography: paper.bib
---

# Summary

Understanding the complex spatial-temporal organization of seismicity is critical for both fundamental geophysics and hazard assessment. **Sysmic** (formerly SFA) is a comprehensive, open-source computational framework that transcends traditional fractal estimation. It integrates high-precision **Fractal Tomography**, **Bayesian Inference**, **Topological Data Analysis (TDA)**, and **Real-Time Monitoring** into a unified intelligence platform.

Designed to handle the massive datasets of modern high-sensitivity networks (e.g., NIED Hi-Net), Sysmic resolves the long-standing "geometric projection paradox" in seismology by providing precision-calibrated tools to distinguish genuine fault structures from observational artifacts. Beyond retrospective analysis, the framework includes the **Gravitas** and **Nexus** engines, enabling predictive modeling and automated research intelligence.

# Statement of Need

Quantitative seismology faces a "data deluge" where traditional, catalog-agnostic statistical methods often fail to extract meaningful signals from noise-dominated datasets. Researchers specifically lack:
1.  **Robustness against precision artifacts:** Standard algorithms often yield spurious "volumetric" dimensions ($D \approx 3$) simply due to location uncertainty ($\sigma$).
2.  **Integrated multi-paradigm analysis:** Tools are typically siloed—fractal analysis, topological clustering, and Bayesian inference rarely coexist in a single workflow.
3.  **Real-time capability:** Most fractal codes are designed for static, retrospective catalog analysis, not continuous monitoring streams.

Sysmic addresses these needs by providing a modular ecosystem that treats data quality ($\sigma$, $M_c$, Network $Q_{eff}$) as a first-class citizen in the inference process. It was originally developed to support the breakthroughs in "Fractal Tomography and the Depth-Dependent Planarization of Seismicity" (Firmenich et al., 2026), but has evolved into a general-purpose engine for advanced seismic analysis.

# Architecture and Key Modules

The Sysmic framework is structured into high-performance modules designed for scalability and extensibility:

### 1. Core Engines (`sysmic.core`, `sysmic.bayesian`)
*   **Precision-Calibrated Fractal Estimator:** Implements an optimized Grassberger-Procaccia algorithm ($O(N \log N)$ via k-d trees) capable of processing >200,000 events in seconds.
*   **Fisher Information Barrier Detection:** Automatically calculates the critical precision threshold $\sigma_c$ where inference becomes prior-dominated.
*   **Bayesian Inference Suite:** MCMC-based estimation of latent dimensionality ($D_3$) using `emcee` and `dynesty` samplers, incorporating "Triple-Validation" (KL Divergence, Boundary Concentration, Scale Invariance).

### 2. Gravitas: Real-Time Intelligence (`sysmic.gravitas`)
*   **Continuous Monitoring:** A dedicated engine for real-time calculation of fractal dimension and $b$-value evolution, capable of ingesting streaming data.
*   **Adaptive RL Integration:** Experimental support for Reinforcement Learning (Thompson Sampling) to optimize monitoring window parameters dynamically.
*   **Predictive Analytics:** Implements precursor detection algorithms based on fractal dimension fluctuations ($\Delta D_2$) and entropy measures.

### 3. Nexus: Research Automation (`sysmic.nexus`)
*   **Automated Insight Generation:** Algorithms to scan vast parameter spaces and identify statistically significant anomalies without manual intervention.
*   **Project Intelligence:** A meta-analysis layer that integrates results across different tectonic regions and scales, facilitating global comparative studies.

### 4. Advanced Interoperability (`sysmic.integration`)
*   **Global Seismic Grid (WSG):** Connectors for standard formats (ObsPy, QuakeML) and major catalogs (USGS, ISC-GEM, JMA).
*   **Planetary Seismology Ready:** Modules designed for ingest of planetary data (e.g., InSight Mars, upcoming Dragonfly), leveraging the framework's robustness to sparse/noisy data.
*   **Topological Analysis:** Integration with graph theory libraries to compute network modularity and topological dimension gaps ($\Delta D = D_2 - D_{graph}$).

### 5. Interactive Interfaces
*   **CLI & TUI:** Professional command-line interface with ASCII art branding for rapid deployment.
*   **Web Dashboard:** Streamlit-based graphical interface for data exploration and result visualization.
*   **Cloud Studios:** Optimized Colab notebooks for cloud-native analysis.

# Pedagogical Resources

To democratize advanced geophysics, Sysmic includes a `resources/multimedia` library containing:
*   **Didactic Videos:** visualizations of 8-bit vs. 4K earth geometry, seismic illusions, and fractal architecture.
*   **Technical Papers:** PDF reports on fractal resolution, high-resolution geometry, and seismic tomography.
*   **Multilingual Support:** Materials available in English, Spanish, and Portuguese to support global education.

# Computational Efficiency and the Democratization of Science

A core design philosophy of Sysmic is the removal of the specific barrier between high-impact scientific capacity and access to supercomputing infrastructure. To empirically demonstrate this, the entire development, testing, and execution of this framework—including the processing of >190,000 Hi-Net events, Bayesian MCMC inference, and real-time Gravitas simulations—was strictly constrained to a **mid-range 2016 consumer laptop** (Intel Core i5-6200U, 8GB DDR3 RAM, 128MB dedicated GPU).

This deliberate hardware constraint serves as a validation of the framework's optimization (e.g., k-d tree spatial indexing, vectorized broadcasting) and challenges the assumption that "Big Data" seismology requires massive computational resources. Sysmic proves that with optimized algorithms, tier-1 scientific discovery is accessible on ubiquitous legacy hardware.

# Mathematics and Validation

Sysmic goes beyond black-box estimation by implementing rigorous mathematical safeguards:

**The Fisher Information Barrier:**
$$ \sigma_c = -\frac{\lambda}{2} \log\left( \frac{k \cdot \mathcal{I}_{prior} \cdot \sigma_{D_2}^2 \cdot N_{pairs}}{Q_{eff} \cdot N_{eff}} \right) $$
This derived criterion prevents the "Bayesian Saturation" artifact, ensuring that outputs reflect physical reality rather than prior assumptions.

**Spectral Graph Correction:**
$$ D_{graph} \approx 2 + \frac{\log N_{planes}}{\log(L/\ell)} - \beta_{topo} \lambda_{gap} $$
Enables the reconciliation of geometric and topological dimensionality.

# Example Use Cases

1.  **High-Precision Tomography:** Resolving the "deck of cards" multi-planar structure of the Japan subduction zone (Hi-Net validation).
2.  **Global Tectonic Classification:** Discriminating tectonic regimes (Rifting vs. Collision) based on their specific fractal signatures.
3.  **Seismic Nowcasting:** Monitoring $D_2$ drops in real-time as a potential proxy for stress accumulation (Gravitas engine).

# Availability and Open Science

Sysmic is open-source (GPLv3) and designed for the Open Science era. It includes comprehensive unit tests, synthetic validation datasets (isotropic vs. anisotropic noise), and Docker support for reproducible environments. This framework aims to democratize access to high-end statistical seismology tools, bridging the gap between theoretical complexity and operational utility.

# Acknowledgements

This research was made possible by the open data policies of the **United States Geological Survey (USGS)**. We explicitly acknowledge and thank the **National Research Institute for Earth Science and Disaster Resilience (NIED)**, Japan, for granting access to the high-sensitivity seismograph network data (Hi-net/F-net) under registered user protocols. The precision of this framework in resolving deep slab structures is directly attributable to the quality of the NIED data.

# Statement of AI-Assisted Development and Critical Reflection

The development of this framework utilized several Large Language Models (LLMs) to accelerate prototyping and structure intermediate outputs. The models used, in descending order of intensity and relevance, were:

1.  **Antigravity (powered by Gemini 3 Pro)**
2.  **DeepSeek v3.2**
3.  **Claude 4.5 Sonnet**
4.  **Google NotebookLM**
5.  **Qwen 3 Max**

**Exclusion of ChatGPT 5.2:**
Additionally, **ChatGPT 5.2** was evaluated. This instance is noteworthy solely for its underwhelming performance. Despite being the only utilized tool supported by a paid subscription (entry-level tier, ~$20-25/mo), its persistent lack of reliability and high hallucination rate necessitated its definitive exclusion from the active toolset.

**Critical Assessment:**
While these tools provided undeniable utility in scaffolding code structures and articulating initial drafts, accelerating production significantly, their contribution was strictly bounded by significant limitations. Despite explicit and severe instructions regarding data integrity, the models exhibited a persistent tendency towards hallucination and the fabrication of fictitious data. Consequently, they proved conducive only under constant, line-by-line supervision of all code, translations, and academic writing. Furthermore, contrary to provider specifications, we observed narrow effective context windows and a progressive degradation of long-term memory within sessions. This experience delineates a sharp frontier between the *expected* net contribution of AI agents and the *genuine* value they currently offer, which remains contingent on exhaustive human verification.

# References

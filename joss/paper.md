---
title: 'Sysmic: A Python Framework for Precision-Calibrated Fractal Tomography of Seismicity'
tags:
  - Python
  - seismology
  - fractal dimension
  - bayesian inference
  - fisher information
  - open science
authors:
  - name: Facundo Firmenich
    orcid: 0009-0002-6578-3811
    affiliation: 1
  - name: Pau Firmenich
    affiliation: 1
  - name: León Firmenich
    affiliation: 1
affiliations:
  - name: Centro de Estudios del Sur (CEDESUR), Barcelona, Spain & Buenos Aires, Argentina
    index: 1
date: 30 March 2026
bibliography: paper.bib
---

# Summary

The spatial organization of seismicity encodes fundamental information about fault geometry,
stress distribution, and rupture mechanics. The correlation dimension $D_2$, estimated via
the Grassberger-Procaccia algorithm [@Grassberger1983], has long served as a quantitative
descriptor of this organization. However, a critical and underappreciated artifact afflicts
all correlation dimension estimates: when the location uncertainty $\sigma_h$ of the catalog
exceeds a critical threshold $\sigma_c$, the algorithm systematically overestimates $D_2$
toward the Euclidean bound ($D = 3$), rendering the measurement physically meaningless.

**Sysmic** resolves this problem by providing:

1. A precision-calibrated Grassberger-Procaccia estimator that explicitly quantifies the
   impact of $\sigma_h$ on $D_2$.
2. A Bayesian MCMC framework (built on `emcee` [@ForemanMackey2013]) that infers the
   latent three-dimensional fractal dimension $D_3$ from the observed $D_2$, correcting
   for the geometric projection effect.
3. An analytic criterion — the **Fisher Information Barrier** — that identifies the
   critical precision threshold $\sigma_c = 2.3 \pm 0.4$ km, above which inference
   degenerates and outputs carry no physical information.

# Statement of Need

Standard seismological software (e.g., ZMAP, SeismoStat) provides $D_2$ estimates without
uncertainty quantification relative to network precision. This leads to systematic errors
in tectonic classification: a subduction zone imaged with $\sigma_h = 5$ km will appear
volumetric ($D_2 \approx 3$), indistinguishable from isotropic noise, regardless of the
true fault geometry. Sysmic is, to our knowledge, the first open-source tool to treat
location precision as a first-class parameter in fractal dimension inference.

The framework was validated on six independent networks spanning four tectonic regimes
(Japan Hi-Net, USGS Cascadia, GeoNet New Zealand, GEOFON Sumatra, Swiss SED, ISC-GEM),
reproducing the predicted saturation behavior above $\sigma_c$ and resolving genuine
sub-planar fault structures in precision catalogs [@Firmenich2026].

# Core Methodology

## Grassberger-Procaccia Estimator

The correlation integral is computed as:

$$C(r) = \frac{2}{N(N-1)} \sum_{i<j} \Theta(r - \|\mathbf{x}_i - \mathbf{x}_j\|)$$

where $\Theta$ is the Heaviside function and $\|\cdot\|$ is the Euclidean distance in
metric coordinates. The correlation dimension is:

$$D_2 = \lim_{r \to 0} \frac{d \log C(r)}{d \log r}$$

estimated via Theil-Sen robust regression on the linear scaling region.
Computational complexity: $O(N \log N)$ using `scipy.spatial.cKDTree`.

## Fisher Information Barrier

For a uniform prior over $[\theta_{\min}, \theta_{\max}]$, the boundary mass concentration:

$$P_{\rm bnd}(\sigma_h) = \frac{1}{1 + \exp\left[-k(\sigma_h - \sigma_c)\right]}$$

rises from $< 5\%$ (resolved inference) to $> 90\%$ (prior-dominated saturation) as
$\sigma_h$ crosses $\sigma_c = 2.3$ km, empirically validated on the SCSN California
catalog [@Hauksson2012].

## Bayesian Inference of $D_3$

The latent three-dimensional dimension $D_3$ is inferred via the correlation
integral likelihood:

$$\log \mathcal{L} \propto -\frac{1}{2\hat{s}^2}\sum_k \left[\log C(r_k) - D_3 \log r_k - \hat{a}\right]^2$$

where $C(r_k)$ are the observed correlation integral values and $\hat{a}$ is
the log-amplitude nuisance parameter marginalized analytically. The prior is
$\pi(D_3) = \mathrm{Uniform}[1.5, 3.0]$ (Fisher prior regularization:
$\mathcal{I}_{\rm prior} = 1/(1.5)^2 = 0.444$). MCMC uses `emcee`
(32 walkers, 10,000 steps, 2,000 burn-in) with convergence validated by
$\hat{R} < 1.01$ and ESS $> 5{,}000$.

# Empirical Validation

All results presented here derive exclusively from empirical catalog data. No synthetic
datasets are used in any published figure.

| Noto, Japan (Hi-Net)    | <0.5  | 2.12 | $2.82 \pm 0.05$ | Resolved  |
| Cascadia (USGS)         | 6.1   | 2.21 | $\to 3.0$       | Saturated |
| Cook Strait, NZ         | 0.9   | 2.24 | $2.53 \pm 0.17$ | Resolved  |
| Bay of Plenty, NZ       | 1.1   | 1.91 | $2.20 \pm 0.20$ | Resolved  |
| Valais, Switzerland     | 0.8--1.2 | 1.50 | $1.911 \pm 0.066$ | Resolved |
| Sumatra (GEOFON)        | 7.5   | 2.21 | $\to 3.0$       | Saturated |

# Acknowledgements

We acknowledge the open data policies of the United States Geological Survey (USGS),
the GeoNet New Zealand network, the GEOFON Program (GFZ Potsdam), and the Swiss
Seismological Service (SED-ETHZ). Hi-Net data were provided by the National Research
Institute for Earth Science and Disaster Resilience (NIED), Japan [@Okada2004], under
a registered researcher agreement.

# References

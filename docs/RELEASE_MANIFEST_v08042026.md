# RELEASE MANIFEST v08042026

## 1) Alcance del release

Este manifiesto define el contrato de consistencia entre:

- Manuscrito maestro: `Paper02_08042026/main.tex`
- Repositorio de software: `github_08042026/`

Objetivo: garantizar trazabilidad **claim → tabla/figura → CSV → script**.

---

## 2) Figuras requeridas por manuscrito

El manuscrito referencia `fig01`–`fig19`.

### 2.1 Generadas por `github_08042026/figs_gen.py`

- `fig01_loglog_scaling_regions`
- `fig02_posterior_distributions`
- `fig03_precision_degradation`
- `fig04_tectonic_hierarchy`
- `fig05_depth_stratification`
- `fig06_nz_validation`
- `fig07_hinet_spatial_sensitivity`
- `fig08_precision_drift`
- `fig09_mc3d_fisher_surface`
- `fig10_diagnostic_summary`
- `fig11_prior_sensitivity`
- `fig12_zaccagnino_stability`
- `fig13_topological_distance_sketch`
- `fig14_vrml_validation`
- `fig15_empirical_saturation_transition`
- `fig16_cross_scale_synthesis`
- `fig17_oos_posterior_contrast`

### 2.2 Generadas por `github_08042026/wp2_wp3_generate_assets.py`

- `fig18_delta_tohoku_bootstrap`
- `fig19_delta_sensitivity_heatmap`

---

## 3) CSV canónicos mínimos

### 3.1 Núcleo de figuras (carpeta `github_08042026/data/`)

- `noto_correlation.csv`
- `cascadia_correlation.csv`
- `swiss_sed_posterior.csv`
- `sumatra_posterior.csv`
- `scsn_degradation.csv`
- `tectonic_hierarchy.csv`
- `depth_stratification.csv`
- `nz_validation.csv`
- `hinet_radius_sensitivity.csv`
- `precision_drift.csv`
- `gisborne_pathology.csv`
- `japan_mc_depth.csv`
- `prior_sensitivity.csv`
- `zaccagnino_scores.csv`
- `vrml_correlation.csv`
- `mc3d_surface.csv`

### 3.2 WP2/WP3

- `delta_tohoku_bootstrap.csv`
- `delta_tohoku_sensitivity_grid.csv`
- `open_catalogs_paradox2_extension.csv`
- `claim_evidence_matrix.csv`

---

## 4) Estado de artefactos no vinculados

- `fig20_l1_l4_independence_dag.pdf`: presente en `github_08042026/figures`,
  no referenciada en `Paper02_08042026/main.tex` actual.

Se conserva como artefacto auxiliar, fuera del contrato mínimo de `main.tex`.

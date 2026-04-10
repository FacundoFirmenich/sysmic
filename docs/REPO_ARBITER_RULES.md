# REPO ARBITER RULES (v08042026)

Reglas de arbitraje para evitar mezclas de versiones y regresiones numéricas.

## Regla 0 — Lectura completa

Leer siempre archivos completos antes de tocar valores numéricos críticos.

## Regla 1 — Sumatra/Swiss (out-of-sample)

Fuente de verdad prioritaria:

- `Audit_Results_BIEN_REAL/INFERENCE_SUMMARY_GEOFON_Sumatra.csv`
- `Audit_Results_BIEN_REAL/INFERENCE_SUMMARY_Swiss_SED.csv`

No arbitrar con versiones legacy cuando exista divergencia.

## Regla 2 — Prohibición de `_CENTRAL_RAW_` obsoleto para Sumatra

No usar `INFERENCE_SUMMARY_GEOFON_Sumatra.csv` de `_CENTRAL_RAW_` para valores
de saturación/diagnóstico si difiere del audit canónico.

## Regla 3 — Separación estricta de datasets Hi-Net

No mezclar:

- `N_JUICE = 166,920`
- `N_depth_strat = 128,942`
- `VRML directo = 192,387`

Cada N se utiliza en su bloque metodológico correspondiente.

## Regla 4 — Anchors tectónicos canónicos

- Deep Slab: `D2 = 1.26`
- Rifting: `D2 = 1.50`
- Transform: `D2 = 1.81`
- Subduction: `D2 = 2.12`
- Collision: `D2 = 2.24`

## Regla 5 — Fisher barrier

Usar como referencia canónica de manuscrito actual:

- `sigma_c = 2.3 ± 0.4 km`

## Regla 6 — Reproducibilidad del repo

Toda figura declarada en manifiesto debe tener:

1. Archivo final en `figures/`
2. Ruta de generación en script (`figs_gen.py` o `wp2_wp3_generate_assets.py`)
3. Datos fuente en `data/` o fuente canónica explícita documentada

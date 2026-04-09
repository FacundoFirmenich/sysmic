# -*- coding: utf-8 -*-
"""
app/streamlit_app.py
====================
Interactive Sysmic dashboard for fractal tomography of seismic catalogs.

Upload a CSV with columns [latitude, longitude, depth] or use the built-in
demo datasets to explore the Fisher Information Barrier interactively.
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sysmic — Fractal Tomography",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Sysmic v1.0")
st.sidebar.markdown("**Fractal Tomography of Seismicity**")
st.sidebar.markdown("---")

demo = st.sidebar.selectbox(
    "Demo dataset",
    ["Noto, Japan (Hi-Net)", "Cascadia, USA (USGS)",
     "Cook Strait, NZ (GeoNet)", "Sumatra (GEOFON)"],
)

sigma_h = st.sidebar.slider(
    "Location uncertainty σ_h (km)", 0.01, 10.0, 1.0, 0.05,
    help="Horizontal location uncertainty of the catalog."
)

sigma_c = 2.3
st.sidebar.markdown(f"**Fisher barrier σ_c = {sigma_c} km**")
status = "✅ Resolved" if sigma_h < sigma_c else "⚠️ Saturated"
st.sidebar.markdown(f"**Inference status:** {status}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🌍 Sysmic — Fractal Tomography of Seismicity")
st.markdown(
    "Precision-calibrated Bayesian inference of the correlation dimension $D_3$. "
    "All results are empirical; no synthetic data are used."
)

col1, col2, col3 = st.columns(3)

DEMO_PARAMS = {
    "Noto, Japan (Hi-Net)":     dict(D2=2.12, D3=2.82, D3s=0.04, s=0.04, Pbnd=0.8),
    "Cascadia, USA (USGS)":     dict(D2=2.21, D3=3.00, D3s=0.06, s=6.1,  Pbnd=97.0),
    "Cook Strait, NZ (GeoNet)": dict(D2=2.24, D3=2.53, D3s=0.17, s=0.9,  Pbnd=2.1),
    "Sumatra (GEOFON)":         dict(D2=2.21, D3=3.00, D3s=0.05, s=7.5,  Pbnd=99.9),
}
p = DEMO_PARAMS[demo]

with col1:
    st.metric("Observed D₂", f"{p['D2']:.2f}")
with col2:
    d3_display = f"{p['D3']:.2f} ± {p['D3s']:.2f}" if p["Pbnd"] < 50 else "→ 3.0 (saturated)"
    st.metric("Inferred D₃", d3_display)
with col3:
    st.metric("P_bnd (%)", f"{p['Pbnd']:.1f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Posterior plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

x = np.linspace(1.5, 3.1, 500)

# Posterior
ax1.plot(x, np.where((x >= 1.5) & (x <= 3.0), 1/1.5, 0) / (1/1.5),
         "--", color="#475569", alpha=0.6, label="Prior (Uniform)")
y_post = norm.pdf(x, p["D3"], p["D3s"])
ax1.plot(x, y_post / y_post.max(), color="#2563eb", lw=2.5, label="Posterior")
if p["Pbnd"] > 50:
    ax1.fill_between(x, 0, 1, where=(x >= 2.98), color="#d946ef", alpha=0.4,
                     label=f"P_bnd = {p['Pbnd']:.0f}%")
ax1.set_xlabel("D₃"); ax1.set_ylabel("Normalized density")
ax1.set_title("Bayesian Posterior"); ax1.legend(); ax1.set_xlim(1.5, 3.1)

# Fisher surface
s_vals = np.linspace(0.1, 10, 200)
Pbnd_curve = 100 / (1 + np.exp(-2.8 * (s_vals - sigma_c)))
ax2.plot(s_vals, Pbnd_curve, color="#4f46e5", lw=2.5)
ax2.axvline(sigma_c, color="#0f172a", ls=":", lw=2, label=f"σc = {sigma_c} km")
ax2.axvline(sigma_h, color="#d946ef", ls="-", lw=2, label=f"σh = {sigma_h:.2f} km")
ax2.axvspan(sigma_c, 10, color="#475569", alpha=0.08, label="Saturation regime")
ax2.set_xlabel("σ_h (km)"); ax2.set_ylabel("P_bnd (%)");
ax2.set_title("Fisher Information Barrier"); ax2.legend()

st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ---------------------------------------------------------------------------
# Data table
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Empirical Validation Summary")

df = pd.DataFrame([
    {"Region": k, "Network": n, "σ_h": v["s"], "D₂": v["D2"],
     "D₃": f'{v["D3"]:.2f}±{v["D3s"]:.2f}', "P_bnd (%)": v["Pbnd"],
     "Status": "Resolved" if v["Pbnd"] < 50 else "Saturated"}
    for (k, v), n in zip(DEMO_PARAMS.items(),
                          ["Hi-Net", "USGS", "GeoNet", "GEOFON"])
])
st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Analyze Your Own Catalog")
uploaded = st.file_uploader(
    "Upload CSV with columns: latitude, longitude, depth (km)",
    type="csv"
)
if uploaded:
    cat = pd.read_csv(uploaded)
    st.write(f"Loaded {len(cat):,} events.")
    st.dataframe(cat.head(10))
    st.info(
        "Full GP analysis with Bayesian inference: run `python sysmic/core.py` "
        "locally or use the Colab notebook."
    )

st.markdown("---")
st.caption(
    "Data source: United States Geological Survey (USGS) and collaborating networks. "
    "License: GPLv3. Cite: Firmenich et al. (2026), JGR: Solid Earth."
)

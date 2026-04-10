import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize


HERE = Path(__file__).resolve().parent
OUT_FIG = HERE / "figures"
OUT_DATA = HERE / "data"


def _ensure_outdirs() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)


def compute_delta_from_d3(d3: float, L: float = 600.0, ell: float = 3.0) -> float:
    """Invert deck-of-cards formula for delta using sequence-level D3."""
    return (2.0 * L) / (3.0 * (math.exp((d3 - 2.0) * math.log(L / ell)) - 1.0))


def build_wp2_delta_bootstrap() -> pd.DataFrame:
    # Canonical sequence values from manuscript Table Hi-Net
    seq = [
        ("Noto", 2.82, 0.05),
        ("Tohoku", 2.95, 0.03),
        ("Tokachi", 2.39, 0.07),
    ]

    rng = np.random.default_rng(42)
    B = 50_000
    rows = []

    for name, mu, sigma in seq:
        delta_hat = compute_delta_from_d3(mu)

        samples = rng.normal(mu, sigma, B)
        samples = samples[(samples > 2.01) & (samples < 2.999)]
        deltas = (2.0 * 600.0) / (3.0 * (np.exp((samples - 2.0) * np.log(600.0 / 3.0)) - 1.0))

        rows.append(
            {
                "sequence": name,
                "D3_mean": mu,
                "D3_std": sigma,
                "delta_km_hat": float(delta_hat),
                "delta_km_mean_boot": float(np.mean(deltas)),
                "delta_km_std_boot": float(np.std(deltas, ddof=1)),
                "delta_km_p2.5": float(np.percentile(deltas, 2.5)),
                "delta_km_p50": float(np.percentile(deltas, 50.0)),
                "delta_km_p97.5": float(np.percentile(deltas, 97.5)),
                "n_boot": int(len(deltas)),
            }
        )

    return pd.DataFrame(rows)


def save_wp2_figure(df: pd.DataFrame) -> None:
    order = ["Noto", "Tohoku", "Tokachi"]
    p = df.set_index("sequence").loc[order].reset_index()

    x = np.arange(len(p))
    y = p["delta_km_p50"].to_numpy()
    yerr_low = y - p["delta_km_p2.5"].to_numpy()
    yerr_high = p["delta_km_p97.5"].to_numpy() - y

    # Two-scale layout to avoid visually empty panel (Tokachi dominates range).
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.4, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.15], "hspace": 0.05}
    )

    seq_colors = ["#1E88E5", "#7A1FA2", "#0D47A1"]

    def _draw_panel(ax):
        # Credible interval stems
        for i in range(len(x)):
            ax.vlines(
                x[i],
                p["delta_km_p2.5"].iloc[i],
                p["delta_km_p97.5"].iloc[i],
                color=seq_colors[i],
                lw=4.4,
                alpha=0.25,
                zorder=1,
            )

        # Median trend + points
        ax.plot(x, y, "-", color="#113A6B", lw=1.8, alpha=0.85, zorder=2)
        for i in range(len(x)):
            ax.scatter(
                x[i], y[i],
                s=95, color=seq_colors[i], edgecolor="white", linewidth=1.2,
                zorder=3,
            )

        ax.grid(alpha=0.25, ls="--")

    _draw_panel(ax_top)
    _draw_panel(ax_bot)

    # Split ranges
    y_top_min = max(12.0, float(np.min(p["delta_km_p97.5"]) + 4.0))
    y_top_max = float(np.max(p["delta_km_p97.5"]) + 10.0)
    ax_top.set_ylim(y_top_min, y_top_max)
    ax_bot.set_ylim(0, 12)

    # Axis break marks
    d = 0.009
    kwargs = dict(transform=ax_top.transAxes, color="#444444", clip_on=False, lw=1.2)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Annotations (median + CI)
    # User-requested placement:
    #   - Noto / Tohoku: move labels right
    #   - Tokachi: move label left
    x_offsets = {"Noto": 0.22, "Tohoku": 0.22, "Tokachi": -0.26}
    for i, row in p.iterrows():
        label = (
            f"{row['delta_km_p50']:.2f} km\n"
            f"[{row['delta_km_p2.5']:.2f}, {row['delta_km_p97.5']:.2f}]"
        )
        seq_name = row["sequence"]
        x_pos = i + x_offsets.get(seq_name, 0.0)
        target_ax = ax_bot if row["delta_km_p50"] <= 12 else ax_top
        y_shift = 0.35 if target_ax is ax_bot else 2.0
        target_ax.text(
            x_pos, row["delta_km_p50"] + y_shift, label,
            ha="center", va="bottom", fontsize=8.6, color="#102A43",
            fontweight="bold",
            # Requested: transparent container background, no border
            bbox=dict(boxstyle="round,pad=0.18", fc="none", ec="none", alpha=1.0),
            zorder=4,
        )

    ax_top.set_ylabel(r"Inverted fault spacing $\delta$ (km)")
    ax_bot.set_ylabel(r"$\delta$ (zoom)")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(p["sequence"])
    ax_bot.set_xlabel("Hi-Net sequence")

    fig.suptitle(
        r"Bootstrap inversion of $\delta$ from sequence-level $D_3$ "
        r"(95% credible intervals)",
        fontsize=11,
        y=0.98,
    )
    # Avoid tight_layout warning with broken-axis styling
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.18, top=0.91, hspace=0.05)

    # Light-blue explanatory strip (consistent style with figs_gen outputs)
    fig.text(
        0.5,
        0.052,
        (
            "Fig. 18 — Bootstrap inversion of fault spacing $\\delta$ from sequence-level $D_3$ "
            "(95% credible intervals)."
        ),
        ha="center",
        va="center",
        fontsize=7.8,
        color="#0A2342",
        bbox=dict(boxstyle="round,pad=0.24", fc="#DCEEFF", ec="#B0B8C4", lw=0.65, alpha=0.95),
    )

    out_paths = [
        OUT_FIG / "fig18_delta_tohoku_bootstrap.pdf",
        OUT_FIG / "fig18_delta_tohoku_bootstrap.png",
    ]
    for path in out_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)


def build_wp2_delta_sensitivity(B: int = 20_000) -> pd.DataFrame:
    """
    Robustness grid for delta inversion against geometric assumptions.

    We vary:
      - L in {400, 600, 800} km
      - ell in {2, 3, 5} km
    and propagate sequence-level D3 uncertainty via bootstrap.
    """
    seq = [
        ("Noto", 2.82, 0.05),
        ("Tohoku", 2.95, 0.03),
        ("Tokachi", 2.39, 0.07),
    ]
    L_values = [400.0, 600.0, 800.0]
    ell_values = [2.0, 3.0, 5.0]

    rng = np.random.default_rng(123)
    rows = []

    for L in L_values:
        for ell in ell_values:
            for name, mu, sigma in seq:
                samples = rng.normal(mu, sigma, B)
                samples = samples[(samples > 2.01) & (samples < 2.999)]
                deltas = (2.0 * L) / (3.0 * (np.exp((samples - 2.0) * np.log(L / ell)) - 1.0))
                rows.append(
                    {
                        "sequence": name,
                        "L_km": L,
                        "ell_km": ell,
                        "delta_p50": float(np.percentile(deltas, 50.0)),
                        "delta_p2.5": float(np.percentile(deltas, 2.5)),
                        "delta_p97.5": float(np.percentile(deltas, 97.5)),
                        "delta_mean": float(np.mean(deltas)),
                        "delta_std": float(np.std(deltas, ddof=1)),
                        "n_boot": int(len(deltas)),
                    }
                )

    return pd.DataFrame(rows)


def save_wp2_sensitivity_figure(df_sens: pd.DataFrame) -> None:
    """Heatmap-style reviewer figure focused on Tohoku robustness."""
    toh = df_sens[df_sens["sequence"] == "Tohoku"].copy()
    pivot = toh.pivot(index="L_km", columns="ell_km", values="delta_p50")

    L_sorted = sorted(pivot.index)
    ell_sorted = sorted(pivot.columns)
    Z = pivot.loc[L_sorted, ell_sorted].to_numpy()

    # Requested palette (dark -> light):
    # morado -> violeta -> azul fuerte -> azul -> celeste
    cmap_wp2 = LinearSegmentedColormap.from_list(
        "wp2_delta_blueviolet",
        ["#4B0082", "#7A1FA2", "#0D47A1", "#1E88E5", "#81D4FA"],
        N=256,
    )

    # Keep upper values in cyan family so the ~5 km level maps to celeste
    # (no green/yellow transition).
    norm = Normalize(vmin=float(np.nanmin(Z)), vmax=5.0, clip=True)

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap=cmap_wp2, norm=norm)
    for i, L in enumerate(L_sorted):
        for j, ell in enumerate(ell_sorted):
            txt_col = "#ffffff" if Z[i, j] < 3.8 else "#0A2342"
            ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", color=txt_col,
                    fontsize=9, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, label=r"Median inverted $\delta_{\mathrm{Tohoku}}$ (km)")
    cbar.set_ticks([2.0, 3.0, 4.0, 5.0])
    ax.set_xticks(np.arange(len(ell_sorted)), [f"{e:.0f}" for e in ell_sorted])
    ax.set_yticks(np.arange(len(L_sorted)), [f"{L:.0f}" for L in L_sorted])
    ax.set_xlabel(r"Micro-scale cutoff $\ell$ (km)")
    ax.set_ylabel(r"Macro-scale extent $L$ (km)")
    ax.set_title(r"Sensitivity of inverted $\delta$ to $(L,\ell)$ assumptions", fontsize=11)

    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.205)
    fig.text(
        0.5,
        0.058,
        (
            "Fig. 19 — Sensitivity of inverted $\\delta_{\\mathrm{Tohoku}}$ to geometric assumptions "
            "$(L,\\ell)$\n"
            "Each cell reports posterior median under bootstrap propagation."
        ),
        ha="center",
        va="center",
        fontsize=7.4,
        color="#0A2342",
        bbox=dict(boxstyle="round,pad=0.22", fc="#DCEEFF", ec="#B0B8C4", lw=0.65, alpha=0.95),
    )

    out_paths = [
        OUT_FIG / "fig19_delta_sensitivity_heatmap.pdf",
        OUT_FIG / "fig19_delta_sensitivity_heatmap.png",
    ]
    for path in out_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300)


def build_claim_evidence_matrix() -> pd.DataFrame:
    rows = [
        {
            "claim_id": "C1",
            "claim": "Fisher barrier at sigma_c=2.3±0.4 km separates resolved vs saturated inference",
            "evidence_main": "Tab. scsn; Fig. scsn_degradation; Fig. precision_drift",
            "data_source": "SCSN degradation + Swiss-SED + GEOFON",
            "stat_criterion": "Pbnd transition + posterior shift + uncertainty propagation",
            "inference_level": "Calibrated + externally supported",
        },
        {
            "claim_id": "C2",
            "claim": "Below-barrier catalogs recover sub-volumetric D3",
            "evidence_main": "Tab. hinet; Fig. posteriors; Fig. oos_posterior_contrast",
            "data_source": "Hi-Net JUICE + Swiss-SED",
            "stat_criterion": "Pbnd=0%; KL>10 nats; bounded posterior away from D3=3",
            "inference_level": "Empirically confirmed",
        },
        {
            "claim_id": "C3",
            "claim": "Depth-dependent planarization within Japan is monotonic",
            "evidence_main": "Tab. depth; Tab. depth_stats; Fig. depth_stratification",
            "data_source": "Hi-Net 2001-2005 depth-strat",
            "stat_criterion": "p<1e-15; Hedges g=1.76 [1.52,2.00]",
            "inference_level": "Empirically confirmed",
        },
        {
            "claim_id": "C4",
            "claim": "Deck-of-cards inversion provides sequence-resolved spacing estimates",
            "evidence_main": "Tab. delta_tohoku_bootstrap; Fig. delta_tohoku_bootstrap; Fig. delta_sensitivity_heatmap",
            "data_source": "Hi-Net JUICE D3 posteriors",
            "stat_criterion": "Bootstrap CI + sensitivity to (L,ell)",
            "inference_level": "Model-based, stress-tested",
        },
        {
            "claim_id": "C5",
            "claim": "Preliminary tectonic hierarchy is portable to open catalogs",
            "evidence_main": "Tab. hierarchy; Tab. open_catalogs_paradox2",
            "data_source": "Pan-American + Swiss-SED + GeoNet + GEOFON",
            "stat_criterion": "Resolved/saturated consistency under precision conditioning",
            "inference_level": "Preliminary/hypothesis-generating",
        },
    ]
    return pd.DataFrame(rows)


def build_wp3_open_catalogs() -> pd.DataFrame:
    # Uses available open catalogs without authentication/registration workflows.
    rows = [
        {
            "catalog": "Swiss-SED Valais",
            "access": "EIDA open endpoint",
            "tectonic_setting": "Rifting",
            "D2": 1.50,
            "D3": 1.911,
            "D3_std": 0.066,
            "sigma_h_km": 1.2,
            "Pbnd_pct": 0.0,
            "status": "resolved",
        },
        {
            "catalog": "GeoNet Bay of Plenty",
            "access": "GeoNet open web services",
            "tectonic_setting": "Back-arc rifting",
            "D2": 1.91,
            "D3": 2.1980360531425096,
            "D3_std": 0.19747251179402414,
            "sigma_h_km": 1.1,
            "Pbnd_pct": 0.0,
            "status": "resolved",
        },
        {
            "catalog": "GeoNet Cook Strait",
            "access": "GeoNet open web services",
            "tectonic_setting": "Transform/thrust",
            "D2": 2.24,
            "D3": 2.528892001091647,
            "D3_std": 0.17294456260817404,
            "sigma_h_km": 0.9,
            "Pbnd_pct": 0.0,
            "status": "resolved",
        },
        {
            "catalog": "GEOFON Sumatra",
            "access": "GEOFON open endpoint",
            "tectonic_setting": "Subduction",
            "D2": 2.2126,
            "D3": 2.998,
            "D3_std": 0.002,
            "sigma_h_km": 7.5,
            "Pbnd_pct": 100.0,
            "status": "saturated",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    _ensure_outdirs()

    wp2 = build_wp2_delta_bootstrap()
    for path in [
        OUT_DATA / "delta_tohoku_bootstrap.csv",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        wp2.to_csv(path, index=False)
    save_wp2_figure(wp2)

    wp2_sens = build_wp2_delta_sensitivity(B=20_000)
    for path in [
        OUT_DATA / "delta_tohoku_sensitivity_grid.csv",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        wp2_sens.to_csv(path, index=False)
    save_wp2_sensitivity_figure(wp2_sens)

    wp3 = build_wp3_open_catalogs()
    for path in [
        OUT_DATA / "open_catalogs_paradox2_extension.csv",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        wp3.to_csv(path, index=False)

    claims = build_claim_evidence_matrix()
    for path in [
        OUT_DATA / "claim_evidence_matrix.csv",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        claims.to_csv(path, index=False)

    print("[OK] WP2 CSV:")
    print(wp2.to_string(index=False))
    print("\n[OK] WP2 sensitivity grid (head):")
    print(wp2_sens.head(9).to_string(index=False))
    print("\n[OK] WP3 CSV:")
    print(wp3.to_string(index=False))
    print("\n[OK] Claim-Evidence matrix:")
    print(claims.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Visualization module for Seismic Fractal Analysis.
Implements publication-quality plotting with 'Premium Scientific' styling.
"""

from typing import Any, Dict, List, Union

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle


class StyleManager:
    """Manages matplotlib styles for publication-quality figures."""

    @staticmethod
    def set_premium_style():
        """Apply 'Premium Scientific' style settings."""
        # Reset to defaults first to avoid contamination
        plt.rcdefaults()

        # Base configuration
        plt.rcParams.update(
            {
                # Fonts
                "font.family": "sans-serif",
                # "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], # Let MPL choose best available
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "axes.titleweight": "bold",
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                # Figure layout
                "figure.figsize": (8, 6),
                "figure.dpi": 300,
                "figure.autolayout": True,
                "figure.facecolor": "white",
                # Axes
                "axes.linewidth": 1.0,
                "axes.grid": True,
                "axes.axisbelow": True,
                "axes.spines.top": False,
                "axes.spines.right": False,
                # Grid
                "grid.color": "#E0E0E0",
                "grid.linestyle": "-",
                "grid.linewidth": 0.5,
                "grid.alpha": 1.0,
                # Lines & Markers
                "lines.linewidth": 2.0,
                "lines.markersize": 6,
                "lines.markeredgewidth": 0.0,
                # Saving
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.1,
                "savefig.dpi": 300,
                "savefig.transparent": False,
            }
        )

    @staticmethod
    def get_palette(n_colors: int = 5) -> list:
        """Return a professional color palette (Okabe-Ito inspired)."""
        # High contrast, colorblind safe
        colors = [
            "#0072B2",  # Blue
            "#D55E00",  # Vermilion
            "#009E73",  # Bluish Green
            "#F0E442",  # Yellow
            "#CC79A7",  # Reddish Purple
            "#56B4E9",  # Sky Blue
            "#E69F00",  # Orange
        ]
        return colors[:n_colors]


class FractalPlotter:
    """Generates specific scientific plots for SFA."""

    @staticmethod
    def plot_spatial_distribution(
        coordinates: np.ndarray, magnitudes: np.ndarray, region_name: str
    ) -> plt.Figure:
        """
        2D Map view with depth coloring + 3D inset or side view.
        Focuses on clear geographic representation.
        """
        StyleManager.set_premium_style()

        # Create a layout with a main map and a depth profile
        fig = plt.figure(figsize=(10, 6))
        gs = GridSpec(1, 2, width_ratios=[2, 1], wspace=0.15)

        # 1. Map View (Lon vs Lat)
        ax_map = fig.add_subplot(gs[0])

        # Normalize depth for colormap
        depths = coordinates[:, 2]
        sc = ax_map.scatter(
            coordinates[:, 0],  # Lon
            coordinates[:, 1],  # Lat
            c=depths,
            cmap="viridis_r",  # Deep = Darker/Purple, Shallow = Yellow/Green
            s=np.exp(magnitudes / 2) * 3,
            alpha=0.7,
            edgecolor="none",
        )

        ax_map.set_xlabel("Longitude (°)")
        ax_map.set_ylabel("Latitude (°)")
        ax_map.set_title(f"Seismicity Map: {region_name}")
        ax_map.set_aspect("equal")
        ax_map.grid(True, linestyle=":", alpha=0.6)

        # Add colorbar
        cbar = plt.colorbar(
            sc, ax=ax_map, orientation="horizontal", pad=0.15, fraction=0.05
        )
        cbar.set_label("Depth (km)")

        # 2. Depth Profile (Depth vs Lat or Lon depending on aspect)
        # Let's do Depth vs Latitude (Cross-section)
        ax_profile = fig.add_subplot(gs[1], sharey=ax_map)

        ax_profile.scatter(
            depths,
            coordinates[:, 1],
            c=depths,
            cmap="viridis_r",
            s=np.exp(magnitudes / 2) * 3,
            alpha=0.7,
            edgecolor="none",
        )

        ax_profile.set_xlabel("Depth (km)")
        # ax_profile.set_ylabel("Latitude (°)") # Shared
        ax_profile.tick_params(labelleft=False)
        ax_profile.set_title("Cross-section")
        ax_profile.invert_xaxis()  # Surface on left/right? Usually depth increases downwards or to right.
        # Let's keep 0 on left.
        ax_profile.set_xlim(0, max(depths) * 1.1)
        ax_profile.grid(True, linestyle=":", alpha=0.6)

        return fig

    @staticmethod
    def plot_correlation_integral(
        log_r: np.ndarray,
        log_c: np.ndarray,
        slope: float,
        valid_mask: np.ndarray,
        region_name: str,
    ) -> plt.Figure:
        """Log-log plot of Correlation Integral with scaling region."""
        StyleManager.set_premium_style()
        fig, ax = plt.subplots(figsize=(7, 6))

        # Plot all data
        ax.scatter(
            log_r,
            log_c,
            color="#95a5a6",  # Grey
            alpha=0.4,
            label="Raw Data",
            s=15,
            edgecolor="none",
        )

        # Highlight scaling region
        if np.any(valid_mask):
            ax.scatter(
                log_r[valid_mask],
                log_c[valid_mask],
                color="#e74c3c",  # Red
                label="Scaling Region",
                s=25,
                zorder=5,
                edgecolor="none",
            )

            # Plot regression line
            x_fit = log_r[valid_mask]
            y_fit = slope * x_fit + (
                np.mean(log_c[valid_mask]) - slope * np.mean(x_fit)
            )
            ax.plot(
                x_fit,
                y_fit,
                color="black",
                linestyle="-",
                linewidth=2,
                label=f"Fit ($D_2={slope:.2f}$)",
                zorder=6,
            )

        ax.set_xlabel("$\log_{10}(r)$")
        ax.set_ylabel("$\log_{10}(C(r))$")
        ax.set_title(f"Fractal Dimension Estimation: {region_name}")
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

        return fig

    @staticmethod
    def plot_comparative_analysis(
        results: Dict[str, Dict[str, Any]],
    ) -> plt.Figure:
        """Bar chart comparing D2 across regions."""
        StyleManager.set_premium_style()
        fig, ax = plt.subplots(figsize=(10, 5))

        regions = list(results.keys())
        d2_values = [results[r]["d2_mean"] for r in regions]
        d2_errors = [results[r]["d2_std"] for r in regions]

        # Sort by D2 for better readability
        sorted_indices = np.argsort(d2_values)
        regions = [regions[i] for i in sorted_indices]
        d2_values = [d2_values[i] for i in sorted_indices]
        d2_errors = [d2_errors[i] for i in sorted_indices]

        palette = StyleManager.get_palette(len(regions))
        bars = ax.barh(
            regions,
            d2_values,
            xerr=d2_errors,
            capsize=4,
            color="#3498db",
            alpha=0.9,
            edgecolor="none",
            height=0.6,
        )

        ax.set_xlabel("Fractal Dimension ($D_2$)")
        ax.set_title("Comparative Fractal Dimension Analysis")
        ax.set_xlim(0, 3.0)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        # Add value labels
        for rect in bars:
            width = rect.get_width()
            ax.text(
                width + 0.1,
                rect.get_y() + rect.get_height() / 2.0,
                f"{width:.2f}",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=9,
            )

        return fig

    @staticmethod
    def plot_bayesian_density(
        results: Dict[str, Dict[str, Any]],
    ) -> plt.Figure:
        """
        Alternative Bayesian visualization: Overlapping Density Plots (Ridge-like).
        Cleaner and more 'scientific' than violin plots.
        """
        StyleManager.set_premium_style()
        fig, ax = plt.subplots(figsize=(10, 6))

        regions = list(results.keys())
        palette = StyleManager.get_palette(len(regions))

        # Sort by mean D2 for visual hierarchy
        means = [results[r]["mean"] for r in regions]
        sorted_indices = np.argsort(means)
        regions = [regions[i] for i in sorted_indices]
        palette = [palette[i] for i in sorted_indices]

        for i, region in enumerate(regions):
            samples = results[region]["samples"]
            if len(samples) > 0:
                # Kernel Density Estimate
                sns.kdeplot(
                    samples,
                    ax=ax,
                    fill=True,
                    alpha=0.3,
                    linewidth=2,
                    color=palette[i],
                    label=rf"{region} ($\mu={np.mean(samples):.2f}$)",
                )
                # Add a small rug plot for detail
                # sns.rugplot(samples, ax=ax, color=palette[i], alpha=0.5, height=0.05)

        ax.set_xlabel("Fractal Dimension ($D_2$)")
        ax.set_ylabel("Posterior Density")
        ax.set_title("Bayesian Posterior Distributions")
        ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        ax.set_xlim(0, 3.0)

        return fig

    @staticmethod
    def plot_computational_tradeoff_classic(
        sample_sizes: List[int],
        errors: List[float],
        times: List[float],
        title: str = "Computational Trade-off (Classic View)",
    ) -> plt.Figure:
        """
        Classic Dual-Axis Plot: Time and Error vs Sample Size.
        More intuitive for understanding trends.
        """
        StyleManager.set_premium_style()
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # Plot Time (Left Axis)
        color = "tab:blue"
        ax1.set_xlabel("Sample Size (N)")
        ax1.set_ylabel("Computation Time (s)", color=color)
        ax1.plot(sample_sizes, times, color=color, marker="o", label="Time")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Plot Error (Right Axis)
        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Relative Error (%)", color=color)
        ax2.plot(
            sample_sizes, errors, color=color, marker="s", linestyle="--", label="Error"
        )
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title(title)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_computational_tradeoff(
        sample_sizes: List[int],
        errors: List[float],
        times: List[float],
        title: str = "Computational Trade-off",
    ) -> plt.Figure:
        """
        Pareto-style Scatter Plot: Time vs Error.
        Point size/color indicates Sample Size (N).
        """
        StyleManager.set_premium_style()
        fig, ax = plt.subplots(figsize=(8, 6))

        # Normalize N for sizing
        sizes = np.array(sample_sizes)
        norm_sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min())
        marker_sizes = 50 + norm_sizes * 200

        scatter = ax.scatter(
            times,
            errors,
            s=marker_sizes,
            c=sizes,
            cmap="viridis",
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
            zorder=5,
        )

        # Connect points with a line to show the trend
        ax.plot(times, errors, linestyle="--", color="gray", alpha=0.5, zorder=1)

        # Annotate points with N
        for t, e, n in zip(times, errors, sample_sizes):
            ax.annotate(
                f"N={n}", (t, e), xytext=(5, 5), textcoords="offset points", fontsize=9
            )

        ax.set_xlabel("Computation Time (s)")
        ax.set_ylabel("Relative Error (%)")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Log scale might be useful if ranges are large, but let's stick to linear for now unless requested
        # ax.set_xscale('log')
        # ax.set_yscale('log')

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Sample Size (N)")

        return fig

    @staticmethod
    def save_plot(
        fig: Union[plt.Figure, go.Figure],
        filename_base: str,
        output_dir: str = "figures",
    ):
        """Save plot to PNG and PDF formats."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename_base)

        if isinstance(fig, plt.Figure):
            fig.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
            fig.savefig(f"{output_path}.pdf", format="pdf", bbox_inches="tight")
            print(f"Saved {output_path}.png and {output_path}.pdf")
            plt.close(fig)  # Close to free memory

        elif isinstance(fig, go.Figure):
            try:
                fig.write_image(f"{output_path}.png", scale=3)
                fig.write_image(f"{output_path}.pdf", format="pdf")
                print(f"Saved {output_path}.png and {output_path}.pdf")
            except (ValueError, IOError) as e:
                print(f"Could not save Plotly figure: {e}")

    @staticmethod
    def render_table(df: pd.DataFrame, filename_base: str, title: str = ""):
        """Render a DataFrame as a publication-ready table image."""
        StyleManager.set_premium_style()

        fig, ax = plt.subplots(figsize=(8, len(df) * 0.5 + 1))
        ax.axis("off")

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            loc="center",
            cellLoc="center",
            bbox=[0, 0, 1, 1],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # Style headers
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#2c3e50")
            else:
                cell.set_facecolor("#f8f9fa" if row % 2 else "white")

        if title:
            plt.title(title, pad=10, fontweight="bold")

        plt.tight_layout()

        # Save
        fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{filename_base}.pdf", format="pdf", bbox_inches="tight")
        print(f"Saved table {filename_base}")
        plt.close(fig)
        return fig


class InteractivePlotter:
    """Generates interactive Plotly figures for the dashboard."""

    @staticmethod
    def plot_3d_spatial(
        coordinates: np.ndarray, magnitudes: np.ndarray, region_name: str
    ) -> go.Figure:
        """Interactive 3D scatter plot."""
        depths = coordinates[:, 2]

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=coordinates[:, 0],
                    y=coordinates[:, 1],
                    z=-depths,  # Negative depth for intuitive visualization
                    mode="markers",
                    marker=dict(
                        size=np.exp(magnitudes / 2),  # Scale size by magnitude
                        color=magnitudes,
                        colorscale="Plasma",
                        opacity=0.8,
                        colorbar=dict(title="Magnitude (Mw)"),
                    ),
                    text=[
                        f"Mag: {m:.1f}<br>Depth: {d:.1f}km"
                        for m, d in zip(magnitudes, depths)
                    ],
                    hoverinfo="text",
                )
            ]
        )

        fig.update_layout(
            title=f"3D Seismicity: {region_name}",
            scene=dict(
                xaxis_title="East-West (km)",
                yaxis_title="North-South (km)",
                zaxis_title="Depth (km)",
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            template="plotly_dark",
        )
        return fig

    @staticmethod
    def plot_correlation_integral(
        log_r: np.ndarray,
        log_c: np.ndarray,
        slope: float,
        valid_mask: np.ndarray,
        region_name: str,
    ) -> go.Figure:
        """Interactive Log-Log plot."""
        fig = go.Figure()

        # All data
        fig.add_trace(
            go.Scatter(
                x=log_r,
                y=log_c,
                mode="markers",
                name="Raw Data",
                marker=dict(color="gray", opacity=0.5, size=6),
            )
        )

        # Scaling region
        if np.any(valid_mask):
            fig.add_trace(
                go.Scatter(
                    x=log_r[valid_mask],
                    y=log_c[valid_mask],
                    mode="markers",
                    name="Scaling Region",
                    marker=dict(color="#E74C3C", size=8),
                )
            )

            # Regression line
            x_fit = log_r[valid_mask]
            y_fit = slope * x_fit + (
                np.mean(log_c[valid_mask]) - slope * np.mean(x_fit)
            )

            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode="lines",
                    name=f"Fit (D₂={slope:.3f})",
                    line=dict(color="white", dash="dash", width=2),
                )
            )

        fig.update_layout(
            title=f"Correlation Integral: {region_name}",
            xaxis_title="log(r)",
            yaxis_title="log(C(r))",
            template="plotly_dark",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        return fig


class AdvancedPlotters:
    """
    Additional plotting utilities for LaTeX export and global visualization.
    """

    @staticmethod
    def save_latex_table(
        df: pd.DataFrame,
        filename: str,
        caption: str = "",
        label: str = "tab:results",
    ) -> str:
        """
        Export DataFrame to LaTeX table format.

        Args:
            df: DataFrame to export
            filename: Output filename (without extension)
            caption: Table caption
            label: LaTeX label for cross-referencing

        Returns:
            LaTeX table string
        """
        # Format floats to 3 decimal places
        df_formatted = df.copy()
        for col in df_formatted.columns:
            if df_formatted[col].dtype in [np.float64, np.float32]:
                df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.3f}")
        latex_str = df_formatted.to_latex(
            index=False,
            escape=False,
            caption=caption if caption else None,
            label=label if caption else None,
        )

        # Save to file
        with open(f"{filename}.tex", "w", encoding="utf-8") as f:
            f.write(latex_str)

        print(f"Saved LaTeX table: {filename}.tex")
        return latex_str

    @staticmethod
    def world_map(
        regions: List[Dict[str, Any]],
        filename_base: str = "region_map",
    ) -> plt.Figure:
        """
        Plot world map with regional bounding boxes.
        Uses a clean, flat aesthetic.
        """
        StyleManager.set_premium_style()
        fig, ax = plt.subplots(figsize=(12, 6))

        # Setup Map
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

            ax.add_feature(cfeature.LAND, facecolor="#F5F5DC")  # Light beige for land
            ax.add_feature(cfeature.OCEAN, facecolor="#EBF5FB")  # Light blue for ocean
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.7)

            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
            ax.gridlines(
                draw_labels=True,
                dms=True,
                x_inline=False,
                y_inline=False,
                alpha=0.3,
                linestyle="--",
            )

        except ImportError:
            # Fallback if cartopy not installed
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel("Longitude (°)")
            ax.set_ylabel("Latitude (°)")
            ax.grid(True, alpha=0.4, linestyle="--")

        ax.set_title("Regional Selection for Seismic Fractal Analysis")

        # Generate colors
        cmap = plt.get_cmap("tab20")
        colors = cmap(np.linspace(0, 1, len(regions)))

        for reg, col in zip(regions, colors):
            minlon = reg.get("minlon", reg.get("min_lon", -180))
            maxlon = reg.get("maxlon", reg.get("max_lon", 180))
            minlat = reg.get("minlat", reg.get("min_lat", -90))
            maxlat = reg.get("maxlat", reg.get("max_lat", 90))

            # Handle dateline crossing if necessary (simple version)
            width = maxlon - minlon
            height = maxlat - minlat

            rect = Rectangle(
                (minlon, minlat),
                width,
                height,
                facecolor=col,
                alpha=0.5,
                edgecolor="black",
                linewidth=1,
                zorder=10,
            )
            ax.add_patch(rect)

            # Label
            label = reg.get("code", reg.get("name", "?"))
            ax.text(
                (minlon + maxlon) / 2,
                (minlat + maxlat) / 2,
                label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                zorder=11,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="none",
                ),
            )

        plt.tight_layout()

        # Save
        fig.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{filename_base}.pdf", format="pdf", bbox_inches="tight")
        print(f"Saved {filename_base}")
        plt.close(fig)

        return fig

"""
Publication-Oriented Plotting Utilities (Nature MI style).

This module centralizes figure generation for the textual perturbation study. The guiding goal
is to produce consistent, publication-ready visuals with minimal manual post-processing:
- Standard typography and layout (Nature-style defaults)
- Accessible colors and readable annotations
- Deterministic saving to PNG (and SVG when possible)

Most functions follow a simple contract: take data + a target filename, write the figure, and
close the matplotlib state to prevent cross-figure contamination in batch runs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import seaborn as sns
from PIL import Image
import pandas as pd
import xml.etree.ElementTree as ET
from matplotlib.colors import LinearSegmentedColormap
from project.config import PUBLISH_DPI
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Nature-style configuration (refined)
# Fonts: Helvetica/Arial; clean axes; consistent lettered panels
# Colors: Accessible palette aligned with Nature MI figures

def configure_nature_style():
    """
    Apply consistent Nature-like matplotlib/seaborn style defaults.

    Scientific Visualization Standards:
    1. Typography: Helvetica/Arial (sans-serif) is standard for scientific figures.
    2. Vector Graphics: PDF fonttype 42 ensures text remains editable in Illustrator/Inkscape.
    3. Resolution: 300 DPI minimum for raster outputs (PNG).
    4. Accessibility: Clean backgrounds (white), no grid clutter, high-contrast lines.
    """
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'Nimbus Sans', 'DejaVu Sans']
    rcParams['font.size'] = 8
    rcParams['axes.titlesize'] = 9
    rcParams['axes.labelsize'] = 8
    rcParams['xtick.labelsize'] = 7
    rcParams['ytick.labelsize'] = 7
    rcParams['legend.fontsize'] = 7
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['svg.fonttype'] = 'none'
    rcParams['savefig.facecolor'] = 'white'
    rcParams['axes.facecolor'] = 'white'
    rcParams['lines.linewidth'] = 1.0
    rcParams['lines.markersize'] = 3.0
    rcParams['axes.linewidth'] = 0.6
    rcParams['grid.linewidth'] = 0.5
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams['xtick.top'] = False
    rcParams['xtick.bottom'] = True
    rcParams['ytick.right'] = False
    rcParams['ytick.left'] = True
    rcParams['figure.figsize'] = (3.5, 3.0)
    rcParams['figure.constrained_layout.use'] = False
    try:
        import seaborn as _sns
        _sns.set_context("paper", font_scale=0.9)
        _sns.set_style("white")
    except Exception:
        pass

def apply_nature_axes(ax, title=None):
    """
    Apply axis-level styling consistent with the configured publication theme.

    Args:
        ax: Matplotlib Axes to style.
        title: Optional axes title.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(
        axis='both',
        which='both',
        labelsize=7,
        length=3,
        width=0.6,
        direction='out',
        bottom=True,
        left=True,
        top=False,
        right=False
    )
    if title is not None:
        ax.set_title(title)
    return ax

# Palette
COLOR_PRIMARY = "#0072B2"   # blue
COLOR_SECONDARY = "#E69F00" # orange
COLOR_NEUTRAL = "#666666"   # dark grey
COLOR_LIGHT = "#bdbdbd"     # light grey
COLOR_DARK = "#212121"      # almost black
GROUP_PALETTE = {"Typo": COLOR_PRIMARY, "Whitespace": COLOR_SECONDARY}

def _save_both(filename):
    base, _ = os.path.splitext(filename)
    plt.savefig(filename, bbox_inches='tight', dpi=PUBLISH_DPI)
    try:
        plt.savefig(base + ".svg", bbox_inches='tight', format='svg')
    except Exception:
        pass
    plt.close()

def select_max_change_pair(df: pd.DataFrame) -> tuple[str | None, str | None]:
    return None, None

#
# mean_radar_chart removed per visualization policy
def plot_holistic_radar(categories, values, title, filename):
    configure_nature_style()
    import numpy as np
    import matplotlib.pyplot as plt
    n = len(categories)
    angles = [k / float(n) * 2 * np.pi for k in range(n)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(3.8, 3.3), subplot_kw={"projection": "polar"})
    # Orientation: place first category at top, clockwise order
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], [""] * (n), color="black", size=7)
    ax.tick_params(axis="x", pad=6)
    # Grid aesthetics
    ax.grid(color="#d0d0d0", linestyle="--", linewidth=0.5)
    ax.spines["polar"].set_linewidth(0.8)
    # Data range (adaptive, non-fixed)
    vals = np.array(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(vals.min())
        vmax = float(vals.max())
        if vmin == vmax:
            d = abs(vmax) if vmax != 0 else 1.0
            vmin = vmax - 0.1 * d
            vmax = vmax + 0.1 * d
    # Add margin to avoid polygon hugging grid
    rng = max(1e-9, (vmax - vmin))
    vmin -= 0.08 * rng
    vmax += 0.08 * rng
    # Nice ticks (4 rings using 1-2-5 rule)
    def nice_step(r, target=4):
        raw = r / target
        exp = np.floor(np.log10(raw))
        base = raw / (10 ** exp)
        if base < 1.5:
            step = 1.0
        elif base < 3.5:
            step = 2.0
        else:
            step = 5.0
        return step * (10 ** exp)
    step = nice_step(vmax - vmin, target=4)
    start = np.ceil(vmin / step) * step
    ticks = np.arange(start, vmax + step * 0.5, step)
    fmt = "{:.1f}" if (vmax - vmin) < 0.1 else "{:.1f}"
    ax.set_rlabel_position(90)
    _labels = [fmt.format(t) for t in ticks]
    _labels = [("0" + s[2:]) if s.startswith("-0") else s for s in _labels]
    plt.yticks(ticks, _labels, color="grey", size=6)
    plt.ylim(vmin, vmax)
    # Uniform category labels at fixed radial offset
    r_label = vmax + 0.10 * rng
    for th, lab in zip(angles[:-1], categories):
        # Dynamic alignment based on angle to avoid overlap
        angle_deg = np.degrees(th)
        if angle_deg == 0:
            ha, va = "center", "bottom"
        elif angle_deg == 180:
            ha, va = "center", "top"
        elif 0 < angle_deg < 180:
            ha, va = "left", "center"
        else:
            ha, va = "right", "center"
        ax.text(th, r_label, lab, fontsize=8, ha=ha, va=va, color="black")
    vals_closed = list(values) + [values[0]]
    ax.plot(angles, vals_closed, linewidth=1.6, linestyle="solid", color=COLOR_PRIMARY)
    ax.scatter(angles, vals_closed, s=8, color=COLOR_PRIMARY, zorder=3)
    ax.fill(angles, vals_closed, color=COLOR_PRIMARY, alpha=0.12)
    plt.title(title, y=1.1)
    plt.tight_layout()
    _save_both(filename)

def plot_metric_clustermap(corr, filename, category_map: dict | None = None, category_order: list[str] | None = None, category_palette: dict | None = None, cbar_loc: str = "left", label_style: str = "text", title: str | None = None):
    """
    Plot a metric taxonomy heatmap from a correlation matrix.

    Scientific Rationale:
    - Hypothesis Validation: We expect high intra-group correlation (diagonal blocks) 
      and low inter-group correlation, confirming the theoretical independence of 
      linguistic dimensions (e.g., Syntax vs Semantics).
    - Taxonomy Alignment: When `category_map` is provided, we strictly enforce the 
      theoretical ordering (row_cluster=False) rather than data-driven hierarchical 
      clustering. This imposes the theoretical framework onto the empirical data to 
      visualize deviations.

    Args:
        corr: Square correlation matrix (metrics x metrics).
        filename: Output path for the saved figure (PNG; SVG attempted).
        title: Optional figure title (ignored; no overall title is drawn).
        category_map: Optional metric -> category label mapping.
        category_order: Optional explicit ordering of category labels.
        category_palette: Optional category -> color mapping for label styling.
        cbar_loc: Colorbar placement ("left", "right", or "bottom").
        label_style: How to indicate categories ("text" headers/brackets or "bars").
    """
    configure_nature_style()
    import seaborn as sns
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FixedLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    # Standard metric display names
    metric_display_names = {
        "normalized_levenshtein": "Normalized Levenshtein (NL)",
        "char_ngram_jaccard": "Char n-gram Jaccard (CNJ)",
        "compression_delta": "Compression Delta (CD)",
        "token_count_change": "Token Count Change (TCC)",
        "fragmentation_index": "Fragmentation Index (FI)",
        "normalized_entropy_delta": "Normalized Entropy Delta (NED)",
        "contextual_embedding_distance": "Contextual Embedding Distance (CED)",
        "semantic_entailment_score": "Semantic Entailment Score (SES)",
        "lm_surprisal_delta": "Language Model Surprisal Delta (LMSD)",
        "dependency_overlap_score": "Dependency Overlap Score (DOS)",
        "tree_depth_change": "Tree Depth Change (TDC)",
        "pos_divergence": "Part-of-Speech Divergence (POSD)",
        "syllable_count_change": "Syllable Count Change (SCC)",
        "stress_pattern_divergence": "Stress Pattern Divergence (SPD)",
        "ssim_distance": "Structural Similarity Distance (SSIMD)",
        "glyph_displacement": "Glyph Displacement (GD)",
        "spatial_dispersion_salience_score": "Spatial Dispersion Salience Score (SDSS)"
    }

    try:
        cmap = plt.cm.viridis
        data = corr.copy()
        
        # Apply renaming if applicable
        data.rename(index=metric_display_names, columns=metric_display_names, inplace=True)
        
        # Update category_map to match new names
        effective_category_map = {}
        if category_map:
            for old_name, cat in category_map.items():
                new_name = metric_display_names.get(old_name, old_name)
                effective_category_map[new_name] = cat
        else:
            effective_category_map = None

        row_colors = None
        col_colors = None
        if effective_category_map:
            order = category_order or ["Informatics", "Tokenization", "Semantics", "Syntax", "Rhythm", "Visualization"]
            
            # Ensure strict ordering by group first, then by original order
            # This is crucial to prevent clustering from shuffling metrics across groups
            sorted_cols = []
            for grp in order:
                grp_cols = [c for c in data.columns if effective_category_map.get(c) == grp]
                sorted_cols.extend(grp_cols)
            
            # Add any remaining columns not in the category map or order
            remaining = [c for c in data.columns if c not in sorted_cols]
            sorted_cols.extend(remaining)
            
            data = data.loc[sorted_cols, sorted_cols]
            
            palette = category_palette or {
                "Informatics": "#1f77b4",
                "Tokenization": "#ff7f0e",
                "Semantics": "#2ca02c",
                "Syntax": "#d62728",
                "Rhythm": "#9467bd",
                "Visualization": "#8c564b"
            }
            if label_style == "bars":
                row_colors = [palette.get(effective_category_map.get(r, ""), "#bdbdbd") for r in data.index]
                col_colors = [palette.get(effective_category_map.get(c, ""), "#bdbdbd") for c in data.columns]
            else:
                row_colors = None
                col_colors = None
        
        # Common font size for consistency between values and labels
        font_size = 8
        
        cbar_pos = None
        g = sns.clustermap(
            data, 
            cmap=cmap, 
            center=0, 
            vmin=-1, 
            vmax=1, 
            figsize=(8, 8), 
            dendrogram_ratio=(.02, .02), 
            cbar_pos=cbar_pos, 
            tree_kws={'linewidth': 0.8},
            linewidths=0.5,
            linecolor='white',
            row_cluster=False,  # Strictly disable clustering to preserve group order
            col_cluster=False,  # Strictly disable clustering to preserve group order
            row_colors=row_colors,
            col_colors=col_colors,
            annot=True,  # Display values in cells
            fmt=".2f",  # Format to two decimal places
            annot_kws={"size": font_size}  # Adjust font size to fit
        )
        # Adjust axis labels font size and padding
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=font_size, rotation=30, ha="right")
        g.ax_heatmap.tick_params(axis='x', pad=0.01)  # Reduce x-axis tick label padding
        
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=font_size)
        g.ax_heatmap.tick_params(axis='y', pad=0.01)  # Reduce y-axis tick label padding
        
        # Remove axis labels (x and y titles) as they are redundant with tick labels
        g.ax_heatmap.set_xlabel("")
        g.ax_heatmap.set_ylabel("")
        # Hide dendrogram axes for a cleaner publication layout when clustering is disabled
        try:
            if not effective_category_map:
                pass
            else:
                g.ax_row_dendrogram.set_visible(False)
                g.ax_col_dendrogram.set_visible(False)
        except Exception:
            pass
        pos = g.ax_heatmap.get_position()
        try:
            norm = plt.Normalize(-1.0, 1.0)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
            sm.set_array([])
            if cbar_loc == "left":
                cax = g.fig.add_axes([pos.x0 - 0.06, pos.y0, 0.0165, pos.height])
                cb = g.fig.colorbar(sm, cax=cax, orientation="vertical")
                ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
                cax.yaxis.set_major_locator(FixedLocator(ticks))
                cax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                cax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False)
                cax.yaxis.set_label_position("left")
                cax.set_ylabel("Correlation", fontsize=font_size)
                try:
                    cb.outline.set_visible(False)
                except Exception:
                    pass
            elif cbar_loc == "right":
                cax = g.fig.add_axes([pos.x1 + 0.015, pos.y0, 0.022, pos.height])
                cb = g.fig.colorbar(sm, cax=cax, orientation="vertical")
                try:
                    cb.outline.set_visible(False)
                except Exception:
                    pass
            else:
                cax = g.fig.add_axes([pos.x0, pos.y0 - 0.045, pos.width, 0.02])
                cb = g.fig.colorbar(sm, cax=cax, orientation="horizontal")
                try:
                    cb.outline.set_visible(False)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            g.ax_heatmap.tick_params(right=False, top=False)
        except Exception:
            pass
        
        # Add group headers and base tick colors aligned to categories
        if effective_category_map:
            order = category_order or ["Informatics", "Tokenization", "Semantics", "Syntax", "Rhythm", "Visualization"]
            idx_groups = []
            for grp in order:
                idx_groups.append([i for i, name in enumerate(data.index) if effective_category_map.get(name) == grp])
            pal = category_palette or {
                "Informatics": "#1f77b4",
                "Tokenization": "#ff7f0e",
                "Semantics": "#2ca02c",
                "Syntax": "#d62728",
                "Rhythm": "#9467bd",
                "Visualization": "#8c564b"
            }
            legend_y = min(0.98, pos.y1 + 0.005)
            try:
                xt = g.ax_heatmap.get_xticklabels()
                for i, name in enumerate(data.columns):
                    c = pal.get(effective_category_map.get(name, ""), "#4d4d4d")
                    xt[i].set_color(c)
                    try:
                        xt[i].set_fontweight("normal")
                    except Exception:
                        pass
                yt = g.ax_heatmap.get_yticklabels()
                for i, name in enumerate(data.index):
                    c = pal.get(effective_category_map.get(name, ""), "#4d4d4d")
                    yt[i].set_color(c)
                    try:
                        yt[i].set_fontweight("normal")
                    except Exception:
                        pass
            except Exception:
                pass
            if label_style == "text":
                try:
                    ax = g.ax_heatmap
                    # Top headers
                    header_offset_y = 0.3  # Closer to heatmap top
                    bracket_height = 0.3   # Height of the bracket legs
                    text_offset_y = 0.4    # Reduced offset to move text closer to brackets
                    for k, grp in enumerate([grp for grp in order if len([i for i, name in enumerate(data.columns) if effective_category_map.get(name) == grp]) > 0]):
                        idx = [i for i, name in enumerate(data.columns) if effective_category_map.get(name) == grp]
                        if not idx: continue
                        
                        # Verify strict continuity of the group indices
                        if idx != list(range(min(idx), max(idx) + 1)):
                            # Skip drawing messy lines if metrics are not contiguous
                            continue
                            
                        start, end = min(idx), max(idx)
                        center = (start + end) / 2
                        
                        # Draw bracket using data coordinates (y < 0 is above)
                        line_y = -header_offset_y
                        # Horizontal line
                        ax.plot([start, end+1], [line_y, line_y], color="#9e9e9e", linewidth=1.0, clip_on=False, zorder=10)
                        # Vertical legs at both ends
                        ax.plot([start, start], [line_y, line_y + bracket_height], color="#9e9e9e", linewidth=1.0, clip_on=False, zorder=10)
                        ax.plot([end+1, end+1], [line_y, line_y + bracket_height], color="#9e9e9e", linewidth=1.0, clip_on=False, zorder=10)
                        
                        ax.text(center+0.5, -text_offset_y, grp, color=pal.get(grp, "#4d4d4d"), ha="center", va="bottom", fontsize=9)

                    # Left grouping lines and labels
                    left_offset_x = 0.3  # Reduced offset to keep bracket closer to heatmap
                    bracket_width = 0.3
                    text_offset_x = 0.1  # Offset relative to bracket line
                    for grp in order:
                        idx = [i for i, name in enumerate(data.index) if effective_category_map.get(name) == grp]
                        if not idx: continue
                        
                        # Verify strict continuity of the group indices
                        if idx != list(range(min(idx), max(idx) + 1)):
                            continue
                            
                        start, end = min(idx), max(idx)
                        center = (start + end) / 2
                        
                        # Draw bracket using data coordinates (x < 0 is left)
                        # Heatmap row i spans y=i to y=i+1 (top-down)
                        line_x = -left_offset_x
                        ax.plot([line_x, line_x], [start, end+1], color="#9e9e9e", linewidth=1.0, clip_on=False, zorder=10)
                        ax.plot([line_x, line_x+bracket_width], [start, start], color="#9e9e9e", linewidth=1.0, clip_on=False, zorder=10)
                        ax.plot([line_x, line_x+bracket_width], [end+1, end+1], color="#9e9e9e", linewidth=1.0, clip_on=False, zorder=10)
                        
                        # Add rotated text label centered on the bracket
                        ax.text(line_x - text_offset_x, center + 0.5, grp, 
                                color=pal.get(grp, "#4d4d4d"), 
                                ha="right", va="center", 
                                rotation=90, fontsize=9)
                except Exception:
                    pass
        title_y = min(0.995, legend_y + 0.055)
        if title is not None:
            g.fig.suptitle(title, y=title_y, fontsize=13, fontweight='bold')
        _save_both(filename)
        print(f"Clustermap saved to {filename}")
    except Exception:
        pass

# plot_compound_overview function removed per user request

def enforce_publication_style(fig=None):
    configure_nature_style()
    if fig is None:
        fig = plt.gcf()
    for ax in fig.get_axes():
        apply_nature_axes(ax, None)
    return fig
def figure_quality_report(dir_path, outfile=None):
    rows = []
    if not os.path.exists(dir_path):
        return pd.DataFrame(rows)
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        if os.path.isfile(p) and name.lower().endswith(".png"):
            try:
                im = Image.open(p)
                w, h = im.size
                size = os.path.getsize(p)
                svg = os.path.exists(os.path.splitext(p)[0] + ".svg")
                rows.append({"file": name, "width_px": w, "height_px": h, "filesize_bytes": size, "has_svg": svg})
            except Exception:
                rows.append({"file": name, "width_px": None, "height_px": None, "filesize_bytes": None, "has_svg": os.path.exists(os.path.splitext(p)[0] + ".svg")})
    df = pd.DataFrame(rows)
    if outfile:
        try:
            df.to_csv(outfile, index=False)
        except Exception:
            pass
    return df

def plot_metric_ridgeline(df: pd.DataFrame, filename: str, category_map: dict | None = None, category_order: list[str] | None = None):
    """
    Generate a 'Series Ridgeline' plot (stacked time-series/stream style) for metrics.
    
    Scientific Visualization (Optimized):
    - X-axis: Sample Index (Sequence).
    - Y-axis: Normalized Metric Delta (stacked).
    - Style: Scatter points (colored by category), semi-transparent.
    - Layout: Compact, spines and ticks visible.
    
    Args:
        df: DataFrame containing metric values.
        filename: Output filename (PNG/SVG).
        category_map: Dictionary mapping metric names to dimensions.
        category_order: List of dimensions in desired order.
    """
    configure_nature_style()
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    
    # Metric Display Names
    metric_display_names = { 
         "normalized_levenshtein": "Normalized Levenshtein (NL)", 
         "char_ngram_jaccard": "Char n-gram Jaccard (CNJ)", 
         "compression_delta": "Compression Delta (CD)", 
         "token_count_change": "Token Count Change (TCC)", 
         "fragmentation_index": "Fragmentation Index (FI)", 
         "normalized_entropy_delta": "Normalized Entropy Delta (NED)", 
         "contextual_embedding_distance": "Contextual Embedding Distance (CED)", 
         "semantic_entailment_score": "Semantic Entailment Score (SES)", 
         "lm_surprisal_delta": "Language Model Surprisal Delta (LMSD)", 
         "dependency_overlap_score": "Dependency Overlap Score (DOS)", 
         "tree_depth_change": "Tree Depth Change (TDC)", 
         "pos_divergence": "Part-of-Speech Divergence (POSD)", 
         "syllable_count_change": "Syllable Count Change (SCC)", 
         "stress_pattern_divergence": "Stress Pattern Divergence (SPD)", 
         "ssim_distance": "Structural Similarity Distance (SSIMD)", 
         "glyph_displacement": "Glyph Displacement (GD)", 
         "spatial_dispersion_salience_score": "Spatial Dispersion Salience Score (SDSS)" 
    }

    # Filter and Sort Metrics
    if category_map is None:
        return
    metrics = [m for m in category_map.keys() if m in df.columns]
    if not metrics:
        return
        
    if category_order:
        sorted_metrics = []
        for dim in category_order:
            dim_metrics = [m for m in metrics if category_map[m] == dim]
            sorted_metrics.extend(dim_metrics)
        metrics = sorted_metrics
    
    # Prepare Data: Normalize each metric to [0, 1]
    plot_df = df[metrics].copy()
    plot_df = plot_df.fillna(0)
    for col in plot_df.columns:
        min_val = plot_df[col].min()
        max_val = plot_df[col].max()
        if max_val - min_val > 1e-9:
            plot_df[col] = (plot_df[col] - min_val) / (max_val - min_val)
        else:
            plot_df[col] = 0.0

    # Palette for Labels
    palette = {
        "Informatics": "#1f77b4",
        "Tokenization": "#ff7f0e",
        "Semantics": "#2ca02c",
        "Syntax": "#d62728",
        "Rhythm": "#9467bd",
        "Visualization": "#8c564b"
    }

    # Plot Setup
    n_metrics = len(metrics)
    fig, ax = plt.subplots(figsize=(7.0, 8.5)) 
    
    # Parameters for stacking
    # Reduced step to 0.4 for a more compact, dense visualization (half height per panel)
    y_offset_step = 0.4
    
    x = np.arange(len(plot_df))
    
    yticks = []
    yticklabels = []
    ytick_colors = []
    
    # Iterate
    for i, m in enumerate(metrics):
        dim = category_map.get(m, "Unknown")
        label_color = palette.get(dim, "#333333")
        
        # Calculate offset (Top to Bottom visual order)
        base_y = (n_metrics - 1 - i) * y_offset_step
        
        y_values = plot_df[m].values
        
        # Add Jitter to break straight lines (visual artifact of discrete/constant data)
        # Using a small uniform noise to spread points vertically without altering interpretation
        rng = np.random.default_rng(42 + i)
        y_jitter = rng.uniform(-0.02, 0.02, size=len(y_values))
        
        # Scale values to fit the compact panel height (0.35 max amplitude)
        # Apply scaling to both values and jitter
        y_scaled = (y_values + y_jitter) * 0.35
        
        # Plot Scatter Points (Colored by Category, Semi-transparent)
        # s=2 for visibility, alpha=0.5 for density
        ax.scatter(x, base_y + y_scaled, s=2, color=label_color, alpha=0.5, zorder=i, edgecolors='none')
        
        # Store Tick Info - Align tick with the VISUAL CENTER of the panel
        # Panel height is ~0.35, so center is base_y + 0.175
        yticks.append(base_y + 0.175)
        yticklabels.append(metric_display_names.get(m, m.replace("_", " ").title()))
        ytick_colors.append(label_color)

    # Styling Axes
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    
    # Apply colors to Y labels
    for t, color in zip(ax.get_yticklabels(), ytick_colors):
        t.set_color(color)
        
    ax.set_xlabel("Pairwise Samples", fontsize=10)
    
    # Spines and Ticks (Standard Style)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    # Ticks on both axes
    ax.tick_params(axis='both', which='major', direction='out', length=3, width=0.6, bottom=True, left=True)
    
    # Tight X-axis alignment
    ax.set_xlim(x.min(), x.max())
    
    # Adjust Y-limits (Tightened to remove top whitespace)
    # Max data height is approx base_y + 0.4
    top_base_y = (n_metrics - 1) * y_offset_step
    ax.set_ylim(-0.05, top_base_y + 0.4)
    
    plt.tight_layout()
    _save_both(filename)
    print(f"Series Ridgeline plot saved to {filename}")


def plot_tokenizer_comparison(csv_path: str, output_filename: str):
    configure_nature_style()
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    try:
        details_path = csv_path.replace(".csv", "_details.csv")
        use_details = os.path.exists(details_path)
        df = pd.read_csv(details_path if use_details else csv_path)
        need = {"type", "original_score", "perturbed_score", "delta"}
        if not need.issubset(set(df.columns)):
            print("CSV missing required columns for pre/post comparison.")
            return
        base_types = ["WordPiece", "BBPE", "Unigram"]
        df = df[df["type"].isin(base_types)].copy()
        means = df.groupby("type", dropna=False)["delta"].mean().to_dict()
        order = sorted(base_types, key=lambda t: (-float(means.get(t, float("-inf"))), base_types.index(t)))
        df["type"] = pd.Categorical(df["type"], categories=order, ordered=True)
        df = df.sort_values("type")
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        apply_nature_axes(ax)
        x_all = df["delta"].values
        if x_all.size > 0:
            xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
            rng = max(1e-9, xmax - xmin)
            pad = 0.05 * rng
            ax.set_xlim(xmin - pad, xmax + pad)
        else:
            ax.set_xlim(-1.0, 1.0)
        y_base = {t: i for i, t in enumerate(order)}
        edge_color = "#333333"
        baseline_color = "#9e9e9e"
        rank_colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
        
        for t in order:
            sub = df[df["type"] == t]
            y0 = y_base[t]
            n = len(sub)
            if n > 0:
                import seaborn as sns
                fill_color = rank_colors[min(y0, len(rank_colors) - 1)]
                before = len(ax.collections)
                sns.violinplot(
                    x=sub["delta"],
                    y=[t] * n,
                    orient="h",
                    inner=None,
                    cut=0,
                    linewidth=0.6,
                    color=fill_color,
                    alpha=0.65,
                    ax=ax,
                    zorder=1,
                )
                for col in ax.collections[before:]:
                    try:
                        col.set_edgecolor("#6e6e6e")
                        col.set_linewidth(0.6)
                    except Exception:
                        pass
                sns.boxplot(
                    x=sub["delta"],
                    y=[t] * n,
                    orient="h",
                    width=0.2,
                    showfliers=False,
                    boxprops={"facecolor": "white", "edgecolor": edge_color, "alpha": 1.0, "linewidth": 0.8},
                    whiskerprops={"color": edge_color, "linewidth": 0.8},
                    capprops={"color": edge_color, "linewidth": 0.8},
                    medianprops={"color": edge_color, "linewidth": 1.2},
                    ax=ax,
                    zorder=2,
                )

        ax.set_yticks(list(range(len(order))))
        ax.set_yticklabels(order)
        ax.set_xlabel(r"Difference in Rényi efficiency ($\Delta H_{\alpha}$)", labelpad=2)
        ax.axvline(0.0, color=baseline_color, linewidth=0.6, linestyle="--", zorder=0)
        
        ax.xaxis.grid(False) # Nature prefers no gridlines or extremely subtle ones
        try:
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5]))
        except Exception:
            pass
        ax.tick_params(axis="both", which="major", length=3, width=0.6, direction="out", pad=2)
        ax.tick_params(axis="x", bottom=True, top=False)
        ax.tick_params(axis="y", left=True, right=False)
        plt.tight_layout()
        _save_both(output_filename)
        print(f"Tokenizer comparison plot saved to {output_filename}")
    except Exception as e:
        print(f"Failed to plot tokenizer comparison: {e}")

def plot_perturbation_pipeline(original_text, perturbed_text, filename, colormap: str = "viridis", show_colorbar: bool = True, vmin: float | None = None, vmax: float | None = None, scale_mode: str = "symmetric", panel_letter: str | None = None, figsize: tuple[float, float] | None = None):
    """
    Visualization: Differential Wavelet Energy (Δ|W|^2), single-panel figure.
    Layout: single-column; right-side Y axis; right-side vertical colorbar.
    Wavelet: cmor1.5-1.0, scales 1–64, sampling_period = 1 token.
    """
    configure_nature_style()
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from project.utils.wavelet import TextWaveletAnalyzer
    import pywt
    
    wa = TextWaveletAnalyzer()
    
    # Compute Wavelets
    res = wa.compute_pair_wavelets(original_text, perturbed_text)
    
    # Differential Energy: E_perturbed - E_original
    E_diff = res["energy_scalogram_diff"]
    n_scales = E_diff.shape[0]
    n_time = E_diff.shape[1]
    
    # Scientific color scale
    # scale_mode: "auto" (percentile 1–99), "symmetric" (±p99(abs)), "fixed" (use vmin/vmax as given)
    if vmin is None or vmax is None:
        if scale_mode == "symmetric":
            p = np.percentile(np.abs(E_diff), 99)
            vmin = -p
            vmax = p
        elif scale_mode == "fixed":
            vmin = vmin if vmin is not None else -30.0
            vmax = vmax if vmax is not None else 10.0
        else:
            p1 = np.percentile(E_diff, 1)
            p99 = np.percentile(E_diff, 99)
            vmin = p1
            vmax = p99
    
    # Nature single-column panel size suitable for triptych (3 side-by-side)
    fig, ax = plt.subplots(figsize=(figsize if figsize is not None else (3.0, 2.0)))
    
    # Plot Heatmap
    # Extent: Time [0, 1], Scale [0, n_scales]
    cmap_used = colormap if isinstance(colormap, str) else "viridis"
    im = ax.imshow(E_diff, aspect='auto', cmap=cmap_used, origin='lower', 
                   extent=[0, 1, 0.5, n_scales + 0.5], vmin=vmin, vmax=vmax)
    tscore = np.sum(np.abs(E_diff), axis=0)
    t_idx = int(np.argmax(tscore)) if n_time > 0 else 0
    x_peak = (t_idx / (n_time - 1)) if n_time > 1 else 0.0
    ax.axvline(x_peak, color="#616161", linestyle="--", linewidth=0.9, zorder=4)
    try:
        xs = np.linspace(0, 1, n_time)
        ys = np.linspace(1, n_scales, n_scales)
        lvls = np.linspace(vmin, vmax, 6)
        ax.contour(xs, ys, E_diff, levels=lvls, colors="#424242", linewidths=0.4, alpha=0.15)
    except Exception:
        pass
    try:
        ax.text(x_peak, n_scales + 0.2, "Peak", ha="center", va="bottom", fontsize=7, color="#616161", clip_on=False)
    except Exception:
        pass
    
    # Scientific Title & Labels
    # Nature-style: omit heavy title; rely on caption/panel lettering externally
    # ax.set_title("Differential Energy Spectrum", fontsize=9, pad=4)
    try:
        if panel_letter:
            ax.text(0.02, 0.98, panel_letter, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="top")
    except Exception:
        pass
    
    ax.set_xlabel("Normalized Time", fontsize=8)
    ax.set_ylabel("Wavelet Scale", fontsize=8, labelpad=2)
    
    # Ticks formatting
    # X-axis: 0.0 to 1.0
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6)
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    
    # Y-axis: 1 to Max Scale (single scale; disable minor ticks)
    y_ticks = np.linspace(1, n_scales, 5, dtype=int)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(t)) for t in y_ticks], fontsize=6)
    ax.tick_params(axis='y', which='major', left=True, right=False, labelleft=True, pad=1)
    try:
        ax.minorticks_off()
    except Exception:
        pass
    
    # Axis ticks style (major only)
    ax.tick_params(axis='both', which='major', length=3.0, width=0.6, direction='out', labelsize=6)
    
    # Spines and Y label side
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.7)
    
    # Standard Scientific Layout: Y-axis Left, Colorbar Right
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['right'].set_visible(False)

    # Heatmap border (thin frame in data coords)
    rect = patches.Rectangle((0, 0.5), 1, n_scales, linewidth=0.7, edgecolor="#757575", facecolor='none')
    ax.add_patch(rect)

    # Right-side vertical Colorbar
    # Adjust layout to make room on the right
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.22, top=0.90)
    
    if show_colorbar:
        from matplotlib.ticker import MaxNLocator, FormatStrFormatter
        pos = ax.get_position()
        # Place cbar on the right side
        cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.012, pos.height])
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        
        # Restore Title
        cbar.set_label("$\Delta$ Energy", rotation=270, labelpad=6, fontsize=6)
        
        cbar.ax.tick_params(labelsize=6, length=2, width=0.4, direction="out", pad=2, left=False, right=True, labelleft=False, labelright=True)
        try:
            cbar.ax.minorticks_off()
        except Exception:
            pass
        try:
            absmax = float(max(abs(vmin), abs(vmax)))
            if absmax == 0:
                absmax = 1.0
            vmin, vmax = -absmax, absmax
            im.set_clim(vmin, vmax)
            def _nice_step(r, target=5):
                raw = r / max(1, (target - 1))
                exp = np.floor(np.log10(raw)) if raw > 0 else 0.0
                base = raw / (10 ** exp) if raw > 0 else 1.0
                if base < 1.5:
                    step = 1.0
                elif base < 3.5:
                    step = 2.0
                else:
                    step = 5.0
                return step * (10 ** exp)
            step = _nice_step(vmax - vmin, target=5)
            ticks = np.arange(-absmax, absmax + step * 0.5, step)
            labels = [("{:.2f}".format(t)).replace("-0.00", "0.00") for t in ticks]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(labels)
        except Exception:
            pass
        try:
            cbar.outline.set_visible(False)
        except Exception:
            pass
        try:
            cbar.locator = MaxNLocator(nbins=5, steps=[1, 2, 5])
            cbar.update_ticks()
        except Exception:
            pass
    
    # No bottom formula/caption per Nature layout; keep figure clean
    _save_both(filename)
    print(f"Differential spectrum saved to {filename}")

# plot_compound_overview function removed per user request
def plot_tokenizer_cwt_triptych(delta_energy_by_tokenizer: dict, filename: str, colormap: str = "viridis", clip_percentile: float = 90.0):
    """
    Triptych of CWT differential energy (ΔE) for different tokenizers.

    Content:
    - Panels a/b/c correspond to WordPiece, byte-level BPE, and Unigram tokenizers.
    - Each panel visualizes the scale-time spectrum of
          ΔE = |W_pert|^2 - |W_orig|^2
      with origin="lower" so that scale 1 appears at the bottom.
    - The left y-axis encodes wavelet scale s, and the x-axis shows normalized time τ.
    - The panel titles start with bold a/b/c letters to match common journal style.

    Color and ticks:
    - A symmetric colour scale is used, with a single vertical colorbar on the right.
    - Axis ticks are drawn only on the left and bottom; right and top ticks are disabled.

    Args:
        delta_energy_by_tokenizer: dict mapping tokenizer name to a ΔE array [S, T].
        filename: output path; both PNG and SVG are written.
        colormap: name of the Matplotlib colormap to use.
        clip_percentile: placeholder for potential future clipping logic (currently unused).
    """
    configure_nature_style()
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    titles = [("WordPiece", "a"), ("byte-level BPE", "b"), ("Unigram", "c")]
    mats = []
    for key, _ in titles:
        m = delta_energy_by_tokenizer.get(key)
        if m is None:
            mats.append(np.zeros((64, 2), dtype=float))
        else:
            mats.append(np.asarray(m, dtype=float))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True)
    im_handles = []

    # SymLog scaling [-1000, 1000] per user request for huge differences
    vmin, vmax = -1000.0, 1000.0
    linthresh = 1.0
    norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax, base=10)

    for ax, (title, letter), m in zip(axes, titles, mats):
        apply_nature_axes(ax)
        im = ax.imshow(
            m,
            aspect="auto",
            cmap=colormap,
            origin="lower",
            extent=[0, 1, 0.5, m.shape[0] + 0.5],
            norm=norm,
            interpolation="nearest",
        )
        im_handles.append(im)
        
        # Removed peak line (ax.axvline) per user request to clean up visualization
        
        try:
            xs = np.linspace(0, 1, m.shape[1])
            ys = np.linspace(1, m.shape[0], m.shape[0])
            # Contour levels for SymLog? Might be tricky. Use fixed log-like levels or skip.
            # Using simple linspace might not be informative for log data, but keeping it simple for now or removing.
            # Let's use a few levels that match the ticks roughly.
            lvls = [-1000, -100, -10, -1, 1, 10, 100, 1000]
            ax.contour(xs, ys, m, levels=lvls, colors="#424242", linewidths=0.4, alpha=0.12)
        except Exception:
            pass
        # Nature requirement: Bold a/b/c labels
        ax.set_title(rf"$\bf{{{letter}}}$ {title}", fontsize=8, pad=4, loc="left")
        ax.set_xlabel(r"Normalized time ($\tau$)", fontsize=8, labelpad=2)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(["0.0", "0.5", "1.0"], fontsize=7)
        ax.tick_params(
            axis="both",
            which="both",
            length=4,
            width=0.8,
            direction="out",
            pad=2,
            bottom=True,
            left=True,
            top=False,
            right=False,
        )
    try:
        divider = make_axes_locatable(axes[-1])
        cax = divider.append_axes("right", size="5%", pad=0.10)
        base_im = im_handles[-1] if im_handles else None
        if base_im is not None:
            cbar = fig.colorbar(base_im, cax=cax, orientation="vertical")
            # Colorbar ticks for SymLog [-1000, 1000]
            ticks = [-1000, -100, -10, -1, 0, 1, 10, 100, 1000]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([str(t) for t in ticks])
            cbar.set_label(r"$\Delta E = |W_{\mathrm{pert}}|^2 - |W_{\mathrm{orig}}|^2$ (a.u.)", rotation=270, labelpad=11, fontsize=7)
            cbar.ax.tick_params(labelsize=6, length=2, width=0.4, direction="out", pad=1)
            try:
                cbar.outline.set_visible(False)
            except Exception:
                pass
    except Exception:
        pass
    axes[0].set_ylabel(r"Wavelet scale ($s$)", fontsize=8, labelpad=2)
    axes[0].set_yticks([1, 16, 32, 48, 64])
    axes[0].set_yticklabels(["1", "16", "32", "48", "64"], fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.18, top=0.90, wspace=0.08)
    _save_both(filename)
    print(f"Tokenizer CWT triptych saved to {filename}")

def _hex_from_style(style):
    if not style:
        return []
    out = []
    parts = style.split(';')
    for kv in parts:
        if ':' in kv:
            k, v = kv.split(':', 1)
            v = v.strip()
            if v.startswith('#') and len(v) in (4, 7):
                out.append(v.lower())
    return out
def _is_gray(hexcolor):
    if not hexcolor or not hexcolor.startswith('#') or len(hexcolor) != 7:
        return False
    try:
        r = int(hexcolor[1:3], 16)
        g = int(hexcolor[3:5], 16)
        b = int(hexcolor[5:7], 16)
        return abs(r - g) <= 8 and abs(g - b) <= 8 and abs(r - b) <= 8
    except Exception:
        return False
def _parse_svg_numeric(v):
    if v is None:
        return None
    s = ''.join(ch for ch in v if (ch.isdigit() or ch == '.' ))
    try:
        return float(s) if s else None
    except Exception:
        return None
def _extract_svg_info(path):
    info = {"width": None, "height": None, "colors": set(), "font_sizes": [], "panel_letters": []}
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        w = _parse_svg_numeric(root.attrib.get('width'))
        h = _parse_svg_numeric(root.attrib.get('height'))
        if (w is None or h is None) and 'viewBox' in root.attrib:
            vb = root.attrib.get('viewBox').split()
            if len(vb) == 4:
                w = float(vb[2])
                h = float(vb[3])
        info["width"] = w
        info["height"] = h
        for el in root.iter():
            style = el.attrib.get('style')
            for c in _hex_from_style(style):
                info["colors"].add(c)
            fill = el.attrib.get('fill')
            stroke = el.attrib.get('stroke')
            if fill and fill.startswith('#'):
                info["colors"].add(fill.lower())
            if stroke and stroke.startswith('#'):
                info["colors"].add(stroke.lower())
            fs = el.attrib.get('font-size')
            if fs:
                v = _parse_svg_numeric(fs)
                if v:
                    info["font_sizes"].append(v)
            if el.tag.endswith('text'):
                txt = ''.join(el.itertext()).strip()
                if txt in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                    x = _parse_svg_numeric(el.attrib.get('x'))
                    y = _parse_svg_numeric(el.attrib.get('y'))
                    info["panel_letters"].append({"ch": txt, "x": x, "y": y})
    except Exception:
        pass
    return info
def qc_palette(svg_path, allowed=None):
    if allowed is None:
        allowed = {COLOR_PRIMARY.lower(), COLOR_SECONDARY.lower(), COLOR_NEUTRAL.lower(), COLOR_LIGHT.lower()}
    info = _extract_svg_info(svg_path)
    used = set([c.lower() for c in info["colors"] if isinstance(c, str)])
    non_compliant = [c for c in used if c not in allowed and not _is_gray(c)]
    return {"ok": len(non_compliant) == 0, "non_compliant": non_compliant, "used": sorted(list(used))}
def qc_min_font(svg_path, min_size=7.0):
    info = _extract_svg_info(svg_path)
    sizes = info["font_sizes"]
    if not sizes:
        return {"ok": False, "min_found": None}
    mn = min(sizes)
    return {"ok": mn >= min_size, "min_found": mn}
def qc_panel_alignment(svg_path, letters=None, y_top_fraction=0.2, x_left_fraction=0.2, y_tolerance=0.1):
    if letters is None:
        letters = list("ABCDEF")
    info = _extract_svg_info(svg_path)
    w = info["width"] or 1.0
    h = info["height"] or 1.0
    pos = []
    for d in info["panel_letters"]:
        if d["ch"] in letters and d["x"] is not None and d["y"] is not None:
            xn = d["x"] / w
            yn = d["y"] / h
            pos.append((d["ch"], xn, yn))
    if not pos:
        return {"ok": False, "details": "no_letters"}
    ys = [p[2] for p in pos]
    xs = [p[1] for p in pos]
    top_ok = all(y <= y_top_fraction for y in ys)
    left_ok = all(x <= x_left_fraction for x in xs)
    y_std = np.std(ys) if len(ys) >= 2 else 0.0
    align_ok = y_std <= y_tolerance
    return {"ok": top_ok and left_ok and align_ok, "top_ok": top_ok, "left_ok": left_ok, "y_std": y_std, "count": len(pos)}
def generate_qc_report(dir_path, outfile_md=None, outfile_csv=None, min_font=7.0):
    rows = []
    if not os.path.exists(dir_path):
        return pd.DataFrame(rows)
    for name in os.listdir(dir_path):
        if name.lower().endswith(".svg"):
            p = os.path.join(dir_path, name)
            pal = qc_palette(p)
            fnt = qc_min_font(p, min_size=min_font)
            pnl = qc_panel_alignment(p)
            rows.append({
                "file": name,
                "palette_ok": pal["ok"],
                "min_font_ok": fnt["ok"],
                "panel_align_ok": pnl["ok"],
                "min_font_found": fnt.get("min_found"),
                "non_compliant_colors": ";".join(pal.get("non_compliant", [])),
                "panel_count": pnl.get("count", 0),
                "panel_y_std": pnl.get("y_std", None)
            })
    df = pd.DataFrame(rows)
    if outfile_csv:
        try:
            df.to_csv(outfile_csv, index=False)
        except Exception:
            pass
    if outfile_md:
        try:
            lines = []
            lines.append("# Figure QC Report")
            lines.append("")
            for _, r in df.iterrows():
                lines.append(f"- {r['file']}: palette_ok={r['palette_ok']}, min_font_ok={r['min_font_ok']}, panel_align_ok={r['panel_align_ok']}, min_font_found={r['min_font_found']}, non_compliant_colors={r['non_compliant_colors']}, panel_count={r['panel_count']}, panel_y_std={r['panel_y_std']}")
            with open(outfile_md, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception:
            pass
    return df

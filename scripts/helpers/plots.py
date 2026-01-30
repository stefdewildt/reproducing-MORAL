from typing import Dict, Optional, Tuple
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re

                                        # set font equal to TMLR font
mpl.rcParams.update({
    "text.usetex": True,                # LaTeX compiler; comment out
                                        # for faster rendering
    "font.family" : "serif",
    "font.serif" : ["Latin Modern Roman"],
    "mathtext.fontset" : "cm",
    "axes.labelsize" : 11,
    "font.size" : 11,
    "xtick.labelsize" : 10,
    "ytick.labelsize" : 10,
})


def _parse_params_from_row(
        row: Dict[str, str]
    ) -> Optional[Tuple[float, float, float]]:
    """
    Parse fm, hMM, hmm from a .csv row
    """
    text = row.get("dataset") or ""
    text = re.sub(r"_(\d+)$", "", text)
    
    m = re.search(r"fm([\d\.]+)_hMM([\d\.]+)_hmm([\d\.]+)", text)
    if not m:
        return None
    try:
        fm = float(m.group(1))
        hMM = float(m.group(2))
        hmm = float(m.group(3))
        return fm, hMM, hmm
    except ValueError:
        return None


def cmyk_to_rgb(c, m, y, k):
    """
    Convert CMYK color to RGB color
    """
    r = (1 - c) * (1 - k)
    g = (1 - m) * (1 - k)
    b = (1 - y) * (1 - k)
    return (r, g, b)


def add_h_edge_to_results(
        csv_out_dpah,
        results,
        key_from_row_fn = None
    ):
    """
    Calculate the homophily based on edge types from the .csv
    and add it to the results dictionary
    """
    csv_path = Path(csv_out_dpah)

    with csv_path.open("r", newline = "", encoding = "utf-8") as f:
        rows = list(csv.DictReader(f))
    if key_from_row_fn is None:
        def key_from_row_fn(row):
            fm, h_MM, h_mm = _parse_params_from_row(row)
            return f"dpah_fm{fm:.2f}_hMM{h_MM:.2f}_hmm{h_mm:.2f}"
    for row in rows:
        key = key_from_row_fn(row)
        if key not in results:
            continue

        p00 = float(row["global_p00"])
        p01 = float(row["global_p01"])
        p11 = float(row["global_p11"])
        denom = p00 + p01 + p11
        h_edge = (p00 + p11) / denom if denom != 0.0 else float("nan")
        results[key]["h_edge"] = h_edge

        # Use Newman's method 
        fm, _, _ = _parse_params_from_row(row)
        p1 = float(fm)
        p0 = 1.0 - p1
        h_edge_rand = p0**2 + p1**2
        results[key]["h_edge_rand"] = h_edge_rand
        results[key]["delta_h_edge"] = float(h_edge - h_edge_rand)

    return results


def calculate_graph_homophily(
        global_p00,                     # fraction of class 00 edges
        global_p01,                     # fraction of class 01 edges
        global_p11,                     # fraction of class 11 edges
        *,                          
        eps = 0.0                       # epsilon for zero division
    ):
    """
    h_edge = (p00 + p11) / (p00 + p01 + p11)
    """
    p00 = float(global_p00)
    p01 = float(global_p01)
    p11 = float(global_p11)
    return (p00 + p11) / (p00 + p01 + p11 + eps)


def plot_results_heatmap(
        results,
        fm_values,
        h_MM_values,
        h_mm_values,
        metric = 'reranked_NDKL',
        label = 'reranked NDKL',
        colour_map = 'inferno'
    ):
    """
    Plots a heatmap of the results for different h_MM and h_mm values,
    for each fixed fm value
    """
    h_MM_values = np.array(h_MM_values)
    h_mm_values = np.array(h_mm_values)

    fig, axs = plt.subplots(
        1, 6,                           # one row, six columns
        figsize = (28, 4),
        sharey = True,                  # share both axes
        sharex = True)
    
    for fm in fm_values:
        Z = np.full(                    # init the heatmap
            (len(h_MM_values), len(h_mm_values)),
            np.nan
        )
                                        # fill the heatmap
        for i, h_MM in enumerate(h_MM_values):
            for j, h_mm in enumerate(h_mm_values):
                key = f'dpah_fm{fm:.2f}_hMM{h_MM:.2f}_hmm{h_mm:.2f}'
                m = results[key][metric]
                value = m[0] if isinstance(m, (list, tuple, np.ndarray)) else float(m)

                Z[i, j] = value
    
        Z = Z.T                         # transpose to get h_MM on x-axis

        axs[fm_values.index(fm)].imshow(
            Z,
            origin = "lower",
            extent = [
                h_MM_values.min(), h_MM_values.max(),  
                h_mm_values.min(), h_mm_values.max(),
            ],
            aspect = "auto",
            cmap = colour_map,           # https://matplotlib.org/stable/users/explain/colors/colormaps.html
            
)

        
    fig.colorbar(
        axs[fm_values.index(fm)].images[0],
        ax = axs[fm_values.index(fm)],
        label = label
    )
    axs[0].set_ylabel("h_mm")
    axs[len(axs) // 2].set_xlabel("h_MM")

    plt.tight_layout()
    plt.savefig(f"../figures/dpah_heatmap_{metric}.pdf")
    plt.show()


def plot_results_heatmap_paper(
        results,
        fm_values,
        h_MM_values,
        h_mm_values,
        metric,
        label,
        fs = 12,
        colour_map = "inferno"
    ):
    h_MM_values = np.array(h_MM_values)
    h_mm_values = np.array(h_mm_values)
    n_fm = len(fm_values)

    fig, axs = plt.subplots(
        1,
        n_fm,
        figsize = (4.6 * n_fm, 4),
        sharey = True,
        sharex = True
    )

    if n_fm == 1:                       # convert type if juse one f_m
        axs = [axs]
    
    for ax, fm in zip(axs, fm_values):
                                        # init the heatmap with nans
        Z = np.full((len(h_MM_values), len(h_mm_values)), np.nan)

        for idx, h_MM in enumerate(h_MM_values):
            for jdx, h_mm in enumerate(h_mm_values):
                key = f'dpah_fm{fm:.2f}_hMM{h_MM:.2f}_hmm{h_mm:.2f}'
                                        # homophily is stored without
                                        # std (i.e. as int, not tuple)
                m = results[key][metric]
                value = m[0] if isinstance(m, (list, tuple, np.ndarray)) else float(m)
                Z[idx, jdx] = value
    
        vmin_hat = None                 # set dynamic ranges
        vmax_hat = None
        if metric == 'h_edge':
            vmin_hat = 0.0
            vmax_hat = 1.0
        elif metric == 'delta_h_edge':
            vmin_hat = -1.0
            vmax_hat = 1.0
        elif metric == 'h_edge_rand':
            vmin_hat = 0.0
            vmax_hat = 1.0

        ax.imshow(
            Z.T, origin = "lower", extent = [
                h_mm_values.min(), h_mm_values.max(),
                h_MM_values.min(), h_MM_values.max(),
            ],
            aspect = "auto",
            # cmap = "inferno",
            # cmap = "seismic",
            cmap = colour_map,
            vmin = vmin_hat,
            vmax = vmax_hat
        )
        ax.set_title(rf"$f_\mathrm{{m}} = {fm:.2f}$", fontsize = fs + 6)
        
    cbar = fig.colorbar(
        axs[fm_values.index(fm)].images[0],
        ax = axs[fm_values.index(fm)]
    )
    cbar.set_label(
        label,
        fontsize = fs + 10
    )
    cbar.ax.tick_params(labelsize = fs)
    axs[0].set_ylabel(
        r"$h_\textrm{mm}$",
        fontsize = fs + 4
    )
    axs[len(axs) // 2].set_xlabel(
        r"$h_\mathrm{MM}$",
        fontsize = fs + 8
    )
    for ax in axs:
        ax.tick_params(axis = "both", labelsize = fs)
    plt.tight_layout()
    plt.savefig(
        f"../figures/dpah_heatmap_{metric}.pdf",
        format = "pdf",
        bbox_inches = "tight",
        pad_inches = 0.015
    )
    plt.show()


def plot_three_heatmaps_compact(
    results,
    fm_values,
    h_MM_values,
    h_mm_values,
    metric_top = "h_edge",
    label_top = rf"$h_\mathrm{{edge}}$",
    metric_mid = "reranked_NDKL",
    label_mid = rf"$\mathrm{{NDKL}}$",
    metric_bottom = "precision_at_k",
    label_bottom = rf"$\mathrm{{Precision}}$",
    fs = 16,
    colour_map = "inferno",
    savepath = None,
):
                                        # https://matplotlib.org/stable/users/explain/colors/colormaps.html
    h_MM_values = np.asarray(h_MM_values)
    h_mm_values = np.asarray(h_mm_values)
    fm_values = list(fm_values)
    n = len(fm_values)

    fig = plt.figure(figsize = (4.6 * n + 0.8, 10.0))
    gs = fig.add_gridspec(
        3, n + 1,
        width_ratios = [1] * n + [0.035],
        wspace = 0.12,
        hspace = 0.12
    )

    axs_top = [fig.add_subplot(gs[0, j]) for j in range(n)]
    axs_mid = [fig.add_subplot(gs[1, j], sharex = axs_top[0], sharey = axs_top[0]) for j in range(n)]
    axs_bot = [fig.add_subplot(gs[2, j], sharex = axs_top[0], sharey = axs_top[0]) for j in range(n)]

    cax_top = fig.add_subplot(gs[0, -1])
    cax_mid = fig.add_subplot(gs[1, -1])
    cax_bot = fig.add_subplot(gs[2, -1])

                                        # shift all colorbars slightly left 
    for cax in (cax_top, cax_mid, cax_bot):
        pos = cax.get_position()
        cax.set_position([pos.x0 - 0.008, pos.y0, pos.width, pos.height])

    def get_val(key, metric):
        m = results[key][metric]
        return m[0] if isinstance(m, (list, tuple, np.ndarray)) else float(m)

    def make_Z(fm, metric):
        Z = np.full((len(h_MM_values), len(h_mm_values)), np.nan)
        for i, h_MM in enumerate(h_MM_values):
            for j, h_mm in enumerate(h_mm_values):
                key = f"dpah_fm{fm:.2f}_hMM{h_MM:.2f}_hmm{h_mm:.2f}"
                if key not in results or metric not in results[key]:
                    continue
                Z[i, j] = get_val(key, metric)
        return Z.T 

    extent = [h_mm_values.min(), h_mm_values.max(), h_MM_values.min(), h_MM_values.max()]

    im_top = None
    for ax, fm in zip(axs_top, fm_values):
        im_top = ax.imshow(
            make_Z(fm, metric_top),
            origin = "lower",
            extent = extent,
            aspect = "auto",
            cmap = colour_map
        )
        ax.set_title(rf"$f_\mathrm{{m}} = {fm:.2f}$", fontsize = fs + 10)
        ax.tick_params(labelsize = fs)

    im_mid = None
    for ax, fm in zip(axs_mid, fm_values):
        im_mid = ax.imshow(
            make_Z(fm, metric_mid),
            origin = "lower",
            extent = extent,
            aspect = "auto",
            cmap = colour_map
        )
        ax.tick_params(labelsize = fs)

    im_bot = None
    for ax, fm in zip(axs_bot, fm_values):
        im_bot = ax.imshow(
            make_Z(fm, metric_bottom),
            origin = "lower",
            extent = extent,
            aspect = "auto",
            cmap = colour_map
        )
        ax.tick_params(labelsize = fs)

    axs_mid[0].set_ylabel(r"minority $h_\mathrm{mm}$", fontsize = fs + 10)

    for j in range(1, n):
        axs_top[j].tick_params(labelleft = False)
        axs_mid[j].tick_params(labelleft = False)
        axs_bot[j].tick_params(labelleft = False)

    xticks = np.arange(0.0, 1.01, 0.2)
    for ax in axs_bot:
        ax.set_xticks(xticks)
        ax.set_xticklabels([rf"${x:.1f}$" for x in xticks])

    mid = n // 2
    axs_bot[mid].set_xlabel(r"majority $h_\mathrm{MM}$", fontsize = fs + 10)
    for j, ax in enumerate(axs_bot):
        if j != mid:
            ax.set_xlabel("")

    for ax in axs_top:
        ax.tick_params(labelbottom = False)
    for ax in axs_mid:
        ax.tick_params(labelbottom = False)

    cbar1 = fig.colorbar(im_top, cax = cax_top)
    cbar1.set_label(label_top, fontsize = fs + 8)
    cbar1.ax.tick_params(labelsize = fs)

    cbar2 = fig.colorbar(im_mid, cax = cax_mid)
    cbar2.set_label(label_mid, fontsize = fs + 8)
    cbar2.ax.tick_params(labelsize = fs)

    cbar3 = fig.colorbar(im_bot, cax = cax_bot)
    cbar3.set_label(label_bottom, fontsize = fs + 8)
    cbar3.ax.tick_params(labelsize = fs)

    if savepath is not None:
        fig.savefig(
            savepath,
            format = "pdf",
            bbox_inches = "tight",
            pad_inches = 0.01
        )
    plt.show()


def build_plotting_df_from_results(
        results: dict,
        k: int = 1000,
        prefix: str = "raw",
        eps: float = 1e-12
    ) -> pd.DataFrame:
    """
    Builds a plotting df from the results dictionary where
    we now also include the intra-group skew metrics to show
    demographic parity's deficiencies
    """
    rows = []
    for ds, entry in results.items():
                                        # metrics come as (mean, std)
        ndkl_mean, ndkl_std = entry.get(
            f"{prefix}_NDKL",
            (np.nan, np.nan)
        )
        dp_mean, dp_std     = entry.get(
            f"{prefix}_demographic_parity",
            (np.nan, np.nan)
        )                               # hat_pi has no std
        hat_pi = entry.get(
            f"{prefix}_pi_hat_at_k_mean",
            None
        )
        if hat_pi is None:              # just skip if absent
            continue                    
                                        #! maybe reshape here? 
        hat_pi = np.asarray(hat_pi, dtype = float).reshape(-1)
                                        # skip if insufficient data
        if hat_pi.size < 3 or not np.isfinite(hat_pi[:3]).all():
            continue

        hat_pi00, hat_pi01, hat_pi11 = hat_pi[:3]
                                        # calculate rho (ratio of intra-group edges)
        rho = hat_pi00 / (hat_pi11 + eps)
                                        # also its log and just delta_intra
        log_rho = float(np.log(rho + eps))
        delta_intra = float(abs(hat_pi00 - hat_pi11))

        rows.append({                   # add all as a row
            "dataset" : ds,
            "ndkl_mean" : float(ndkl_mean) if ndkl_mean is not None else np.nan,
            "ndkl_std" : float(ndkl_std) if ndkl_std is not None else np.nan,
            "dp_mean" : float(dp_mean) if dp_mean is not None else np.nan,
            "dp_std" : float(dp_std) if dp_std is not None else np.nan,
            "hat_pi00" : hat_pi00,
            "hat_pi01" : hat_pi01,
            "hat_pi11" : hat_pi11,
            "rho" : rho,
            "log_rho" : log_rho,
            "delta_intra" : delta_intra,
        })

    return pd.DataFrame(rows)


def draw_violins(
        ax,
        data_list,
        positions,
        color,
        width = 0.6,
        alpha = 0.25
    ):
    """
    Draws violins onto ax given style parameters
    (taken from internship repo)
    """
    v = ax.violinplot(
        data_list,
        positions = positions,
        showmeans = True,
        showmedians = False,
        showextrema = True,
        widths = width,
    )
    for b in v["bodies"]:
        b.set_facecolor(color)
        b.set_edgecolor(color)
        b.set_alpha(alpha)
        b.set_linewidth(0)
    for key in ["cmeans", "cbars", "cmins", "cmaxes"]:
        v[key].set_color(color)
        v[key].set_linewidth(1.2)
    return v


def _values_by_bin(
        df,
        y_col,
        bin_col,
        bin_labels
    ):
    """ 
    Helper function to get y_col values for each bin in bin_col
    """
    out = []
    for lab in bin_labels:
        vals = df.loc[df[bin_col] == lab, y_col].to_numpy(
            dtype = float
        )
        vals = vals[np.isfinite(vals)]
        out.append(vals if vals.size else np.array([np.nan]))
    return out


def plot_violin(
    df: pd.DataFrame,                   # the data
    *,                                  # keyword-only arguments
    y = "dp_mean",                      # column to plot on y-axis
    y_label = r"$\Delta \mathrm{DP}$",  # y-axis label
    color = "black",                    # violin color
    x = "log_rho",                      # column to bin on x-axis
    n_bins = 10,                        # number of bins            
    dp_band = None,                     # optional (low, high) filter
                                        # on dp_mean
    quantile_bins = True,               # quantile direction for binning
    fs = 12,                            # font size 
    savepath = None,                    # where to save (optional)
                                        # bottom axis tick labels
    ratio_ticks = (0.01, 0.1, 1, 10, 100),
):
    df = df.copy()                      # avoid Pythonic errors

                                        # filter invalid rows, by infinites
                                        # and the low/high band
    df = df[np.isfinite(df[x]) & np.isfinite(df[y])]
    if dp_band is not None:
        lo, hi = dp_band
        df = \
            df[np.isfinite(df["dp_mean"]) & \
               (df["dp_mean"] >= lo) & \
                (df["dp_mean"] <= hi)
            ]

                                        # binning in log-space
    if quantile_bins:
        df["x_bin"] = pd.qcut(
            df[x],
            q = n_bins,
            duplicates = "drop"
        )
    else:
        df["x_bin"] = pd.cut(
            df[x],
            bins = n_bins
        )
    bin_labels = list(df["x_bin"].cat.categories)

                                        # get data per bin
    y_arrays = _values_by_bin(df, y, "x_bin", bin_labels)
    x_pos = np.arange(len(bin_labels))
    mids = np.array(
        [0.5 * (float(b.left) + float(b.right)) \
         for b in bin_labels],
        dtype = float
    )

                                        # do the plot (golden ratio)
    fig, ax = plt.subplots(figsize = (3 * 1.618, 3))
    draw_violins(                       # use same lay-out as JFRM paper
        ax,
        y_arrays,
        x_pos,
        color = color,
        width = 0.7, alpha = 0.25
    )

    if y_label == "$\mathrm{NDKL}$":    # add labels
        ax.set_ylabel(y_label, fontsize = fs + 1)
    else:
        ax.set_ylabel(y_label, fontsize = fs + 3)
    ax.tick_params(axis = "y", labelsize = fs)
                                        # add ticks
    ratio_ticks = np.asarray(ratio_ticks, dtype = float)
    ratio_ticks = ratio_ticks[np.isfinite(ratio_ticks) & (ratio_ticks > 0)]
    ln_targets = np.log(ratio_ticks)

                                        # for each tick value, find the bin
                                        # whose log-midpoint is closest, and
                                        # remember the bin and distance
    snapped = []                       
    for r, ln_r in zip(ratio_ticks, ln_targets):
        j = int(np.argmin(np.abs(mids - ln_r)))
        err = float(np.abs(mids[j] - ln_r))
        snapped.append((j, err, r))

    best_per_pos = {}                   #!TODO: deduplicate bins 
    for j, err, r in snapped:
        if (j not in best_per_pos) or (err < best_per_pos[j][0]):
            best_per_pos[j] = (err, r)
                                        # set ticks and labels
    tick_pos = np.array(sorted(best_per_pos.keys()), dtype = int)
    tick_lab = [f"{best_per_pos[j][1]:g}" for j in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize = fs)
    ax.set_xlabel(
        r"binned $\ln(\hat{\pi}_{0\!-\!0}@K \,/\, \hat{\pi}_{1\!-\!1}@K)$",
        fontsize = fs + 1)
    ax.set_axisbelow(True)
    ax.grid(                            # log-grid lines
        True,
        axis = "y",
        linestyle = "--",
        linewidth = 0.8,
        alpha = 0.5
    )
    ax.grid(
        True,
        axis = "x",
        linestyle = ":",
        linewidth = 0.9,
        alpha = 0.9
    )

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(
            savepath,
            bbox_inches = "tight",
            pad_inches = 0.015,
            dpi = 300
        )
    plt.show()


def plot_eight_heatmaps_compact(
    results,
    fm_values,
    h_MM_values,
    h_mm_values,
    fs=16,
    savepath=None,
    colour_map = "inferno",
):
    """
    plots eight heatmaps in a compact grid layout
    """
                                        # get values for plotting
    h_MM_values = np.asarray(h_MM_values, dtype=float)
    h_mm_values = np.asarray(h_mm_values, dtype=float)
    fm_values = list(fm_values)
    n = len(fm_values)

    metrics = [
        "h_edge",
        "h_edge_rand",
        "delta_h_edge",
        "reranked_demographic_parity",
        "reranked_NDKL",
        "precision_at_k",
        "reranked_AWRF",
        "NDCG_at_k",
    ]
    
    labels = [
        r"$h_\mathrm{edge}$",
        r"$h_\mathrm{edge}^\mathrm{rand}$",
        r"$\Delta h_\mathrm{edge}$",
        r"$\Delta_\mathrm{DP}$",
        r"$\mathrm{NDKL}$",
        r"$\mathrm{Precision}$",
        r"$\mathrm{AWRF}$",
        r"$\mathrm{NDCG}$",
    ]

    R = len(metrics)
    fig = plt.figure(figsize = (4.6 * n + 0.8, 3.0 * R))
    gs = fig.add_gridspec(
        R, n + 1,
        width_ratios = [1] * n + [0.035],
        wspace = 0.12,
        hspace = 0.12
    )

    axs = [[fig.add_subplot(gs[r, j]) for j in range(n)] for r in range(R)]
    for r in range(1, R):
        for j in range(n):
            axs[r][j].sharex(axs[0][0])
            axs[r][j].sharey(axs[0][0])

    caxs = [fig.add_subplot(gs[r, -1]) for r in range(R)]

                                        # shift all colorbars slightly left
    for cax in caxs:
        pos = cax.get_position()
        cax.set_position([pos.x0 - 0.008, pos.y0, pos.width, pos.height])

    def get_val(key, metric):
        m = results[key][metric]
        return m[0] if isinstance(m, (list, tuple, np.ndarray)) else float(m)

    def make_Z(fm, metric):
        Z = np.full((len(h_MM_values), len(h_mm_values)), np.nan)
        for i, h_MM in enumerate(h_MM_values):
            for j, h_mm in enumerate(h_mm_values):
                key = f"dpah_fm{fm:.2f}_hMM{h_MM:.2f}_hmm{h_mm:.2f}"
                if key not in results or metric not in results[key]:
                    continue
                Z[i, j] = get_val(key, metric)
        return Z.T                      # x = h_MM, y = h_mm

    extent = [h_mm_values.min(), h_mm_values.max(), h_MM_values.min(), h_MM_values.max()]

    def global_minmax(metric):
        vals = []
        for fm in fm_values:
            Z = make_Z(fm, metric)
            vals.append(Z)
        allZ = np.stack(vals, axis = 0)
        vmin = np.nanmin(allZ)
        vmax = np.nanmax(allZ)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None, None
        if vmin == vmax:
            eps = 1e-6 if vmin == 0 else 1e-3 * abs(vmin)
            return vmin - eps, vmax + eps
        return float(vmin), float(vmax)

                                        # per-metric normalization
    norm = {}
    for m in metrics:
        vmin, vmax = global_minmax(m)

                                        # special cases for homophily metrics
        if m == "h_edge":
            vmin, vmax = 0.0, 1.0
        elif m == "h_edge_rand":
                                        # baseline in [0,1], adapted if needed
            vmin, vmax = (0.0, 1.0) if (vmin is None) else (max(0.0, vmin), min(1.0, vmax))
        elif m == "delta_h_edge":
                                        # symmetric around 0 for interpretability (diverging cmap)
            if vmin is None:
                vmin, vmax = -1.0, 1.0
            else:
                M = max(abs(vmin), abs(vmax))
                vmin, vmax = -M, M

        norm[m] = (vmin, vmax)

    ims = [None] * R
    for r in range(R):
        metric = metrics[r]
        vmin, vmax = norm[metric]
        for j, fm in enumerate(fm_values):
            ims[r] = axs[r][j].imshow(
                make_Z(fm, metric),
                origin = "lower",
                extent = extent,
                aspect = "auto",
                cmap = colour_map,
                vmin = vmin,
                vmax = vmax,
            )
            axs[r][j].tick_params(labelsize = fs)
            if r == 0:
                axs[r][j].set_title(
                    rf"$f_\mathrm{{m}} = {fm:.2f}$",
                    fontsize = fs + 10)

                                        # only left-most column shows y tick labels
    for r in range(R):
        for j in range(1, n):
            axs[r][j].tick_params(labelleft=False)

                                        # x ticks every 0.2. Apply to bottom row, all share x
    xticks = np.arange(0.0, 1.01, 0.2)
    for ax in axs[-1]:
        ax.set_xticks(xticks)
        ax.set_xticklabels([rf"${x:.1f}$" for x in xticks])

                                        # x-label only once in bottom middle
    mid = n // 2
    axs[-1][mid].set_xlabel(r"majority $h_\mathrm{MM}$", fontsize=fs + 10)
    for j, ax in enumerate(axs[-1]):
        if j != mid:
            ax.set_xlabel("")

                                        # hide x-tick labels on all rows except the last
    for r in range(R - 1):
        for ax in axs[r]:
            ax.tick_params(labelbottom=False)

                                        # floating y-label between 4th and 5th row,
                                        # between ΔDP and NDKL rows
    pos4 = axs[3][0].get_position()     # row index 3
    pos5 = axs[4][0].get_position()     # row index 4
    y_between = 0.5 * (pos4.y0 + pos5.y1)
    x_left = pos4.x0 - 0.04
    fig.text(
        x_left, y_between, r"minority $h_\mathrm{mm}$",
        rotation=90, va="center", ha="center",
        fontsize=fs + 10
    )

                                        # colorbars (one per row)
    for r in range(R):
        cb = fig.colorbar(ims[r], cax=caxs[r])
        cb.set_label(labels[r], fontsize=fs + 8)
        cb.ax.tick_params(labelsize=fs)

    if savepath is not None:
        fig.savefig(savepath, format="pdf", bbox_inches="tight", pad_inches=0.01)

    plt.show()


def plot_zscores_with_hedge_bottom(
    results,
    fm_values,
    h_MM_values,
    h_mm_values,
    fs = 16,
    savepath = None,
    key_prec = "precision_at_k",
    key_ndkl = "reranked_NDKL",
    key_awrf = "reranked_AWRF",
    key_hedge = "h_edge",
    interpolate_nans = True,
    vline_style = dict(color = "black", linestyle = "--", linewidth = 0.9, alpha = 0.45),
):
    """
    Plots z-score plots of precision and NDKL on top,
    and h_edge on the bottom for each fm value
    """
    ms = 8                              # marker size
    fm_values   = list(fm_values)
    h_MM_values = np.asarray(list(h_MM_values), dtype = float)
    h_mm_values = list(h_mm_values)

    n = len(fm_values)
    fig, axs = plt.subplots(
        2, n,
        figsize = (4.2 * n, 6.2),
        sharex = True,
        gridspec_kw = {                 # separate subplots
            "hspace" : 0.12,
            "wspace" : 0.08
        },
    )
    if n == 1:
        axs = np.array([[axs[0]], [axs[1]]])

    def meanish(v):
        return v[0] if isinstance(v, (list, tuple, np.ndarray)) else float(v)

    def nan_interp(x, xs):
        x = np.asarray(x, dtype = float)
        mask = np.isfinite(x)
        if mask.sum() < 2:
            return x
        return np.interp(xs, xs[mask], x[mask])

    def zscore_nan(x, xs):
        x = np.asarray(x, dtype = float)
        if interpolate_nans:
            x = nan_interp(x, xs)
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0.0:
            return np.zeros_like(x)
        return (x - mu) / sd

    xticks = np.arange(0.0, 1.01, 0.2)

                                        # calculate global z-max for y-limits
    zmax = 0.0
    for fm in fm_values:
        for metric in (key_prec, key_ndkl, key_awrf):

            vals = []
            for h_MM in h_MM_values:
                per_hmm = []
                for h_mm in h_mm_values:
                    k = f"dpah_fm{fm:.2f}_hMM{h_MM:.2f}_hmm{h_mm:.2f}"
                    if k in results and metric in results[k]:
                        per_hmm.append(meanish(results[k][metric]))
                vals.append(np.mean(per_hmm) if len(per_hmm) else np.nan)
            z = zscore_nan(vals, h_MM_values)
            zmax = max(zmax, np.nanmax(np.abs(z)))


    for j, fm in enumerate(fm_values):
        ax_top = axs[0, j]
        ax_bottom = axs[1, j]

        def avg_over_hmm(metric_key):
            vals = []
            for h_MM in h_MM_values:
                per_hmm = []
                for h_mm in h_mm_values:
                    k = f"dpah_fm{fm:.2f}_hMM{h_MM:.2f}_hmm{h_mm:.2f}"
                    if k in results and metric_key in results[k]:
                        per_hmm.append(meanish(results[k][metric_key]))
                vals.append(np.mean(per_hmm) if len(per_hmm) else np.nan)
            return np.asarray(vals, dtype=float)

        prec  = avg_over_hmm(key_prec)
        ndkl  = avg_over_hmm(key_ndkl)
        hedge = avg_over_hmm(key_hedge)
        awrf  = avg_over_hmm(key_awrf)


                                        # find peak location of NDKL
        prec_i  = nan_interp(prec,  h_MM_values) if interpolate_nans else prec
        ndkl_i  = nan_interp(ndkl,  h_MM_values) if interpolate_nans else ndkl
        hedge_i = nan_interp(hedge, h_MM_values) if interpolate_nans else hedge
        z_prec = zscore_nan(prec, h_MM_values)
        z_ndkl = zscore_nan(ndkl, h_MM_values)
        z_awrf = zscore_nan(awrf, h_MM_values)

        peak_idx = int(np.nanargmax(ndkl_i))
        peak_hMM = float(h_MM_values[peak_idx])
        peak_hedge = float(hedge_i[peak_idx])

                                        # z-scores on top
        ax_top.axhline(0.0, color = "black", linewidth = 0.6, alpha = 0.25, zorder = 0)

                                        # http://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
        ax_top.plot(
            h_MM_values,
            z_prec,
            color = "blue",
            marker = "+",
            linestyle = "-.",
            linewidth = 1.1,
            markersize = ms,
            zorder = 3
        )
        ax_top.plot(
            h_MM_values,
            z_ndkl,
            color = "red",
            marker = "x",
            linewidth = 1.1,
            markersize = ms,
            zorder = 4
        )

        ax_top.plot(
            h_MM_values,
            z_awrf,
            color = "magenta",
            marker = "2",
            linestyle = "--",
            linewidth = 1.1,
            markersize = ms + 4,
            zorder = 1
        )


                                        # vertical line at NDKL peak (same x in both rows)
        ax_top.axvline(peak_hMM, **vline_style)
        ax_bottom.axvline(peak_hMM, **vline_style)

        ax_top.set_title(rf"$f_\mathrm{{m}} = {fm:.2f}$", fontsize = fs + 6)
        ax_top.tick_params(labelsize = fs)

        ax_top.set_ylim(-zmax, zmax)
        ax_top.set_yticks([-2, -1, 0, 1, 2])

                                        # homophily bottom row
        ax_bottom.plot(
            h_MM_values, hedge_i,
            color = "black",
            marker = "2",
            linewidth = 0.9,
            markersize = ms + 4
        )
        ax_bottom.set_ylim(0.0, 1.0)
        ax_bottom.set_xticks(xticks)
        ax_bottom.tick_params(labelsize = fs)

                                        # shift annotation right for the panels on the right 
                                        # to avoid overlap
        x_text = peak_hMM
        if j >= n - 3:
            x_text = min(peak_hMM + 0.2, h_MM_values.max())

        ax_bottom.text(                 # annotate the homophily level at the peak
            x_text, 0.02,
            rf"$h_\mathrm{{edge}}\approx {peak_hedge:.2f}$",
            ha = "center",
            va = "bottom",
            fontsize = fs + 6,
        )

        if j != 0:
            ax_top.tick_params(labelleft = False)
            ax_bottom.tick_params(labelleft = False)

    fig.supxlabel(r"majority $h_\mathrm{MM}$", fontsize = fs + 6, y = 0.00)
    axs[0, 0].set_ylabel(r"$z$-score (per $f_\mathrm{m}$)", fontsize = fs + 6)
    axs[1, 0].set_ylabel(r"homophily $h_\mathrm{edge}$", fontsize = fs + 6)

    legend_handles = [
        Line2D([0], [0], color = "red", marker = "x", linewidth = 1.1,
                markersize = ms, label = "NDKL"),
        Line2D([0], [0], color = "magenta", linestyle = "--",
               marker = "2", linewidth = 1.1,
                markersize = ms + 4, label = "AWRF"),
        Line2D([0], [0], color = "blue", linestyle = "-.",
               marker = "+", linewidth = 1.1,
               markersize = ms, label = "Precision"),
        Line2D([0], [0], color = "black", marker = "2", linewidth = 0.9,
               markersize = ms + 4, label = r"$h_\mathrm{edge}$"),
        Line2D([0], [0], **vline_style, label = "NDKL peak"),
    ]
    
    fig.legend(
        handles = legend_handles,
        loc = "upper center",
        ncol = 5,
        frameon = False,
        fontsize = fs + 4,
        bbox_to_anchor = (0.5, 1.045),
    )

    plt.tight_layout()

    if savepath:
        fig.savefig(
            savepath,
            format = "pdf",
            bbox_inches = "tight",
            pad_inches = 0.015
        )
        
    plt.show()


def plot_zscores_with_homophily_rows(
    results,
    fm_values,
    h_MM_values,
    h_mm_values,
    fs=16,
    savepath=None,
    key_prec="precision_at_k",
    key_ndkl="reranked_NDKL",
    key_hedge="h_edge",
    key_delta="delta_h_edge",
    label_ndkl = r"$\mathrm{NDKL}$",
    label_prec = r"$\mathrm{Precision}$",
    interpolate_nans=True,
    vline_style=dict(color="black", linestyle="--", linewidth=0.9, alpha=0.45),
):
    """
    Plots z-score plots of precision and NDKL on top,
    and homophily on the bottom for each fm value
    """
    ms = 6  # marker size
    if label_ndkl is None:
        label_ndkl = key_ndkl
    if label_prec is None:
        label_prec = key_prec

    fm_values = list(fm_values)
    h_MM_values = np.asarray(list(h_MM_values), dtype=float)
    h_mm_values = list(h_mm_values)
    n = len(fm_values)

    fig, axs = plt.subplots(
        3, n,
        figsize=(4.2 * n, 9.0),
        sharex=True,
        gridspec_kw={"hspace": 0.12, "wspace": 0.08},
    )
    if n == 1:
        axs = np.array([[axs[0]], [axs[1]], [axs[2]]])

    def meanish(v):
        return v[0] if isinstance(v, (list, tuple, np.ndarray)) else float(v)

    def nan_interp(x, xs):
        x = np.asarray(x, dtype=float)
        mask = np.isfinite(x)
        if mask.sum() < 2:
            return x
        return np.interp(xs, xs[mask], x[mask])

    def zscore_nan(x, xs):
        x = np.asarray(x, dtype=float)
        if interpolate_nans:
            x = nan_interp(x, xs)
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0.0:
            return np.zeros_like(x)
        return (x - mu) / sd

    def avg_over_hmm(fm, metric_key):
        vals = []
        for h_MM in h_MM_values:
            per_hmm = []
            for h_mm in h_mm_values:
                k = f"dpah_fm{fm:.2f}_hMM{h_MM:.2f}_hmm{h_mm:.2f}"
                if k in results and metric_key in results[k]:
                    per_hmm.append(meanish(results[k][metric_key]))
            vals.append(np.mean(per_hmm) if len(per_hmm) else np.nan)
        return np.asarray(vals, dtype=float)

    xticks = np.arange(0.0, 1.01, 0.2)

                                        # global z-score limits top row
    zmax = 0.0
    for fm in fm_values:
        for metric in (key_prec, key_ndkl):
            vals = avg_over_hmm(fm, metric)
            z = zscore_nan(vals, h_MM_values)
            zmax = max(zmax, float(np.nanmax(np.abs(z))) if np.isfinite(np.nanmax(np.abs(z))) else 0.0)

                                        # global delta-h limits bottom row
    dmin, dmax = np.inf, -np.inf
    for fm in fm_values:
        vals = avg_over_hmm(fm, key_delta)
        vals_i = nan_interp(vals, h_MM_values) if interpolate_nans else vals
        if np.any(np.isfinite(vals_i)):
            dmin = min(dmin, float(np.nanmin(vals_i)))
            dmax = max(dmax, float(np.nanmax(vals_i)))
    if not np.isfinite(dmin) or not np.isfinite(dmax):
        dmin, dmax = -1.0, 1.0
    pad = 0.05 * (dmax - dmin) if (dmax - dmin) > 0 else 0.01
    dmin, dmax = dmin - pad, dmax + pad

                                        # plot panels 
    for j, fm in enumerate(fm_values):
        ax_top = axs[0, j]
        ax_mid = axs[1, j]
        ax_bot = axs[2, j]

        prec = avg_over_hmm(fm, key_prec)
        ndkl = avg_over_hmm(fm, key_ndkl)
        hedge = avg_over_hmm(fm, key_hedge)
        delta = avg_over_hmm(fm, key_delta)

        prec_i = nan_interp(prec, h_MM_values) if interpolate_nans else prec
        ndkl_i = nan_interp(ndkl, h_MM_values) if interpolate_nans else ndkl
        hedge_i = nan_interp(hedge, h_MM_values) if interpolate_nans else hedge
        delta_i = nan_interp(delta, h_MM_values) if interpolate_nans else delta

        z_prec = zscore_nan(prec, h_MM_values)
        z_ndkl = zscore_nan(ndkl, h_MM_values)

                                        # peak location based on (interpolated) NDKL
        peak_idx = int(np.nanargmax(ndkl_i))
        peak_hMM = float(h_MM_values[peak_idx])

        peak_hedge = float(hedge_i[peak_idx]) if np.isfinite(hedge_i[peak_idx]) else float("nan")
        peak_delta = float(delta_i[peak_idx]) if np.isfinite(delta_i[peak_idx]) else float("nan")

                                        # z-scores on top
        color1 = "blue"
        color2 = "red"
        if label_ndkl == "AWRF":
            color1 = "magenta"
            color2 = "green"

        ax_top.axhline(0.0, color="black", linewidth=0.6, alpha=0.25, zorder=0)
        ax_top.plot(h_MM_values, z_ndkl, color=color1, marker="x",
                    linewidth=1.1, markersize=ms, zorder=4)
        ax_top.plot(h_MM_values, z_prec, color=color2, marker="+",
                    linewidth=1.1, markersize=ms, zorder=3)

        ax_top.axvline(peak_hMM, **vline_style)
        ax_top.set_ylim(-zmax, zmax)
        ax_top.set_yticks([-2, -1, 0, 1, 2])
        ax_top.set_title(rf"$f_\mathrm{{m}} = {fm:.2f}$", fontsize=fs + 6)
        ax_top.tick_params(labelsize=fs)

                                        # middle row: h_edge 
        ax_mid.plot(h_MM_values, hedge_i, color="black", marker="2",
                    linewidth=0.9, markersize=ms + 4)
        ax_mid.axvline(peak_hMM, **vline_style)
        ax_mid.set_ylim(0.0, 1.0)
        ax_mid.tick_params(labelsize=fs)

                                        # h_edge annotation, shifted right for right-most 3 to avoid overlap
        x_text = peak_hMM
        if j >= n - 3:
            x_text = min(peak_hMM + 0.2, float(h_MM_values.max()))
        ax_mid.text(
            x_text, 0.02,
            rf"$h_\mathrm{{edge}}\approx {peak_hedge:.2f}$",
            ha="center", va="bottom", fontsize=fs + 6,
        )

                                        # bottom row: delta_h_edge 
        ax_bot.plot(h_MM_values, delta_i, color="black", marker="2",
                    linewidth=0.9, markersize=ms + 4)
        ax_bot.axvline(peak_hMM, **vline_style)
        ax_bot.set_ylim(dmin, dmax)
        ax_bot.set_xticks(xticks)
        ax_bot.tick_params(labelsize=fs)

                                        # delta_h annotation with shift rule
        ax_bot.text(
            x_text, dmin + 0.10 * (dmax - dmin),
            rf"$\Delta h_\mathrm{{edge}}\approx {peak_delta:.2f}$",
            ha="center", va="bottom", fontsize=fs + 6,
        )

                                        # show y tick labels only on first column
        if j != 0:
            ax_top.tick_params(labelleft=False)
            ax_mid.tick_params(labelleft=False)
            ax_bot.tick_params(labelleft=False)

    fig.supxlabel(r"majority $h_\mathrm{MM}$", fontsize=fs + 6, y=0.03)
    axs[0, 0].set_ylabel(r"$z$-score (per $f_\mathrm{m}$)", fontsize=fs + 6)
    axs[1, 0].set_ylabel(r"homophily $h_\mathrm{edge}$", fontsize=fs + 6)
    axs[2, 0].set_ylabel(r"$\Delta h_\mathrm{edge}$", fontsize=fs + 6)

    legend_handles = [
        Line2D([0], [0], color=color1, marker="x", linewidth=1.1, markersize=ms, label=label_ndkl),
        Line2D([0], [0], color=color2,  marker="+", linewidth=1.1, markersize=ms, label=label_prec),
        Line2D([0], [0], color="black", marker="2", linewidth=0.9, markersize=ms + 4,
            label=r"$h_\mathrm{edge}$ / $\Delta h_\mathrm{edge}$"),
        Line2D([0], [0], **vline_style, label=rf"{label_ndkl} peak"),
    ]
    
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=fs + 4,
        bbox_to_anchor=(0.5, 1.045),
    )

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, format="pdf", bbox_inches="tight", pad_inches=0.015)
    plt.show()

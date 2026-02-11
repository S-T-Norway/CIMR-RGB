# --- Imports & plotting config ---
import matplotlib
matplotlib.use("TkAgg")  # comment out if not using interactive windows
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import UnivariateSpline

mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (0, 0),
    "axes.formatter.useoffset": False
})

# ---------- Helpers ----------

def elbow_max_curvature_spline(x, y, bg_smoothing, smoothing=0,
                               exclude_frac=0.05, grid_multiplier=8):
    """
    Find elbow by maximum curvature κ(x) using a cubic spline, evaluated on a dense grid,
    excluding a small fraction of endpoints to avoid edge artifacts.

    Returns: x_star, y_star, p_star, idx_nearest
    """
    x = np.asarray(x); y = np.asarray(y); p = np.asarray(bg_smoothing)

    # Sort by x (monotonic relation)
    order = np.argsort(x)
    x_s, y_s, p_s = x[order], y[order], p[order]

    # Remove duplicate x (required by spline)
    keep = np.r_[True, np.diff(x_s) != 0]
    x_s, y_s, p_s = x_s[keep], y_s[keep], p_s[keep]

    # Spline through data (small smoothing helps noisy derivatives)
    sp = UnivariateSpline(x_s, y_s, k=3, s=smoothing)

    # Dense grid for stable curvature
    n = len(x_s)
    m = max(grid_multiplier * n, 200)  # ensure reasonably dense even for small n
    xx = np.linspace(x_s.min(), x_s.max(), m)

    y1 = sp.derivative(1)(xx)
    y2 = sp.derivative(2)(xx)
    kappa = np.abs(y2) / (1 + y1**2)**1.5

    # Exclude a margin near the endpoints (default 5% each side)
    L = len(xx)
    left = int(np.floor(exclude_frac * L))
    right = int(np.ceil((1 - exclude_frac) * L))
    if right - left < 3:  # fallback if too few points
        left, right = 1, L - 1

    i_star_grid = left + int(np.argmax(kappa[left:right]))
    x_star = float(xx[i_star_grid])
    y_star = float(sp(x_star))

    # Map to nearest original point to retrieve bg_smoothing
    idx_nearest = int(np.argmin(np.abs(x_s - x_star)))
    p_star = float(p_s[idx_nearest])

    test =0

    return x_star, y_star, p_star, idx_nearest


def elbow_max_distance_to_chord(x, y, bg_smoothing):
    """
    Robust L-curve 'corner': point with maximum perpendicular distance
    to the straight chord connecting endpoints. No derivatives needed.
    Returns: x_star, y_star, p_star, idx_max
    """
    x = np.asarray(x); y = np.asarray(y); p = np.asarray(bg_smoothing)
    order = np.argsort(x)
    x_s, y_s, p_s = x[order], y[order], p[order]

    x0, y0 = x_s[0],   y_s[0]
    x1, y1 = x_s[-1],  y_s[-1]

    # Vector along chord and to each point
    vx, vy = x1 - x0, y1 - y0
    denom = np.hypot(vx, vy)
    if denom == 0:
        # Degenerate curve
        idx_max = 0
        return float(x_s[idx_max]), float(y_s[idx_max]), float(p_s[idx_max]), idx_max

    # Perpendicular distance from each point to chord
    # area * 2 / |chord| = |(x-x0, y-y0) x (vx, vy)| / |(vx,vy)|
    dx = x_s - x0
    dy = y_s - y0
    dist = np.abs(dx * vy - dy * vx) / denom

    # Ignore endpoints
    if len(dist) > 2:
        dist[[0, -1]] = -np.inf
    idx_max = int(np.argmax(dist))

    return float(x_s[idx_max]), float(y_s[idx_max]), float(p_s[idx_max]), idx_max

def max_curvature_discrete(x, y, ignore_ends=1):
    x = np.asarray(x); y = np.asarray(y)
    order = np.argsort(x); x, y = x[order], y[order]
    X = np.stack([x, y], axis=1)
    a = X[1:-1] - X[:-2]
    b = X[2:]   - X[1:-1]
    c = X[2:]   - X[:-2]
    area2 = np.abs(a[:,0]*b[:,1] - a[:,1]*b[:,0])          # 2*triangle area
    denom = np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)*np.linalg.norm(c,axis=1)
    kappa = np.zeros(len(x)); kappa[1:-1] = 2*area2/denom   # curvature per interior point

    i0 = max(ignore_ends, 1); i1 = len(x)-max(ignore_ends,1)
    i_star_rel = np.argmax(kappa[i0:i1])
    i_star = i0 + i_star_rel
    return x[i_star], y[i_star], kappa[i_star], i_star

def elbow_max_distance(x, y, use_loglog=True, normalize=True, ignore_ends=1):
    """
    Elbow via maximum perpendicular distance to the chord between endpoints,
    after optional log-log transform and normalization.

    Returns
    -------
    i_star : int, x_star, y_star : float
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    order = np.argsort(x)
    x, y = x[order], y[order]

    X = np.log10(x) if use_loglog else x.copy()
    Y = np.log10(y) if use_loglog else y.copy()

    if normalize:
        X = (X - X.min()) / (X.max() - X.min() + 1e-15)
        Y = (Y - Y.min()) / (Y.max() - Y.min() + 1e-15)

    x0, y0 = X[0], Y[0]
    x1, y1 = X[-1], Y[-1]
    vx, vy = x1 - x0, y1 - y0
    denom = np.hypot(vx, vy) + 1e-15

    dx = X - x0
    dy = Y - y0
    dist = np.abs(dx * vy - dy * vx) / denom

    # avoid picking the exact endpoints
    if len(dist) > 2:
        dist[:ignore_ends] = -np.inf
        dist[-ignore_ends:] = -np.inf

    i_star = int(np.argmax(dist))
    return i_star, float(x[ i_star ]), float(y[ i_star ])


# ---------- Script ----------

# Load your CSV (expects columns listed below)
df = pd.read_csv("l_curve_results_for_joey_700.csv")
# Required columns: scan_direction, max_neighbours, noise_error, fit_error, bg_smoothing
print(df.head())

# # Choose scan direction and panels
# Comment out if scan_direction is turned off
df_fore = df
# df_fore = df[df["scan_direction"] == "_fore"]
neighbour_values = [700]
#
# # Color range across all panels
vmin = df_fore["bg_smoothing"].min()
vmax = df_fore["bg_smoothing"].max()
#
if len(neighbour_values) == 1:
    max_neighbour_value = neighbour_values[0]
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    scatter_ref = None

    df_sub = df_fore[df_fore["max_neighbours"] == max_neighbour_value]
    df_sub = df_sub.sort_values("noise_error")  # monotonic x for plotting
    # ----- pick ONE of the elbows below -----
    # 1) Maximum curvature via spline (recommended). Tweak smoothing/exclude_frac if needed.
    x_star, y_star, p_star, idx_star = elbow_max_curvature_spline(
        df_sub["noise_error"].to_numpy(),
        df_sub["fit_error"].to_numpy(),
        df_sub["bg_smoothing"].to_numpy(),
        smoothing=0,  # try small >0 if noisy, e.g. 1e-6 * len(df_sub)
        exclude_frac=0.05,  # ignore 5% at each end
        grid_multiplier=8
    )
    x, y, kappy, i = max_curvature_discrete(
        df_sub["noise_error"].to_numpy(),
        df_sub["fit_error"].to_numpy(),
        ignore_ends=1
    )

    i, x, y = elbow_max_distance(
        df_sub["noise_error"].to_numpy(),
        df_sub["fit_error"].to_numpy(),
        use_loglog=True, normalize=True, ignore_ends=1
    )

    # 2) Or: “L-curve corner” fallback (max distance to chord)
    # x_star, y_star, p_star, idx_star = elbow_max_distance_to_chord(
    #     df_sub["noise_error"].to_numpy(),
    #     df_sub["fit_error"].to_numpy(),
    #     df_sub["bg_smoothing"].to_numpy()
    # )

    # Scatter colored by bg_smoothing
    sc = ax.scatter(
        df_sub["noise_error"], df_sub["fit_error"],
        c=df_sub["bg_smoothing"], cmap="magma",
        vmin=vmin, vmax=vmax, s=60, edgecolor="k", zorder=2
    )


    if scatter_ref is None:
        scatter_ref = sc

    # Connect points
    ax.plot(df_sub["noise_error"], df_sub["fit_error"],
            color="gray", linewidth=1, zorder=1)

    # # Mark optimal and label
    # ax.scatter([x], [y],
    #            marker="x", s=100, facecolor="red",
    #            linewidths=2, zorder=3,
    #            label=f"Optimal bg_smoothing = {p_star:.4g}")
    # ax.legend(loc="upper right", frameon=True)

    # Label each black dot with its bg_smoothing value
    for x, y, bg in zip(df_sub["noise_error"], df_sub["fit_error"], df_sub["bg_smoothing"]):
        plt.annotate(
            f"{bg:.3f}",  # label text
            (x, y),  # point location
            textcoords="offset points",
            xytext=(0, 5),  # move label 5 points above the dot
            ha='center',
            fontsize=7
        )

    ax.set_title(f"max_neighbours = {max_neighbour_value}")
    ax.set_xlabel("Noise error [K]")
    ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # y-labels for left column
    ax.set_ylabel("Fit error [-]")
    # Shared colorbar
    if scatter_ref is not None:
        cbar = fig.colorbar(scatter_ref, ax=ax,
                            orientation="vertical", fraction=0.035, pad=0.02)
        cbar.set_label("bg_smoothing [-]")

    fig.suptitle("Fit error vs Noise error (colored by bg_smoothing)")
    plt.show()


else:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    axes = axes.ravel()
    scatter_ref = None

    for ax, n in zip(axes, neighbour_values):
        df_sub = df_fore[df_fore["max_neighbours"] == n]

        if df_sub.empty:
            ax.set_title(f"max_neighbours = {n}\n(no data)")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        df_sub = df_sub.sort_values("noise_error")  # monotonic x for plotting

        # ----- pick ONE of the elbows below -----
        # 1) Maximum curvature via spline (recommended). Tweak smoothing/exclude_frac if needed.
        x_star, y_star, p_star, idx_star = elbow_max_curvature_spline(
            df_sub["noise_error"].to_numpy(),
            df_sub["fit_error"].to_numpy(),
            df_sub["bg_smoothing"].to_numpy(),
            smoothing=0,          # try small >0 if noisy, e.g. 1e-6 * len(df_sub)
            exclude_frac=0.05,    # ignore 5% at each end
            grid_multiplier=8
        )
        x, y, kappy, i = max_curvature_discrete(
            df_sub["noise_error"].to_numpy(),
            df_sub["fit_error"].to_numpy(),
            ignore_ends=1
        )

        i, x, y = elbow_max_distance(
            df_sub["noise_error"].to_numpy(),
            df_sub["fit_error"].to_numpy(),
            use_loglog=True, normalize=True, ignore_ends=1
        )

        # 2) Or: “L-curve corner” fallback (max distance to chord)
        # x_star, y_star, p_star, idx_star = elbow_max_distance_to_chord(
        #     df_sub["noise_error"].to_numpy(),
        #     df_sub["fit_error"].to_numpy(),
        #     df_sub["bg_smoothing"].to_numpy()
        # )

        # Scatter colored by bg_smoothing
        sc = ax.scatter(
            df_sub["noise_error"], df_sub["fit_error"],
            c=df_sub["bg_smoothing"], cmap="magma",
            vmin=vmin, vmax=vmax, s=60, edgecolor="k", zorder=2
        )
        if scatter_ref is None:
            scatter_ref = sc

        # Connect points
        ax.plot(df_sub["noise_error"], df_sub["fit_error"],
                color="gray", linewidth=1, zorder=1)

        # Mark optimal and label
        ax.scatter([x], [y],
                   marker="x", s=100, facecolor="red",
                   linewidths=2, zorder=3,
                   label=f"Optimal bg_smoothing = {p_star:.4g}")
        ax.legend(loc="upper right", frameon=True)

        ax.set_title(f"max_neighbours = {n}")
        ax.set_xlabel("Noise error [K]")
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # y-labels for left column
    axes[0].set_ylabel("Fit error [-]")
    axes[3].set_ylabel("Fit error [-]")

    # Hide unused 6th subplot
    if len(neighbour_values) < len(axes):
        axes[len(neighbour_values)].axis("off")

    # Shared colorbar
    if scatter_ref is not None:
        cbar = fig.colorbar(scatter_ref, ax=axes.tolist(),
                            orientation="vertical", fraction=0.035, pad=0.02)
        cbar.set_label("bg_smoothing [-]")

    fig.suptitle("Fit error vs Noise error (colored by bg_smoothing)")
    plt.show()



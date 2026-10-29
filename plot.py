import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import LogFormatterMathtext, LogLocator


ALGO_COLORS = {"SA": "forestgreen", "PA": "crimson", "GA": "royalblue"}
SAMPLER_COLORS = ALGO_COLORS
PLOT_DPI = 300
DEFAULT_EPS_ZERO = 1e-7
DEFAULT_RUN_TIME_LIMIT = 100.0

_SAMPLER_MODE_LABELS = {
    "SA": "SA",
    "PA": "PA",
    "GA": "GA",
    "local": "GA local only",
    "local_global": "GA local + global (frozen)",
    "local_global_retrain_50": "GA local + global + retrain/50",
    "local_global_retrain_10": "GA local + global + retrain/10",
}
_SAMPLER_MODE_COLORS = {
    "SA": ALGO_COLORS["SA"],
    "PA": ALGO_COLORS["PA"],
    "GA": ALGO_COLORS["GA"],
    "local": ALGO_COLORS["GA"],
    "local_global": "darkorange",
    "local_global_retrain_50": "purple",
    "local_global_retrain_10": "teal",
}
_RETRAIN_FALLBACK_COLORS = ["slategray", "olive", "brown", "navy"]


def _compute_energy_for_plot(population, J):
    """Compute EA energies for histogram plots when no notebook callback is passed."""
    return -torch.einsum("ki,ik->k", population, torch.einsum("ij,kj->ik", J, population)) / 2


def _retrain_variant_name(k):
    """Return the canonical sampler variant name for a retrain interval."""
    return f"local_global_retrain_{k}"


def apply_axis_style(ax, grid=True, minor_grid=False, legend=False, legend_kwargs=None,
                     facecolor=None, spine_color="#4d4d4d"):
    """Apply the shared grid, legend and spine styling used by notebook plots."""
    if facecolor is not None:
        ax.set_facecolor(facecolor)
    if grid:
        ax.grid(True, which="major", alpha=0.4)
    if minor_grid:
        ax.grid(True, which="minor", alpha=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color(spine_color)
    if legend:
        kwargs = {"loc": "best", "frameon": True}
        if legend_kwargs is not None:
            kwargs.update(legend_kwargs)
        leg = ax.legend(**kwargs)
        if leg is not None:
            leg.get_frame().set_edgecolor("#bfbfbf")
            leg.get_frame().set_linewidth(1.0)
    return ax


def finalize_plot(output_path=None, fig=None, pdf=False, bbox_inches="tight",
                  show=True, close=False, tight_kwargs=None):
    """Tighten layout, save PNG/PDF outputs, optionally show and close the figure."""
    if fig is None:
        fig = plt.gcf()
    if tight_kwargs is None:
        tight_kwargs = {}
    fig.tight_layout(**tight_kwargs)
    if output_path is not None:
        fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches=bbox_inches)
        print(f"Plot saved to: {output_path}")
        if pdf and str(output_path).endswith(".png"):
            pdf_path = str(output_path).replace(".png", ".pdf")
            fig.savefig(pdf_path, bbox_inches=bbox_inches)
            print(f"Plot saved to: {pdf_path}")
    if show:
        plt.show()
    if close:
        plt.close(fig)


def plot_algos_comparison(results_SA, results_PA, results_GA, output_path="fig1b_replica.png",
                          normalize_by_N=True, N=None, ymin=0.0, ymax=3e-1,
                          eps_zero=DEFAULT_EPS_ZERO):
    """
    Plot Delta E versus runtime for SA, PA and GA on a single panel.

    Values with Delta E / N <= eps_zero are rounded to zero after the optional
    normalization by N. A symlog axis is used when zero should be visible.
    """
    if normalize_by_N and N is None:
        raise ValueError("If normalize_by_N=True, you must provide N.")

    eps_zero_plot = eps_zero if normalize_by_N else eps_zero * N
    use_symlog_zero = ymin is not None and ymin <= 0
    if ymin is None:
        ymin = eps_zero_plot / 10

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "font.family": "sans-serif",
    })

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.4), dpi=150)
    fig.patch.set_facecolor("#e9e9e9")
    ax.set_facecolor("#e9e9e9")

    series = [
        (results_SA, "SA", ALGO_COLORS["SA"]),
        (results_PA, "PA", ALGO_COLORS["PA"]),
        (results_GA, "GA", ALGO_COLORS["GA"]),
    ]
    max_time = 0.0

    for results, label, color in series:
        td = results["trace_data"]
        times = np.asarray(td["time"], dtype=float)
        delta_E = np.asarray(td["delta_E"], dtype=float)

        if len(times) == 0:
            print(f"Skipping {label}: no recorded points.")
            continue

        if normalize_by_N:
            delta_E = delta_E / N

        delta_E_plot = np.where(delta_E <= eps_zero_plot, 0.0, delta_E)
        if not use_symlog_zero:
            delta_E_plot = np.maximum(delta_E_plot, ymin)

        ax.plot(times, delta_E_plot, color=color, linewidth=2.8, label=label)
        max_time = max(max_time, float(times[-1]))

    if use_symlog_zero:
        ax.set_yscale("symlog", linthresh=eps_zero_plot, linscale=0.5)
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

    ax.set_xlim(0, max_time)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel(r"$\Delta E / N$" if normalize_by_N else r"$\Delta E$")
    ax.set_title("SA vs PA vs GA -- full schedule, no time cutoff", pad=8)
    ax.grid(True, which="major", color="#9e9e9e", alpha=0.75, linewidth=1.0)
    ax.grid(True, which="minor", color="#bdbdbd", alpha=0.35, linewidth=0.6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#4d4d4d")

    leg = ax.legend(loc="lower left", frameon=True, fancybox=True, framealpha=0.9)
    leg.get_frame().set_edgecolor("#bfbfbf")
    leg.get_frame().set_linewidth(1.0)
    finalize_plot(output_path, fig=fig)


def plot_success_probability(ga_stats, pa_stats, ga_imp, pa_imp, total_seeds,
                             output_path="fig3_success_prob.png"):
    """
    Plot success probability versus runtime for GA and PA.

    The plot includes 25th-75th percentile bands plus best/worst seed traces.
    """

    def _band(stats, imp, color, label):
        """Draw imputed success-probability quartiles for one algorithm."""
        lowers, uppers, medians = [], [], []
        for _, row in imp.iterrows():
            nT = row["nTemps"]
            cnt = int(row["count"])
            actual = stats.loc[stats["nTemps"] == nT, "succ_prob"].values
            arr = np.concatenate([actual, np.ones(total_seeds - cnt)])
            lowers.append(np.percentile(arr, 25))
            uppers.append(np.percentile(arr, 75))
            medians.append(np.median(arr))

        x = imp["avg_time"].values
        plt.fill_between(x, lowers, uppers, color=color, alpha=0.2)
        plt.plot(x, lowers, color=color, ls=":", lw=2, label="_nolegend_")
        plt.plot(x, uppers, color=color, ls=":", lw=2, label="_nolegend_")
        plt.plot(x, medians, color=color, ls="-", lw=3, label=label)

    def _best_worst(stats, threshold=0.9, tail_n=10):
        """Return labels for the easiest and hardest seeds in the plot."""
        per_seed = []
        for _, sdf in stats.groupby("seed"):
            sdf = sdf.sort_values("avg_time")
            times = sdf["avg_time"].values
            succs = sdf["succ_prob"].values
            hit = np.where(succs >= threshold)[0]
            per_seed.append({
                "times": times,
                "succs": succs,
                "t90": times[hit[0]] if len(hit) else None,
                "end": succs[-tail_n:].mean() if len(succs) else np.nan,
            })
        with_t90 = [p for p in per_seed if p["t90"] is not None]
        none_t90 = [p for p in per_seed if p["t90"] is None]
        best = min(with_t90, key=lambda p: p["t90"]) if with_t90 else max(per_seed, key=lambda p: p["end"])
        worst = min(none_t90, key=lambda p: p["end"]) if none_t90 else max(with_t90, key=lambda p: p["t90"])
        return best, worst

    plt.figure(figsize=(5, 4))
    _band(pa_stats, pa_imp, "firebrick", "PA")
    _band(ga_stats, ga_imp, "royalblue", r"$\mathrm{GA}_{15}$")

    for stats, color in [(ga_stats, "royalblue"), (pa_stats, "firebrick")]:
        best, worst = _best_worst(stats)
        plt.plot(best["times"], best["succs"], color=color, ls="-.", lw=1.0, alpha=0.7)
        plt.plot(worst["times"], worst["succs"], color=color, ls="--", lw=1.0, alpha=0.7)

    plt.axhline(0.9, ls="--", color="black", lw=2)
    plt.xscale("log")
    plt.xlim(1.5, 550)
    plt.ylim(-0.01, 1.01)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Success Probability", fontsize=12)
    apply_axis_style(plt.gca(), legend=True, legend_kwargs={"loc": "center left", "fontsize": 10})
    finalize_plot(output_path, pdf=True, close=True, bbox_inches=None)


def plot_scatter_pa_vs_ga(pa_dir, ga_dir, pa_files, output_path="fig4_scatter.png",
                          run_time_limit=DEFAULT_RUN_TIME_LIMIT):
    """
    Plot log-log scatter of per-seed average runtimes for PA versus GA.

    Points below the diagonal indicate seeds where GA is faster. Color encodes
    the GA/PA time ratio; timed-out points use a cross marker.
    """
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    def _avg_last_runtime(path, runtime_col, n=10):
        """Average runtime over the last n rows of a result file."""
        try:
            df = pd.read_csv(path, header=None, dtype=str, sep=r"\s+", comment="#")
            if df.empty:
                return None
            vals = pd.to_numeric(df.iloc[-n:, runtime_col], errors="coerce")
            return float(vals.dropna().mean())
        except Exception as e:
            print(f"  Error reading {path}: {e}")
            return None

    pa_times, ga_times = [], []
    for fname in pa_files:
        seed = fname.split("_")[1].replace(".txt", "")
        pa_path = os.path.join(pa_dir, fname)
        ga_path = os.path.join(ga_dir, f"results_{seed}.txt")
        if not os.path.exists(ga_path):
            continue
        pa_t = _avg_last_runtime(pa_path, runtime_col=5)
        ga_t = _avg_last_runtime(ga_path, runtime_col=6)
        if pa_t is not None and ga_t is not None:
            pa_times.append(pa_t)
            ga_times.append(ga_t)

    if not pa_times:
        print("No valid seeds to plot.")
        return

    pa_arr = np.array(pa_times)
    ga_arr = np.array(ga_times)
    ratios = np.clip(ga_arr / pa_arr, 0.2, 5.0)
    sunset = [
        "#a50026", "#d73027", "#f46d43", "#fdae61", "#fee090",
        "#e0f3f8", "#abd9e9", "#74add1", "#4575b4", "#313695",
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("sunset", sunset, N=256).reversed()
    norm = mcolors.TwoSlopeNorm(vmin=0.2, vcenter=1.0, vmax=5.0)
    timeout = (pa_arr >= run_time_limit) | (ga_arr >= run_time_limit)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot2grid((4, 4), (1, 0), rowspan=3, colspan=3)
    ax_top = plt.subplot2grid((4, 4), (0, 0), colspan=3, sharex=ax)
    ax_right = plt.subplot2grid((4, 4), (1, 3), rowspan=3, sharey=ax)

    for axis in [ax, ax_top, ax_right]:
        axis.grid(True, which="both", ls="--", alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, run_time_limit * 1.5)
    ax.set_ylim(1, run_time_limit * 1.5)

    sc = ax.scatter(
        pa_arr[~timeout],
        ga_arr[~timeout],
        c=ratios[~timeout],
        cmap=cmap,
        norm=norm,
        s=80,
        alpha=1.0,
        zorder=3,
        marker="o",
        edgecolors="black",
        linewidths=0.25,
    )
    ax.scatter(
        pa_arr[timeout],
        ga_arr[timeout],
        c=ratios[timeout],
        cmap=cmap,
        norm=norm,
        s=100,
        alpha=1.0,
        zorder=3,
        marker="x",
        linewidths=3.0,
    )

    lim = run_time_limit * 1.5
    ax.plot([1, lim], [1, lim], color="black", ls="--", lw=3)
    ax.set_xlabel("PA time (s)", fontsize=20)
    ax.set_ylabel(r"$\mathrm{GA}_{15}$ time (s)", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.yaxis.set_major_formatter(plt.ScalarFormatter())

    cbar_ax = inset_axes(
        ax,
        width="100%",
        height="5%",
        bbox_to_anchor=(0.4, -0.35, 0.5, 0.6),
        bbox_transform=ax.transAxes,
    )
    cbar = plt.colorbar(sc, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0.2, 1.0, 5.0])
    cbar.set_label(r"$\mathrm{GA}_{15}$/PA time ratio", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.xaxis.set_label_position("bottom")

    log_bins = np.logspace(0, np.log10(run_time_limit * 2), 30)
    ax_top.hist(pa_times, bins=log_bins, color="firebrick", alpha=0.5, edgecolor="none")
    ax_top.hist(pa_times, bins=log_bins, histtype="step", edgecolor="firebrick", lw=3)
    ax_top.set_xscale("log")
    ax_top.tick_params(labelbottom=False)
    ax_top.set_ylabel("Count", fontsize=14)

    ax_right.hist(ga_times, bins=log_bins, orientation="horizontal", color="royalblue", alpha=0.5, edgecolor="none")
    ax_right.hist(ga_times, bins=log_bins, orientation="horizontal", histtype="step", edgecolor="royalblue", lw=3)
    ax_right.set_yscale("log")
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel("Count", fontsize=14)
    finalize_plot(output_path, fig=fig, pdf=True, close=True, bbox_inches=None)


def _sampler_mode_style(variant):
    """Return (label, color) for an SA/PA sampler or GA sampler variant."""
    if variant in _SAMPLER_MODE_LABELS:
        return _SAMPLER_MODE_LABELS[variant], _SAMPLER_MODE_COLORS[variant]
    if variant.startswith("local_global_retrain_"):
        k = variant.rsplit("_", 1)[-1]
        color = _RETRAIN_FALLBACK_COLORS[sum(ord(c) for c in variant) % len(_RETRAIN_FALLBACK_COLORS)]
        return f"local + global + retrain/{k}", color
    return variant, "gray"


def _ordered_variants(keys):
    """Return a stable sampler ordering, appending unknown variants instead of dropping them."""

    def _retrain_sort_key(variant):
        """Sort retrain variants by numeric interval when possible."""
        try:
            return int(variant.rsplit("_", 1)[-1])
        except ValueError:
            return np.inf

    keys = list(keys)
    order = [v for v in ["SA", "PA", "GA", "local", "local_global"] if v in keys]
    retrain_keys = [v for v in keys if v.startswith("local_global_retrain_")]
    retrain_keys.sort(key=_retrain_sort_key)
    known = set(order) | set(retrain_keys)
    extra_keys = sorted(v for v in keys if v not in known)
    return order + retrain_keys + extra_keys


def _sampler_color(key):
    """Return the plotting color associated with a sampler key."""
    if key in SAMPLER_COLORS:
        return SAMPLER_COLORS[key]
    return _sampler_mode_style(key)[1]


def plot_energy_histogram(sampler_results, J, N, T_sample, bins=50,
                          output_path="sampler_energy_hist.png", energy_fn=None):
    """
    Plot the energy-per-spin distribution of the final population for each algorithm.

    A good sampler produces a narrow distribution centered on the thermal mean;
    a trapped sampler is shifted toward higher energies or multiple basins.
    """
    if energy_fn is None:
        energy_fn = _compute_energy_for_plot

    fig, ax = plt.subplots(figsize=(7, 4))
    for alg, res in sampler_results.items():
        with torch.no_grad():
            E = energy_fn(res["population"], J).cpu().numpy() / N
        color = _sampler_color(alg)
        ax.hist(E, bins=bins, density=True, alpha=0.35, color=color, label=alg)
        ax.hist(E, bins=bins, density=True, histtype="step", color=color, lw=2)

    ax.set_xlabel(r"$E / N$", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"Final population energy distribution  ($T_{{\\mathrm{{sample}}}} = {T_sample}$)", fontsize=12)
    apply_axis_style(ax, legend=True, legend_kwargs={"fontsize": 11})
    finalize_plot(output_path, fig=fig, close=True)


def plot_three_algos_three_panels_paper_style(
    results_SA,
    results_PA,
    results_GA,
    time_limits,
    output_path="fig1b_replica.png",
    normalize_by_N=True,
    N=None,
    ymin=0.0,
    ymax=3e-1,
    ground_state_tol=1e-6,
    figure_title=None,
):
    """
    Plot Delta E versus runtime for SA, PA and GA in one panel per displayed time window. These values only control the x-axis and panel titles; they do not stop the algorithms.

    Values with Delta E / N <= ground_state_tol are shown as zero after
    normalization by N. A symlog scale keeps zero visible.
    """
    if normalize_by_N and N is None:
        raise ValueError("If normalize_by_N=True, you must provide N.")

    zero_tol_plot = ground_state_tol if normalize_by_N else ground_state_tol * N
    use_symlog_zero = ymin <= 0

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "font.family": "sans-serif",
    })
    fig, axes = plt.subplots(1, len(time_limits), figsize=(14.2, 3.6), dpi=150, sharey=True)
    if len(time_limits) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#e9e9e9")
    series = [
        (results_SA, "SA", ALGO_COLORS["SA"]),
        (results_PA, "PA", ALGO_COLORS["PA"]),
        (results_GA, "GA", ALGO_COLORS["GA"]),
    ]

    for k, (ax, tl) in enumerate(zip(axes, time_limits)):
        ax.set_facecolor("#e9e9e9")

        for results, label, color in series:
            td = results[k]["trace_data"]
            times = np.asarray(td["time"], dtype=float)
            delta_E = np.asarray(td["delta_E"], dtype=float)

            if len(times) == 0:
                print(f"Skipping {label} at {tl} s: no recorded points.")
                continue

            if normalize_by_N:
                delta_E = delta_E / N

            mask = times <= tl
            times_plot = times[mask]
            delta_E_plot = delta_E[mask]

            after = np.flatnonzero(times > tl)
            if len(after) > 0 and (len(times_plot) == 0 or times_plot[-1] < tl):
                idx_after = after[0]
                times_plot = np.append(times_plot, tl)
                delta_E_plot = np.append(delta_E_plot, delta_E[idx_after])

            if len(times_plot) == 0:
                print(f"Skipping {label} at {tl} s: no points within the displayed time window.")
                continue

            delta_E_plot = np.where(delta_E_plot <= zero_tol_plot, 0.0, delta_E_plot)
            if not use_symlog_zero:
                delta_E_plot = np.maximum(delta_E_plot, ymin)

            ax.plot(times_plot, delta_E_plot, color=color, linewidth=2.8, label=label)

        if use_symlog_zero:
            ax.set_yscale("symlog", linthresh=zero_tol_plot, linscale=0.5)
        else:
            ax.set_yscale("log")
        ax.set_xlim(0, tl)
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"Time window: {int(tl)} s", pad=8)
        ax.set_xlabel("Runtime (s)")

        if k == 0:
            ax.set_ylabel(r"$\Delta E / N$" if normalize_by_N else r"$\Delta E$")

        if tl == 15:
            xticks = np.arange(0, 15.1, 2.5)
        elif tl == 50:
            xticks = np.arange(0, 50.1, 10)
        elif tl == 150:
            xticks = np.arange(0, 150.1, 25)
        else:
            xticks = np.linspace(0, tl, 6)
        ax.set_xticks(xticks)

        if not use_symlog_zero:
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

        ax.grid(True, which="major", color="#9e9e9e", alpha=0.75, linewidth=1.0)
        ax.grid(True, which="minor", color="#bdbdbd", alpha=0.35, linewidth=0.6)

        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
            spine.set_color("#4d4d4d")

        leg = ax.legend(loc="lower left", frameon=True, fancybox=True, framealpha=0.9)
        leg.get_frame().set_edgecolor("#bfbfbf")
        leg.get_frame().set_linewidth(1.0)

    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=15, y=1.06)

    axes[0].text(-0.23, 1.13, "(b)", transform=axes[0].transAxes, fontsize=20, fontweight="bold", va="top", ha="left")
    finalize_plot(output_path, fig=fig, tight_kwargs={"w_pad": 1.8})


def plot_ga_overlap_trace(sampler_mode_results, output_path="ga_sampler_modes_overlap_trace.png"):
    """Plot mean two-halves spin overlap q(t) for GA sampler variants."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    for variant in _ordered_variants(sampler_mode_results.keys()):
        data = sampler_mode_results[variant]
        label, color = _sampler_mode_style(variant)
        ax.plot(data["time"], data["overlap_mean"], color=color, linewidth=2.0, label=label)
        ax.fill_between(
            data["time"],
            data["overlap_mean"] - data["overlap_std"],
            data["overlap_mean"] + data["overlap_std"],
            color=color,
            alpha=0.15,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Overlap $q$ (two halves of the population)")
    ax.set_title("Population overlap vs. sampling time")
    apply_axis_style(ax, legend=True)
    finalize_plot(output_path, fig=fig)


def plot_acceptance_rate_retrain_effect(sampler_mode_results, retrain_intervals=None, seed=None,
                                        output_path="ga_acceptance_vs_retrain.png"):
    """
    Plot MADE acceptance rate versus update index for one disorder instance.

    The figure compares the frozen-model GA sampler with retraining intervals.
    """
    frozen_acc = np.asarray(sampler_mode_results["local_global"]["acc_rate"], dtype=float)
    n_updates = frozen_acc.shape[0]
    steps = np.arange(1, n_updates + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    label, color = _sampler_mode_style("local_global")
    ax.plot(steps, frozen_acc, color=color, linewidth=2.0, label=label)

    if retrain_intervals is None:
        retrain_intervals = sorted(
            int(v.rsplit("_", 1)[-1])
            for v in sampler_mode_results
            if v.startswith("local_global_retrain_")
        )

    for k in sorted(retrain_intervals):
        v = _retrain_variant_name(k)
        if v not in sampler_mode_results:
            continue
        label, color = _sampler_mode_style(v)
        acc = np.asarray(sampler_mode_results[v]["acc_rate"], dtype=float)
        ax.plot(steps, acc, color=color, linewidth=2.0, label=label)
        for x in range(k, n_updates + 1, k):
            ax.axvline(x, color=color, linestyle="--", linewidth=0.6, alpha=0.4)

    ax.set_xlabel("Update")
    ax.set_ylabel("MADE acceptance rate")
    title = "Acceptance rate for one disorder instance"
    if seed is not None:
        title += f" (seed {seed})"
    ax.set_title(title)
    apply_axis_style(ax, legend=True)
    finalize_plot(output_path, fig=fig)


def plot_pq_histograms(pq_data, T_sample, seed, exact_reference=None,
                       output_path="overlap_Pq_T077_SA_PA_GA_reference.png", bins=60):
    """
    Plot P(q) panels for sampler populations.

    The exact/reference P(q) is drawn as a grey background in every panel when
    available.
    """
    variants = _ordered_variants(pq_data.keys())
    fig, axes = plt.subplots(1, len(variants), figsize=(4.0 * len(variants), 4), dpi=150, sharey=True)
    if len(variants) == 1:
        axes = [axes]

    ref_label = None
    ref_q = None
    if exact_reference is not None:
        ref_q = exact_reference["q"]
        ref_label = f"exact/reference T={exact_reference['T']:.2f}"

    for ax, variant in zip(axes, variants):
        if ref_q is not None:
            ax.hist(ref_q, bins=bins, range=(-1, 1), density=True, color="0.75", alpha=0.45, label=ref_label)
            ax.hist(ref_q, bins=bins, range=(-1, 1), density=True, histtype="step", color="0.35", linewidth=1.5)

        q = pq_data[variant]
        label, color = _sampler_mode_style(variant)
        ax.hist(q, bins=bins, range=(-1, 1), density=True, color=color, alpha=0.35, label=label)
        ax.hist(q, bins=bins, range=(-1, 1), density=True, histtype="step", color=color, linewidth=2.0)
        ax.set_title(label)
        ax.set_xlabel("$q$")
        ax.set_xlim(-1, 1)
        apply_axis_style(ax, legend=True, legend_kwargs={"fontsize": 8, "loc": "upper left"})

    axes[0].set_ylabel("$P(q)$")
    fig.suptitle(f"Overlap distribution, seed {seed}, T = {T_sample:.2f}")
    finalize_plot(output_path, fig=fig)


def plot_energy_comparison(
    results_SA,
    results_PA,
    results_GA,
    output_path="SK_energy_vs_time.png",
    normalize_by_N=True,
    N=None,
):
    """
    Plot the minimum energy found versus runtime for SA, PA and GA.
    """

    if normalize_by_N and N is None:
        raise ValueError(
            "If normalize_by_N=True, N must be provided."
        )

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "font.family": "sans-serif",
    })

    fig, ax = plt.subplots(
        figsize=(6.4, 4.4),
        dpi=150,
    )

    fig.patch.set_facecolor("#e9e9e9")
    ax.set_facecolor("#e9e9e9")

    series = [
        (results_SA, "SA", ALGO_COLORS["SA"]),
        (results_PA, "PA", ALGO_COLORS["PA"]),
        (results_GA, "GA", ALGO_COLORS["GA"]),
    ]

    max_time = 0.0

    for results, label, color in series:

        td = results["trace_data"]

        times = np.asarray(
            td["time"],
            dtype=float,
        )

        energies = np.asarray(
            td["min_energy"],
            dtype=float,
        )

        if normalize_by_N:
            energies = energies / N

        ax.plot(
            times,
            energies,
            color=color,
            linewidth=2.8,
            label=label,
        )

        max_time = max(
            max_time,
            float(times[-1]),
        )

    ax.set_xlim(0, max_time)

    ax.set_xlabel("Runtime (s)")

    ax.set_ylabel(
        r"$E_{\min}/N$"
        if normalize_by_N
        else r"$E_{\min}$"
    )

    ax.set_title(
        "SK model -- minimum energy vs runtime",
        pad=8,
    )

    ax.grid(
        True,
        which="major",
        color="#9e9e9e",
        alpha=0.75,
        linewidth=1.0,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#4d4d4d")

    leg = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
    )

    leg.get_frame().set_edgecolor("#bfbfbf")
    leg.get_frame().set_linewidth(1.0)

    finalize_plot(
        output_path,
        fig=fig,
    )
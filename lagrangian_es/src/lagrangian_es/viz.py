"""Figures.

Plant-agnostic: trajectories are read through `system.task_position`, never
through a raw state key, so the same plotting code serves the quadrotor and the
arm.  `tests/test_lint_seam.py` enforces that.
"""
from __future__ import annotations

import pathlib
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ACCENT = "#0B7A85"
WARN = "#C0490F"
TARGET = "#5B4BC4"
MUTED = "#7A8896"
CYCLE = [ACCENT, WARN, TARGET, "#1F7A45", "#B3261E", "#8A6D0B"]


def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, loc="left", pad=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.22, linewidth=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def _save(fig, out: Optional[str]):
    fig.tight_layout()
    if out:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out
    return fig


# --------------------------------------------------------------------------- #
def learning_curves(histories: Dict[str, List[dict]], out: Optional[str] = None,
                    keys: Sequence[str] = ("fitness_elite", "crash_rate",
                                           "legB_err", "sigma"),
                    targets: Optional[Dict[str, float]] = None):
    """One panel per tracked quantity, one line per run."""
    targets = targets or {"legB_err": 0.15, "crash_rate": 0.0}
    keys = [k for k in keys if any(k in h[0] for h in histories.values() if h)]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 2.9))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, keys):
        for i, (label, hist) in enumerate(histories.items()):
            ys = [h[key] for h in hist if key in h]
            xs = [h["gen"] for h in hist if key in h]
            ax.plot(xs, ys, lw=1.7, color=CYCLE[i % len(CYCLE)], label=label)
        if key in targets:
            ax.axhline(targets[key], ls="--", lw=1, color=MUTED, zorder=0)
        _style(ax, "generation", key.replace("_", " "))
    axes[0].legend(fontsize=8, frameon=False)
    return _save(fig, out)


def metric_panel(history: List[dict], out: Optional[str] = None):
    """The three diagnostics from section 7 that turn a silent bug into a visible one.

    * `metric_dist_I` near zero means the whitened arm has quietly become the
      isotropic baseline -- you would be comparing an arm to itself.
    * `metric_sat` rising means jacrev is seeing saturated clamps, so G is
      rank-deficient in exactly the directions that matter and the ridge hides it.
    * `th_timescale_sep` falling means the attitude loop can no longer keep up
      with the potential, which shows up later as a rising crash rate.
    """
    rows = [h for h in history if "metric_cond" in h]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    if rows:
        g = [r["gen"] for r in rows]
        axes[0].semilogy(g, [r["metric_cond"] for r in rows], color=ACCENT, lw=1.7)
        _style(axes[0], "generation", "cond(G)", "Metric anisotropy")
        axes[1].plot(g, [r["metric_dist_I"] for r in rows], color=ACCENT, lw=1.7)
        axes[1].axhline(0, ls="--", lw=1, color=WARN)
        _style(axes[1], "generation", r"$\|P-I\|_F$", "Distance from the baseline")
        axes[2].plot(g, [r["metric_sat"] for r in rows], color=WARN, lw=1.7,
                     label="actuator saturation")
    sep = [(h["gen"], h["th_timescale_sep"]) for h in history if "th_timescale_sep" in h]
    if sep:
        ax2 = axes[2].twinx()
        ax2.plot(*zip(*sep), color=TARGET, lw=1.7, ls="--")
        ax2.set_ylabel(r"$\omega_{att}/\omega_{pos}$", fontsize=9, color=TARGET)
        ax2.tick_params(labelsize=8, colors=TARGET)
    _style(axes[2], "generation", "saturated fraction", "Saturation & timescales")
    return _save(fig, out)


def trajectories(system, traces, goals, out: Optional[str] = None,
                 labels: Optional[Sequence[str]] = None, max_eps: int = 6):
    """Task-space paths.  Reads positions through the system's accessor only."""
    labels = labels or [f"run {i}" for i in range(len(traces))]
    n = min(max_eps, traces[0].goals.shape[1])
    d = system.task_dim
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(4.0 * ncol, 3.4 * nrow))
    for b in range(n):
        if d >= 3:
            ax = fig.add_subplot(nrow, ncol, b + 1, projection="3d")
        else:
            ax = fig.add_subplot(nrow, ncol, b + 1)
        for i, tr in enumerate(traces):
            p = system.task_position(tr.states).detach().cpu().numpy()[:, b]
            c = CYCLE[i % len(CYCLE)]
            if d >= 3:
                ax.plot(p[:, 0], p[:, 1], p[:, 2], lw=1.4, color=c, label=labels[i])
            else:
                ax.plot(p[:, 0], p[:, 1], lw=1.4, color=c, label=labels[i])
        gb = goals[b].detach().cpu().numpy()
        for j, g in enumerate(gb):
            if d >= 3:
                ax.scatter(*g[:3], s=42, color=TARGET, marker="o" if j == 0 else "^",
                           depthshade=False)
            else:
                ax.scatter(g[0], g[1], s=42, color=TARGET, marker="o" if j == 0 else "^")
        ax.set_title(f"episode {b}", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        if b == 0:
            ax.legend(fontsize=8, frameon=False)
    return _save(fig, out)


def ablation_grid(cells: Dict[str, Dict[str, list]], out: Optional[str] = None,
                  metric: str = "legB_err", target: Optional[float] = 0.15):
    """The 2x2, as paired per-seed points plus the cell mean.

    Individual seeds are drawn because single-seed results are not evidence and a
    bar chart hides exactly that.
    """
    names = list(cells)
    fig, ax = plt.subplots(figsize=(1.9 * len(names) + 2.2, 3.6))
    for i, name in enumerate(names):
        vals = np.asarray(cells[name][metric], dtype=float)
        jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.22
        ax.scatter(np.full_like(vals, i) + jitter, vals, s=34,
                   color=CYCLE[i % len(CYCLE)], alpha=0.75, zorder=3)
        ax.hlines(vals.mean(), i - 0.3, i + 0.3, color="black", lw=2, zorder=4)
        ax.text(i, vals.mean(), f"  {vals.mean():.3f}", fontsize=8, va="bottom")
    if target is not None:
        ax.axhline(target, ls="--", lw=1, color=MUTED, zorder=0)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" x ", "\n") for n in names], fontsize=8)
    _style(ax, "", metric.replace("_", " "), "Ablation cells (one point per seed)")
    return _save(fig, out)


def paired_differences(diffs: Dict[str, list], out: Optional[str] = None,
                       metric: str = "legB_err"):
    """Per-seed paired differences -- the comparison the ablation actually rests on."""
    fig, ax = plt.subplots(figsize=(2.0 * len(diffs) + 2.0, 3.4))
    for i, (name, vals) in enumerate(diffs.items()):
        v = np.asarray(vals, dtype=float)
        ax.scatter(np.full_like(v, i), v, s=34, color=CYCLE[i % len(CYCLE)], zorder=3)
        ax.hlines(v.mean(), i - 0.3, i + 0.3, color="black", lw=2, zorder=4)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(diffs)))
    ax.set_xticklabels(list(diffs), fontsize=8)
    _style(ax, "", f"paired delta {metric}  (negative = whitening helps)",
           "Whitened minus isotropic, per seed")
    return _save(fig, out)

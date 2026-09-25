"""Unified visualization of detection and graph source localization."""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genpareto
from .pipeline import PipelineResult
from .evt_analysis import EVTAnalyzer, GEVFit, GPDFit


def plot_time_series_with_extremes(
    time_series: np.ndarray,
    extreme_mask: np.ndarray,
    save_path: str | Path | None = None,
):
    """Plot a one-dimensional series and mark injected extreme points."""
    values = np.asarray(time_series)
    mask = np.asarray(extreme_mask, dtype=bool)
    if values.ndim != 1 or mask.shape != values.shape:
        raise ValueError("time_series and extreme_mask must be one-dimensional and aligned")
    fig, ax = plt.subplots(figsize=(12, 4))
    time = np.arange(len(values))
    ax.plot(time, values, color="#2a78d6", lw=1, label="временной ряд")
    ax.scatter(time[mask], values[mask], color="#eb6834", s=18, label="экстремальные события")
    ax.set(xlabel="время", ylabel="значение", title="Синтетический ряд и экстремальные события")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_gev_distribution(
    fit: GEVFit,
    save_path: str | Path | None = None,
):
    """Plot block maxima with the fitted GEV density."""
    diagnostics = EVTAnalyzer().diagnostics(fit)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(fit.block_maxima, bins="auto", density=True, alpha=0.35, label="максимумы блоков")
    ax.plot(
        diagnostics["density_grid"],
        diagnostics["fitted_density"],
        color="#eb6834",
        lw=2,
        label="подогнанная плотность GEV",
    )
    ax.set(xlabel="максимум блока", ylabel="плотность", title="Распределение максимумов блоков")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_return_levels(
    levels,
    save_path: str | Path | None = None,
):
    """Plot GEV return levels and their bootstrap confidence intervals."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    periods = np.asarray(levels["return_period"], dtype=float)
    values = np.asarray(levels["return_level"], dtype=float)
    ax.plot(periods, values, "o-", color="#2a78d6", label="возвратный уровень")
    ax.fill_between(
        periods,
        np.asarray(levels["ci_lower"], dtype=float),
        np.asarray(levels["ci_upper"], dtype=float),
        color="#2a78d6",
        alpha=0.18,
        label="доверительный интервал",
    )
    ax.set_xscale("log")
    ax.set(xlabel="период возврата (блоки)", ylabel="уровень", title="Возвратные уровни GEV")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_gpd_exceedances(
    fit: GPDFit,
    save_path: str | Path | None = None,
):
    """Plot empirical GPD exceedances and their fitted survival curve."""
    exceedances = np.sort(np.asarray(fit.exceedances, dtype=float))
    empirical_survival = (len(exceedances) - np.arange(len(exceedances))) / len(exceedances)
    grid = np.linspace(0, float(exceedances.max()), 200)
    fitted_survival = genpareto.sf(grid, fit.xi, loc=0, scale=fit.sigma_u)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.step(exceedances, empirical_survival, where="post", label="эмпирическая хвостовая вероятность")
    ax.plot(grid, fitted_survival, color="#eb6834", lw=2, label="подогнанная GPD")
    ax.set_yscale("log")
    ax.set(xlabel="превышение над порогом", ylabel="P(Y > y)", title="Превышения над порогом POT")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_mean_excess(
    time_series: np.ndarray,
    thresholds: np.ndarray,
    save_path: str | Path | None = None,
):
    """Plot the empirical mean excess function over candidate thresholds."""
    values = np.asarray(time_series, dtype=float)
    levels = np.asarray(thresholds, dtype=float)
    if values.ndim != 1 or levels.ndim != 1 or not len(levels):
        raise ValueError("time_series and thresholds must be one-dimensional")
    mean_excess = np.array(
        [np.mean(values[values > level] - level) if np.any(values > level) else np.nan for level in levels]
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(levels, mean_excess, "o-", color="#2a78d6", ms=3, label="среднее превышение")
    ax.set(xlabel="порог", ylabel="E[X-u | X>u]", title="Mean Excess Plot")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150, bbox_inches="tight")
    return fig, ax


def compare_evt_vs_autoregression(
    time_series: np.ndarray,
    truth: np.ndarray,
    evt_mask: np.ndarray,
    ar_mask: np.ndarray,
    save_path: str | Path | None = None,
):
    """Compare injected events with POT and AR-residual detections."""
    values = np.asarray(time_series, dtype=float)
    truth_mask = np.asarray(truth, dtype=bool)
    evt = np.asarray(evt_mask, dtype=bool)
    ar = np.asarray(ar_mask, dtype=bool)
    if values.ndim != 1 or any(mask.shape != values.shape for mask in (truth_mask, evt, ar)):
        raise ValueError("time_series and all masks must be one-dimensional and aligned")

    time = np.arange(len(values))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time, values, color="#52514e", lw=0.8, label="временной ряд")
    axes[0].scatter(time[truth_mask], values[truth_mask], color="#eb6834", s=18, label="истинные события")
    axes[0].set_title("Сигнал и истинные экстремальные события")
    axes[1].plot(time, values, color="#b9b5ac", lw=0.7)
    axes[1].scatter(time[evt], values[evt], color="#2a78d6", s=20, label="детекция EVT")
    axes[1].set_title("POT/GPD")
    axes[2].plot(time, values, color="#b9b5ac", lw=0.7)
    axes[2].scatter(time[ar], values[ar], color="#1baf7a", s=20, label="AR-остатки")
    axes[2].set_title("Авторегрессионная baseline-модель")
    axes[2].set_xlabel("время")
    for axis in axes:
        axis.set_ylabel("значение")
        axis.legend(loc="upper right")
    fig.tight_layout()
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150, bbox_inches="tight")
    return fig, axes


def plot_pipeline(result: PipelineResult, save_path: str | Path | None = None):
    """Show signal/EVT moment and the graph/GAT source ranking in one figure."""
    series, det, cv = result.example, result.detection, result.cv
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.plot(det.indicator, color="#2a78d6", lw=1, label="графовый индикатор")
    ax.axhline(det.threshold, color="#eb6834", ls="--", label="POT/GPD-порог")
    ax.axvline(series.event_onset, color="#52514e", ls=":", label="истинный момент")
    if det.detected_at is not None:
        ax.axvline(det.detected_at, color="#1baf7a", label="детекция EVT")
    ax.set(xlabel="время", ylabel="индикатор", title="1. Детекция момента")
    ax.legend(fontsize=8)
    ax = axes[1]
    n = len(series.groups)
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.column_stack([np.cos(angle), np.sin(angle)])
    for left, right in np.argwhere(np.triu(series.adjacency, 1)):
        ax.plot(
            xy[[left, right], 0], xy[[left, right], 1], color="#dcdad4", lw=1, zorder=1
        )
    probabilities = cv.probabilities[0]
    nodes = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=probabilities,
        cmap="viridis",
        s=150,
        edgecolor="#52514e",
        zorder=2,
    )
    ax.scatter(
        *xy[series.source_node],
        s=330,
        facecolors="none",
        edgecolors="#eb6834",
        lw=2,
        label="истинный источник",
        zorder=3,
    )
    for node, (x, y) in enumerate(xy):
        ax.text(x, y, str(node), ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(nodes, ax=ax, label="OOF-вероятность GAT")
    ax.legend(fontsize=8)
    ax.axis("off")
    ax.set_title("2. Локализация источника на графе")
    fig.text(
        0.5,
        0.01,
        "GAT обучает веса внимания к соседям: ранний сильный отклик и согласованность по рёбрам повышают ранг узла-источника. CV разделяет независимые события.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150, bbox_inches="tight")
    return fig, axes

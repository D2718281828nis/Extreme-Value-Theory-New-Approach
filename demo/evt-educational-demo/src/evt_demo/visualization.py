"""Unified visualization of detection and graph source localization."""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .pipeline import PipelineResult


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

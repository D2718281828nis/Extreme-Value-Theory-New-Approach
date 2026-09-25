"""Unified visualization of detection and graph source localization."""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .pipeline import PipelineResult
from .evaluation import DetectionComparison
from .graph_model import GraphSample, TrainingTrace


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


def plot_detection_quality(
    comparison: DetectionComparison, save_path: str | Path | None = None
):
    """Plot precision, recall, and F1 next to the existing alarm visualization."""
    metrics = np.vstack([comparison.precision, comparison.recall, comparison.f1]).T
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    width = 0.34
    for index, name in enumerate(comparison.names):
        ax.bar(x + (index - 0.5) * width, metrics[index], width, label=name)
    ax.set_xticks(x, ["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("качество точечной детекции")
    ax.set_title("EVT и AR: качество относительно известного интервала события")
    ax.legend()
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=8)
    fig.tight_layout()
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150, bbox_inches="tight")
    return fig, ax


def plot_gat_training(
    sample: GraphSample, trace: TrainingTrace, save_path: str | Path | None = None
):
    """Visualize optimization and the learned attention weights on graph edges."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(np.arange(1, len(trace.losses) + 1), trace.losses, color="#2a78d6")
    axes[0].set(xlabel="эпоха", ylabel="cross-entropy", title="Обучение GAT")
    n_nodes = len(sample.features)
    angle = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    xy = np.column_stack([np.cos(angle), np.sin(angle)])
    weights = trace.attention_weights
    normalized = (weights - weights.min()) / max(float(np.ptp(weights)), 1e-8)
    for edge, weight in zip(trace.attention_edge_index.T, normalized):
        left, right = map(int, edge)
        if left == right:
            continue
        axes[1].plot(
            xy[[left, right], 0],
            xy[[left, right], 1],
            color=plt.cm.Blues(0.2 + 0.8 * weight),
            lw=0.5 + 3 * weight,
            alpha=0.8,
        )
    points = axes[1].scatter(
        xy[:, 0],
        xy[:, 1],
        c=trace.probabilities,
        cmap="viridis",
        s=150,
        edgecolor="#52514e",
        zorder=3,
    )
    axes[1].scatter(
        *xy[sample.source_node],
        s=330,
        facecolors="none",
        edgecolors="#eb6834",
        lw=2,
        label="источник",
    )
    for node, (x, y) in enumerate(xy):
        axes[1].text(
            x, y, str(node), ha="center", va="center", fontsize=7, color="white"
        )
    fig.colorbar(points, ax=axes[1], label="вероятность источника")
    axes[1].axis("off")
    axes[1].set_title("GAT: внимание вдоль рёбер")
    axes[1].legend()
    fig.tight_layout()
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150, bbox_inches="tight")
    return fig, axes


def plot_source_search(result: PipelineResult, save_path: str | Path | None = None):
    """Show event propagation through channels and the final OOF source ranking."""
    series = result.example
    start, stop = series.event_onset - 30, min(
        len(series.values), int(series.arrival_times.max()) + 35
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    image = axes[0].imshow(
        series.values[start:stop].T,
        aspect="auto",
        cmap="coolwarm",
        extent=[start, stop, len(series.groups) - 0.5, -0.5],
    )
    axes[0].plot(
        series.arrival_times,
        np.arange(len(series.groups)),
        color="black",
        lw=1.2,
        label="истинный фронт",
    )
    axes[0].axvline(
        result.detection.detected_at, color="#1baf7a", ls="--", label="EVT-детекция"
    )
    axes[0].set(
        xlabel="время", ylabel="узел", title="Распространение события по каналам"
    )
    axes[0].legend(fontsize=8)
    fig.colorbar(image, ax=axes[0], label="значение ряда")
    probabilities = result.cv.probabilities[0]
    axes[1].bar(np.arange(len(probabilities)), probabilities, color="#2a78d6")
    axes[1].axvline(
        series.source_node, color="#eb6834", lw=2, label="истинный источник"
    )
    axes[1].set(
        xlabel="узел", ylabel="OOF-вероятность GAT", title="Ранжирование источника"
    )
    axes[1].legend()
    fig.tight_layout()
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=150, bbox_inches="tight")
    return fig, axes

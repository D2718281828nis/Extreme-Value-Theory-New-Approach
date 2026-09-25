"""Synthetic multichannel graph time series with a propagating extreme event."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class GraphTimeSeries:
    values: np.ndarray  # [time, nodes]
    adjacency: np.ndarray  # [nodes, nodes]
    groups: np.ndarray  # structural node groups
    source_node: int
    event_onset: int
    arrival_times: np.ndarray


def generate_time_series_with_extreme_events(
    n_points: int = 2_000,
    phi: float = 0.8,
    sigma: float = 1.0,
    n_extreme_events: int = 3,
    extreme_magnitude: float = 8.0,
    extreme_duration: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a reproducible AR(1) series with injected extreme events."""
    if n_points < 100 or not -1 < phi < 1 or sigma <= 0:
        raise ValueError("Require n_points>=100, abs(phi)<1 and sigma>0")
    if n_extreme_events < 1 or extreme_magnitude <= 0 or extreme_duration < 1:
        raise ValueError("Require positive event count, magnitude and duration")
    warmup = max(20, int(5 / (1 - abs(phi))))
    available = n_points - warmup
    if n_extreme_events * extreme_duration > available:
        raise ValueError("Events do not fit without overlap after warmup")

    rng = np.random.default_rng(seed)
    series = np.zeros(n_points + warmup, dtype=float)
    unconditional_std = sigma / np.sqrt(1 - phi**2)
    series[0] = rng.normal(0, unconditional_std)
    innovations = rng.normal(0, sigma, len(series))
    for time in range(1, len(series)):
        series[time] = phi * series[time - 1] + innovations[time]
    series = series[warmup:]

    starts = np.sort(rng.choice(available - extreme_duration + 1, n_extreme_events, replace=False))
    mask = np.zeros(n_points, dtype=bool)
    for start in starts:
        stop = start + extreme_duration
        series[start:stop] += extreme_magnitude * unconditional_std
        mask[start:stop] = True
    return series, mask


def save_to_csv(
    time_series: np.ndarray, extreme_mask: np.ndarray, path: str | Path
) -> None:
    """Save a generated series and its injected-event mask as CSV."""
    values = np.asarray(time_series, dtype=float)
    mask = np.asarray(extreme_mask, dtype=bool)
    if values.ndim != 1 or mask.shape != values.shape:
        raise ValueError("time_series and extreme_mask must be one-dimensional and aligned")
    frame = np.column_stack((np.arange(len(values)), values, mask.astype(int)))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(target, frame, delimiter=",", header="timestamp,value,is_extreme", comments="", fmt=["%d", "%.17g", "%d"])


def _graph_distances(adjacency: np.ndarray, source: int) -> np.ndarray:
    distance = np.full(len(adjacency), np.inf)
    distance[source] = 0
    frontier = [source]
    while frontier:
        node = frontier.pop(0)
        for neighbour in np.flatnonzero(adjacency[node]):
            if not np.isfinite(distance[neighbour]):
                distance[neighbour] = distance[node] + 1
                frontier.append(int(neighbour))
    return distance


def generate_graph_time_series(
    n_points: int = 1_200,
    n_nodes: int = 18,
    n_groups: int = 3,
    phi: float = 0.72,
    sigma: float = 0.7,
    event_onset: int = 850,
    source_node: int | None = None,
    magnitude: float = 8.0,
    seed: int = 42,
) -> GraphTimeSeries:
    """Generate correlated AR(1) channels and an event spreading over graph edges."""
    if n_points < 300 or n_nodes < 6 or not 2 <= n_groups <= n_nodes:
        raise ValueError("Need n_points>=300, n_nodes>=6 and 2<=n_groups<=n_nodes")
    if not -1 < phi < 1 or sigma <= 0 or magnitude <= 0:
        raise ValueError("Require abs(phi)<1 and positive sigma/magnitude")
    if not n_points // 2 <= event_onset < n_points - 80:
        raise ValueError("event_onset must leave baseline and event windows")
    rng = np.random.default_rng(seed)
    groups = np.arange(n_nodes) * n_groups // n_nodes
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for group in range(n_groups):
        ids = np.flatnonzero(groups == group)
        for left, right in zip(ids[:-1], ids[1:]):
            adjacency[left, right] = adjacency[right, left] = 1
    for group in range(n_groups - 1):
        left = np.flatnonzero(groups == group)[-1]
        right = np.flatnonzero(groups == group + 1)[0]
        adjacency[left, right] = adjacency[right, left] = 1
    source = int(rng.integers(n_nodes) if source_node is None else source_node)
    if not 0 <= source < n_nodes:
        raise ValueError("source_node is outside graph")
    innovations = rng.normal(0, sigma, (n_points, n_nodes))
    shared = rng.normal(0, sigma * 0.35, (n_points, n_groups))
    values = np.zeros_like(innovations)
    values[0] = rng.normal(0, sigma / np.sqrt(1 - phi**2), n_nodes)
    for time in range(1, n_points):
        values[time] = phi * values[time - 1] + innovations[time] + shared[time, groups]
    distances = _graph_distances(adjacency, source).astype(int)
    arrival = event_onset + 3 * distances
    width = 18
    for node in range(n_nodes):
        start = arrival[node]
        stop = min(start + width, n_points)
        pulse = np.sin(np.linspace(0, np.pi, stop - start))
        values[start:stop, node] += magnitude * np.exp(-0.22 * distances[node]) * pulse
    return GraphTimeSeries(values, adjacency, groups, source, event_onset, arrival)

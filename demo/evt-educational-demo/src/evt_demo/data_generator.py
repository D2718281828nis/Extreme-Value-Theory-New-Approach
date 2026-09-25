"""Synthetic multichannel graph time series with a propagating extreme event."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GraphTimeSeries:
    values: np.ndarray  # [time, nodes]
    adjacency: np.ndarray  # [nodes, nodes]
    groups: np.ndarray  # structural node groups
    source_node: int
    event_onset: int
    arrival_times: np.ndarray


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

import numpy as np
from evt_demo import generate_time_series_with_extreme_events
from evt_demo.data_generator import generate_graph_time_series
from evt_demo.visualization import plot_time_series_with_extremes


def test_reproducible_graph_event():
    left = generate_graph_time_series(seed=3)
    right = generate_graph_time_series(seed=3)
    assert np.array_equal(left.values, right.values)
    assert np.array_equal(left.adjacency, left.adjacency.T)
    assert left.arrival_times[left.source_node] == left.event_onset
    assert np.all(np.diff(left.arrival_times[np.argsort(left.arrival_times)]) >= 0)


def test_legacy_notebook_imports(tmp_path):
    series, mask = generate_time_series_with_extreme_events(seed=42)
    assert series.shape == mask.shape == (2_000,)
    figure, _ = plot_time_series_with_extremes(
        series, mask, save_path=tmp_path / "intro.png"
    )
    assert (tmp_path / "intro.png").exists()
    figure.clear()

import numpy as np
from evt_demo.data_generator import generate_graph_time_series


def test_reproducible_graph_event():
    left = generate_graph_time_series(seed=3)
    right = generate_graph_time_series(seed=3)
    assert np.array_equal(left.values, right.values)
    assert np.array_equal(left.adjacency, left.adjacency.T)
    assert left.arrival_times[left.source_node] == left.event_onset
    assert np.all(np.diff(left.arrival_times[np.argsort(left.arrival_times)]) >= 0)

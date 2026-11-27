import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ecg_incart_pipeline as E


def word(typ, dt=0):
    return ((typ << 10) | dt).to_bytes(2, "little")


def test_mit_annotation_reader_handles_pdp11_skip(tmp_path):
    absolute = 2000
    data = bytearray(word(1, 12) + word(5, 200))
    data += word(59) + (absolute >> 16).to_bytes(2, "little") + (absolute & 0xFFFF).to_bytes(2, "little")
    data += word(5, 80) + word(5, 80) + word(28, 80) + word(0)
    path = tmp_path / "sample.atr"; path.write_bytes(data)
    ann = E.read_annotations(path)
    assert [(t, typ) for t, typ, _ in ann] == [(12, 1), (212, 5), (2080, 5), (2160, 5), (2240, 28)]


def test_vt_episode_rule_requires_run_duration_and_rate():
    fs = 100
    ann = [(t, 5 if i < 6 else 1, "") for i, t in enumerate(range(0, 600, 50))]
    ann += [(1000 + 50 * i, 5, "") for i in range(4)]
    ann += [(2000 + 100 * i, 5, "") for i in range(8)]
    episodes = E.vt_episodes(ann, fs=fs)
    assert len(episodes) == 1
    assert episodes[0]["onset_sample"] == 0 and episodes[0]["n_beats"] == 6
    assert episodes[0]["mean_hr"] == 120


def test_ecg_detector_spec_is_available():
    spec = E.SPEC
    assert spec["background"] == 400 and spec["blind_pot"]["r"] == 3
    assert spec["blind_gev"]["block"] == 10


def _synthetic_ecg():
    n = 4 * E.FS; t = np.arange(n); y = np.zeros((n, 12)); peaks = np.array([500, 800])
    for p in peaks:
        p_wave = .12 * np.exp(-.5 * ((t - (p - 45)) / 4) ** 2)
        qrs = 1.2 * np.exp(-.5 * ((t - p) / 1.3) ** 2)
        y[:, 1] += p_wave + qrs
        y[:, 0] += qrs; y[:, 5] += qrs
        y[p + 17:p + 23, 1] += .4
    return y, peaks


def test_beat_parameters_recovers_pr_st_and_cross_lead_axis():
    y, peaks = _synthetic_ecg()
    beats = E.beat_parameters(y, peaks)
    assert np.isclose(beats["rr"][-1], 300 / E.FS)
    assert .12 < beats["pr"][0] < .24
    assert beats["st"][0] > .1
    assert np.allclose(beats["qrs_axis"], 45, atol=1e-6)
    assert (beats["pr_quality"] > E.PR_QUALITY_MIN).all()


def test_derived_output_has_exactly_four_one_second_series():
    y, peaks = _synthetic_ecg()
    series, beats, aux = E.derived_series(y, peaks)
    assert series.shape == (4, 4)
    assert np.isfinite(series).all()
    assert aux["pr_measured"].shape == (4,)
    assert np.isclose(series[1, 0], 300 / E.FS)


def test_axis_uses_circular_deviation_and_state_coordinates():
    assert np.isclose(E.circular_deviation(-179, 179), 2)
    series = np.array([[.7, .15, 0, 179], [.7, .15, 0, -181]])
    state = E.state_matrix(series)
    assert np.allclose(state[0], state[1], atol=1e-12)
    centered = np.full((E.B + 1, 4), [.7, .15, 0, 179.0])
    centered[-1, 3] = -179
    X = E.analysis_series(centered)
    assert np.isclose(X[-1, 3] - X[-2, 3], 2)


def test_graph_is_defined_on_four_derived_parameters():
    assert E.PARAMETERS == ["RR", "PR", "ST", "QRS_axis"]
    assert E.physiological_edges() == [(0, 1), (2, 3)]


def test_common_result_panel_handles_empty_ecg_cascade():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import case_data as D
    import case_figures as F
    c = D.Case("D", "ECG", "s")
    c.groups = np.array(E.PARAMETERS); c.onset = np.full(4, np.nan)
    c.source = c.source_mean = "RR"; c.view = (350, 580); c.det_t = 450
    fig, ax = plt.subplots()
    F.p_result(ax, c)
    labels = [label.get_text() for label in ax.get_yticklabels()]
    assert labels == [f"{name} (n=0)" for name in E.PARAMETERS]
    assert len(ax.texts) == 1 and "no parameter passes" in ax.texts[0].get_text()
    plt.close(fig)


def test_cascade_uses_two_250ms_windows_at_six_mad():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((7200, 4))
    event = {"onset": 468.43579766536965, "end": 527.9066147859922}
    baseline = int(E.B / .25)
    median = np.median(X[:baseline, 2]); mad = np.median(np.abs(X[:baseline, 2] - median)) * 1.4826
    first = int(445.5 / .25)
    X[first:first + 2, 2] = median + 7 * mad
    lat, _, tw, _, _ = E.cascade(X, event, .25)
    assert np.isclose(tw[first], 445.5)
    assert np.isclose(lat[2], 445.5 - event["onset"])


def test_multi_episode_graphs_have_no_cross_record_edges():
    edges = [{"physiological": [(0, 1)]}, {"physiological": [(2, 3)]}]
    assert E.multi_edges(edges, "physiological") == [(0, 1), (6, 7)]


def test_localization_allows_fewer_than_three_involved_parameters():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((7200, 4))
    event = {"onset": 468.43579766536965, "end": 527.9066147859922}
    lat = np.array([-5.0, np.nan, np.nan, np.nan])
    _, df, _ = E.localization(lat, np.zeros(4), X, np.arange(7200) * .25, event, np.full(4, np.nan))
    assert df.involved.sum() == 1

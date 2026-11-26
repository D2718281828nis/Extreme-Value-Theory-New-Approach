import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common as C
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


def test_interval_features_recovers_synthetic_p_delay_and_st_shift():
    n = 1000; y = np.zeros((n, 12)); t = np.arange(n)
    y += 0.1 * np.exp(-0.5 * ((t - 200) / 3) ** 2)[:, None]
    y[277:288] += 0.5
    seconds, beats = E.interval_features(y, np.array([257]))
    assert ((beats["st"][0] > 0.35) & (beats["st"][0] < 0.55)).all()
    assert abs(beats["pr"][0, 0] - (257 - 200) / E.FS) < 0.03
    assert (beats["pq"][0] > 1).all()
    assert seconds["st"].shape == (n // E.FS, 12)
    assert seconds["pr"].shape == seconds["pq"].shape == seconds["st"].shape

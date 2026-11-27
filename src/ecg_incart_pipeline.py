"""Все 75 записей INCART: четыре ряда из сырой ЭКГ, EVT и каскады по эпизодам ЖТ.

R-пики находятся по отведению II без использования аннотаций. По II измеряются RR, клинический PR
(начало P — начало QRS) и подъём ST на J+60…80 мс. Четвёртый ряд — средняя фронтальная ось QRS,
atan2 от чистых площадей QRS в aVF и I. Узлы графа — RR, PR, ST и QRS axis, а не 12 отведений.
Аннотации используются только для эталона ЖТ и последующей проверки положений R-пиков.

Запуск: python ecg_incart_pipeline.py --data ../data/INCART --out ../results/ecg
"""
from __future__ import annotations
import argparse, hashlib, json, os, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import signal, stats
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
import common as C
import folds as FO

warnings.filterwarnings("ignore")
RECORD = "I42"
FS = 257
N = 462600
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PARAMETERS = ["RR", "PR", "ST", "QRS_axis"]
GROUPS = np.array(["rhythm", "conduction", "repolarization", "spatial_axis"])
UNITS = ["s", "s", "mV", "degree"]
SHA256 = {
    "I42.dat": "1ed5c6039a2bbb6dd6932a6c45e05ecad2fedb054a479d2a0f444a9134429ede",
    "I42.hea": "eb72e7ae6e2275f53f2b5e38c62374b8ab1a288e6e1758b40db5c26ac23d02bc",
    "I42.atr": "7421ccbac2269f57a768633ee301929d48ed8eade5336d080c0f3aba376f9109",
}
B = 400
WH = 30
MAD_THR = 6.0
PR_QUALITY_MIN = 3.0
SPEC = C.load_spec("ecg", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "ecg.json"))
BEAT_CODES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 25, 34, 35, 38, 41}


def read_annotations(path):
    """MIT/WFDB annotations; SKIP хранит абсолютное время как PDP-11 long (старшие 16 бит первыми)."""
    b = Path(path).read_bytes(); i = t = 0; out = []; aux = ""
    while i + 1 < len(b):
        w = int.from_bytes(b[i:i + 2], "little"); i += 2
        dt, typ = w & 1023, w >> 10
        if typ == 0:
            break
        if typ == 59:
            hi = int.from_bytes(b[i:i + 2], "little"); lo = int.from_bytes(b[i + 2:i + 4], "little"); i += 4
            t = (hi << 16) | lo
            if t >= 1 << 31:
                t -= 1 << 32
            continue
        t += dt
        if typ in (60, 61, 62):
            i += 2
        elif typ == 63:
            n = int.from_bytes(b[i:i + 2], "little") & 1023; i += 2
            aux = b[i:i + n].decode("ascii", errors="replace"); i += n + (n & 1)
        out.append((t, typ, aux)); aux = ""
    return out


def vt_episodes(annotations, fs=FS, min_beats=5, min_hr=100):
    runs, cur = [], []
    for t, typ, _ in annotations:
        if typ == 5:
            cur.append(t)
        else:
            if cur:
                runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        if len(r) < min_beats:
            continue
        rr = np.diff(r) / fs
        hr = 60 / rr.mean()
        if hr > min_hr:
            out.append({"onset_sample": int(r[0]), "end_sample": int(r[-1]), "onset": r[0] / fs,
                        "end": r[-1] / fs, "duration": (r[-1] - r[0]) / fs, "n_beats": len(r), "mean_hr": float(hr),
                        "min_rr": float(rr.min()), "max_rr": float(rr.max())})
    return out


def load_record(path, record):
    root = Path(path); expected = {}
    checksums = root / "SHA256SUMS.txt"
    if checksums.exists():
        for line in checksums.read_text().splitlines():
            if line.strip():
                digest, name = line.split(maxsplit=1); expected[name.lstrip("*")] = digest
    for extension in ("dat", "hea", "atr"):
        name = f"{record}.{extension}"; want = expected.get(name, SHA256.get(name))
        assert want is not None, f"Нет официального SHA-256 для {name}"
        got = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert got == want, f"SHA-256 {name} не совпал: {got}"
    lines = (root / f"{record}.hea").read_text().splitlines()
    _, nsig, fs, n = lines[0].split()[:4]
    assert int(nsig) == 12 and int(fs) == FS
    n = int(n)
    fmt = [s.split()[1] for s in lines[1:13]]
    assert set(fmt) == {"16"}, "Ожидается WFDB format 16"
    gain = np.array([int(s.split()[2]) for s in lines[1:13]])
    baseline = np.array([int(s.split()[4]) for s in lines[1:13]])
    raw = np.fromfile(root / f"{record}.dat", dtype="<i2").reshape(n, 12)
    x = (raw - baseline) / gain
    ann = read_annotations(root / f"{record}.atr")
    return x, ann, vt_episodes(ann)


def load(path):
    x, ann, episodes = load_record(path, RECORD)
    assert len(x) == N and episodes, "Эпизоды ЖТ не найдены"
    event = max(episodes, key=lambda e: e["duration"])
    assert 460 < event["onset"] < 480 and 55 < event["duration"] < 65, "Основное событие I42 изменилось"
    return x, ann, episodes, event


def filtered(x):
    sos = signal.butter(3, [0.5, 45], btype="band", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def detect_r_peaks(y):
    """Отведение II выбрано заранее как ритмическое: одновременно различимы P- и R-волны."""
    peaks, _ = signal.find_peaks(y[:, 1], distance=int(.28 * FS), prominence=1.0)
    return peaks


def _robust_sigma(x):
    return float(np.median(np.abs(x - np.median(x))) * 1.4826 + 1e-6)


def _delineate_ii(low, p):
    """Границы P/QRS и качество P на II; индексы относятся к исходной частоте дискретизации."""
    base_seg = low[p - 90:p - 60, 1]
    base = float(np.median(base_seg)); noise = _robust_sigma(base_seg)
    qrs = low[p - 25:p + 40, 1] - base
    r_index = 25
    threshold = max(3 * noise, .08 * abs(qrs[r_index]), 1e-6)
    if abs(qrs[r_index]) <= threshold:
        return None
    left = right = r_index
    while left > 0 and abs(qrs[left - 1]) > threshold:
        left -= 1
    while right + 1 < len(qrs) and abs(qrs[right + 1]) > threshold:
        right += 1
    qrs_onset, qrs_offset = p - 25 + left, p - 25 + right
    p_end = qrs_onset - 2
    p_start = p - 77
    if p_end - p_start < 20:
        return qrs_onset, qrs_offset, np.nan, 0.0
    p_wave = low[p_start:p_end, 1] - base
    peak = int(np.argmax(np.abs(p_wave)))
    amplitude = float(abs(p_wave[peak])); quality = amplitude / noise
    threshold_p = max(2 * noise, .2 * amplitude, 1e-6)
    onset = peak; below = 0
    for k in range(peak - 1, -1, -1):
        if abs(p_wave[k]) < threshold_p:
            below += 1
            if below == 3:
                onset = k + 3
                break
        else:
            below = 0; onset = k
    pr = (qrs_onset - (p_start + onset)) / FS
    if quality < PR_QUALITY_MIN or not .08 <= pr <= .32:
        pr = np.nan
    return qrs_onset, qrs_offset, pr, quality


def beat_parameters(y, peaks):
    """Показатели каждого сокращения; PR остаётся NaN при недостаточном качестве P."""
    low = signal.sosfiltfilt(signal.butter(3, 15, btype="low", fs=FS, output="sos"), y, axis=0)
    peaks = np.asarray(peaks, int)
    peaks = peaks[(peaks > 100) & (peaks + 65 < len(y))]
    beats = {k: [] for k in ["r_sample", "rr", "pr", "pr_quality", "st", "qrs_axis"]}
    for k, p in enumerate(peaks):
        delineation = _delineate_ii(low, p)
        if delineation is None:
            continue
        qrs_onset, qrs_offset, pr, quality = delineation
        baseline = np.median(low[p - 90:p - 60], axis=0)
        j_point = qrs_offset
        st_slice = low[j_point + 15:j_point + 21, 1]
        if len(st_slice) != 6:
            continue
        st = float(np.median(st_slice) - baseline[1])
        qrs = low[qrs_onset:qrs_offset + 1][:, [0, 5]] - baseline[[0, 5]]
        area_i, area_avf = qrs.sum(axis=0) / FS
        axis = float(np.degrees(np.arctan2(area_avf, area_i)))
        beats["r_sample"].append(int(p))
        beats["rr"].append(float((p - peaks[k - 1]) / FS) if k else np.nan)
        beats["pr"].append(pr); beats["pr_quality"].append(quality)
        beats["st"].append(st); beats["qrs_axis"].append(axis)
    return {k: np.asarray(v, float) for k, v in beats.items()}


def _fill_linear(values, baseline=B):
    values = np.asarray(values, float)
    fallback = np.nanmedian(values[:baseline])
    if not np.isfinite(fallback):
        fallback = np.nanmedian(values)
    if not np.isfinite(fallback):
        raise ValueError("Недостаточно измерений для временного ряда")
    return pd.Series(values).ffill(limit=1).fillna(fallback).to_numpy()


def _circular_mean_deg(values):
    z = np.exp(1j * np.radians(values)); z = z[np.isfinite(z)]
    return float(np.degrees(np.angle(z.mean()))) if len(z) else np.nan


def _fill_axis(values, baseline=B):
    values = np.asarray(values, float)
    fallback = _circular_mean_deg(values[:baseline])
    if not np.isfinite(fallback):
        fallback = _circular_mean_deg(values)
    if not np.isfinite(fallback):
        raise ValueError("Недостаточно измерений оси QRS")
    return pd.Series(values).ffill(limit=1).fillna(fallback).to_numpy()


def per_window_series(beats, n_windows, window_s=1.0):
    """Медианы сокращений в окне; один пропуск заполняется прошлым значением, остальные — медианой фона."""
    pt = beats["r_sample"] / FS
    baseline = int(round(B / window_s))
    raw = {k: np.full(n_windows, np.nan) for k in ["rr", "pr", "st", "qrs_axis"]}
    quality = np.full(n_windows, np.nan); measured_pr = np.zeros(n_windows, bool)
    for t in range(n_windows):
        mask = (pt >= t * window_s) & (pt < (t + 1) * window_s)
        if not mask.any():
            continue
        for k in ["rr", "pr", "st"]:
            values = beats[k][mask]
            if np.isfinite(values).any():
                raw[k][t] = np.nanmedian(values)
        raw["qrs_axis"][t] = _circular_mean_deg(beats["qrs_axis"][mask])
        quality[t] = np.nanmedian(beats["pr_quality"][mask])
        measured_pr[t] = np.isfinite(beats["pr"][mask]).any()
    series = np.column_stack([_fill_linear(raw[k], baseline) for k in ["rr", "pr", "st"]]
                             + [_fill_axis(raw["qrs_axis"], baseline)])
    return series, raw, quality, measured_pr


def per_second_series(beats, n_seconds):
    return per_window_series(beats, n_seconds, 1.0)


def derived_series(y, peaks=None):
    peaks = detect_r_peaks(y) if peaks is None else np.asarray(peaks, int)
    beats = beat_parameters(y, peaks)
    series, raw, quality, measured_pr = per_second_series(beats, len(y) // FS)
    return series, beats, {"raw": raw, "pr_quality": quality, "pr_measured": measured_pr, "peaks": peaks}


def circular_deviation(values, center):
    return (np.asarray(values) - center + 180) % 360 - 180


def analysis_series(series, baseline=B):
    """Четыре ряда для графа и каскада; ось представлена кратчайшим отклонением от центра фона."""
    center = _circular_mean_deg(series[:baseline, 3])
    return np.column_stack([series[:, :3], circular_deviation(series[:, 3], center)])


def state_matrix(series):
    """Махalanobis для угла использует sin/cos: искусственного скачка на ±180° нет."""
    angle = np.radians(series[:, 3])
    return np.column_stack([series[:, :3], np.sin(angle), np.cos(angle)])


def quality_summary(beats, event):
    pt = beats["r_sample"] / FS; t0, t1 = event["onset"], event["end"]
    intervals = {"background": (0, B), "pre_event": (B, t0), "vt": (t0, t1), "post_recovery": (t1 + 60, N / FS)}
    rows = []
    for name, (a, b) in intervals.items():
        mask = (pt >= a) & (pt < b); pr = beats["pr"][mask]; pq = beats["pr_quality"][mask]
        valid = np.isfinite(pr)
        rows.append({"interval": name, "seconds": [a, b], "n_beats": int(mask.sum()),
                     "pr_valid_fraction": float(valid.mean()) if len(valid) else 0.0,
                     "pr_median_s": float(np.nanmedian(pr)) if valid.any() else None,
                     "pr_iqr_s": np.quantile(pr[valid], [.25, .75]).tolist() if valid.any() else None,
                     "p_quality_median": float(np.nanmedian(pq)) if len(pq) else None,
                     "p_quality_gt3_fraction": float((pq > PR_QUALITY_MIN).mean()) if len(pq) else 0.0})
    return rows


def indicator(F, b0, b1):
    x = F[b0:b1]
    mu = x.mean(0); scale = x.std(0) + 1e-9
    z = (F - mu) / scale
    precision = LedoitWolf().fit(z[b0:b1]).precision_
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", z, precision, z), 0))


def alarm_episodes(starts, gap=10):
    ep = []
    for s in starts:
        if not ep or s - ep[-1][-1] > gap:
            ep.append([s])
        else:
            ep[-1].append(s)
    return [e[0] for e in ep]


def detection(series, event):
    t0, t1 = event["onset"], event["end"]
    pre = (int(t0) - 300, int(t0) - 30)
    F = state_matrix(series)
    hi = indicator(F, 0, B)
    out = {}
    for method, spec in [("blind_pot", SPEC["blind_pot"]), ("blind_gev", SPEC["blind_gev"])]:
        d = C.detect(hi, spec, B)
        eps = alarm_episodes(d["alarms"])
        ev = [e for e in eps if t0 - 60 <= e <= t1]
        normal_seconds = (t0 - 60 - B) + (len(hi) - (t1 + 60))
        false = [e for e in eps if (B <= e < t0 - 60) or e > t1 + 60]
        recovery = [e for e in eps if t1 < e <= t1 + 60]
        out[method] = {"thr": d["thr"], "xi": d["xi"], "episodes": eps,
                       "event_alarm": ev[0] if ev else None,
                       "dt": ev[0] - t0 if ev else None,
                       "early": bool(ev and ev[0] < t0),
                       "false_alarms": false, "recovery_alarms": recovery,
                       "false_alarms_per_hour": len(false) / (normal_seconds / 3600),
                       "normal_test_seconds": normal_seconds,
                       "block_s": spec.get("block", 1)}
    hl = indicator(F, *pre)
    dl = C.detect(hl, SPEC["targeted_local"], pre, start=pre[1])
    first = dl["first"]
    out["targeted_local"] = {"baseline": list(pre), "search_from": pre[1], "thr": dl["thr"], "xi": dl["xi"],
                             "first_alarm": first, "dt": None if first is None else first - t0,
                             "early": first is not None and first < t0}
    base = hi[:B] / np.median(hi[:B]); full = hi / np.median(hi[:B])
    out["heavy_tail"] = {k: {"kurtosis": float(stats.kurtosis(x)), "q999": float(np.quantile(x, .999))}
                         for k, x in [("baseline", base), ("full", full)]}
    return out, hi, hl, pre


def hypothesis(X, event):
    """Предвестник по четырём производным рядам, без спектральных признаков исходных отведений."""
    noise = np.vstack([np.zeros((1, X.shape[1])), np.diff(X, axis=0) ** 2])
    variance = pd.DataFrame(X).rolling(5, center=True, min_periods=1).var().to_numpy()
    end = int(event["onset"])
    def at(e, M):
        return np.array([M[e - (4 - k) * WH:e - (3 - k) * WH].mean(0) for k in range(4)])
    def slope(m):
        return np.polyfit(np.arange(4), np.log(np.median(m / m[0], 1) + 1e-12), 1)[0]
    observed = {"noise": at(end, noise), "variance": at(end, variance)}
    cand = [e for e in range(4 * WH + 30, len(X) - WH, 60) if abs(e - end) > 300]
    surrogates = {k: [slope(at(e, M)) for e in cand] for k, M in [("noise", noise), ("variance", variance)]}
    result = {"window_s": WH, "n_surrogate": len(cand), "series": PARAMETERS}
    for name, m in observed.items():
        obs = slope(m); subj = np.repeat(np.arange(4), 4); win = np.tile(np.arange(4), 4)
        result[name] = {"median_by_window": np.median(m / (m[0] + 1e-12), 1).tolist(),
                        "frac_parameters_increase_T1_T4": float((m[3] > m[0]).mean()),
                        "rm_corr_parameters": C.rm_corr(subj, win, np.log(m.T.ravel() + 1e-12)),
                        "slope_obs": float(obs), "slope_surrogate_mean": float(np.mean(surrogates[name])),
                        "p_surrogate_greater": float((np.array(surrogates[name]) >= obs).mean()),
                        "p_surrogate_less": float((np.array(surrogates[name]) <= obs).mean())}
    return result, observed["noise"], observed["variance"]


def cascade(X, event, window_s=1.0, search_from=B):
    t1 = event["end"]; tw = np.arange(len(X), dtype=float) * window_s
    baseline = int(round(B / window_s))
    med = np.median(X[:baseline], 0); mad = np.median(np.abs(X[:baseline] - med), 0) * 1.4826 + 1e-6
    z = np.abs((X - med) / mad)
    lat = np.full(4, np.nan)
    for j in range(4):
        mask = (z[:, j] > MAD_THR) & (tw >= search_from) & (tw <= t1)
        k = C.first_persistent(mask, 0, 2)
        if k is not None:
            lat[j] = tw[k] - event["onset"]
    dev = np.abs(X - med); foot = []
    for j in range(4):
        thr, _ = C.pot_threshold(dev[:baseline, j], .9, .99)
        foot.append(float((dev[(tw >= event["onset"]) & (tw <= t1), j] > thr).mean()))
    return lat, z, tw, np.array(foot), (0, B)


def node_features(X, pre):
    a, b = pre; p = X[a:b]
    rows = []
    for j in range(4):
        x = p[:, j]
        dfa = C.dfa_alpha(x, scales=[4, 6, 8, 12, 16, 24, 32, 48, 64])
        ac1 = np.corrcoef(x[:-1], x[1:])[0, 1]
        rows.append([x.mean(), x.std(), dfa, ac1, np.median(np.abs(np.diff(x)))])
    names = ["mean", "std", "dfa", "ac1", "mad_diff"]
    return np.nan_to_num(np.asarray(rows)).astype(np.float32), names


def physiological_edges():
    return [(0, 1), (2, 3)]


def localization(lat, foot, X, tw, event, oof):
    def safe_corr(x, y):
        valid = np.isfinite(x) & np.isfinite(y)
        xv, yv = np.asarray(x)[valid], np.asarray(y)[valid]
        if len(xv) < 3 or len(np.unique(xv)) < 2 or len(np.unique(yv)) < 2:
            return np.nan
        return float(stats.spearmanr(xv, yv)[0])
    t0, t1 = event["onset"], event["end"]
    pre = (int(t0) - 300, int(t0) - 30)
    ip = (tw >= pre[0]) & (tw < pre[1]); ie = (tw >= t0) & (tw <= t1)
    xp = X[ip]; common = np.median(xp, 1)
    beta = xp.T @ common / (common @ common + 1e-12)
    idio = (xp - np.outer(common, beta)).std(0) + 1e-9
    rz = ((X[ie].mean(0) - xp.mean(0)) - beta * (X[ie].mean() - common.mean())) / idio
    early = np.isfinite(lat) & (lat <= np.nanpercentile(lat, 10))
    df = pd.DataFrame({"parameter": PARAMETERS, "group": GROUPS, "unit": UNITS, "latency": lat,
                       "involved": np.isfinite(lat).astype(int), "footprint": foot, "residual_z": rz,
                       "gnn_oof": oof, "early10": early})
    by = df.set_index("parameter")[["group", "latency", "footprint", "residual_z", "gnn_oof", "early10"]].copy()
    lead = {"footprint": PARAMETERS[int(np.argmax(foot))], "residual_z": PARAMETERS[int(np.argmax(rz))],
            "cascade": PARAMETERS[int(np.nanargmin(lat))] if np.isfinite(lat).any() else None,
            "gnn_oof": PARAMETERS[int(np.nanargmax(oof))] if np.isfinite(oof).any() else None}
    out = {"interpretation": "доминирующий физиологический параметр, не анатомическая локализация очага ЖТ",
           "leading_parameter": lead,
           "leading_parameter_agree": int(sum(v == lead["footprint"] for v in lead.values())),
           "earliest_parameters": df.dropna(subset=["latency"]).sort_values("latency")[["parameter", "latency"]].round(3).values.tolist(),
           "n_involved": int(np.isfinite(lat).sum()), "n_early10": int(early.sum()),
           "footprint_by_parameter": dict(zip(PARAMETERS, np.round(foot, 4))),
           "residual_z_by_parameter": dict(zip(PARAMETERS, np.round(rz, 3))),
           "spearman": {"latency_vs_footprint": safe_corr(lat, foot),
                        "latency_vs_residual": safe_corr(lat, rz),
                        "footprint_vs_residual": safe_corr(foot, rz)}}
    return out, df, by


def run_protocols(F, y, E, seeds, epochs):
    import torch
    _, NodeGAT, MLP = C.make_models()
    rows, oof = [], {}
    for seed in seeds:
        for split in ["random", "node"]:
            folds = list(KFold(4, shuffle=True, random_state=seed).split(F)) if split == "random" else list(LeaveOneOut().split(F))
            for model, gname in [("gat", "physiological"), ("gat", "functional"), ("gat", "knn"),
                                 ("gat", "empty"), ("mlp", "-"), ("lr", "-")]:
                pred = np.zeros(len(y))
                for tr, te in folds:
                    sc = StandardScaler().fit(F[tr]); fs = sc.transform(F).astype(np.float32)
                    if len(np.unique(y[tr])) < 2:
                        pred[te] = float(y[tr].mean()); continue
                    if model == "lr":
                        pred[te] = LogisticRegression(max_iter=3000).fit(fs[tr], y[tr]).predict_proba(fs[te])[:, 1]; continue
                    torch.manual_seed(seed); np.random.seed(seed)
                    yt = torch.tensor(y[tr], dtype=torch.float32)
                    pw = torch.tensor((1 - y[tr].mean()) / max(y[tr].mean(), 1e-3), dtype=torch.float32)
                    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
                    if model == "mlp":
                        net = MLP(fs.shape[1]); opt = torch.optim.Adam(net.parameters(), 5e-3, weight_decay=5e-4)
                        xt = torch.tensor(fs[tr])
                        for _ in range(epochs):
                            net.train(); opt.zero_grad(); lossf(net(xt), yt).backward(); opt.step()
                        net.eval()
                        with torch.no_grad(): pred[te] = torch.sigmoid(net(torch.tensor(fs[te]))).numpy()
                    else:
                        ei = C.edge_index_from(E[gname], len(y)); x = torch.tensor(fs)
                        mask = torch.zeros(len(y), dtype=torch.bool); mask[tr] = True
                        ya = torch.tensor(y, dtype=torch.float32)
                        net = NodeGAT(fs.shape[1]); opt = torch.optim.Adam(net.parameters(), 5e-3, weight_decay=5e-4)
                        for _ in range(epochs):
                            net.train(); opt.zero_grad(); lossf(net(x, ei)[mask], ya[mask]).backward(); opt.step()
                        net.eval()
                        with torch.no_grad(): pred[te] = torch.sigmoid(net(x, ei)).numpy()[te]
                rows.append({"seed": seed, "split": split, "model": model, "graph": gname,
                             "auroc": float(roc_auc_score(y, pred)), "auprc": float(average_precision_score(y, pred))})
                oof[(seed, split, model, gname)] = pred
                print(rows[-1], flush=True)
    return pd.DataFrame(rows), oof


def make_figures(y, hi, det, z, tw, lat, event, out, series, quality, measured_pr):
    import matplotlib.pyplot as plt
    t0, t1 = event["onset"], event["end"]
    fig, axs = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    taxis = np.arange(len(y)) / FS
    for ax, j in zip(axs[:3], [1, 0, 5]):
        ax.plot(taxis, y[:, j], lw=.5); ax.set_ylabel(LEADS[j] + ", mV"); ax.grid(alpha=.2)
        ax.axvspan(t0, t1, color="r", alpha=.1); ax.axvline(t0, color="r", ls="--", lw=.8)
    axs[3].plot(hi, lw=.8, label="4-parameter health indicator")
    axs[3].axhline(det["blind_pot"]["thr"], color="r", ls="--", label="POT threshold")
    axs[3].axhline(det["blind_gev"]["thr"], color="orange", ls=":", label="GEV threshold")
    axs[3].axvspan(t0, t1, color="r", alpha=.1); axs[3].set_ylabel("indicator")
    axs[3].set_xlim(350, 650)
    axs[3].set_xlabel("time, s"); axs[3].legend(); axs[3].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / "ecg_evt_detection.png", dpi=160); plt.close(fig)
    sel = (tw >= 380) & (tw <= 570)
    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(z[sel].T, aspect="auto", origin="lower", extent=[tw[sel][0], tw[sel][-1], 0, 4],
                   vmin=0, vmax=15, cmap="magma")
    ax.axvline(t0, color="cyan", ls="--", lw=1); ax.axvline(t1, color="cyan", ls="--", lw=1)
    ax.set_yticks(np.arange(4) + .5, PARAMETERS); ax.set_xlabel("time, s"); ax.set_ylabel("parameter")
    ax.set_title("Four derived ECG series: absolute robust deviation and VT cascade")
    fig.colorbar(im, ax=ax, label="|z|"); fig.tight_layout(); fig.savefig(out / "ecg_vt_cascade.png", dpi=160); plt.close(fig)

    view = (350, 580); series_t = np.arange(len(series), dtype=float)
    mask = (series_t >= view[0]) & (series_t <= view[1]); tt = series_t[mask]
    labels = ["RR interval, s", "PR interval, s", "ST elevation at J+60…80 ms, mV", "Frontal QRS axis, degree"]
    fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    for j, ax in enumerate(axs):
        ax.plot(tt, series[mask, j], lw=.9, color="#4c78a8")
        if np.isfinite(lat[j]):
            ax.scatter(t0 + lat[j], series[int(round(t0 + lat[j])), j], s=40, facecolors="white", edgecolors="red", zorder=4)
        ax.set_ylabel(labels[j]); ax.grid(alpha=.2)
        ax.axvline(det["blind_pot"]["event_alarm"], color="red", lw=1)
        ax.axvline(det["blind_gev"]["event_alarm"], color="orange", lw=1)
        ax.axvline(t0, color="black", ls="--", lw=.8); ax.set_xlim(*view)
    reliable = measured_pr[mask]
    axs[1].scatter(tt[reliable], series[mask, 1][reliable], s=5, color="#1baf7a", label="seconds with measured PR")
    axs[1].legend(loc="upper left"); axs[1].text(.99, .05, f"median P quality in view: {np.nanmedian(quality[mask]):.1f}",
                                               transform=axs[1].transAxes, ha="right", fontsize=8)
    axs[-1].set_xlabel("time, s")
    fig.suptitle("Four time series from raw ECG: RR, PR, ST and cross-lead QRS axis")
    fig.tight_layout(); fig.savefig(out / "ecg_parameter_cascade.png", dpi=160); plt.close(fig)


def single_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/INCART"); ap.add_argument("--out", default="../results/ecg")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); start = time.time()
    x, annotations, episodes, event = load(a.data); y = filtered(x)
    R = {"dataset": "St Petersburg INCART 12-lead Arrhythmia Database v1.0.0, PhysioNet DOI 10.13026/C2V88N",
         "record": RECORD, "n_raw_leads": 12, "n_parameters": 4, "parameters": PARAMETERS, "fs": FS,
         "duration_s": N / FS, "sha256": SHA256, "event": event, "all_vt_episodes": episodes, "detection_spec": SPEC}
    series, beats, aux = derived_series(y)
    peaks = aux["peaks"]
    ref = np.array([t for t, typ, _ in annotations if typ in BEAT_CODES])
    coverage = float(np.mean([np.min(np.abs(peaks - t)) <= int(.08 * FS) for t in ref]))
    R["lead_choice"] = {
        "selected": "II", "reason": "стандартное ритмическое отведение: синусовые P обычно положительны и различимы, R выражен; один канал синхронизирует RR, PR и ST",
        "selection_rule": "выбрано по физиологии до анализа события, не по качеству детекции ЖТ",
        "cross_lead_parameter": "atan2(чистая площадь QRS aVF, чистая площадь QRS I), градусы"}
    R["signal_qrs"] = {"lead": "II", "min_distance_s": .28, "prominence_mv": 1.0,
                       "n_detected": len(peaks), "reference_position_coverage_80ms": coverage,
                       "note": "позиции и типы аннотаций не входят в признаки; покрытие — последующая проверка"}
    R["derived_signals"] = {
        "RR": "медиана RR по сокращениям II в каждой секунде, с",
        "PR": "начало P до начала QRS на II; quality≥3 и диапазон 80–320 мс; ненадёжные измерения — пропуски до заполнения",
        "ST": "J+60…80 мс относительно изоэлектрической базы на II, мВ",
        "QRS_axis": "atan2(aVF, I) чистых площадей QRS, градусы; круговая обработка в графе, каскаде и EVT",
        "missing_values": "медиана за секунду → каузальное заполнение прошлым значением максимум на 1 с → медиана фона; доля измеренного PR сохранена"}
    X = analysis_series(series)
    cascade_raw, _, _, _ = per_window_series(beats, int(round(N / FS / .25)), .25)
    Xc = analysis_series(cascade_raw, int(round(B / .25)))
    R["detection"], hi, hl, pre = detection(series, event)
    R["H0_noise"], I, V = hypothesis(X, event)
    lat, z, tw, foot, cascade_base = cascade(Xc, event, .25)
    R["cascade"] = {"feature": "absolute robust deviation of each derived parameter", "baseline": list(cascade_base),
                    "search_from": cascade_base[1], "window_s": .25, "threshold_mad": MAD_THR, "persistent_windows": 2,
                    "rationale": "250 мс × 2 = 500 мс: каскад разрешает внутрисекундную динамику ЭКГ; основной EVT-индикатор остаётся односекундным",
                    "latency_by_parameter": dict(zip(PARAMETERS, np.round(lat, 3))),
                    "recruited_parameters": [PARAMETERS[j] for j in range(4) if np.isfinite(lat[j])]}
    R["pr_quality"] = quality_summary(beats, event)
    pd.DataFrame(R["pr_quality"]).to_csv(out / "pr_quality.csv", index=False)
    Fnode, node_names = node_features(X, pre)
    ef = C.functional_edges(X[:B], k=1); ek = C.knn_edges(Fnode, k=1); ep = physiological_edges()
    E = {"physiological": ep, "functional": ef, "knn": ek, "empty": []}
    labels = np.isfinite(lat).astype(int)
    R["graphs"] = {"nodes": PARAMETERS, "n_edges": {k: len(v) for k, v in E.items() if k != "empty"},
                   "edges": {k: [[PARAMETERS[i], PARAMETERS[j]] for i, j in v] for k, v in E.items() if k != "empty"},
                   "rules": {"physiological": "RR–PR (ритм/проведение), ST–QRS axis (реполяризация/пространственная морфология)",
                             "functional": "top-1 |corr| по фону четырёх рядов", "knn": "top-1 по стандартизированным признакам узлов"},
                   "jaccard": {"physiological-functional": C.jaccard(ep, ef), "physiological-knn": C.jaccard(ep, ek),
                               "functional-knn": C.jaccard(ef, ek)}}
    R["labels"] = {"involved": int(labels.sum()), "n": len(labels), "by_parameter": dict(zip(PARAMETERS, labels.tolist()))}
    if len(np.unique(labels)) == 2:
        runs, oof = run_protocols(Fnode, labels, E, a.seeds, a.epochs)
        runs.to_csv(out / "gnn_runs.csv", index=False)
        g = runs.groupby(["split", "model", "graph"])[["auroc", "auprc"]].agg(["mean", "std"]).round(4)
        g.columns = ["_".join(c) for c in g.columns]; R["gnn_summary"] = g.reset_index().to_dict("records")
        oof_gnn = np.mean([oof[(s, "node", "gat", "physiological")] for s in a.seeds], 0)
        tables = [FO.fold_table(PARAMETERS, GROUPS, FO.random_kfold(len(labels), 4, s), "random", s) for s in a.seeds]
        tables.append(FO.fold_table(PARAMETERS, GROUPS, list(LeaveOneOut().split(Fnode)), "node"))
        R["folds_md5"] = FO.save_fold_map(out / "folds.csv", tables)
    else:
        R["gnn_skipped"] = "Каскад дал только один класс параметров; классификация четырёх узлов не определена"
        oof_gnn = np.full(4, np.nan)
    R["localization"], loc, by = localization(lat, foot, Xc, tw, event, oof_gnn)
    loc.to_csv(out / "localization_parameters.csv", index=False); by.to_csv(out / "localization_groups.csv")
    pd.DataFrame(Fnode, columns=node_names).assign(parameter=PARAMETERS, group=GROUPS, involved=labels).to_csv(out / "node_features.csv", index=False)
    np.savez(out / "indicators.npz", hi=hi, hl=hl, z=z.astype(np.float32), tw=tw, lat=lat,
             series=series, analysis_series=X, cascade_series=Xc, rr=series[:, 0], pr=series[:, 1], st=series[:, 2], qrs_axis=series[:, 3],
             pr_quality=aux["pr_quality"], pr_measured=aux["pr_measured"], I=I, V=V)
    make_figures(y, hi, R["detection"], z, tw, lat, event, out, series, aux["pr_quality"], aux["pr_measured"])
    R["runtime_s"] = round(time.time() - start, 1)
    (out / "results.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in ["event", "detection", "H0_noise", "graphs", "labels", "localization"]},
                     ensure_ascii=False, indent=1, default=str)[:9000])


DERIVATION_VERSION = "four-series-v1"


def derived_record(root, record, cache_dir):
    """Извлечение одной записи с дисковым кэшем; исходные SHA-256 проверяются до построения кэша."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{record}.npz"
    if cache.exists():
        d = np.load(cache)
        if str(d["version"]) == DERIVATION_VERSION:
            beats = {k: d[k] for k in ["r_sample", "rr", "pr", "pr_quality", "st", "qrs_axis"]}
            return {"series": d["series"], "beats": beats, "episodes": json.loads(str(d["episodes"])),
                    "coverage": float(d["coverage"])}
    x, annotations, episodes = load_record(root, record)
    y = filtered(x)
    series, beats, aux = derived_series(y)
    ref = np.array([t for t, typ, _ in annotations if typ in BEAT_CODES])
    peaks = aux["peaks"]
    coverage = float(np.mean([np.min(np.abs(peaks - t)) <= int(.08 * FS) for t in ref]))
    np.savez_compressed(cache, version=DERIVATION_VERSION, series=series, episodes=json.dumps(episodes),
                        coverage=coverage, **beats)
    return {"series": series, "beats": beats, "episodes": episodes, "coverage": coverage}


def multi_edges(E, graph):
    return [(4 * k + i, 4 * k + j) for k in range(len(E)) for i, j in E[k][graph]]


def _node_indices(episodes):
    return np.concatenate([np.arange(4 * e, 4 * e + 4) for e in episodes]) if len(episodes) else np.array([], int)


def run_multi_protocols(F, Y, groups, E, seeds, epochs):
    """Независимые четырёхузловые графы эпизодов; grouped split исключает пересечение записей."""
    import torch
    _, NodeGAT, MLP = C.make_models()
    n_ep = len(F); F = F.reshape(n_ep * 4, -1); Y = Y.reshape(-1)
    rows, oof = [], {}
    for seed in seeds:
        for split in ["random", "record"]:
            folds = FO.random_kfold(n_ep, 5, seed) if split == "random" else list(FO.stable_group_kfold(groups))
            for model, gname in [("gat", "physiological"), ("gat", "functional"), ("gat", "knn"),
                                 ("gat", "empty"), ("mlp", "-"), ("lr", "-")]:
                pred = np.zeros(len(Y))
                for tr_ep, te_ep in folds:
                    tr, te = _node_indices(tr_ep), _node_indices(te_ep)
                    sc = StandardScaler().fit(F[tr]); fs = sc.transform(F).astype(np.float32)
                    if len(np.unique(Y[tr])) < 2:
                        pred[te] = float(Y[tr].mean()); continue
                    if model == "lr":
                        pred[te] = LogisticRegression(max_iter=3000).fit(fs[tr], Y[tr]).predict_proba(fs[te])[:, 1]; continue
                    torch.manual_seed(seed); np.random.seed(seed)
                    yt = torch.tensor(Y[tr], dtype=torch.float32)
                    pw = torch.tensor((1 - Y[tr].mean()) / max(Y[tr].mean(), 1e-3), dtype=torch.float32)
                    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
                    if model == "mlp":
                        net = MLP(fs.shape[1]); opt = torch.optim.Adam(net.parameters(), 5e-3, weight_decay=5e-4)
                        xt = torch.tensor(fs[tr])
                        for _ in range(epochs):
                            net.train(); opt.zero_grad(); lossf(net(xt), yt).backward(); opt.step()
                        net.eval()
                        with torch.no_grad(): pred[te] = torch.sigmoid(net(torch.tensor(fs[te]))).numpy()
                    else:
                        ei = C.edge_index_from(multi_edges(E, gname), len(Y)); x = torch.tensor(fs)
                        mask = torch.zeros(len(Y), dtype=torch.bool); mask[tr] = True
                        ya = torch.tensor(Y, dtype=torch.float32)
                        net = NodeGAT(fs.shape[1]); opt = torch.optim.Adam(net.parameters(), 5e-3, weight_decay=5e-4)
                        for _ in range(epochs):
                            net.train(); opt.zero_grad(); lossf(net(x, ei)[mask], ya[mask]).backward(); opt.step()
                        net.eval()
                        with torch.no_grad(): pred[te] = torch.sigmoid(net(x, ei)).numpy()[te]
                rows.append({"seed": seed, "split": split, "model": model, "graph": gname,
                             "auroc": float(roc_auc_score(Y, pred)), "auprc": float(average_precision_score(Y, pred))})
                oof[(seed, split, model, gname)] = pred
                print(rows[-1], flush=True)
    return pd.DataFrame(rows), oof


def summarize_localization(df):
    by = df.groupby("parameter", sort=False).agg(
        n_points=("episode_id", "size"), n_involved=("involved", "sum"),
        latency_median=("latency", "median"), footprint=("footprint", "mean"),
        residual_z=("residual_z", "mean"), gnn=("gnn_oof", "mean"))
    finite_latency = by.latency_median.dropna()
    lead = {"footprint": by.footprint.idxmax(), "residual_z": by.residual_z.idxmax(),
            "cascade": finite_latency.idxmin() if len(finite_latency) else None,
            "gnn_oof": by.gnn.idxmax() if by.gnn.notna().any() else None}
    out = {"interpretation": "усреднение по независимым эпизодам и записям, не анатомическая локализация очага ЖТ",
           "leading_parameter": lead,
           "leading_parameter_agree": int(sum(v == lead["footprint"] for v in lead.values())),
           "n_points": int(len(df)), "n_involved": int(df.involved.sum()),
           "by_parameter": by.round(4).reset_index().to_dict("records")}
    return out, by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/INCART"); ap.add_argument("--out", default="../results/ecg")
    ap.add_argument("--records", nargs="+", default=[f"I{k:02d}" for k in range(1, 76)])
    ap.add_argument("--cache", default=None, help="кэш производных рядов; по умолчанию data/INCART/derived")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args(); root = Path(a.data); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); start = time.time()
    cache = Path(a.cache) if a.cache else root / "derived"
    latency_rows, features, labels, record_groups, episode_ids, edge_sets = [], [], [], [], [], []
    baseline_corrs, episode_summaries = [], []
    excluded = {"no_annotated_vt": 0, "onset_before_400s": 0, "beyond_record": 0}
    representative = None
    for record in a.records:
        d = derived_record(root, record, cache)
        duration = len(d["series"]); all_episodes = sorted(d["episodes"], key=lambda e: e["onset"])
        if not all_episodes:
            excluded["no_annotated_vt"] += 1
        episodes = []
        for e in all_episodes:
            if e["onset"] < B:
                excluded["onset_before_400s"] += 1
            elif e["end"] > duration:
                excluded["beyond_record"] += 1
            else:
                episodes.append(e)
        print(f"{record}: VT всего {len(all_episodes)}, подходящих {len(episodes)}", flush=True)
        if not episodes:
            continue
        series = d["series"]; X = analysis_series(series)
        cascade_raw, _, _, _ = per_window_series(d["beats"], int(round(duration / .25)), .25)
        Xc = analysis_series(cascade_raw, int(round(B / .25)))
        baseline_corrs.append(np.nan_to_num(np.abs(np.corrcoef(X[:B].T))))
        main_episode = max(range(len(episodes)), key=lambda k: episodes[k]["duration"])
        previous_end = None
        for k, event in enumerate(episodes):
            search_from = max(B, event["onset"] - 120)
            if previous_end is not None:
                search_from = max(search_from, previous_end + 60)
            previous_end = event["end"]
            lat, z, tw, foot, cascade_base = cascade(Xc, event, .25, search_from)
            pre = (int(event["onset"]) - 300, int(event["onset"]) - 30)
            Fnode, node_names = node_features(X, pre)
            det, hi, hl, _ = detection(series, event)
            loc, loc_df, _ = localization(lat, foot, Xc, tw, event, np.full(4, np.nan))
            episode_id = f"{record}_VT{k + 1:02d}"
            episode_ids.append(episode_id); record_groups.append(record)
            features.append(Fnode); labels.append(np.isfinite(lat).astype(int))
            ef = C.functional_edges(X[:B], k=1); ek = C.knn_edges(Fnode, k=1)
            edge_sets.append({"physiological": physiological_edges(), "functional": ef, "knn": ek, "empty": []})
            episode_summaries.append({"episode_id": episode_id, "record": record, "onset": event["onset"],
                                      "end": event["end"], "duration": event["duration"], "search_from": search_from,
                                      "n_involved": int(np.isfinite(lat).sum()),
                                      "pot_alarm": det["blind_pot"]["event_alarm"], "pot_dt": det["blind_pot"]["dt"],
                                      "gev_alarm": det["blind_gev"]["event_alarm"], "gev_dt": det["blind_gev"]["dt"],
                                      "pot_false_alarms": len(det["blind_pot"]["false_alarms"]),
                                      "gev_false_alarms": len(det["blind_gev"]["false_alarms"])})
            for row in loc_df.to_dict("records"):
                row.update({"episode_id": episode_id, "record": record, "event_onset": event["onset"],
                            "event_end": event["end"], "search_from": search_from,
                            "detected_s": det["blind_pot"]["event_alarm"]})
                latency_rows.append(row)
            if representative is None or (record == RECORD and k == main_episode):
                representative = {"record": record, "event": event, "series": series, "beats": d["beats"], "X": X,
                                  "Xc": Xc, "lat": lat, "z": z, "tw": tw, "det": det, "hi": hi, "hl": hl,
                                  "coverage": d["coverage"], "search_from": search_from}
    if not episode_ids:
        raise RuntimeError("Нет эпизодов ЖТ с началом после 400 с")
    F = np.stack(features); Y = np.stack(labels); groups = np.asarray(record_groups)
    latencies = pd.DataFrame(latency_rows)
    latencies.to_csv(out / "episode_latencies.csv", index=False)
    pd.DataFrame(episode_summaries).to_csv(out / "episode_detection.csv", index=False)
    corr = np.mean(baseline_corrs, axis=0)
    ep = physiological_edges(); ef = C.topk_edges(corr, 1); ek = C.knn_edges(F.mean(0), 1)
    R = {"dataset": "St Petersburg INCART 12-lead Arrhythmia Database v1.0.0, PhysioNet DOI 10.13026/C2V88N",
         "n_records_requested": len(a.records), "n_records": len(a.records), "records": a.records,
         "n_records_with_eligible_vt": int(len(np.unique(groups))), "n_vt_episodes": len(episode_ids),
         "excluded_episodes": excluded, "n_parameters": 4, "parameters": PARAMETERS, "fs": FS,
         "record": representative["record"], "event": representative["event"],
         "detection": representative["det"], "detection_spec": SPEC,
         "lead_choice": {"selected": "II", "reason": "стандартное ритмическое отведение с различимыми P и R",
                         "cross_lead_parameter": "atan2(чистая площадь QRS aVF, чистая площадь QRS I), градусы"},
         "signal_qrs": {"reference_position_coverage_80ms": representative["coverage"]},
         "cascade": {"window_s": .25, "threshold_mad": MAD_THR, "persistent_windows": 2,
                     "context_before_event_s": 120, "gap_after_previous_vt_s": 60,
                     "baseline_s": [0, B], "representative_record": representative["record"],
                     "latency_by_parameter": dict(zip(PARAMETERS, np.round(representative["lat"], 3)))}}
    R["H0_noise"], I, V = hypothesis(representative["X"], representative["event"])
    R["pr_quality"] = quality_summary(representative["beats"], representative["event"])
    pd.DataFrame(R["pr_quality"]).to_csv(out / "pr_quality.csv", index=False)
    R["graphs"] = {"nodes": PARAMETERS, "n_edges": {"physiological": len(ep), "functional": len(ef), "knn": len(ek)},
                   "edges": {k: [[PARAMETERS[i], PARAMETERS[j]] for i, j in v]
                             for k, v in [("physiological", ep), ("functional", ef), ("knn", ek)]},
                   "rules": {"functional": "top-1 по средней |corr| фона всех записей",
                             "knn": "top-1 по средним признакам всех эпизодов", "physiological": "RR–PR и ST–QRS_axis"},
                   "jaccard": {"physiological-functional": C.jaccard(ep, ef), "physiological-knn": C.jaccard(ep, ek),
                               "functional-knn": C.jaccard(ef, ek)}}
    R["labels"] = {"n": int(Y.size), "involved": int(Y.sum()),
                   "by_parameter": {p: int(Y[:, j].sum()) for j, p in enumerate(PARAMETERS)}}
    if len(np.unique(Y)) == 2:
        runs, oof = run_multi_protocols(F, Y, groups, edge_sets, a.seeds, a.epochs)
        runs.to_csv(out / "gnn_runs.csv", index=False)
        g = runs.groupby(["split", "model", "graph"])[["auroc", "auprc"]].agg(["mean", "std"]).round(4)
        g.columns = ["_".join(c) for c in g.columns]; R["gnn_summary"] = g.reset_index().to_dict("records")
        oof_flat = np.mean([oof[(s, "record", "gat", "physiological")] for s in a.seeds], 0)
        latencies["gnn_oof"] = oof_flat
        tables = [FO.fold_table(episode_ids, groups, FO.random_kfold(len(F), 5, s), "random", s) for s in a.seeds]
        tables.append(FO.fold_table(episode_ids, groups, list(FO.stable_group_kfold(groups)), "record"))
        R["folds_md5"] = FO.save_fold_map(out / "folds.csv", tables)
    else:
        latencies["gnn_oof"] = np.nan; R["gnn_skipped"] = "Метки содержат только один класс"
    latencies.to_csv(out / "episode_latencies.csv", index=False)
    R["localization"], by = summarize_localization(latencies)
    latencies.to_csv(out / "localization_parameters.csv", index=False); by.to_csv(out / "localization_groups.csv")
    feature_df = []
    for e, record, f, y in zip(episode_ids, groups, F, Y):
        for j, parameter in enumerate(PARAMETERS):
            feature_df.append(dict(zip(node_names, f[j])) | {"episode_id": e, "record": record,
                                                             "parameter": parameter, "involved": int(y[j])})
    pd.DataFrame(feature_df).to_csv(out / "node_features.csv", index=False)
    rep = representative
    np.savez(out / "indicators.npz", hi=rep["hi"], hl=rep["hl"], z=rep["z"].astype(np.float32), tw=rep["tw"], lat=rep["lat"],
             series=rep["series"], analysis_series=rep["X"], cascade_series=rep["Xc"], graph_corr=corr,
             rr=rep["series"][:, 0], pr=rep["series"][:, 1], st=rep["series"][:, 2], qrs_axis=rep["series"][:, 3], I=I, V=V)
    x, _, _ = load_record(root, rep["record"]); y = filtered(x)
    _, _, quality, measured_pr = per_second_series(rep["beats"], len(rep["series"]))
    make_figures(y, rep["hi"], rep["det"], rep["z"], rep["tw"], rep["lat"], rep["event"], out,
                 rep["series"], quality, measured_pr)
    R["runtime_s"] = round(time.time() - start, 1)
    (out / "results.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in ["n_records", "n_records_with_eligible_vt", "n_vt_episodes",
                                        "excluded_episodes", "labels", "localization"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

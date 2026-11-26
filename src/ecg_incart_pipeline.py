"""Одна 12-канальная ЭКГ INCART: развитие нетипичной активности, EVT-детекция ЖТ и каскад отведений.

Основная запись I42 выбрана аудитом аннотаций: устойчивая ЖТ на 468.436–527.907 с (175 сокращений).
В I30 самая длинная серия составляет только 4.514 с, поэтому она не используется. Аннотации задают только
эталон события; входной индикатор строится по сигналу без типов сокращений. Подход повторяет пример стерео-ЭЭГ:
гипотеза -> слепая и локальная POT/GEV-детекция -> каскад вовлечения -> граф отведений -> сравнение моделей.

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
GROUPS = np.array(["limb"] * 6 + ["precordial"] * 6)
SHA256 = {
    "I42.dat": "1ed5c6039a2bbb6dd6932a6c45e05ecad2fedb054a479d2a0f444a9134429ede",
    "I42.hea": "eb72e7ae6e2275f53f2b5e38c62374b8ab1a288e6e1758b40db5c26ac23d02bc",
    "I42.atr": "7421ccbac2269f57a768633ee301929d48ed8eade5336d080c0f3aba376f9109",
}
B = 400
WH = 30
MAD_THR = 6.0
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
    """Кандидаты ЖТ: min_beats последовательных V и средняя ЧСС выше min_hr; без склейки серий."""
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


def load(path):
    root = Path(path)
    for name, want in SHA256.items():
        got = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert got == want, f"SHA-256 {name} не совпал: {got}"
    lines = (root / f"{RECORD}.hea").read_text().splitlines()
    _, nsig, fs, n = lines[0].split()[:4]
    assert int(nsig) == 12 and int(fs) == FS and int(n) == N
    fmt = [s.split()[1] for s in lines[1:13]]
    assert set(fmt) == {"16"}, "Ожидается WFDB format 16"
    gain = np.array([int(s.split()[2]) for s in lines[1:13]])
    baseline = np.array([int(s.split()[4]) for s in lines[1:13]])
    raw = np.fromfile(root / f"{RECORD}.dat", dtype="<i2").reshape(N, 12)
    x = (raw - baseline) / gain
    ann = read_annotations(root / f"{RECORD}.atr")
    episodes = vt_episodes(ann)
    assert episodes, "Эпизоды ЖТ не найдены"
    event = max(episodes, key=lambda e: e["duration"])
    assert 460 < event["onset"] < 480 and 55 < event["duration"] < 65, "Основное событие I42 изменилось"
    return x, ann, episodes, event


def filtered(x):
    sos = signal.butter(3, [0.5, 45], btype="band", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def window_features(y):
    """Односекундные признаки только по сигналу: время x (отведение x признак)."""
    nw = len(y) // FS
    x = y[:nw * FS].reshape(nw, FS, 12)
    f, p = signal.periodogram(x, fs=FS, axis=1, window="hann", detrend="linear")
    bands = [(0.5, 4), (4, 8), (8, 15), (15, 30), (30, 45)]
    bp = np.stack([np.log10(p[:, (f >= a) & (f < b), :].sum(1) + 1e-20) for a, b in bands], 2)
    detr = signal.detrend(x, axis=1)
    sd = detr.std(1)
    ll = np.abs(np.diff(x, axis=1)).mean(1)
    xc = x - x.mean(1, keepdims=True)
    ac1 = (xc[:, :-1] * xc[:, 1:]).mean(1) / (x.var(1) + 1e-12)
    sk = stats.skew(x, axis=1)
    ku = stats.kurtosis(x, axis=1)
    bb = np.log10(p.sum(1) + 1e-20)
    logp = np.log10(p + 1e-20); common = np.median(logp, axis=2)
    corr = np.empty_like(bb)
    for j in range(12):
        corr[:, j] = [np.corrcoef(logp[t, :, j], common[t])[0, 1] for t in range(nw)]
    F = np.concatenate([bp, sd[:, :, None], ll[:, :, None], ac1[:, :, None], sk[:, :, None],
                        ku[:, :, None], corr[:, :, None]], axis=2)
    names = ["power_0.5_4", "power_4_8", "power_8_15", "power_15_30", "power_30_45",
             "std_detr", "linelen", "ac1", "skew", "kurtosis", "corr_common"]
    return np.nan_to_num(F), bb, names


def rhythm_morphology(y):
    """Сигнальные R-пики, каузальные ЧСС/RR-CV и расстояние морфологии до шаблона фона 0–400 с."""
    peaks, _ = signal.find_peaks(y[:, 1], distance=int(.28 * FS), prominence=1.0)
    pt = peaks / FS; nw = len(y) // FS
    hr = np.full(nw, np.nan); cv = np.full(nw, np.nan)
    for t in range(nw):
        p = pt[(pt >= max(0, t - 3)) & (pt < t + 1)]
        if len(p) >= 3:
            rr = np.diff(p); hr[t] = 60 / np.median(rr); cv[t] = rr.std() / rr.mean()
    hr = np.where(np.isfinite(hr), hr, np.nanmedian(hr[:B])); cv = np.where(np.isfinite(cv), cv, np.nanmedian(cv[:B]))
    pre, post = 51, 103
    ok = peaks[(peaks >= pre) & (peaks + post < len(y))]
    seg = np.stack([y[p - pre:p + post] for p in ok]); base = ok < B * FS
    template = np.median(seg[base], axis=0)
    a = seg - seg.mean(1, keepdims=True); b = template - template.mean(0, keepdims=True)
    corr = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(0) + 1e-20)
    dist = 1 - corr; morph = np.full((nw, 12), np.nan)
    for t in range(nw):
        vals = dist[(ok / FS >= t) & (ok / FS < t + 1)]
        if len(vals):
            morph[t] = np.median(vals, axis=0)
    for j in range(12):
        morph[:, j] = pd.Series(morph[:, j]).interpolate(limit=1).to_numpy()
        morph[:, j] = np.where(np.isfinite(morph[:, j]), morph[:, j], np.nanmedian(morph[:B, j]))
    return np.c_[hr, -cv, morph], morph, hr, cv, peaks


def interval_features(y, peaks):
    """ST и описательный P–R proxy по сигналу, агрегированные в односекундные ряды."""
    low = signal.sosfiltfilt(signal.butter(3, 15, btype="low", fs=FS, output="sos"), y, axis=0)
    peaks = peaks[(peaks > 100) & (peaks + 40 < len(y))]
    st = np.zeros((len(peaks), 12)); pr = np.zeros_like(st); pq = np.zeros_like(st)
    for k, p in enumerate(peaks):
        base = np.median(low[p - 90:p - 60], axis=0)
        st[k] = np.median(low[p + 20:p + 31], axis=0) - base
        seg = low[p - 77:p - 20] - base
        mad = np.median(np.abs(seg - np.median(seg, axis=0)), axis=0) * 1.4826 + 1e-6
        j = np.argmax(np.abs(seg), axis=0)
        pr[k] = (77 - j) / FS
        pq[k] = np.abs(seg[j, np.arange(12)]) / mad
    pt = peaks / FS; nw = len(y) // FS; seconds = {}
    for name, values in [("st", st), ("pr", pr), ("pq", pq)]:
        out = np.full((nw, 12), np.nan)
        for t in range(nw):
            mask = (pt >= t) & (pt < t + 1)
            if mask.any():
                out[t] = np.median(values[mask], axis=0)
        for j in range(12):
            out[:, j] = pd.Series(out[:, j]).interpolate(limit=1).to_numpy()
            out[:, j] = np.where(np.isfinite(out[:, j]), out[:, j], np.nanmedian(out[:B, j]))
        seconds[name] = out
    return seconds, {"peaks": peaks, "st": st, "pr": pr, "pq": pq}


def quality_summary(beats, event):
    pt = beats["peaks"] / FS; t0, t1 = event["onset"], event["end"]
    intervals = {"background": (0, B), "pre_event": (B, t0), "vt": (t0, t1), "post_recovery": (t1 + 60, N / FS)}
    rows = []
    for name, (a, b) in intervals.items():
        mask = (pt >= a) & (pt < b); pr = beats["pr"][mask]; pq = beats["pq"][mask]
        rows.append({"interval": name, "seconds": [a, b], "n_beats": int(mask.sum()),
                     "pr_proxy_median_s": float(np.median(pr)), "pr_proxy_iqr_s": np.quantile(pr, [.25, .75]).tolist(),
                     "p_quality_median": float(np.median(pq)), "p_quality_gt3_fraction": float((pq > 3).mean())})
    return rows


def indicator(F, b0, b1):
    x = F[b0:b1].reshape(b1 - b0, -1)
    mu = x.mean(0); precision = LedoitWolf().fit(x).precision_
    z = F.reshape(len(F), -1) - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", z, precision, z), 0))


def alarm_episodes(starts, gap=10):
    ep = []
    for s in starts:
        if not ep or s - ep[-1][-1] > gap:
            ep.append([s])
        else:
            ep[-1].append(s)
    return [e[0] for e in ep]


def detection(F, event):
    t0, t1 = event["onset"], event["end"]
    pre = (int(t0) - 300, int(t0) - 30)
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


def hypothesis(F, event, annotations):
    bp = F[:, :, :5]
    high = 10 ** bp[:, :, 3] + 10 ** bp[:, :, 4] + 1e-20
    variance = F[:, :, 5] ** 2
    end = int(event["onset"])
    def at(e):
        I = np.array([high[e - (4 - k) * WH:e - (3 - k) * WH].mean(0) for k in range(4)])
        V = np.array([variance[e - (4 - k) * WH:e - (3 - k) * WH].mean(0) for k in range(4)])
        return I, V
    def slope(m):
        return np.polyfit(np.arange(4), np.log(np.median(m / m[0], 1)), 1)[0]
    I, V = at(end)
    cand = [e for e in range(4 * WH + 30, len(F) - WH, 60) if abs(e - end) > 300]
    sur = {"noise": [], "variance": []}
    for e in cand:
        i, v = at(e); sur["noise"].append(slope(i)); sur["variance"].append(slope(v))
    beats = np.array([t / FS for t, typ, _ in annotations if typ in BEAT_CODES])
    hr, rmssd = [], []
    for k in range(4):
        b = beats[(beats >= end - (4 - k) * WH) & (beats < end - (3 - k) * WH)]
        rr = np.diff(b); hr.append(60 / np.median(rr) if len(rr) else np.nan)
        rmssd.append(np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else np.nan)
    result = {"window_s": WH, "n_surrogate": len(cand), "hr_median_by_window": hr, "rmssd_by_window": rmssd}
    for name, m in [("noise", I), ("variance", V)]:
        obs = slope(m); subj = np.repeat(np.arange(12), 4); win = np.tile(np.arange(4), 12)
        result[name] = {"median_by_window": np.median(m / m[0], 1).tolist(),
                        "frac_channels_increase_T1_T4": float((m[3] > m[0]).mean()),
                        "rm_corr_channels": C.rm_corr(subj, win, np.log(m.T.ravel())),
                        "slope_obs": float(obs), "slope_surrogate_mean": float(np.mean(sur[name])),
                        "p_surrogate_greater": float((np.array(sur[name]) >= obs).mean()),
                        "p_surrogate_less": float((np.array(sur[name]) <= obs).mean())}
    return result, I, V


def cascade(st, event):
    """Каскад по абсолютному ST-отклонению от медианы раннего фона."""
    t0, t1 = event["onset"], event["end"]
    base = (0, B); tw = np.arange(len(st), dtype=float)
    med = np.median(st[:B], 0); mad = np.median(np.abs(st[:B] - med), 0) * 1.4826 + 1e-6
    z = np.abs((st - med) / mad)
    lat = np.full(12, np.nan)
    for j in range(12):
        mask = (z[:, j] > MAD_THR) & (tw >= B) & (tw <= t1)
        k = C.first_persistent(mask, 0, 3)
        if k is not None:
            lat[j] = tw[k] - t0
    dev = np.abs(st - med); foot = []
    for j in range(12):
        thr, _ = C.pot_threshold(dev[:B, j], .9, .99)
        foot.append(float((dev[(tw >= t0) & (tw <= t1), j] > thr).mean()))
    return lat, z, tw, np.array(foot), base


def node_features(F, bb, st, pre):
    a, b = pre
    p = F[a:b]
    dfa = np.array([C.dfa_alpha(bb[a:b, j], scales=[4, 6, 8, 12, 16, 24, 32, 48, 64]) for j in range(12)])
    means = p.mean(0)
    Fnode = np.c_[means, bb[a:b].std(0), dfa, st[a:b].mean(0), st[a:b].std(0)]
    names = ["power_0.5_4", "power_4_8", "power_8_15", "power_15_30", "power_30_45", "std_detr",
             "linelen", "ac1", "skew", "kurtosis", "corr_common", "bb_std", "dfa", "st_mean", "st_std"]
    return np.nan_to_num(Fnode).astype(np.float32), names


def structural_edges():
    return [(i, j) for group in [range(6), range(6, 12)] for i in group for j in group if i < j]


def localization(lat, foot, e, tw, event, oof):
    t0, t1 = event["onset"], event["end"]
    pre = (int(t0) - 300, int(t0) - 30)
    ip = (tw >= pre[0]) & (tw < pre[1]); ie = (tw >= t0) & (tw <= t1)
    xp = e[ip]; mp = np.median(xp, 1)
    beta = xp.T @ mp / (mp @ mp + 1e-12)
    idio = (xp - np.outer(mp, beta)).std(0) + 1e-9
    rz = ((e[ie].mean(0) - xp.mean(0)) - beta * (e[ie].mean() - mp.mean())) / idio
    early = np.isfinite(lat) & (lat <= np.nanpercentile(lat, 10))
    df = pd.DataFrame({"lead": LEADS, "group": GROUPS, "latency": lat, "involved": np.isfinite(lat).astype(int),
                       "footprint": foot, "residual_z": rz, "gnn_oof": oof, "early10": early})
    by = df.groupby("group").agg(footprint=("footprint", "mean"), residual_z=("residual_z", "mean"),
                                 latency_min=("latency", "min"), gnn=("gnn_oof", "mean"), n=("lead", "size"),
                                 n_early=("early10", "sum"))
    lead = {"footprint": by.footprint.idxmax(), "residual_z": by.residual_z.idxmax(),
            "cascade_early10": by.n_early.idxmax(), "gnn_oof": by.gnn.idxmax() if by.gnn.notna().any() else None}
    out = {"interpretation": "информационный след на поверхностных отведениях, не анатомическая локализация очага ЖТ",
           "leading_group": lead, "leading_group_agree": int(sum(v == lead["footprint"] for v in lead.values())),
           "earliest_leads": df.dropna(subset=["latency"]).sort_values("latency").head(6)[["lead", "latency"]].round(3).values.tolist(),
           "n_involved": int(np.isfinite(lat).sum()), "n_early10": int(early.sum()),
           "footprint_by_lead": dict(zip(LEADS, np.round(foot, 4))),
           "residual_z_by_lead": dict(zip(LEADS, np.round(rz, 3))),
           "spearman": {"latency_vs_footprint": float(stats.spearmanr(lat, foot, nan_policy="omit")[0]),
                        "latency_vs_residual": float(stats.spearmanr(lat, rz, nan_policy="omit")[0]),
                        "footprint_vs_residual": float(stats.spearmanr(foot, rz)[0])},
           "by_group": by.round(4).reset_index().to_dict("records")}
    return out, df, by


def run_protocols(F, y, E, seeds, epochs):
    import torch
    _, NodeGAT, MLP = C.make_models()
    rows, oof = [], {}
    for seed in seeds:
        for split in ["random", "lead"]:
            folds = list(KFold(5, shuffle=True, random_state=seed).split(F)) if split == "random" else list(LeaveOneOut().split(F))
            for model, gname in [("gat", "structural"), ("gat", "functional"), ("gat", "knn"),
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


def make_figures(y, hi, det, z, tw, lat, event, out, st, pr, pq, hr, cv):
    import matplotlib.pyplot as plt
    t0, t1 = event["onset"], event["end"]
    fig, axs = plt.subplots(4, 1, figsize=(15, 11), sharex=False)
    taxis = np.arange(len(y)) / FS
    for ax, j in zip(axs[:3], [1, 6, 10]):
        ax.plot(taxis, y[:, j], lw=.5); ax.set_xlim(350, 580); ax.set_ylabel(LEADS[j] + ", mV"); ax.grid(alpha=.2)
        ax.axvspan(t0, t1, color="r", alpha=.1); ax.axvline(t0, color="r", ls="--", lw=.8)
    axs[3].plot(hi, lw=.8, label="health indicator")
    axs[3].axhline(det["blind_pot"]["thr"], color="r", ls="--", label="POT threshold")
    axs[3].axhline(det["blind_gev"]["thr"], color="orange", ls=":", label="GEV threshold")
    axs[3].axvspan(t0, t1, color="r", alpha=.1); axs[3].set_xlim(350, 650); axs[3].set_ylabel("indicator")
    axs[3].set_xlabel("time, s"); axs[3].legend(); axs[3].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / "ecg_evt_detection.png", dpi=160); plt.close(fig)
    sel = (tw >= 380) & (tw <= 570)
    fig, ax = plt.subplots(figsize=(15, 6))
    im = ax.imshow(z[sel].T, aspect="auto", origin="lower", extent=[tw[sel][0], tw[sel][-1], 0, 12],
                   vmin=0, vmax=15, cmap="magma")
    ax.axvline(t0, color="cyan", ls="--", lw=1); ax.axvline(t1, color="cyan", ls="--", lw=1)
    ax.set_yticks(np.arange(12) + .5, LEADS); ax.set_xlabel("time, s"); ax.set_ylabel("lead")
    ax.set_title("Absolute ST-shift deviation z-score and VT cascade")
    fig.colorbar(im, ax=ax, label="z"); fig.tight_layout(); fig.savefig(out / "ecg_vt_cascade.png", dpi=160); plt.close(fig)

    tt = np.arange(len(st)); view = (350, 580)
    fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    axs[0].plot(tt, hr, label="heart rate", lw=.8); axs[0].plot(tt, cv * 100, label="RR-CV × 100", lw=.8)
    axs[0].set_ylabel("bpm / CV×100"); axs[0].legend(loc="upper left")
    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    for j in range(12):
        axs[1].plot(tt, st[:, j], color=colors[j], lw=.6, alpha=.85, label=LEADS[j])
        if np.isfinite(lat[j]):
            axs[1].scatter(t0 + lat[j], st[int(round(t0 + lat[j])), j], s=35,
                           facecolors="white", edgecolors="red", zorder=4)
    axs[1].set_ylabel("ST shift, mV"); axs[1].legend(ncol=6, fontsize=7, loc="upper left")
    reliable = np.where(pq > 3, pr, np.nan)
    for j in range(12):
        axs[2].plot(tt, reliable[:, j], color=colors[j], lw=.6, alpha=.65)
        axs[3].plot(tt, pq[:, j], color=colors[j], lw=.6, alpha=.65)
    axs[2].set_ylabel("P–R proxy, s\nonly P quality > 3")
    axs[3].axhline(3, color="orange", ls="--", lw=.8); axs[3].set_ylim(0, 15)
    axs[3].set_ylabel("P quality, MAD"); axs[3].set_xlabel("time, s")
    for ax in axs:
        ax.axvline(det["blind_pot"]["event_alarm"], color="red", lw=1, label="POT" if ax is axs[0] else None)
        ax.axvline(det["blind_gev"]["event_alarm"], color="orange", lw=1, label="GEV" if ax is axs[0] else None)
        ax.axvline(t0, color="black", ls="--", lw=.8, label="VT onset" if ax is axs[0] else None)
        ax.set_xlim(*view); ax.grid(alpha=.2)
    axs[0].legend(loc="upper left", ncol=4)
    fig.suptitle("Event cascade: RR, ST, P–R proxy and P quality across all 12 leads")
    fig.tight_layout(); fig.savefig(out / "ecg_parameter_cascade.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/INCART"); ap.add_argument("--out", default="../results/ecg")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); start = time.time()
    x, annotations, episodes, event = load(a.data); y = filtered(x)
    R = {"dataset": "St Petersburg INCART 12-lead Arrhythmia Database v1.0.0, PhysioNet DOI 10.13026/C2V88N",
         "record": RECORD, "n_leads": 12, "fs": FS, "duration_s": N / FS, "sha256": SHA256,
         "record_selection": {"I30_rejected": "максимальная серия ЖТ 4.514 с", "I42_selected": "одиночная ЖТ 59.471 с"},
         "event": event, "all_vt_episodes": episodes, "detection_spec": SPEC}
    F, bb, fnames = window_features(y)
    _, _, hr, cv, peaks = rhythm_morphology(y)
    seconds, beats = interval_features(y, peaks)
    st, pr, pq = seconds["st"], seconds["pr"], seconds["pq"]
    ref = np.array([t for t, typ, _ in annotations if typ in BEAT_CODES])
    coverage = float(np.mean([np.min(np.abs(peaks - t)) <= int(.08 * FS) for t in ref]))
    R["signal_qrs"] = {"lead": "II", "min_distance_s": .28, "prominence_mv": 1.0,
                       "n_detected": len(peaks), "reference_position_coverage_80ms": coverage,
                       "note": "позиции и типы аннотаций не входят в индикатор; покрытие — последующая проверка"}
    R["interval_features"] = {
        "rr": "сигнальные R-пики на II; каузальные окна 4 с",
        "st": "медиана ЭКГ 78–121 мс после R минус база 350–233 мс до R, мВ",
        "pr_proxy": "задержка P-кандидата до R; не клинический PR и не входит в EVT-индикатор",
        "p_quality": "амплитуда P-кандидата / MAD участка перед ним; низкое качество не равно PR=0"}
    R["indicator_features"] = ["heart_rate", "negative_rr_cv"] + ["st_" + n for n in LEADS]
    R["detection"], hi, hl, pre = detection(np.c_[hr, -cv, st], event)
    R["H0_noise"], I, V = hypothesis(F, event, annotations)
    lat, z, tw, foot, cascade_base = cascade(st, event)
    R["cascade"] = {"feature": "absolute ST-shift deviation from baseline median", "baseline": list(cascade_base),
                    "search_from": cascade_base[1], "window_s": 1, "threshold_mad": MAD_THR, "persistent_windows": 3,
                    "latency_by_lead": dict(zip(LEADS, np.round(lat, 3))),
                    "recruited_leads": [LEADS[j] for j in range(12) if np.isfinite(lat[j])]}
    R["st_median_by_lead"] = {name: dict(zip(LEADS, np.round(np.median(st[(np.arange(len(st)) >= a) & (np.arange(len(st)) < b)], axis=0), 4)))
                              for name, (a, b) in {"background": (0, B), "pre_event": (B, int(event["onset"])),
                                                    "vt": (int(event["onset"]), int(event["end"]))}.items()}
    R["pr_quality"] = quality_summary(beats, event)
    pd.DataFrame(R["pr_quality"]).to_csv(out / "pr_quality.csv", index=False)
    Fnode, node_names = node_features(F, bb, st, pre)
    ef = C.functional_edges(st[:B], k=3); ek = C.knn_edges(Fnode, k=3); es = structural_edges()
    E = {"structural": es, "functional": ef, "knn": ek, "empty": []}
    labels = np.isfinite(lat).astype(int)
    R["graphs"] = {"n_edges": {k: len(v) for k, v in E.items() if k != "empty"},
                   "jaccard": {"structural-functional": C.jaccard(es, ef), "structural-knn": C.jaccard(es, ek),
                               "functional-knn": C.jaccard(ef, ek)}}
    R["labels"] = {"involved": int(labels.sum()), "n": len(labels), "by_lead": dict(zip(LEADS, labels.tolist()))}
    if len(np.unique(labels)) == 2:
        runs, oof = run_protocols(Fnode, labels, E, a.seeds, a.epochs)
        runs.to_csv(out / "gnn_runs.csv", index=False)
        g = runs.groupby(["split", "model", "graph"])[["auroc", "auprc"]].agg(["mean", "std"]).round(4)
        g.columns = ["_".join(c) for c in g.columns]; R["gnn_summary"] = g.reset_index().to_dict("records")
        oof_gnn = np.mean([oof[(s, "lead", "gat", "structural")] for s in a.seeds], 0)
        tables = [FO.fold_table(LEADS, GROUPS, FO.random_kfold(len(labels), 5, s), "random", s) for s in a.seeds]
        tables.append(FO.fold_table(LEADS, GROUPS, list(LeaveOneOut().split(Fnode)), "lead"))
        R["folds_md5"] = FO.save_fold_map(out / "folds.csv", tables)
    else:
        R["gnn_skipped"] = "Каскад дал только один класс отведений; классификация не определена"
        oof_gnn = np.full(12, np.nan)
    R["localization"], loc, by = localization(lat, foot, st, tw, event, oof_gnn)
    loc.to_csv(out / "localization_leads.csv", index=False); by.to_csv(out / "localization_groups.csv")
    pd.DataFrame(Fnode, columns=node_names).assign(lead=LEADS, group=GROUPS, involved=labels).to_csv(out / "node_features.csv", index=False)
    np.savez(out / "indicators.npz", hi=hi, hl=hl, z=z.astype(np.float32), tw=tw, lat=lat,
             st=st, pr=pr, pq=pq, hr=hr, rr_cv=cv, I=I, V=V)
    make_figures(y, hi, R["detection"], z, tw, lat, event, out, st, pr, pq, hr, cv)
    R["runtime_s"] = round(time.time() - start, 1)
    (out / "results.json").write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in ["event", "detection", "H0_noise", "labels", "localization"]},
                     ensure_ascii=False, indent=1, default=str)[:9000])


if __name__ == "__main__":
    main()

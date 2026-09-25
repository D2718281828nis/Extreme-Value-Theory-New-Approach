"""Пример 1. Биомедицинская сложная система: головной мозг (стерео-ЭЭГ, эпилептический приступ).

Та же последовательность шагов методологии, что в примерах с авиационным двигателем и фондовым рынком:
гипотеза шумового предвестника -> EVT-детекция -> графовое представление -> ГНС с протоколом без утечки ->
локализация источника. Считается по производному набору `sEEG-HFOs-8_cases_derived.npz`, который строит
`brain_extract_edf.py` из открытой записи (Zenodo, DOI 10.5281/zenodo.21967993; MD5 EDF проверяется).

Запуск: python brain_seeg_pipeline.py --data ../data/sEEG/sEEG-HFOs-8_cases_derived.npz --out ../results/brain
"""
from __future__ import annotations
import argparse, json, os, re, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import signal, stats
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import common as C

warnings.filterwarnings("ignore")
EDF_MD5 = "c83e85316860502327f6b7ed7ccb15d2"          # совпадает с MD5 файла на странице Zenodo
ONSET = 10396.445                                      # аннотация врача «приступ + БТКП»
B = 3600                                               # фон: первый час записи (0–3600 с)
PRE = (10096, 10386)                                   # окно признаков узлов: 290 с до события (заканчивается за 10 с до аннотации)
SEIZ = (10396, 10452)                                  # интервал приступа до постиктального подавления (оценка по данным, см. text)
BLOCK = 10                                             # длина блока для блочных максимумов, с
WH = 30                                                # окна T1..T4 по 30 с, как в работе Karpov et al. (2021)
MAD_THR = 6.0                                          # порог вовлечения, как RECRUITMENT_THRESHOLD_MAD в extreme_event_agent


def load(path):
    if str(path).lower().endswith(".csv"):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from generate_brain_toy import build_dataset
        d = build_dataset(Path(path))
    else:
        d = np.load(path)
    keys = d.files if hasattr(d, "files") else d.keys()
    kind = str(d["dataset_kind"]) if "dataset_kind" in keys else "zenodo_derived"
    assert kind == "synthetic_toy" or str(d["edf_md5"]) == EDF_MD5, "MD5 исходного EDF не совпадает с Zenodo"
    names = [n.replace("EEG ", "") for n in d["names"]]
    shaft = [re.match(r"([A-Za-z]+'?)", n).group(1) for n in names]
    num = [int(re.search(r"(\d+)$", n).group(1)) for n in names]
    return d, names, np.array(shaft), np.array(num)


def band_logpower(d, lo, hi):
    f = d["freqs"]; P = 10 ** d["logspec"].astype(np.float32)
    return np.log10(P[:, :, (f >= lo) & (f < hi)].sum(2))


# ------------------------------------------------------------------ 1. гипотеза (дизайн Karpov et al., 2021)
def noise_intensity(Pw, f):
    """I = ∫_1^30 C f^{-alpha} df, степенной закон подогнан в лог-лог координатах к спектру 1–30 Гц."""
    sel = (f >= 1) & (f <= 30); lf = np.log10(f[sel])
    Y = np.log10(Pw[:, sel]); X = np.c_[np.ones_like(lf), lf]
    a, b = np.linalg.lstsq(X, Y.T, rcond=None)[0]; Cc = 10 ** a; al = -b
    return np.where(np.abs(1 - al) > 1e-6, Cc * (30 ** (1 - al) - 1) / (1 - al), Cc * np.log(30))


def hypothesis(d):
    f = d["freqs"]; P = 10 ** d["logspec"].astype(np.float32); std = d["std_detr"]
    def at(end):
        I = np.array([noise_intensity(P[end - (4 - k) * WH:end - (3 - k) * WH].mean(0), f) for k in range(4)])
        V = np.array([(std[end - (4 - k) * WH:end - (3 - k) * WH] ** 2).mean(0) for k in range(4)])
        return I, V
    end = int(ONSET)
    I, V = at(end)
    Cn = I.shape[1]; subj = np.repeat(np.arange(Cn), 4); win = np.tile(np.arange(4), Cn)
    slope = lambda M: np.polyfit(np.arange(4), np.log(np.median(M / M[0], 1)), 1)[0]
    obs_I, obs_V = slope(I), slope(V)
    cand = [e for e in range(4 * WH + 30, 10000, 60) if abs(e - end) > 300]
    sI, sV = [], []
    for e in cand:
        Ii, Vi = at(e); sI.append(slope(Ii)); sV.append(slope(Vi))
    sI, sV = np.array(sI), np.array(sV)
    return {"noise": {"median_by_window": np.median(I / I[0], 1).round(4).tolist(),
                      "frac_channels_increase_T1_T4": float((I[3] > I[0]).mean()),
                      "rm_corr_channels": C.rm_corr(subj, win, np.log(I.T.ravel())),
                      "slope_obs": float(obs_I), "slope_surrogate_mean": float(sI.mean()),
                      "p_surrogate_greater": float((sI >= obs_I).mean()), "p_surrogate_less": float((sI <= obs_I).mean())},
            "variance": {"median_by_window": np.median(V / V[0], 1).round(4).tolist(),
                         "rm_corr_channels": C.rm_corr(subj, win, np.log(V.T.ravel())),
                         "slope_obs": float(obs_V), "p_surrogate_greater": float((sV >= obs_V).mean())},
            "n_surrogate": len(cand), "window_s": WH}, I, V


# ------------------------------------------------------------------ 2. EVT-детекция
def indicator(X, b0, b1):
    mu = X[b0:b1].mean(0); Pm = LedoitWolf().fit(X[b0:b1]).precision_; Z = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", Z, Pm, Z))


def episodes(starts, gap=60):
    ep = []
    for s in starts:
        if not ep or s - ep[-1][-1] > gap: ep.append([s])
        else: ep[-1].append(s)
    return [e[0] for e in ep]


def all_runs(mask, start, k):
    out, run = [], 0
    for i in range(start, len(mask)):
        run = run + 1 if mask[i] else 0
        if run == k: out.append(i - k + 1)
    return out


def detection(d):
    bb = band_logpower(d, 1, 68); T = bb.shape[0]
    hi = indicator(bb, 0, B)
    thr_pot, xi_pot = C.pot_threshold(hi[:B], 0.8, 0.999)
    pot_eps = episodes(all_runs(hi > thr_pot, B, 3))
    bm = hi[: T // BLOCK * BLOCK].reshape(-1, BLOCK).max(1)
    thr_gev, xi_gev = C.gev_threshold(bm[: B // BLOCK], 0.99)
    gev_eps = episodes([k * BLOCK for k in all_runs(bm > thr_gev, B // BLOCK, 2)])
    def classify(eps):
        ev = [e for e in eps if SEIZ[0] - 60 <= e <= SEIZ[1]]
        fa = [e for e in eps if e not in ev]
        return {"episodes": eps, "event_alarm": ev[0] if ev else None, "dt": None if not ev else float(ev[0] - ONSET),
                "false_alarms": fa, "false_alarms_per_hour": len(fa) / ((SEIZ[0] - 60 - B) / 3600)}
    # целевой вариант (Положение 2): полоса 13–67 Гц, фон — локальная база перед событием
    hg = band_logpower(d, 13, 68); hl = indicator(hg, *PRE)
    thr_l, _ = C.pot_threshold(hl[PRE[0]:PRE[1]], 0.8, 0.999); kl = C.first_persistent(hl > thr_l, PRE[1], 3)
    base = hi[:B] / np.median(hi[:B]); full = hi / np.median(hi[:B])
    res = {"blind_pot": {"thr": thr_pot, "xi": xi_pot, **classify(pot_eps)},
           "blind_gev": {"thr": thr_gev, "xi": xi_gev, "block_s": BLOCK, **classify(gev_eps)},
           "targeted_local": {"band": [13, 67], "baseline": list(PRE), "thr": thr_l, "first_alarm": kl,
                              "dt": None if kl is None else float(kl - ONSET)},
           "test_hours": (SEIZ[0] - 60 - B) / 3600,
           "heavy_tail": {k: {"kurtosis": float(stats.kurtosis(x)), "q999": float(np.quantile(x, 0.999))} for k, x in [("baseline", base), ("full", full)]}}
    return res, hi, hl, thr_pot, thr_gev, thr_l


def sensitivity_dropout(d, names):
    """Анализ чувствительности (post hoc): провалы широкополосной мощности контакта ниже фона более чем на 6 робастных σ
    (возможный обрыв контакта) на проверочном интервале; слепая детекция без таких контактов."""
    bb = band_logpower(d, 1, 68); med = np.median(bb[:B], 0); mad = np.median(np.abs(bb[:B] - med), 0) * 1.4826
    low = (bb - med) / mad < -6
    cnt = {names[c]: int(low[B:SEIZ[0] - 60, c].sum()) for c in range(len(names)) if low[B:SEIZ[0] - 60, c].any()}
    out = {"rule": "z(лог-мощность 1–67 Гц) < −6 относительно медианы и 1,4826·MAD первого часа, секунды 3600–10336", "channels": cnt}
    def run(excl):
        keep = [i for i, n in enumerate(names) if n not in excl]; hi = indicator(bb[:, keep], 0, B)
        tp, _ = C.pot_threshold(hi[:B], 0.8, 0.999); T = len(hi)
        bm = hi[: T // BLOCK * BLOCK].reshape(-1, BLOCK).max(1); tg, _ = C.gev_threshold(bm[: B // BLOCK], 0.99)
        pot = episodes(all_runs(hi > tp, B, 3)); gev = episodes([k * BLOCK for k in all_runs(bm > tg, B // BLOCK, 2)])
        ev = lambda e: SEIZ[0] - 60 <= e <= SEIZ[1]
        return {"pot_event": [float(e - ONSET) for e in pot if ev(e)], "pot_false": [e for e in pot if not ev(e)],
                "gev_event": [float(e - ONSET) for e in gev if ev(e)], "gev_false": [e for e in gev if not ev(e)],
                "single_exceed_pre": [t for t in range(int(ONSET) - 60, int(ONSET)) if hi[t] > tp]}
    out["without_PA9"] = run(["PA9"]); out["without_all_dropout_channels"] = run(list(cnt))
    return out


# ------------------------------------------------------------------ 3. каскад вовлечения (250 мс) и метки узлов
def cascade(d, names, base=PRE, s0=None):
    fs = int(d["fs"]); raw = d["raw_event"].astype(np.float32) * d["gain"][:, None].astype(np.float32)
    t0 = int(d["event_raw_start"]); b, a = signal.butter(4, [13, 67], btype="band", fs=fs)
    y = signal.filtfilt(b, a, raw, axis=1); w = fs // 4; nW = y.shape[1] // w
    E = np.log10((y[:, :nW * w].reshape(len(names), nW, w) ** 2).mean(2) + 1e-12); tw = t0 + np.arange(nW) * 0.25
    s0 = base[1] if s0 is None else s0
    ib = (tw >= base[0]) & (tw < base[1]); med = np.median(E[:, ib], 1); mad = np.median(np.abs(E[:, ib] - med[:, None]), 1) * 1.4826
    z = (E - med[:, None]) / mad[:, None]
    lat = np.full(len(names), np.nan)
    for c in range(len(names)):
        k = C.first_persistent((z[c] > MAD_THR) & (tw >= s0) & (tw <= SEIZ[1]), 0, 2)
        if k is not None: lat[c] = tw[k] - ONSET
    return lat, z, tw


def check_311(z, tw, names):
    """Максимальная z-оценка пяти контактов, названных в § 3.11 самыми ранними (латентность −28,934 с), в интервале −36…−21 с."""
    sel = (tw >= ONSET - 36) & (tw <= ONSET - 21)
    return {n: float(z[names.index(n), sel].max()) for n in ["R1", "FD1", "FD2", "FD3", "PA'5"] if n in names}


# ------------------------------------------------------------------ 4. признаки узлов, графы
def node_features(d):
    f = d["freqs"]; P = 10 ** d["logspec"][PRE[0]:PRE[1]].astype(np.float32)
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 68)]
    bp = np.stack([np.log10(P[:, :, (f >= a) & (f < b)].sum(2)).mean(0) for a, b in bands], 1)
    bb = np.log10(P.sum(2)); cm = bb.mean(1)
    dfa = np.array([C.dfa_alpha(bb[:, c], scales=[4, 6, 8, 12, 16, 24, 32, 48, 64]) for c in range(bb.shape[1])])
    corr = np.array([np.corrcoef(bb[:, c], cm)[0, 1] for c in range(bb.shape[1])])
    F = np.c_[bp, bb.std(0), np.log10(np.median(d["linelen"][PRE[0]:PRE[1]], 0) + 1e-6), d["ac1"][PRE[0]:PRE[1]].mean(0), dfa, corr, stats.skew(bb, 0)]
    return np.nan_to_num(F).astype(np.float32), ["delta", "theta", "alpha", "beta", "gamma", "bb_std", "linelen", "ac1", "dfa", "corr_cm", "skew"]


def structural_edges(shaft, num):
    E = []
    for s in np.unique(shaft):
        ids = sorted(np.where(shaft == s)[0], key=lambda i: num[i])
        E += [tuple(sorted((ids[k], ids[k + 1]))) for k in range(len(ids) - 1)]
    return sorted(E)


# ------------------------------------------------------------------ 5. ГНС и протокол без утечки
def run_protocols(F, y, E, groups, seeds, epochs):
    import torch
    _, NodeGAT, MLP = C.make_models()
    rows = []; OOF = {}
    for seed in seeds:
        for split in ["random", "shaft"]:
            folds = list(KFold(5, shuffle=True, random_state=seed).split(F)) if split == "random" else list(GroupKFold(5).split(F, groups=groups))
            for model, gname in [("gat", "structural"), ("gat", "functional"), ("gat", "knn"), ("gat", "empty"), ("mlp", "-"), ("lr", "-")]:
                P = np.zeros(len(y))
                for tr, te in folds:
                    sc = StandardScaler().fit(F[tr]); Fs = sc.transform(F).astype(np.float32)
                    if model == "lr":
                        P[te] = LogisticRegression(max_iter=3000).fit(Fs[tr], y[tr]).predict_proba(Fs[te])[:, 1]; continue
                    torch.manual_seed(seed); np.random.seed(seed)
                    ytr = torch.tensor(y[tr], dtype=torch.float32)
                    pw = torch.tensor((1 - y[tr].mean()) / max(y[tr].mean(), 1e-3), dtype=torch.float32)
                    lf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
                    if model == "mlp":
                        net = MLP(Fs.shape[1]); opt = torch.optim.Adam(net.parameters(), 5e-3, weight_decay=5e-4)
                        xt = torch.tensor(Fs[tr])
                        for _ in range(epochs):
                            net.train(); opt.zero_grad(); lf(net(xt), ytr).backward(); opt.step()
                        net.eval()
                        with torch.no_grad(): P[te] = torch.sigmoid(net(torch.tensor(Fs[te]))).numpy()
                    else:
                        ei = C.edge_index_from(E[gname], len(y)); x = torch.tensor(Fs); mk = torch.zeros(len(y), dtype=torch.bool); mk[tr] = True
                        yt = torch.tensor(y, dtype=torch.float32)
                        net = NodeGAT(Fs.shape[1]); opt = torch.optim.Adam(net.parameters(), 5e-3, weight_decay=5e-4)
                        for _ in range(epochs):
                            net.train(); opt.zero_grad(); lf(net(x, ei)[mk], yt[mk]).backward(); opt.step()
                        net.eval()
                        with torch.no_grad(): P[te] = torch.sigmoid(net(x, ei)).numpy()[te]
                rows.append(dict(seed=seed, split=split, model=model, graph=gname,
                                 auroc=float(roc_auc_score(y, P)), auprc=float(average_precision_score(y, P))))
                OOF[(seed, split, model, gname)] = P
                print(rows[-1], flush=True)
    return pd.DataFrame(rows), OOF


# ------------------------------------------------------------------ 6. локализация
def localization(d, names, shaft, num, lat, oof_gnn):
    hg = band_logpower(d, 13, 68)
    thr = np.array([C.pot_threshold(hg[:B, c], 0.9, 0.99)[0] for c in range(hg.shape[1])])
    s0 = int(ONSET) + 11                    # выраженная фаза приступа (см. текст): +11 ... +56 с
    foot = (hg[s0:SEIZ[1]] > thr).mean(0)
    m = hg.mean(1); pre = slice(*PRE)
    Xp = hg[pre] - hg[pre].mean(0); mp = m[pre] - m[pre].mean(); beta = Xp.T @ mp / (mp @ mp); idio = (Xp - np.outer(mp, beta)).std(0)
    rz = ((hg[s0:SEIZ[1]].mean(0) - hg[pre].mean(0)) - beta * (m[s0:SEIZ[1]].mean() - m[pre].mean())) / idio
    hem = np.array(["L" if s.endswith("'") else "R" for s in shaft])
    prior = np.array([(s == "PM" and 3 <= k <= 8) or (s == "CC" and 8 <= k <= 10) for s, k in zip(shaft, num)])
    early = np.isfinite(lat) & (lat <= np.nanpercentile(lat, 10))
    df = pd.DataFrame({"name": names, "shaft": shaft, "hem": hem, "prior": prior, "latency": lat, "footprint": foot,
                       "residual_z": rz, "gnn_oof": oof_gnn, "early10": early})
    def LI(v):
        r, l = np.nanmean(v[hem == "R"]), np.nanmean(v[hem == "L"]); return float((r - l) / (abs(r) + abs(l)))
    by = df.groupby("shaft").agg(footprint=("footprint", "mean"), residual_z=("residual_z", "mean"),
                                 latency_min=("latency", "min"), gnn=("gnn_oof", "mean"), n=("name", "size"))
    df["early10_f"] = early.astype(float)
    by["early10_share"] = df.groupby("shaft").early10_f.mean(); by["n_early10"] = df.groupby("shaft").early10_f.sum()
    lead = {"footprint": by.footprint.idxmax(), "residual_z": by.residual_z.idxmax(), "cascade_early10": by.n_early10.idxmax(), "gnn_oof": by.gnn.idxmax()}
    fin = by.latency_min.notna()
    out = {"leading_shaft": lead, "leading_shaft_agree": int(sum(v == lead["footprint"] for v in lead.values())),
           "shaft_spearman_latency_vs_footprint": float(stats.spearmanr(by.latency_min[fin], by.footprint[fin])[0]),
           "n_early10": int(early.sum()), "hem_counts": {"R": int((hem == "R").sum()), "L": int((hem == "L").sum())},
           "footprint_top3_shafts": by.footprint.sort_values(ascending=False).head(3).round(3).to_dict(),
           "residual_top3_shafts": by.residual_z.sort_values(ascending=False).head(3).round(2).to_dict(),
           "earliest_contacts": df.dropna(subset=["latency"]).sort_values("latency").head(8)[["name", "latency"]].round(2).values.tolist(),
           "gnn_top3_shafts": by.gnn.sort_values(ascending=False).head(3).round(3).to_dict(),
           "involved_share_by_hem": {"R": float(np.isfinite(lat[hem == "R"]).mean()), "L": float(np.isfinite(lat[hem == "L"]).mean())},
           "LI": {"involved_share": LI(np.isfinite(lat).astype(float)), "footprint": LI(foot), "residual_z": LI(rz), "early10_share": LI(early.astype(float)), "gnn_oof": LI(oof_gnn)},
           "spearman": {"footprint_vs_residual": float(stats.spearmanr(foot, rz)[0]),
                        "latency_vs_residual": float(stats.spearmanr(lat, rz, nan_policy="omit")[0]),
                        "latency_vs_footprint": float(stats.spearmanr(lat, foot, nan_policy="omit")[0]),
                        "shaft_footprint_vs_gnn": float(stats.spearmanr(by.footprint, by.gnn)[0])},
           "prior": {"n": int(prior.sum()), "footprint_prior_vs_other": [float(foot[prior].mean()), float(foot[~prior].mean())],
                     "residual_prior_vs_other": [float(rz[prior].mean()), float(rz[~prior].mean())],
                     "mw_p_residual": float(stats.mannwhitneyu(rz[prior], rz[~prior]).pvalue),
                     "prior_in_early10": int((prior & early).sum()), "involved_prior": int(np.isfinite(lat[prior]).sum())}}
    return out, df, by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/sEEG/sEEG-HFOs-8_cases_derived.npz"); ap.add_argument("--out", default="../results/brain")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    d, names, shaft, num = load(a.data)
    keys = d.files if hasattr(d, "files") else d.keys()
    toy = "dataset_kind" in keys and str(d["dataset_kind"]) == "synthetic_toy"
    R = {"dataset": "synthetic brain toy dataset" if toy else "sEEG-HFOs-8.edf, Zenodo DOI 10.5281/zenodo.21967993",
         "synthetic": toy, "edf_md5": str(d["edf_md5"]), "n_contacts": len(names),
         "n_shafts": int(len(np.unique(shaft))), "fs": int(d["fs"]), "crop_s": int(d["crop"]),
         "annotations": list(zip(d["ann_onset"].tolist(), [str(x) for x in d["ann_text"]])), "onset": ONSET}
    ll = d["linelen"]; flat = np.argwhere(ll <= 1e-6)
    R["data_quality"] = {"flat_seconds": sorted({int(t) for t, _ in flat}), "flat_channels": sorted({names[c] for _, c in flat}),
                         "flat_second_channel": [[int(t), names[c]] for t, c in flat]}
    R["H0_noise"], I, V = hypothesis(d)
    R["detection"], hi, hl, tp, tg, tl = detection(d)
    np.savez(f"{a.out}/indicators.npz", hi=hi, hl=hl, thr_pot=tp, thr_gev=tg, thr_local=tl, I=I, V=V)
    fs_ = set(R["data_quality"]["flat_seconds"])
    for k in ["blind_pot", "blind_gev"]:
        w = R["detection"][k].get("block_s", 1)
        R["detection"][k]["false_alarm_flat_signal"] = [e for e in R["detection"][k]["false_alarms"] if any(e <= t < e + 3 * w for t in fs_)]
    R["detection"]["blind_pot"]["confirm_third_second"] = R["detection"]["blind_pot"]["event_alarm"] + 2 if R["detection"]["blind_pot"]["event_alarm"] else None
    ex = hi > tp; R["detection"]["single_second_exceedances_pre_onset"] = [int(t) for t in range(int(ONSET) - 60, int(ONSET)) if ex[t]]
    R["sensitivity_dropout"] = sensitivity_dropout(d, names)
    lat, z, tw = cascade(d, names)
    R["check_311_early5_maxz"] = check_311(z, tw, names)
    # чувствительность каскада: базовая линия, заканчивающаяся за 96 с до отметки, и поиск вовлечения с −96 с
    lat2, z2, _ = cascade(d, names, base=(10020, 10300))
    o2 = np.argsort(np.where(np.isfinite(lat2), lat2, 1e9)); e2 = np.isfinite(lat2) & (lat2 <= np.nanpercentile(lat2, 10))
    R["cascade_sensitivity"] = {"baseline": [10020, 10300], "search_from_rel": round(10300 - ONSET, 3), "involved": int(np.isfinite(lat2).sum()),
                                "earliest6": [[names[i], round(float(lat2[i]), 2)] for i in o2[:6]],
                                "early10_by_shaft": pd.Series(shaft[e2]).value_counts().to_dict(),
                                "check_311_early5_maxz": check_311(z2, tw, names)}
    y = np.isfinite(lat).astype(int)
    F, fnames = node_features(d)
    Es = structural_edges(shaft, num); Ef = C.topk_edges(np.abs(d["corr_base"]), 3); Ek = C.knn_edges(F, 3)
    E = {"structural": Es, "functional": Ef, "knn": Ek, "empty": []}
    shares = pd.Series(shaft).value_counts(normalize=True)
    R["graphs"] = {"n_edges": {k: len(v) for k, v in E.items() if k != "empty"},
                   "jaccard": {"structural-functional": C.jaccard(Es, Ef), "structural-knn": C.jaccard(Es, Ek), "functional-knn": C.jaccard(Ef, Ek)},
                   "within_shaft_frac": {"functional": float(np.mean([shaft[i] == shaft[j] for i, j in Ef])),
                                         "knn": float(np.mean([shaft[i] == shaft[j] for i, j in Ek]))},
                   "within_shaft_random": float((shares ** 2).sum() - 1 / len(shaft)) ,
                   "label_homophily": {g: float(np.mean([y[i] == y[j] for i, j in E[g]])) for g in ["structural", "functional", "knn"]}}
    R["labels"] = {"involved": int(y.sum()), "n": int(len(y)), "latency_quantiles": np.nanpercentile(lat, [0, 10, 50, 90, 100]).round(2).tolist(),
                   "univariate_auroc": {n: float(roc_auc_score(y, F[:, k])) for k, n in enumerate(fnames)},
                   "involved_by_shaft": {str(sh): [int(y[shaft == sh].sum()), int((shaft == sh).sum())] for sh in np.unique(shaft)}}
    RUNS, OOF = run_protocols(F, y, E, shaft, a.seeds, a.epochs); RUNS.to_csv(f"{a.out}/gnn_runs.csv", index=False)
    G = RUNS.groupby(["split", "model", "graph"])[["auroc", "auprc"]].agg(["mean", "std"]).round(4)
    G.columns = ["_".join(c) for c in G.columns]; R["gnn_summary"] = G.reset_index().to_dict("records")
    piv = RUNS[RUNS.model == "gat"].pivot_table(index=["seed", "split"], columns="graph", values="auroc")
    gain = (piv["structural"] - piv["empty"]).groupby(level="split").agg(["mean", "min", "max"])
    R["graph_gain_structural_minus_empty_auroc"] = gain.round(4).to_dict("index")
    oof = np.mean([OOF[(s, "shaft", "gat", "structural")] for s in a.seeds], 0)
    R["localization"], LOC, BY = localization(d, names, shaft, num, lat, oof)
    LOC.to_csv(f"{a.out}/localization_contacts.csv", index=False); BY.to_csv(f"{a.out}/localization_shafts.csv")
    pd.DataFrame(F, columns=fnames).assign(name=names, shaft=shaft, involved=y).to_csv(f"{a.out}/node_features.csv", index=False)
    np.savez(f"{a.out}/cascade.npz", z=z.astype(np.float32), tw=tw, lat=lat)
    R["runtime_s"] = round(time.time() - t0, 1)
    json.dump(R, open(f"{a.out}/results.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: R[k] for k in ["H0_noise", "detection", "graphs", "labels", "localization"]}, ensure_ascii=False, indent=1, default=str)[:7000])


if __name__ == "__main__":
    main()

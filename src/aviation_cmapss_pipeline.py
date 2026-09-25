"""Пример 1. Авиационная сложная система: турбовентиляторный двигатель (NASA C-MAPSS).

Проверка переносимости методологии исследования (гипотеза -> окна/признаки -> EVT-детекция ->
граф -> ГНС с leakage-safe протоколом -> локализация источника) на небиомедицинском объекте.

Данные: NASA PCoE, C-MAPSS (Saxena et al., 2008), открытый датасет:
  Zenodo  DOI 10.5281/zenodo.15346912 (CMAPSSData.zip, md5 79a22f36e80606c69d0e9e4da5bb2b7a)
  Kaggle  https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
Файлы train_FD00x.txt идентичны побайтно в трёх независимых зеркалах (md5 ниже, проверяются при запуске).

Запуск:  python aviation_cmapss_pipeline.py --data ../data/CMAPSSData --out ../results/aviation
"""
from __future__ import annotations
import argparse, hashlib, json, os, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import GroupKFold, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import common as C

warnings.filterwarnings("ignore")
MD5 = {"train_FD001.txt": "259f340bac32ce6fa8894815600fa757", "train_FD003.txt": "81298e81977a53aa268ad500ac42b2d7"}
COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
# 14 информативных датчиков (в FD001 датчики s1, s5, s6, s10, s16, s18, s19 постоянны)
S = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
NAME = {"s2": "T24", "s3": "T30", "s4": "T50", "s7": "P30", "s8": "Nf", "s9": "Nc", "s11": "Ps30", "s12": "phi",
        "s13": "NRf", "s14": "NRc", "s15": "BPR", "s17": "htBleed", "s20": "W31", "s21": "W32"}
# Узел газовоздушного тракта по описанию выходов C-MAPSS (Saxena et al., 2008, Table 2)
MOD = {"s2": "LPC", "s3": "HPC", "s4": "LPT", "s7": "HPC", "s8": "Fan", "s9": "HPC", "s11": "HPC", "s12": "Burner",
       "s13": "Fan", "s14": "HPC", "s15": "Fan", "s17": "HPC", "s20": "HPT", "s21": "LPT"}
MOD_ADJ = [("Fan", "LPC"), ("LPC", "HPC"), ("HPC", "Burner"), ("Burner", "HPT"), ("HPT", "LPT"),
           ("LPT", "Fan"),  # вал низкого давления
           ("HPC", "HPT")]  # вал высокого давления
B, BL = 60, 3          # фоновый участок (циклы) и длина блока для блочных максимумов
W, STRIDE, RUL_CRIT = 30, 3, 30


def load(path, fd):
    f = os.path.join(path, f"train_{fd}.txt")
    md5 = hashlib.md5(open(f, "rb").read()).hexdigest()
    assert md5 == MD5[f"train_{fd}.txt"], f"контрольная сумма {f} не совпала: {md5}"
    d = pd.read_csv(f, sep=r"\s+", header=None, names=COLS)
    d["rul"] = d.groupby("unit").cycle.transform("max") - d.cycle
    return d, md5


# ------------------------------------------------------------------ 1. гипотеза
def hypothesis_noise(d):
    """H1 (аналог Karpov et al., 2021): интенсивность шума растёт перед экстремальным событием.
    Четыре окна T1..T4 по W циклов, последнее заканчивается отказом."""
    rows = []
    for u, g in d.groupby("unit"):
        n = len(g)
        for k in range(4):
            seg = g.iloc[n - (4 - k) * W: n - (3 - k) * W]
            t = np.arange(W)
            for s in S:
                y = seg[s].values.astype(float)
                r = y - np.polyval(np.polyfit(t, y, 2), t)
                rows.append(dict(unit=u, win=k + 1, s=s, noise=r.std(ddof=3), var_raw=y.var(ddof=1),
                                 ac1=np.corrcoef(r[:-1], r[1:])[0, 1]))
    R = pd.DataFrame(rows)
    for c in ["noise", "var_raw"]:
        R[c + "_rel"] = R[c] / R.groupby(["unit", "s"])[c].transform("first")
    E = R.groupby(["unit", "win"]).agg(noise=("noise_rel", "median"), var_raw=("var_raw_rel", "median"),
                                       ac1=("ac1", "median")).reset_index()
    out = {}
    for c in ["noise", "var_raw", "ac1"]:
        y = np.log(E[c]) if c != "ac1" else E[c]
        out[c] = {"median_by_window": E.groupby("win")[c].median().round(4).tolist(),
                  "rm_corr": C.rm_corr(E.unit, E.win, y),
                  "friedman_p": float(stats.friedmanchisquare(*[E[E.win == k][c].values for k in range(1, 5)]).pvalue)}
    return out, E


# ------------------------------------------------------------------ 2. индикатор состояния и EVT
def health_indicator(X, b=B):
    mu = X[:b].mean(0)
    P = LedoitWolf().fit(X[:b]).precision_
    Z = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", Z, P, Z))


def detect_engine(g):
    X = g[S].values.astype(float); n = len(X)
    hi = health_indicator(X)
    nb = n // BL
    bmax = hi[: nb * BL].reshape(-1, BL).max(1)
    thr_gev, xi_gev = C.gev_threshold(bmax[: B // BL], 0.99)
    kb = C.first_persistent(bmax > thr_gev, B // BL, 2)
    det_gev = None if kb is None else kb * BL
    thr_pot, xi_pot = C.pot_threshold(hi[:B], 0.8, 0.999)
    det_pot = C.first_persistent(hi > thr_pot, B, 3)
    # специфичность на заведомо «здоровом» участке: подгонка по циклам 1-30, проверка на 31-60
    thr_h, _ = C.gev_threshold(bmax[: 30 // BL], 0.99)
    fa_gev = C.first_persistent(bmax[: B // BL] > thr_h, 30 // BL, 2) is not None
    thr_hp, _ = C.pot_threshold(hi[:30], 0.8, 0.999)
    fa_pot = C.first_persistent(hi[:B] > thr_hp, 30, 3) is not None
    return dict(life=n, det_gev=det_gev, det_pot=det_pot,
                lead_gev=None if det_gev is None else n - det_gev,
                lead_pot=None if det_pot is None else n - det_pot,
                xi_gev=xi_gev, xi_pot=xi_pot, fa_gev_healthy=fa_gev, fa_pot_healthy=fa_pot), hi


def heavy_tail(his):
    base = np.concatenate([h[:B] / np.median(h[:B]) for h in his])
    full = np.concatenate([h / np.median(h[:B]) for h in his])
    res = {}
    for k, x in {"baseline": base, "full_life": full}.items():
        u = np.quantile(x, 0.95)
        xi, _, beta = stats.genpareto.fit(x[x > u] - u, floc=0)
        res[k] = {"gpd_xi_top5pct": float(xi), "kurtosis": float(stats.kurtosis(x)), "q999": float(np.quantile(x, 0.999))}
    return res, base, full


# ------------------------------------------------------------------ 3. графы
def physical_edges():
    E = set()
    for i, a in enumerate(S):
        for j, b in enumerate(S):
            if i < j and (MOD[a] == MOD[b] or (MOD[a], MOD[b]) in MOD_ADJ or (MOD[b], MOD[a]) in MOD_ADJ):
                E.add((i, j))
    return sorted(E)


def window_features(d, normalized=True):
    """Окна W циклов с шагом STRIDE; для каждого узла-датчика 4 признака."""
    feats, ys, groups, ruls = [], [], [], []
    t = np.arange(W)
    for u, g in d.groupby("unit"):
        X = g[S].values.astype(float); rul = g.rul.values
        mu, sd = X[:B].mean(0), X[:B].std(0) + 1e-9
        Z = (X - mu) / sd if normalized else X
        for e in range(W, len(X) + 1, STRIDE):
            seg = Z[e - W: e]
            F = []
            for j in range(len(S)):
                y = seg[:, j]
                p = np.polyfit(t, y, 1); r = y - np.polyval(p, t)
                F.append([y.mean(), p[0] * W, r.std(), np.corrcoef(r[:-1], r[1:])[0, 1] if normalized else y.std()])
            feats.append(F); ys.append(int(rul[e - 1] <= RUL_CRIT)); groups.append(u); ruls.append(rul[e - 1])
    return np.nan_to_num(np.array(feats, float)), np.array(ys), np.array(groups), np.array(ruls)


def functional_graph(d):
    Z = []
    for u, g in d.groupby("unit"):
        X = g[S].values.astype(float)
        Z.append((X - X[:B].mean(0)) / (X[:B].std(0) + 1e-9))
    return C.functional_edges(np.vstack(Z), k=3)


def knn_graph(F):
    node_desc = F.mean(0)  # усреднённые по окнам признаки узла
    return C.knn_edges(node_desc, k=3)


# ------------------------------------------------------------------ 4. ГНС и протокол без утечки
def run_cv(F, y, groups, E, split, seed, model="gat", epochs=40):
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    GraphGAT, _, MLP = C.make_models()
    rng = np.random.RandomState(seed)
    if split == "group":
        folds = list(GroupKFold(5).split(F, y, groups))
    else:
        folds = list(KFold(5, shuffle=True, random_state=seed).split(F, y))
    P = np.zeros(len(y))
    ei = C.edge_index_from(E, len(S))
    for tr, te in folds:
        torch.manual_seed(seed); np.random.seed(seed)
        sc = StandardScaler().fit(F[tr].reshape(-1, F.shape[2]))
        Fs = sc.transform(F.reshape(-1, F.shape[2])).reshape(F.shape).astype(np.float32)
        if model == "lr":
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(Fs[tr].reshape(len(tr), -1), y[tr])
            P[te] = clf.predict_proba(Fs[te].reshape(len(te), -1))[:, 1]; continue
        pos_w = torch.tensor((1 - y[tr].mean()) / y[tr].mean(), dtype=torch.float32)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
        if model == "mlp":
            net = MLP(Fs.shape[1] * Fs.shape[2]); opt = torch.optim.Adam(net.parameters(), 5e-3, weight_decay=1e-4)
            Xtr = torch.tensor(Fs[tr].reshape(len(tr), -1)); Ytr = torch.tensor(y[tr], dtype=torch.float32)
            for ep in range(epochs):
                net.train(); perm = torch.randperm(len(tr))
                for i in range(0, len(tr), 256):
                    b = perm[i:i + 256]; opt.zero_grad(); l = lossf(net(Xtr[b]), Ytr[b]); l.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                P[te] = torch.sigmoid(net(torch.tensor(Fs[te].reshape(len(te), -1)))).numpy()
            continue
        data = [Data(x=torch.tensor(Fs[i]), edge_index=ei, y=torch.tensor([float(y[i])])) for i in range(len(y))]
        net = GraphGAT(Fs.shape[2]); opt = torch.optim.Adam(net.parameters(), 5e-3, weight_decay=1e-4)
        dl = DataLoader([data[i] for i in tr], batch_size=256, shuffle=True)
        for ep in range(epochs):
            net.train()
            for bt in dl:
                opt.zero_grad(); l = lossf(net(bt.x, bt.edge_index, bt.batch), bt.y); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            dte = DataLoader([data[i] for i in te], batch_size=1024)
            P[te] = np.concatenate([torch.sigmoid(net(bt.x, bt.edge_index, bt.batch)).numpy() for bt in dte])
    return C.metrics(y, P), P


# ------------------------------------------------------------------ 5. локализация источника (FD003)
def fd001_signature(d1):
    sig = []
    for u, g in d1.groupby("unit"):
        X = g[S].values; sig.append(np.sign(X[-20:].mean(0) - X[:50].mean(0)))
    return np.sign(np.mean(sig, 0))


def localize_fd003(d3, sig, F3, y3, g3, E, seeds):
    DISC = ["s7", "s12", "s15", "s20", "s21"]  # датчики, знак отклика которых в FD003 бимодален
    idx = [S.index(s) for s in DISC]
    rows = []
    for u, g in d3.groupby("unit"):
        X = g[S].values.astype(float); n = len(X)
        z = (X[-20:].mean(0) - X[:50].mean(0)) / X[:50].std(0)
        agree = float(np.mean(np.sign(z[idx]) == sig[idx]))
        # (Б) каскад вовлечения: первое устойчивое EVT-превышение по каждому датчику
        on = {}
        for j, s in enumerate(S):
            zz = np.abs((X[:, j] - X[:B, j].mean()) / X[:B, j].std())
            nb = n // BL; bm = zz[: nb * BL].reshape(-1, BL).max(1)
            th, _ = C.gev_threshold(bm[: B // BL], 0.99)
            k = C.first_persistent(bm > th, B // BL, 3)
            on[s] = n if k is None else k * BL
        fan_on = min(on[s] for s in S if MOD[s] == "Fan"); hpc_on = min(on[s] for s in S if MOD[s] == "HPC")
        rows.append(dict(unit=u, life=n, agree_hpc_signature=agree, source_sig="HPC" if agree >= 0.6 else "Fan",
                         cascade_source="Fan" if fan_on < hpc_on else "HPC",
                         z_Nf=z[S.index("s8")], z_NRf=z[S.index("s13")], z_BPR=z[S.index("s15")], z_P30=z[S.index("s7")],
                         first3=",".join(sorted(S, key=lambda s: on[s])[:3])))
    L = pd.DataFrame(rows)
    # (В) ГНС: окклюзионная значимость узлов-датчиков (обнуление признаков модуля) в критических окнах
    imp = run_gnn_occlusion(F3, y3, g3, E, seeds)
    L = L.merge(imp, on="unit", how="left")
    return L


def run_gnn_occlusion(F, y, groups, E, seeds, epochs=40):
    import torch
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    GraphGAT, _, _ = C.make_models()
    ei = C.edge_index_from(E, len(S))
    fan = [S.index(s) for s in S if MOD[s] == "Fan"]; hpc = [S.index(s) for s in S if MOD[s] == "HPC"]
    acc = {}
    for seed in seeds:
        for tr, te in GroupKFold(5).split(F, y, groups):
            torch.manual_seed(seed)
            sc = StandardScaler().fit(F[tr].reshape(-1, F.shape[2]))
            Fs = sc.transform(F.reshape(-1, F.shape[2])).reshape(F.shape).astype(np.float32)
            zero = sc.transform(np.zeros((1, F.shape[2])))[0].astype(np.float32)  # «здоровое» значение признака
            data = [Data(x=torch.tensor(Fs[i]), edge_index=ei, y=torch.tensor([float(y[i])])) for i in range(len(y))]
            net = GraphGAT(F.shape[2]); opt = torch.optim.Adam(net.parameters(), 5e-3, weight_decay=1e-4)
            pw = torch.tensor((1 - y[tr].mean()) / y[tr].mean(), dtype=torch.float32)
            lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
            dl = DataLoader([data[i] for i in tr], batch_size=256, shuffle=True)
            for ep in range(epochs):
                net.train()
                for bt in dl:
                    opt.zero_grad(); l = lossf(net(bt.x, bt.edge_index, bt.batch), bt.y); l.backward(); opt.step()
            net.eval()
            crit = [i for i in te if y[i] == 1]
            with torch.no_grad():
                for i in crit:
                    xs = [Fs[i].copy() for _ in range(3)]
                    xs[1][fan] = zero; xs[2][hpc] = zero
                    bt = Batch.from_data_list([Data(x=torch.tensor(x), edge_index=ei) for x in xs])
                    lg = net(bt.x, bt.edge_index, bt.batch).numpy()
                    u = groups[i]
                    a = acc.setdefault(u, [[], []])
                    a[0].append((lg[0] - lg[1]) / len(fan)); a[1].append((lg[0] - lg[2]) / len(hpc))
    rows = []
    for u, (f, h) in acc.items():
        f, h = float(np.mean(f)), float(np.mean(h))
        rows.append(dict(unit=u, occl_fan_per_node=f, occl_hpc_per_node=h, gnn_source="Fan" if f > h else "HPC"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/CMAPSSData"); ap.add_argument("--out", default="../results/aviation")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--reuse", action="store_true", help="взять уже посчитанные прогоны ГНС и локализации из CSV")
    ap.add_argument("--stage", default="all", choices=["all", "gnn", "loc", "final"],
                    help="gnn/loc — только прогоны ГНС (для заданных зёрен) или только локализация, для параллельного запуска; final — сборка results.json из CSV")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    t0 = time.time(); R = {"dataset": "NASA C-MAPSS (FD001, FD003)", "zenodo_doi": "10.5281/zenodo.15346912"}
    d1, m1 = load(a.data, "FD001"); d3, m3 = load(a.data, "FD003"); R["md5"] = {"train_FD001.txt": m1, "train_FD003.txt": m3}
    R["n_units"] = {"FD001": int(d1.unit.nunique()), "FD003": int(d3.unit.nunique())}
    R["life_FD001"] = d1.groupby("unit").cycle.max().describe().round(1).to_dict()

    # 1. гипотеза
    R["H1_noise"], E_noise = hypothesis_noise(d1); E_noise.to_csv(f"{a.out}/h1_windows.csv", index=False)
    # 2. EVT-детекция
    det, his = [], []
    for u, g in d1.groupby("unit"):
        r, hi = detect_engine(g); r["unit"] = u; det.append(r); his.append(hi)
    D = pd.DataFrame(det); D.to_csv(f"{a.out}/detection_FD001.csv", index=False)
    np.save(f"{a.out}/hi_FD001.npy", np.array(his, dtype=object), allow_pickle=True)
    R["detection"] = {
        "detected": {k: int(D[f"det_{k}"].notna().sum()) for k in ["gev", "pot"]},
        "lead_median": {k: float(D[f"lead_{k}"].median()) for k in ["gev", "pot"]},
        "lead_iqr": {k: [float(D[f"lead_{k}"].quantile(.25)), float(D[f"lead_{k}"].quantile(.75))] for k in ["gev", "pot"]},
        "lead_min": {k: float(D[f"lead_{k}"].min()) for k in ["gev", "pot"]},
        "lead_frac_life_median_gev": float((D.lead_gev / D.life).median()),
        "spearman_gev_pot": float(stats.spearmanr(D.lead_gev, D.lead_pot, nan_policy="omit")[0]),
        "false_alarm_engines_healthy_31_60": {"gev": int(D.fa_gev_healthy.sum()), "pot": int(D.fa_pot_healthy.sum())},
        "xi_gev_median": float(D.xi_gev.median())}
    R["heavy_tail"], base, full = heavy_tail(his)
    np.save(f"{a.out}/hi_pooled_base.npy", base); np.save(f"{a.out}/hi_pooled_full.npy", full)

    # 3. графы
    F1, y1, g1, r1 = window_features(d1, True)
    Eg = {"physical": physical_edges(), "functional": functional_graph(d1), "knn": knn_graph(F1), "empty": []}
    R["graphs"] = {k: {"n_edges": len(v), "edges": [[S[i], S[j]] for i, j in v]} for k, v in Eg.items()}
    R["jaccard"] = {f"{p}-{q}": C.jaccard(Eg[p], Eg[q]) for p, q in [("physical", "functional"), ("physical", "knn"), ("functional", "knn")]}
    R["windows_FD001"] = {"n": int(len(y1)), "pos_frac": float(y1.mean()), "W": W, "stride": STRIDE, "rul_crit": RUL_CRIT}

    # 4. ГНС + leakage-safe
    Fr, yr, gr, _ = window_features(d1, False)
    runs = []
    if a.stage == "final":
        import glob
        parts = sorted(glob.glob(f"{a.out}/gnn_runs_seed*.csv"))
        if parts:
            pd.concat([pd.read_csv(p) for p in parts]).sort_values(["seed"], kind="stable").to_csv(f"{a.out}/gnn_runs.csv", index=False)
        a.reuse = True
    if a.stage == "loc":
        a.seeds_gnn = []
    elif a.reuse and os.path.exists(f"{a.out}/gnn_runs.csv"):
        runs = pd.read_csv(f"{a.out}/gnn_runs.csv").to_dict("records"); a.seeds_gnn = []
    else:
        a.seeds_gnn = a.seeds
    for seed in a.seeds_gnn:
        for feat, (F, y, g) in {"normalized": (F1, y1, g1), "raw": (Fr, yr, gr)}.items():
            for split in ["group", "random"]:
                for model in ["gat", "mlp", "lr"]:
                    m, _ = run_cv(F, y, g, Eg["physical"], split, seed, model, a.epochs)
                    runs.append(dict(seed=seed, features=feat, split=split, model=model, graph="physical" if model == "gat" else "-", **m))
                    print(runs[-1], flush=True)
        for gname in ["functional", "knn", "empty"]:
            m, _ = run_cv(F1, y1, g1, Eg[gname], "group", seed, "gat", a.epochs)
            runs.append(dict(seed=seed, features="normalized", split="group", model="gat", graph=gname, **m)); print(runs[-1], flush=True)
    RUNS = pd.DataFrame(runs)
    if a.stage == "gnn":
        RUNS.to_csv(f"{a.out}/gnn_runs_seed{'-'.join(map(str, a.seeds))}.csv", index=False); return
    if a.stage == "all":
        RUNS.to_csv(f"{a.out}/gnn_runs.csv", index=False)
    if a.stage == "loc":
        sig = fd001_signature(d1); F3, y3, g3, _ = window_features(d3, True)
        L = localize_fd003(d3, sig, F3, y3, g3, Eg["physical"], a.seeds); L.to_csv(f"{a.out}/localization_FD003.csv", index=False); return
    G = RUNS.groupby(["features", "split", "model", "graph"])[["auprc", "macro_f1", "auroc"]].agg(["mean", "std"]).round(4)
    G.columns = ["_".join(c) for c in G.columns]; R["gnn_summary"] = G.reset_index().to_dict("records")

    # 5. локализация источника на FD003
    sig = fd001_signature(d1)
    F3, y3, g3, _ = window_features(d3, True)
    if a.reuse and os.path.exists(f"{a.out}/localization_FD003.csv"):
        L = pd.read_csv(f"{a.out}/localization_FD003.csv")
    else:
        L = localize_fd003(d3, sig, F3, y3, g3, Eg["physical"], a.seeds); L.to_csv(f"{a.out}/localization_FD003.csv", index=False)
    R["localization"] = {
        "signature_counts": L.source_sig.value_counts().to_dict(),
        "agreement_values": L.agree_hpc_signature.value_counts().sort_index().to_dict(),
        "cascade_vs_signature": float((L.cascade_source == L.source_sig).mean()),
        "gnn_vs_signature": float((L.gnn_source == L.source_sig).mean()),
        "gnn_vs_cascade": float((L.gnn_source == L.cascade_source).mean()),
        "median_by_source": L.groupby("source_sig")[["z_Nf", "z_NRf", "z_BPR", "z_P30", "life"]].median().round(2).to_dict(),
        "mannwhitney_life_p": float(stats.mannwhitneyu(L[L.source_sig == "HPC"].life, L[L.source_sig == "Fan"].life).pvalue)}
    R["runtime_s"] = round(time.time() - t0, 1)
    json.dump(R, open(f"{a.out}/results.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: R[k] for k in ["H1_noise", "detection", "heavy_tail", "jaccard", "localization"]}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()

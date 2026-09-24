"""Пример 2. Финансовая сложная система: фондовый рынок (470 акций индекса S&P 500, 2013-2018).

Та же последовательность шагов методологии, что и в примере 1: гипотеза -> EVT-детекция экстремальных
событий -> графовое представление -> ГНС (узловая классификация) с leakage-safe протоколом -> локализация.

Данные (открытые):
  Kaggle «S&P 500 stock data» (C. Nugent), файл all_stocks_5yr.csv, 2013-02-08 ... 2018-02-07, 505 тикеров:
    https://www.kaggle.com/datasets/camnugent/sandp500 ; md5 6d2f3f2529cf6d8b443c8f4beee5638b
    (побайтно совпадающие копии: github.com/plotly/datasets, github.com/MainakRepositor/Datasets,
     github.com/rishabhzn200/Stock-Data-Visualization-Using-Bokeh)
  Секторы GICS на дату окончания ряда: github.com/datasets/s-and-p-500-companies-financials,
    data/constituents-financials.csv, коммит c9f83a9 от 2018-02-08 (505 из 505 тикеров совпадают);
    md5 8e571d9a5791c6355cd5ef8455d56920

Запуск: python finance_sp500_pipeline.py --data ../data --out ../results/finance
"""
from __future__ import annotations
import argparse, hashlib, json, os, time, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import common as C

warnings.filterwarnings("ignore")
MD5 = {"all_stocks_5yr.csv": "6d2f3f2529cf6d8b443c8f4beee5638b",
       "constituents_financials_2018-02-08.csv": "8e571d9a5791c6355cd5ef8455d56920"}
B = 500            # фоновый (обучающий) период: первые 500 торговых дней, 2013-02-11 ... 2015-02-04
WPRE = 60          # окно признаков перед событием (торговых дней)
WH = 20            # длина окон T1..T4 при проверке гипотезы
MIN_SECTOR = 10    # секторы меньше 10 акций не ранжируются при локализации


def md5(f):
    return hashlib.md5(open(f, "rb").read()).hexdigest()


def load(path):
    f1 = os.path.join(path, "all_stocks_5yr.csv"); f2 = os.path.join(path, "constituents_financials_2018-02-08.csv")
    for f in (f1, f2):
        assert md5(f) == MD5[os.path.basename(f)], f"контрольная сумма {f} не совпала"
    df = pd.read_csv(f1, parse_dates=["date"])
    O = df.pivot(index="date", columns="Name", values="open"); P = df.pivot(index="date", columns="Name", values="close")
    full = P.columns[P.notna().all()]; P, O = P[full], O[full]
    R = np.log(P).diff().iloc[1:]; intra = np.log(P / O).iloc[1:]
    bad = R.abs() > 0.5   # корпоративные действия (выделения компаний) и одна ошибка цены: заменяются внутридневной доходностью
    fixes = [(str(R.index[i].date()), R.columns[j]) for i, j in zip(*np.where(bad.values))]
    R = R.mask(bad, intra)
    sec = pd.read_csv(f2).set_index("Symbol").Sector.reindex(R.columns)
    return R, sec, fixes


# ------------------------------------------------------------------ 1. EVT-детекция
def detect_market(R):
    m = R.mean(1); L = -m
    thr_pot, xi_pot = C.pot_threshold(L.iloc[:B].values, 0.9, 0.998)
    bm = L.iloc[:B].values.reshape(-1, 5).max(1)
    thr_gev, xi_gev = C.gev_threshold(bm, 0.99)
    test = L.iloc[B:]
    ext = test[test > thr_pot]
    wk = test.values[: len(test) // 5 * 5].reshape(-1, 5).max(1)
    wk_dates = test.index[: len(test) // 5 * 5].values.reshape(-1, 5)
    gev_weeks = [str(pd.Timestamp(wk_dates[k, 0]).date()) for k in np.where(wk > thr_gev)[0]]
    base_exc = int((L.iloc[:B] > thr_pot).sum())
    # тяжёлый хвост: GPD-параметр формы верхних 5% потерь фона и всего ряда
    def xi_top(x):
        u = np.quantile(x, 0.95); return float(stats.genpareto.fit(x[x > u] - u, floc=0)[0])
    return dict(thr_pot=thr_pot, xi_pot=xi_pot, thr_gev_weekly=thr_gev, xi_gev=xi_gev,
                gpd_xi_top5pct={"baseline": xi_top(L.iloc[:B].values), "full": xi_top(L.values)},
                kurtosis={"baseline": float(stats.kurtosis(L.iloc[:B])), "full": float(stats.kurtosis(L))},
                extreme_days={str(d.date()): float(v) for d, v in ext.items()},
                n_test_days=int(len(test)), expected_exceedances=float(len(test) * 0.002),
                binom_p=float(stats.binomtest(len(ext), len(test), 0.002, alternative="greater").pvalue),
                baseline_exceedances=base_exc, gev_extreme_weeks=gev_weeks, n_test_weeks=int(len(wk)),
                baseline_period=[str(L.index[0].date()), str(L.index[B - 1].date())]), m


# ------------------------------------------------------------------ 2. гипотеза шумового предвестника
def window_stats(X, m, i0, w=WH):
    Rw = X[i0:i0 + w]; mw = m[i0:i0 + w]
    Rc = Rw - Rw.mean(0); mc = mw - mw.mean()
    beta = Rc.T @ mc / (mc @ mc); res = Rc - np.outer(mc, beta)
    Z = Rc / Rc.std(0); Cm = Z.T @ Z / w; n = Cm.shape[0]
    return dict(vol=float(mw.std()), idio=float(np.median(res.std(0))), corr=float((Cm.sum() - n) / (n * (n - 1))),
                lam1=float(np.linalg.eigvalsh(Cm)[-1] / n), ac1=float(np.corrcoef(mw[:-1], mw[1:])[0, 1]))


def hypothesis_noise(R, m, onsets, extreme_days):
    X = R.values; idx = R.index
    rows = []
    for t0 in onsets:
        i = idx.get_loc(pd.Timestamp(t0))
        for k in range(4):
            rows.append(dict(ep=t0, win=k + 1, **window_stats(X, m.values, i - (4 - k) * WH)))
    E = pd.DataFrame(rows)
    out = {"windows": E.round(5).to_dict("records")}
    ie = [idx.get_loc(pd.Timestamp(t)) for t in extreme_days]
    cand = [i for i in range(4 * WH, len(idx)) if all(abs(i - j) > 60 for j in ie)]
    cache = {}
    def ws(i0):
        if i0 not in cache: cache[i0] = window_stats(X, m.values, i0)
        return cache[i0]
    def slope(i, c):
        v = np.array([ws(i - (4 - k) * WH)[c] for k in range(4)])
        return np.polyfit(np.arange(4), np.log(v) if c != "ac1" else v, 1)[0]
    rng = np.random.RandomState(0)
    for c in ["vol", "idio", "corr", "lam1", "ac1"]:
        y = np.log(E[c]) if c != "ac1" else E[c]
        obs = float(np.mean([slope(idx.get_loc(pd.Timestamp(t)), c) for t in onsets]))
        sur = np.array([slope(i, c) for i in cand])
        null = rng.choice(sur, (20000, len(onsets))).mean(1)
        out[c] = {"rm_corr": C.rm_corr(E.ep, E.win, y), "mean_slope_obs": obs, "mean_slope_surrogate": float(sur.mean()),
                  "p_surrogate_greater": float((null >= obs).mean()), "n_surrogate_dates": len(cand)}
    return out, E


# ------------------------------------------------------------------ 3. признаки, метки, графы
def episode_data(R, m, sec, thr, pre_end, label_day):
    X = R.values; idx = R.index
    i = idx.get_loc(pd.Timestamp(pre_end)); Rw = X[i - WPRE + 1: i + 1]; mw = m.values[i - WPRE + 1: i + 1]
    Rc = Rw - Rw.mean(0); mc = mw - mw.mean()
    beta = Rc.T @ mc / (mc @ mc); res = Rc - np.outer(mc, beta)
    corr = (Rc * mc[:, None]).mean(0) / (Rc.std(0) * mc.std())
    dfa = np.array([C.dfa_alpha(Rw[:, j], scales=[4, 6, 8, 10, 15]) for j in range(Rw.shape[1])])
    ac1 = np.array([np.corrcoef(Rw[:-1, j], Rw[1:, j])[0, 1] for j in range(Rw.shape[1])])
    cum = np.cumsum(Rw, 0); mdd = (np.maximum.accumulate(cum, 0) - cum).max(0)
    F = np.c_[Rw.std(0), beta, corr, res.std(0), dfa, ac1, stats.skew(Rw, 0), mdd, Rw.sum(0), Rw.std(0) / thr]
    j = idx.get_loc(pd.Timestamp(label_day)); y = (-X[j] > thr).astype(int)
    E_fun = C.functional_edges(Rw, k=5)
    E_knn = C.knn_edges(F, k=5)
    return np.nan_to_num(F).astype(np.float32), y, E_fun, E_knn, Rw


def sector_edges(sec):
    s = sec.values; E = []
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if s[i] == s[j]: E.append((i, j))
    return E


# ------------------------------------------------------------------ 4. ГНС и протокол без утечки
def train_node_gat(graphs, train_masks, seed, epochs=200):
    import torch
    _, NodeGAT, _ = C.make_models()
    torch.manual_seed(seed); np.random.seed(seed)
    net = NodeGAT(graphs[0][0].shape[1]); opt = torch.optim.Adam(net.parameters(), 5e-3, weight_decay=5e-4)
    ys = np.concatenate([g[1][mk] for g, mk in zip(graphs, train_masks)])
    pw = torch.tensor((1 - ys.mean()) / max(ys.mean(), 1e-3), dtype=torch.float32)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    T = [(torch.tensor(F), C.edge_index_from(E, len(F)), torch.tensor(y, dtype=torch.float32), torch.tensor(mk))
         for (F, y, E), mk in zip(graphs, train_masks)]
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        loss = sum(lossf(net(x, ei)[mk], y[mk]) for x, ei, y, mk in T if mk.any())
        loss.backward(); opt.step()
    return net


def predict(net, F, E):
    import torch
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(F), C.edge_index_from(E, len(F)))).numpy()


def run_protocols(EPD, sec, seeds, epochs):
    """EPD: {episode: dict(F, y, E=dict(graph->edges))}. Три протокола: случайное разбиение узлов,
    разбиение с группировкой по сектору, перенос на незнакомый эпизод (leave-one-episode-out)."""
    import torch
    sg = sec.values
    rows = []
    eps = list(EPD)
    for seed in seeds:
        for split in ["random", "sector", "loeo"]:
            for model, gname in [("gat", "sector"), ("gat", "functional"), ("gat", "knn"), ("gat", "empty"), ("mlp", "-"), ("lr", "-")]:
                P = {e: np.zeros(len(EPD[e]["y"])) for e in eps}
                for e in eps:
                    F, y = EPD[e]["F"], EPD[e]["y"]
                    if split == "loeo":
                        folds = [(None, np.arange(len(y)))]
                    elif split == "random":
                        folds = [(a, b) for a, b in KFold(5, shuffle=True, random_state=seed).split(F)]
                    else:
                        folds = [(a, b) for a, b in GroupKFold(5).split(F, groups=sg)]
                    for tr, te in folds:
                        if split == "loeo":
                            train_eps = [x for x in eps if x != e]
                            Ftr = np.vstack([EPD[x]["F"] for x in train_eps]); sc = StandardScaler().fit(Ftr)
                            gtr = [(sc.transform(EPD[x]["F"]).astype(np.float32), EPD[x]["y"], EPD[x]["E"].get(gname, [])) for x in train_eps]
                            mtr = [np.ones(len(EPD[x]["y"]), bool) for x in train_eps]
                            Fte = sc.transform(F).astype(np.float32)
                            Xtr = np.vstack([g[0] for g in gtr]); ytr = np.concatenate([g[1] for g in gtr])
                        else:
                            sc = StandardScaler().fit(F[tr]); Fs = sc.transform(F).astype(np.float32)
                            mk = np.zeros(len(y), bool); mk[tr] = True
                            gtr = [(Fs, y, EPD[e]["E"].get(gname, []))]; mtr = [mk]; Fte = Fs
                            Xtr, ytr = Fs[tr], y[tr]
                        if model == "lr":
                            p = LogisticRegression(max_iter=3000).fit(Xtr, ytr).predict_proba(Fte)[:, 1]
                        elif model == "mlp":
                            _, _, MLP = C.make_models(); torch.manual_seed(seed)
                            net = MLP(Xtr.shape[1]); opt = torch.optim.Adam(net.parameters(), 5e-3, weight_decay=5e-4)
                            pw = torch.tensor((1 - ytr.mean()) / ytr.mean(), dtype=torch.float32)
                            lf = torch.nn.BCEWithLogitsLoss(pos_weight=pw); xt = torch.tensor(Xtr); yt = torch.tensor(ytr, dtype=torch.float32)
                            for _ in range(epochs):
                                net.train(); opt.zero_grad(); lf(net(xt), yt).backward(); opt.step()
                            net.eval()
                            with torch.no_grad(): p = torch.sigmoid(net(torch.tensor(Fte))).numpy()
                        else:
                            net = train_node_gat(gtr, mtr, seed, epochs)
                            p = predict(net, Fte, EPD[e]["E"].get(gname, []))
                        P[e][te] = p[te]
                for e in eps:
                    y = EPD[e]["y"]
                    rows.append(dict(seed=seed, split=split, model=model, graph=gname, episode=e,
                                     auroc=float(roc_auc_score(y, P[e])), auprc=float(average_precision_score(y, P[e])), pos_frac=float(y.mean())))
                print(split, model, gname, seed, np.round([r["auroc"] for r in rows[-len(eps):]], 3), flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 5. локализация
def localization(R, sec, thr, episodes):
    X = R.values; idx = R.index
    EXC = pd.DataFrame(-X > thr, index=idx, columns=R.columns)
    big = sec.value_counts(); big = big[big >= MIN_SECTOR].index
    m = X.mean(1)
    out, tables = {}, []
    for e, (pre_end, ld) in episodes.items():
        j = idx.get_loc(pd.Timestamp(ld)); i = idx.get_loc(pd.Timestamp(pre_end))
        # (А) след события: доля акций сектора, превысивших собственный EVT-порог в день пика
        foot = EXC.iloc[j].groupby(sec).mean()
        # (Б) независимая оценка «источника»: остаток сверх рыночной моды (бета и идиосинкратическая
        #     волатильность оценены только по окну до события), средний z-остаток сектора; > 0 = потери сверх беты
        Rw = X[i - WPRE + 1: i + 1]; mw = m[i - WPRE + 1: i + 1]
        Rc = Rw - Rw.mean(0); mc = mw - mw.mean(); beta = Rc.T @ mc / (mc @ mc)
        idio = (Rc - np.outer(mc, beta)).std(0)
        resid_z = pd.Series(-(X[j] - beta * m[j]) / idio, index=R.columns).groupby(sec).mean()
        # (В) раннее вовлечение: доля акций сектора с превышением порога за 3 торговых дня до пика
        pre = EXC.iloc[j - 3: j].any(axis=0).groupby(sec).mean()
        f, rz, p = foot[big].sort_values(ascending=False), resid_z[big].sort_values(ascending=False), pre[big].sort_values(ascending=False)
        out[e] = {"footprint_top3": f.head(3).round(3).to_dict(), "footprint_bottom2": f.tail(2).round(3).to_dict(),
                  "residual_z_top3": rz.head(3).round(2).to_dict(), "residual_z_bottom2": rz.tail(2).round(2).to_dict(),
                  "pre_onset_top3": p.head(3).round(3).to_dict(),
                  "top1_agree_footprint_residual": bool(f.index[0] == rz.index[0]),
                  "spearman_footprint_vs_residual": float(stats.spearmanr(foot[big], resid_z[big])[0]),
                  "spearman_footprint_vs_pre": float(stats.spearmanr(foot[big], pre[big])[0])}
        tables.append(pd.DataFrame({"episode": e, "sector": big, "footprint": foot[big].values,
                                    "residual_z": resid_z[big].values, "pre_onset": pre[big].values}))
    return out, pd.concat(tables)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data"); ap.add_argument("--out", default="../results/finance")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--reuse", action="store_true", help="взять уже посчитанные прогоны ГНС из gnn_runs.csv")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    R, sec, fixes = load(a.data)
    res = {"dataset": "Kaggle S&P 500 stock data (camnugent/sandp500), all_stocks_5yr.csv", "n_stocks": int(R.shape[1]),
           "n_days": int(R.shape[0]), "period": [str(R.index[0].date()), str(R.index[-1].date())],
           "corporate_action_fixes": fixes, "sector_counts": sec.value_counts().to_dict()}
    det, m = detect_market(R); res["detection"] = det
    np.save(f"{a.out}/market_losses.npy", (-m).values); pd.Series(-m.values, index=R.index).to_csv(f"{a.out}/market_losses.csv")
    # эпизоды: пик каждого кластера экстремальных дней; окно признаков заканчивается до первого экстремального дня кластера
    EPIS = {"2015-08": ("2015-08-20", "2015-08-24"), "2016-06": ("2016-06-23", "2016-06-24"),
            "2016-09": ("2016-09-08", "2016-09-09"), "2018-02": ("2018-02-01", "2018-02-05")}
    res["episodes"] = EPIS
    onsets = ["2015-08-21", "2016-06-24", "2016-09-09", "2018-02-05"]
    res["H2_noise"], E_h = hypothesis_noise(R, m, onsets, list(det["extreme_days"])); E_h.to_csv(f"{a.out}/h2_windows.csv", index=False)
    thr = np.array([C.pot_threshold(-R.values[:B, j], 0.9, 0.99)[0] for j in range(R.shape[1])])
    Esec = sector_edges(sec)
    EPD, gstats = {}, {}
    for e, (pe, ld) in EPIS.items():
        F, y, Ef, Ek, Rw = episode_data(R, m, sec, thr, pe, ld)
        EPD[e] = {"F": F, "y": y, "E": {"sector": Esec, "functional": Ef, "knn": Ek, "empty": []}}
        s = sec.values
        gstats[e] = {"pos_frac": float(y.mean()), "n_edges": {"sector": len(Esec), "functional": len(Ef), "knn": len(Ek)},
                     "jaccard": {"sector-functional": C.jaccard(Esec, Ef), "sector-knn": C.jaccard(Esec, Ek), "functional-knn": C.jaccard(Ef, Ek)},
                     "within_sector_frac": {"functional": float(np.mean([s[i] == s[j] for i, j in Ef])), "knn": float(np.mean([s[i] == s[j] for i, j in Ek]))},
                     "label_homophily": {g: float(np.mean([y[i] == y[j] for i, j in E])) for g, E in [("sector", Esec), ("functional", Ef), ("knn", Ek)]}}
    p = sec.value_counts(normalize=True); gstats["within_sector_frac_random_expectation"] = float((p ** 2).sum())
    res["graphs"] = gstats
    if a.reuse and os.path.exists(f"{a.out}/gnn_runs.csv"):
        RUNS = pd.read_csv(f"{a.out}/gnn_runs.csv")
    else:
        RUNS = run_protocols(EPD, sec, a.seeds, a.epochs); RUNS.to_csv(f"{a.out}/gnn_runs.csv", index=False)
    G = RUNS.groupby(["split", "model", "graph"])[["auroc", "auprc"]].agg(["mean", "std"]).round(4)
    G.columns = ["_".join(c) for c in G.columns]; res["gnn_summary"] = G.reset_index().to_dict("records")
    res["localization"], LT = localization(R, sec, thr, EPIS); LT.to_csv(f"{a.out}/localization_sectors.csv", index=False)
    res["runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{a.out}/results.json", "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: res[k] for k in ["detection", "H2_noise", "graphs", "localization"]}, ensure_ascii=False, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()

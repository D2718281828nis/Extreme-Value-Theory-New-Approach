"""Рисунки главы 7 (три примера применения методологии: мозг, авиационный двигатель, фондовый рынок).
Читает ../results/{brain,aviation,finance}."""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(__file__))
import aviation_cmapss_pipeline as A

# палитра: первые три категориальных слота эталонной палитры (проверены на различимость при нарушениях цветовосприятия)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})
RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RES, "figures"); os.makedirs(OUT, exist_ok=True)
DATE = "20260923"


def comma(x, nd=2):
    """Десятичная запятая для подписей (ГОСТ-стиль текста исследования)."""
    return f"{x:.{nd}f}".replace(".", ",")


def comma_axes(fig):
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            if axis.get_scale() == "linear" and isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(FuncFormatter(lambda v, _: (f"{v:g}").replace(".", ",")))


def ccdf(x):
    x = np.sort(x); return x, 1 - np.arange(len(x)) / len(x)


# ================================================================ авиация, рисунок 1
def aviation_fig1():
    R = json.load(open(f"{RES}/aviation/results.json"))
    D = pd.read_csv(f"{RES}/aviation/detection_FD001.csv")
    his = np.load(f"{RES}/aviation/hi_FD001.npy", allow_pickle=True)
    H = pd.read_csv(f"{RES}/aviation/h1_windows.csv")
    base = np.load(f"{RES}/aviation/hi_pooled_base.npy"); full = np.load(f"{RES}/aviation/hi_pooled_full.npy")
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))
    # (а) индикатор состояния трёх двигателей
    a = ax[0, 0]
    for u, c in zip([74, 20, 94], [BLUE, ORANGE, AQUA]):
        hi = his[u - 1]; n = len(hi); t = np.arange(n) - n
        a.plot(t, hi, color=c, lw=1.2, label=f"двигатель {u} (ресурс {n} циклов)")
        d = D[D.unit == u].iloc[0]
        if not np.isnan(d.det_gev):
            a.axvline(d.det_gev - n, color=c, lw=1, ls="--")
    a.axvspan(-400, -400 + 0, color=GRID)
    a.set_yscale("log"); a.set_xlabel("циклы до отказа"); a.set_ylabel("индикатор состояния (Махаланобис)")
    a.set_title("а) Индикатор состояния; пунктир — EVT-детекция"); a.legend(fontsize=8, loc="upper left")
    # (б) гипотеза H1
    a = ax[0, 1]
    pos = np.arange(1, 5)
    for k, (col, lab, c, off) in enumerate([("noise", "шум после удаления тренда", BLUE, -0.17), ("var_raw", "дисперсия исходного сигнала", ORANGE, 0.17)]):
        data = [H[H.win == w][col].values for w in pos]
        bp = a.boxplot(data, positions=pos + off, widths=0.28, patch_artist=True, showfliers=False,
                       medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=c), capprops=dict(color=c))
        for b in bp["boxes"]:
            b.set(facecolor=c, alpha=0.35, edgecolor=c)
        r = R["H1_noise"][col]["rm_corr"]
        ptxt = "p < 10⁻⁶" if r["p"] < 1e-6 else f"p = {comma(r['p'])}"
        a.plot([], [], color=c, lw=6, alpha=0.5, label=f"{lab}: r_rm = {comma(r['r_rm']).replace('-', '−')}, {ptxt}")
    a.axhline(1, color=INK2, lw=0.8, ls=":")
    a.set_xticks(pos); a.set_xticklabels(["T1", "T2", "T3", "T4\n(до отказа)"]); a.set_ylabel("отношение к окну T1 (медиана по датчикам)")
    a.set_title("б) Гипотеза H1 в дизайне работы [1]"); a.legend(fontsize=7.5, loc="upper left")
    a.set_ylim(0.5, 3.2)
    # (в) тяжёлый хвост
    a = ax[1, 0]
    for x, c, lab in [(base, BLUE, "фоновый участок (циклы 1–60)"), (full, ORANGE, "весь ресурс до отказа")]:
        ht = R['heavy_tail']['baseline' if c == BLUE else 'full_life']
        xs, p = ccdf(x); a.plot(xs, p, color=c, lw=1.4, label=f"{lab}: эксцесс {comma(ht['kurtosis'], 1)}, q(0,999) = {comma(ht['q999'], 1)}")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("индикатор / медиана фона"); a.set_ylabel("P(X > x)")
    a.set_title("в) Хвост распределения индикатора состояния"); a.legend(fontsize=8)
    # (г) запас времени до отказа
    a = ax[1, 1]
    for col, c, lab in [("lead_gev", BLUE, "GEV (блочные максимумы)"), ("lead_pot", ORANGE, "POT/GPD (превышения порога)")]:
        x = np.sort(D[col].dropna().values); a.step(x, np.arange(1, len(x) + 1) / 100, where="post", color=c, lw=1.4,
                                                    label=f"{lab}\nмедиана {np.median(x):.0f} циклов, n = {len(x)}")
    a.set_xlabel("запас до отказа в момент детекции, циклов"); a.set_ylabel("доля двигателей с детекцией")
    a.set_title("г) Детекция 100 двигателей FD001 без разметки"); a.legend(fontsize=8, loc="lower right"); a.set_ylim(0, 1.02)
    comma_axes(fig); fig.tight_layout()
    f = f"{OUT}/ch7_7-3_cmapss-evt-detection-noise-hypothesis_disser-text-cases_{DATE}.png"; fig.savefig(f, dpi=200); plt.close(fig); return f


# ================================================================ авиация, рисунок 2
def aviation_fig2():
    R = json.load(open(f"{RES}/aviation/results.json"))
    RUNS = pd.read_csv(f"{RES}/aviation/gnn_runs.csv"); L = pd.read_csv(f"{RES}/aviation/localization_FD003.csv")
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))
    # (а) структурный граф и совпадающие рёбра функционального графа
    a = ax[0, 0]; a.grid(False); a.set_xticks([]); a.set_yticks([])
    for s in ["left", "bottom"]: a.spines[s].set_visible(False)
    order = ["Fan", "LPC", "HPC", "Burner", "HPT", "LPT"]
    xy = {}
    for k, mname in enumerate(order):
        ss = [s for s in A.S if A.MOD[s] == mname]
        for q, s in enumerate(ss):
            xy[s] = (k, (q - (len(ss) - 1) / 2) * 0.8)
    Ep = {tuple(e) for e in R["graphs"]["physical"]["edges"]}; Ef = {tuple(e) for e in R["graphs"]["functional"]["edges"]}
    Ef = Ef | {(b, c) for c, b in Ef}
    for s1, s2 in Ep:
        c, lw, z = (BLUE, 1.6, 2) if (s1, s2) in Ef else (GRID, 0.8, 1)
        a.plot([xy[s1][0], xy[s2][0]], [xy[s1][1], xy[s2][1]], color=c, lw=lw, zorder=z)
    for s, (x, y) in xy.items():
        a.scatter([x], [y], s=380, color="white", edgecolor=INK2, zorder=3)
        a.text(x, y, A.NAME[s], ha="center", va="center", fontsize=6.5, zorder=4, color=INK)
    for k, mname in enumerate(order):
        a.text(k, 2.55, mname, ha="center", fontsize=8, color=INK2, fontweight="bold")
    a.set_ylim(-3.9, 2.9); a.set_xlim(-0.6, 5.6)
    J = R["jaccard"]
    a.set_title("а) Структурный граф датчиков по газовоздушному тракту")
    a.legend(handles=[Line2D([], [], color=BLUE, lw=1.6, label="ребро есть и в функциональном графе"),
                      Line2D([], [], color=GRID, lw=1.2, label="только в структурном")], fontsize=7.5, loc="lower left", ncol=1)
    a.text(5.55, -3.8, f"Жаккар: структ.–функц. {comma(J['physical-functional'])};\nструкт.–kNN {comma(J['physical-knn'])}; функц.–kNN {comma(J['functional-knn'])}",
           ha="right", va="bottom", fontsize=7, color=INK2)
    # (б) утечка: сгруппированное vs случайное разбиение
    a = ax[0, 1]
    G = RUNS[RUNS.graph.isin(["physical", "-"])].groupby(["features", "model", "split"]).auprc.agg(["mean", "std"]).reset_index()
    rows = [("normalized", "gat", "ГНС GATv2, норм."), ("normalized", "mlp", "MLP, норм."), ("normalized", "lr", "Логрегрессия, норм."),
            ("raw", "gat", "ГНС GATv2, сырые"), ("raw", "mlp", "MLP, сырые"), ("raw", "lr", "Логрегрессия, сырые")]
    for i, (f, mdl, lab) in enumerate(rows):
        g = G[(G.features == f) & (G.model == mdl)].set_index("split")
        y = len(rows) - 1 - i
        a.plot([g.loc["group", "mean"], g.loc["random", "mean"]], [y, y], color=GRID, lw=2, zorder=1)
        a.errorbar(g.loc["group", "mean"], y, xerr=g.loc["group", "std"], fmt="o", color=BLUE, ms=6, zorder=2)
        a.errorbar(g.loc["random", "mean"], y, xerr=g.loc["random", "std"], fmt="s", color=ORANGE, ms=6, zorder=2)
    a.set_yticks(range(len(rows))); a.set_yticklabels([r[2] for r in rows][::-1], fontsize=8)
    a.set_xlabel("AUPRC (критическое состояние: RUL ≤ 30 циклов)")
    a.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", label="по двигателям\n(без утечки)"),
                      Line2D([], [], marker="s", color=ORANGE, ls="", label="случайное по окнам\n(утечка)")], fontsize=7, loc="upper left")
    a.set_title("б) Утечка: разбиение по двигателям и случайное")
    # (в) аудит ценности графа
    a = ax[1, 0]
    g = RUNS[(RUNS.model == "gat") & (RUNS.split == "group") & (RUNS.features == "normalized")].groupby("graph").auprc.agg(["mean", "std"])
    order_g = ["physical", "functional", "knn", "empty"]; labs = ["структурный", "функцио-\nнальный", "kNN", "без рёбер"]
    for i, k in enumerate(order_g):
        a.errorbar(i, g.loc[k, "mean"], yerr=g.loc[k, "std"], fmt="o", ms=8, color=BLUE if k != "empty" else INK2, capsize=4, lw=1.4)
    for i, k in enumerate(order_g):
        a.text(i + 0.12, g.loc[k, "mean"], comma(g.loc[k, 'mean'], 3), ha="left", va="center", fontsize=8, color=INK)
    a.set_ylim(0.93, 0.955); a.set_xlim(-0.5, 3.5); a.set_xticks(range(4)); a.set_xticklabels(labs); a.set_ylabel("AUPRC (по двигателям; ± ст. откл.)")
    a.set_title("в) Аудит ценности графа: ГНС GATv2 при разных графах")
    # (г) локализация источника в FD003
    a = ax[1, 1]
    for src, c, lab in [("HPC", BLUE, "КВД (сигнатура FD001)"), ("Fan", ORANGE, "вентилятор (обратная)")]:
        s = L[L.source_sig == src]
        ok = s.gnn_source == s.source_sig
        a.scatter(s[ok].z_BPR, s[ok].z_Nf, s=22, color=c, label=lab, zorder=2)
        a.scatter(s[~ok].z_BPR, s[~ok].z_Nf, s=30, facecolor="white", edgecolor=c, lw=1.4, zorder=3)
    a.scatter([], [], s=30, facecolor="white", edgecolor=INK2, label="ГНС: другой модуль")
    loc = R["localization"]
    a.set_xlabel("z-отклонение BPR (степень двухконтурности) к отказу"); a.set_ylabel("z-отклонение Nf (частота вентилятора)")
    a.set_title("г) Локализация источника неисправности (FD003)")
    a.text(0.47, 0.45, f"согласие с сигнатурой:\nГНС — {loc['gnn_vs_signature']:.0%}, каскад — {loc['cascade_vs_signature']:.0%}", transform=a.transAxes, ha="center", va="center", fontsize=8, color=INK2)
    a.legend(fontsize=7.5, loc="upper center")
    comma_axes(fig); fig.tight_layout()
    f = f"{OUT}/ch7_7-3_cmapss-graph-gnn-leakage-localization_disser-text-cases_{DATE}.png"; fig.savefig(f, dpi=200); plt.close(fig); return f


# ================================================================ финансы, рисунок 1
def finance_fig1():
    R = json.load(open(f"{RES}/finance/results.json"))
    L = pd.read_csv(f"{RES}/finance/market_losses.csv", index_col=0, parse_dates=True).iloc[:, 0]
    H = pd.read_csv(f"{RES}/finance/h2_windows.csv")
    det = R["detection"]
    fig = plt.figure(figsize=(10, 7.2)); gs = fig.add_gridspec(2, 2)
    a = fig.add_subplot(gs[0, :])
    a.plot(L.index, L.values * 100, color=BLUE, lw=0.7)
    a.axvspan(L.index[0], pd.Timestamp(det["baseline_period"][1]), color=GRID, alpha=0.6, lw=0)
    a.axhline(det["thr_pot"] * 100, color=ORANGE, lw=1, ls="--")
    ed = pd.to_datetime(list(det["extreme_days"])); a.scatter(ed, L.loc[ed].values * 100, color=ORANGE, s=22, zorder=3)
    labels = {"2015-08-24": "21.08–28.09.2015 (4 дня)", "2016-06-24": "24.06.2016", "2016-09-09": "09.09.2016", "2018-02-05": "05.02.2018"}
    for d, t in labels.items():
        d = pd.Timestamp(d); a.annotate(t, (d, L.loc[d] * 100), textcoords="offset points", xytext=(4, 3), fontsize=7, color=INK2)
    a.text(L.index[20], det["thr_pot"] * 100 + 0.15, f"порог POT/GPD (q = 0,998), обучен только на фоне: {comma(det['thr_pot']*100)}%", fontsize=7.5, color=INK2)
    a.text(L.index[20], -2.7, "фоновый период (обучение порога)", fontsize=7.5, color=INK2)
    a.set_ylabel("потери рыночной моды, % в день"); a.set_ylim(-3.2, 5)
    a.set_title(f"а) EVT-детекция экстремальных дней: {len(ed)} превышений при ожидаемых {comma(det['expected_exceedances'], 1)} (биномиальный p = {comma(det['binom_p'], 4)})")
    Hn = H.copy()
    for c in ["vol", "corr"]:
        Hn[c + "_rel"] = Hn[c] / Hn.groupby("ep")[c].transform("first")
    for k, (c, lab) in enumerate([("vol", "волатильность рыночной моды"), ("corr", "средняя попарная корреляция")]):
        a = fig.add_subplot(gs[1, k]); cols = [BLUE, ORANGE, AQUA, INK2]
        for (ep, g), cc in zip(Hn.groupby("ep"), cols):
            a.plot(g.win, g[c + "_rel"], marker="o", ms=4, lw=1.3, color=cc, label=pd.Timestamp(ep).strftime("%d.%m.%Y"))
        a.axhline(1, color=INK2, lw=0.8, ls=":")
        h = R["H2_noise"][c]
        a.set_xticks([1, 2, 3, 4]); a.set_xticklabels(["T1", "T2", "T3", "T4\n(до события)"])
        a.set_ylabel(f"{lab}, отн. T1")
        a.set_title(f"{'бв'[k]}) {lab.capitalize()}\nr_rm = {comma(h['rm_corr']['r_rm'])}; суррогатный p = {comma(h['p_surrogate_greater'])}")
        a.legend(fontsize=7.5, title="эпизод (начало)", title_fontsize=7.5)
    comma_axes(fig); fig.tight_layout()
    f = f"{OUT}/ch7_7-4_sp500-evt-detection-noise-hypothesis_disser-text-cases_{DATE}.png"; fig.savefig(f, dpi=200); plt.close(fig); return f


# ================================================================ финансы, рисунок 2
SECT_RU = {"Consumer Discretionary": "Потреб. товары втор. необх.", "Consumer Staples": "Товары перв. необх.", "Energy": "Энергетика",
           "Financials": "Финансы", "Health Care": "Здравоохранение", "Industrials": "Промышленность",
           "Information Technology": "Инф. технологии", "Materials": "Сырьё и материалы", "Real Estate": "Недвижимость", "Utilities": "Коммун. услуги"}


def finance_fig2():
    R = json.load(open(f"{RES}/finance/results.json"))
    T = pd.read_csv(f"{RES}/finance/localization_sectors.csv"); RUNS = pd.read_csv(f"{RES}/finance/gnn_runs.csv")
    eps = list(R["episodes"]); ep_lab = {"2015-08": "24.08.2015", "2016-06": "24.06.2016", "2016-09": "09.09.2016", "2018-02": "05.02.2018"}
    fig = plt.figure(figsize=(10.5, 8.6)); gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 0.8])
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    sect = sorted(T.sector.unique(), key=lambda s: SECT_RU[s])
    for k, (col, cmap, lab, vmin, vmax) in enumerate([("footprint", "Blues", "доля акций сектора за EVT-порогом", 0, 1),
                                                       ("residual_z", "RdBu_r", "потери сверх беты, z", -3, 3)]):
        a = ax[k]; a.grid(False)
        M = T.pivot(index="sector", columns="episode", values=col).loc[sect, eps]
        im = a.imshow(M.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                dark = (col == "footprint" and v > 0.6) or (col == "residual_z" and abs(v) > 2)
                a.text(j, i, comma(v) if col == "footprint" else ("+" if v >= 0 else "−") + comma(abs(v), 1), ha="center", va="center", fontsize=7, color="white" if dark else INK)
        a.set_xticks(range(len(eps))); a.set_xticklabels([ep_lab[e] for e in eps], rotation=30, fontsize=7.5)
        a.set_yticks(range(len(sect))); a.set_yticklabels([SECT_RU[s] for s in sect] if k == 0 else [], fontsize=7.5)
        cb = fig.colorbar(im, ax=a, fraction=0.06, pad=0.03); cb.ax.tick_params(labelsize=7); cb.set_label(lab, fontsize=7.5)
        a.set_title("а) След события: EVT-превышения" if k == 0 else "б) Потери сверх рыночной моды (бета до события)")
    a = ax[2]
    cfg = [("gat", "sector", "ГНС GATv2, секторный граф"), ("gat", "functional", "ГНС GATv2, функц. граф"), ("gat", "knn", "ГНС GATv2, kNN-граф"),
           ("gat", "empty", "ГНС GATv2, без рёбер"), ("mlp", "-", "MLP"), ("lr", "-", "Логрегрессия")]
    G = RUNS.groupby(["model", "graph", "split"]).auroc.mean()
    for i, (mdl, gr, lab) in enumerate(cfg):
        y = len(cfg) - 1 - i
        v = [G.loc[(mdl, gr, s)] for s in ["random", "sector", "loeo"]]
        a.plot([min(v), max(v)], [y, y], color=GRID, lw=2, zorder=1)
        for x, c, mk in zip(v, [ORANGE, BLUE, AQUA], ["s", "o", "D"]):
            a.scatter([x], [y], color=c, marker=mk, s=36, zorder=2)
    a.axvline(0.5, color=INK2, lw=0.8, ls=":")
    a.set_yticks(range(len(cfg))); a.set_yticklabels([c[2] for c in cfg][::-1], fontsize=7.5)
    a.set_xlabel("AUROC вовлечения акции в экстремальное событие (среднее по 4 эпизодам × 3 зёрнам)")
    a.set_xlim(0.45, 0.8)
    a.legend(handles=[Line2D([], [], marker="s", color=ORANGE, ls="", label="случайное разбиение узлов (утечка через сектор)"),
                      Line2D([], [], marker="o", color=BLUE, ls="", label="группировка по сектору"),
                      Line2D([], [], marker="D", color=AQUA, ls="", label="перенос на незнакомый эпизод")], fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    a.set_title("в) Протокол без утечки и перенос на новый эпизод; пунктир — случайное угадывание")
    comma_axes(fig); fig.tight_layout()
    f = f"{OUT}/ch7_7-4_sp500-localization-gnn-leakage_disser-text-cases_{DATE}.png"; fig.savefig(f, dpi=200); plt.close(fig); return f


# ================================================================ мозг (стерео-ЭЭГ), рисунок 1
ONSET_B = 10396.445


def brain_fig1():
    R = json.load(open(f"{RES}/brain/results.json")); Z = np.load(f"{RES}/brain/indicators.npz")
    hi, hl = Z["hi"], Z["hl"]; tp, tg, tl = float(Z["thr_pot"]), float(Z["thr_gev"]), float(Z["thr_local"])
    I, V = Z["I"], Z["V"]; D = R["detection"]; H = R["H0_noise"]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4))
    # (а) вся запись
    a = ax[0, 0]; t = np.arange(len(hi)) / 3600
    a.plot(t, hi, color=BLUE, lw=0.4)
    a.axvspan(0, 1, color=GRID, alpha=0.7, lw=0); a.axhline(tp, color=ORANGE, lw=1, ls="--")
    a.scatter([D["blind_pot"]["event_alarm"] / 3600], [hi[D["blind_pot"]["event_alarm"]:D["blind_pot"]["event_alarm"] + 8].max() * 1.4], marker="v", color=ORANGE, s=40, zorder=3)
    for e in D["blind_gev"]["false_alarms"]:
        a.scatter([e / 3600], [hi[e:e + 20].max() * 1.4], marker="x", color=INK2, s=30, zorder=3)
    a.set_yscale("log"); a.set_xlabel("время записи, ч (запись обрезана на 10 550 с)"); a.set_ylabel("индикатор состояния (Махаланобис)")
    a.text(0.03, 0.93, "фон: 0–1 ч", transform=a.transAxes, fontsize=7.5, color=INK2)
    a.legend(handles=[Line2D([], [], color=ORANGE, ls="--", label=f"порог POT/GPD (q = 0,999): {comma(tp, 1)}"),
                      Line2D([], [], marker="v", color=ORANGE, ls="", label=f"срабатывание POT: +{comma(D['blind_pot']['dt'], 1)} с; ложных — 0 за {comma(D['test_hours'])} ч"),
                      Line2D([], [], marker="x", color=INK2, ls="", label=f"ложные срабатывания GEV (блок 10 с): {len(D['blind_gev']['false_alarms'])} — провалы сигнала PA9")],
             fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    a.set_title("а) Слепая EVT-детекция по всей записи")
    # (б) окрестность события
    a = ax[0, 1]; sel = np.arange(int(ONSET_B) - 60, int(ONSET_B) + 61); x = sel - ONSET_B
    a.plot(x, hi[sel] / tp, color=BLUE, lw=1.2, marker="o", ms=2.2, label="широкополосный, фон 0–1 ч (слепой)")
    a.plot(x, hl[sel] / tl, color=AQUA, lw=1.2, marker="o", ms=2.2, label="13–67 Гц, фон перед событием (целевой)")
    a.axhline(1, color=ORANGE, lw=1, ls="--")
    for ts, txt in zip([x0 for x0, _ in R["annotations"]], ["«где тут начало?»", "«приступ + БТКП»", "«клиника»"]):
        a.axvline(ts - ONSET_B, color=INK2, lw=0.8, ls=":")
    a.text(0, 2.6, "аннотации\nврача", fontsize=7, color=INK2, ha="center")
    for dt, c, txt in [(D["targeted_local"]["dt"], AQUA, f"целевой: +{comma(D['targeted_local']['dt'], 1)} с"), (D["blind_pot"]["dt"], BLUE, f"слепой POT: +{comma(D['blind_pot']['dt'], 1)} с")]:
        a.axvline(dt, color=c, lw=1.1, alpha=0.8)
        a.text(dt + (1.2 if c == BLUE else -1.2), 13, txt, fontsize=7.5, color=c, ha="left" if c == BLUE else "right", va="center")
    a.set_yscale("log"); a.set_ylim(0.12, 25); a.set_xlabel("время относительно отметки «приступ + БТКП», с"); a.set_ylabel("индикатор / собственный порог")
    a.legend(fontsize=7, loc="lower left"); a.set_title("б) Окрестность события (±60 с)")
    # (в) гипотеза шумового предвестника
    a = ax[1, 0]; pos = np.arange(1, 5)
    for M, lab, c, off, key in [(I, "интенсивность шума $I=\\int_1^{30} Cf^{-\\alpha}df$", BLUE, -0.17, "noise"), (V, "дисперсия после удаления тренда", ORANGE, 0.17, "variance")]:
        data = [M[k] / M[0] for k in range(4)]
        bp = a.boxplot(data, positions=pos + off, widths=0.28, patch_artist=True, showfliers=False,
                       medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=c), capprops=dict(color=c))
        for b in bp["boxes"]:
            b.set(facecolor=c, alpha=0.35, edgecolor=c)
        a.plot([], [], color=c, lw=6, alpha=0.5, label=f"{lab}\nсуррогатный p(рост) = {comma(H[key]['p_surrogate_greater'])}")
    a.axhline(1, color=INK2, lw=0.8, ls=":"); a.set_yscale("log")
    a.set_xticks(pos); a.set_xticklabels(["T1", "T2", "T3", "T4\n(до отметки)"]); a.set_ylabel("отношение к окну T1 (100 контактов)")
    a.set_title("в) Гипотеза шумового предвестника в дизайне работы [1]"); a.legend(fontsize=6.8, loc="upper right")
    # (г) тяжёлый хвост
    a = ax[1, 1]; base = hi[:3600] / np.median(hi[:3600]); full = hi / np.median(hi[:3600]); ht = D["heavy_tail"]
    for xv, c, lab, k in [(base, BLUE, "фон (0–1 ч)", "baseline"), (full, ORANGE, "вся запись 0–10 550 с", "full")]:
        xs, pp = ccdf(xv); a.plot(xs, pp, color=c, lw=1.4, label=f"{lab}: эксцесс {comma(ht[k]['kurtosis'], 1)}, q(0,999) = {comma(ht[k]['q999'], 2)}")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("индикатор, в единицах медианы фона"); a.set_ylabel("P(X > x)")
    a.legend(fontsize=7.5, loc="lower left"); a.set_title("г) Хвост распределения индикатора")
    comma_axes(fig); fig.tight_layout()
    f = f"{OUT}/ch7_7-2_seeg-evt-detection-noise-hypothesis_disser-text-cases_{DATE}.png"; fig.savefig(f, dpi=200); plt.close(fig); return f


# ================================================================ мозг (стерео-ЭЭГ), рисунок 2
def brain_fig2():
    R = json.load(open(f"{RES}/brain/results.json")); C3 = np.load(f"{RES}/brain/cascade.npz")
    L = pd.read_csv(f"{RES}/brain/localization_contacts.csv"); S = pd.read_csv(f"{RES}/brain/localization_shafts.csv")
    RUNS = pd.read_csv(f"{RES}/brain/gnn_runs.csv"); loc = R["localization"]
    fig = plt.figure(figsize=(10.5, 8.4)); gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1])
    # (а) каскад вовлечения
    a = fig.add_subplot(gs[0, 0]); a.grid(False)
    z, tw, lat = C3["z"], C3["tw"] - ONSET_B, C3["lat"]
    order = np.lexsort((-L.footprint.values, np.where(np.isfinite(lat), lat, 1e9)))
    ts = (tw >= -20) & (tw <= 60)
    im = a.imshow(np.clip(z[order][:, ts], 0, 15), aspect="auto", cmap="Blues", vmin=0, vmax=15, interpolation="nearest",
                  extent=[tw[ts][0], tw[ts][-1], len(order) - 0.5, -0.5])
    for r, i in enumerate(order):
        if np.isfinite(lat[i]):
            a.plot(lat[i], r, marker="|", color=ORANGE, ms=3.2, mew=0.9)
    lab_idx = [r for r, i in enumerate(order) if L.prior.values[i]]
    a.set_yticks(lab_idx); a.set_yticklabels(["◆ " + L.name.values[order[r]] for r in lab_idx], fontsize=5.8)
    first = "\n".join(f"{L.name.values[i]}: +{comma(lat[i], 1)} с" for i in order[:6])
    a.text(-19, 2, "самые ранние:\n" + first, fontsize=6.5, color=INK, va="top", ha="left",
           bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.3"))
    a.axvline(0, color=INK2, lw=0.8, ls=":"); a.axhline(R["labels"]["involved"] - 0.5, color=INK2, lw=0.7)
    a.text(59, R["labels"]["involved"] + 1, f"ниже — не вовлечены ({R['labels']['n'] - R['labels']['involved']})", ha="right", va="top", fontsize=6.5, color=INK2)
    cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=7); cb.set_label("z (MAD), энергия 13–67 Гц, окно 250 мс", fontsize=7)
    a.set_xlabel("время относительно отметки «приступ + БТКП», с"); a.set_title("а) Каскад вовлечения 100 контактов; ◆ — клинический приор")
    # (б) протокол без утечки
    a = fig.add_subplot(gs[0, 1])
    cfg = [("gat", "structural", "ГНС GATv2, структурный граф"), ("gat", "functional", "ГНС GATv2, функц. граф"), ("gat", "knn", "ГНС GATv2, kNN-граф"),
           ("gat", "empty", "ГНС GATv2, без рёбер"), ("mlp", "-", "MLP"), ("lr", "-", "Логрегрессия")]
    G = RUNS.groupby(["model", "graph", "split"]).auroc.agg(["mean", "std"])
    for i, (mdl, gr, lab) in enumerate(cfg):
        y = len(cfg) - 1 - i; g = G.loc[(mdl, gr)]
        a.plot([g.loc["shaft", "mean"], g.loc["random", "mean"]], [y, y], color=GRID, lw=2, zorder=1)
        a.errorbar(g.loc["shaft", "mean"], y, xerr=g.loc["shaft", "std"], fmt="o", color=BLUE, ms=6, zorder=2)
        a.errorbar(g.loc["random", "mean"], y, xerr=g.loc["random", "std"], fmt="s", color=ORANGE, ms=6, zorder=2)
    a.set_yticks(range(len(cfg))); a.set_yticklabels([c[2] for c in cfg][::-1], fontsize=7.5)
    gg = R["graph_gain_structural_minus_empty_auroc"]
    a.set_xlabel("AUROC вовлечения контакта (3 затравки)")
    a.set_ylim(-0.5, 5.5)
    a.text(0.02, 0.1, f"выигрыш от структурного графа (минус без рёбер):\nслучайное {comma(gg['random']['mean'], 3)}; по стволам {comma(gg['shaft']['mean'], 3)}",
           transform=a.transAxes, fontsize=7, color=INK2, va="bottom")
    a.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", label="по электродным стволам (без утечки)"),
                      Line2D([], [], marker="s", color=ORANGE, ls="", label="случайное по контактам (утечка)")], fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    a.set_xlim(0.8, 0.96); a.set_title("б) Утечка через ствол и аудит ценности графа")
    # (в) локализация по стволам
    a = fig.add_subplot(gs[1, 0])
    for _, r in S.iterrows():
        left = r["shaft"].endswith("'"); c = ORANGE if left else BLUE
        prior = r["shaft"] in ("PM", "CC")
        a.scatter(r.footprint, r.residual_z, s=40 + 90 * r.n_early10, color=c, alpha=0.85, edgecolor=INK if prior else "white", lw=1.6 if prior else 0.6, zorder=3)
        off, ha = {"PM": ((-7, 4), "right"), "CC": ((-7, -9), "right"), "PA": ((5, 6), "left")}.get(r["shaft"], ((5, 3), "left"))
        a.annotate(r["shaft"] + (" (приор)" if prior else ""), (r.footprint, r.residual_z), xytext=off, textcoords="offset points", fontsize=7, color=INK, ha=ha)
    a.axhline(0, color=INK2, lw=0.8, ls=":"); a.set_xlim(0.52, 0.97); a.set_ylim(-1.8, 3.7)
    a.set_xlabel("след: доля секунд +10,6…+55,6 с над собственным EVT-порогом"); a.set_ylabel("остаток сверх общей моды, z")
    a.legend(handles=[Line2D([], [], marker="o", color=ORANGE, ls="", label="левое полушарие (ствол со штрихом)"),
                      Line2D([], [], marker="o", color=BLUE, ls="", label="правое полушарие"),
                      Line2D([], [], marker="o", color="white", markeredgecolor=INK2, ls="", ms=9, label="размер — число ранних контактов (10%)")], fontsize=7, loc="upper left")
    a.set_title(f"в) Локализация по стволам: лидер SA′ по {loc['leading_shaft_agree']} из 4 оценок")
    # (г) индексы латерализации
    a = fig.add_subplot(gs[1, 1])
    li = loc["LI"]
    rows = [("доля вовлечённых контактов", li["involved_share"], BLUE, ""), ("след события", li["footprint"], BLUE, ""), ("остаток сверх моды*", li["residual_z"], BLUE, "//"), ("доля ранних контактов", li["early10_share"], BLUE, ""),
            ("ГНС вне фолда", li["gnn_oof"], BLUE, ""),
            ("§ 3.11: ранние контакты", 0.487, INK2, ""), ("§ 3.11: асимметрия МРТ", -0.110, INK2, ""),
            ("§ 3.11: сила невязки", 0.292, INK2, ""), ("§ 3.11: ранность невязки*", -1.0, INK2, "//")]
    for i, (lab, v, c, h) in enumerate(rows):
        y = len(rows) - 1 - i
        a.barh(y, v, color=c, alpha=0.8 if c == BLUE else 0.45, hatch=h, edgecolor="white")
        a.text(v + (0.03 if v >= 0 else -0.03), y, ("+" if v > 0 else ("−" if v < 0 else "")) + comma(abs(v), 3), va="center", ha="left" if v >= 0 else "right", fontsize=7, color=INK)
    a.axvspan(-0.05, 0.05, color=GRID, alpha=0.7, lw=0); a.axvline(0, color=INK2, lw=0.8)
    a.set_yticks(range(len(rows))); a.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5)
    a.set_xlim(-1.35, 1.0); a.set_xlabel("LI = (R − L)/(|R| + |L|):  < 0 — левое, > 0 — правое")
    a.text(0.98, 0.02, "* вырожденное значение\n(средние разного знака или нуль)", transform=a.transAxes, ha="right", fontsize=6.5, color=INK2)
    a.set_title("г) Индексы латерализации: пересчёт и § 3.11")
    comma_axes(fig); fig.tight_layout()
    f = f"{OUT}/ch7_7-2_seeg-graph-gnn-leakage-localization_disser-text-cases_{DATE}.png"; fig.savefig(f, dpi=200); plt.close(fig); return f


if __name__ == "__main__":
    which = sys.argv[1:] or ["b1", "b2", "a1", "a2", "f1", "f2"]
    fn = {"b1": brain_fig1, "b2": brain_fig2, "a1": aviation_fig1, "a2": aviation_fig2, "f1": finance_fig1, "f2": finance_fig2}
    for w in which:
        print(fn[w]())

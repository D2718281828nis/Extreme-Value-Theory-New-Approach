"""Сводные рисунки, показывающие общность методологии («окна -> признаки -> граф -> ГНС -> EVT ->
локализация -> leakage-safe проверка») на трёх разных предметных областях (мозг, авиадвигатель, рынок).

В отличие от src/make_figures.py (рисунки 57–62 главы 7, которые остаются как были — приложение,
показывающее устройство каждого набора данных), этот скрипт ничего не меняет в src/*.py и не влияет
на results/*/results.json. Он только читает уже посчитанные results/{brain,aviation,finance}/ и рисует
пять новых PNG в results/figures/generality/:

    g0_task-overview-evt-detection.png   короткое пояснение задачи по каждой области: ряд + EVT-детекция
    g1_signal-wavelet-dfa.png            вейвлет-разложение и DFA-масштабирование на представительном ряду
    g2_graph-gnn-gat-leakage.png         структурный граф объекта в каждой области; ГНС/GAT против MLP/
                                          логрегрессии, утечка, ценность графа
    g3_accuracy-confusion-matrix.png     точность и матрица ошибок (блок E § 3.11 + доля верных, авиация)
    g4_cascade-localization.png          каскад вовлечения и локализация источника

Вейвлет-разложение и кривая DFA считаются здесь на месте (функция dfa_alpha из common.py даёт только
показатель альфа, не саму кривую F(n); вейвлет-преобразование в src/*.py не считается вовсе) — это
иллюстрация метода, а не число из текста исследования, поэтому пересчёт не противоречит примечанию
«Файлы src/ не изменены» (сам этот файл в контрольные суммы README.md не входит).
"""
from __future__ import annotations
import json, os, re, sys, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import common as C
import paths as P
import aviation_cmapss_pipeline as A

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#dcdad4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})
RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RES, "figures", "generality"); os.makedirs(OUT, exist_ok=True)
DATE = time.strftime("%Y%m%d")
DOM_COLOR = {"brain": BLUE, "aviation": ORANGE, "finance": AQUA}
DOM_LABEL = {"brain": "мозг (стерео-ЭЭГ)", "aviation": "авиадвигатель (C-MAPSS)", "finance": "рынок (S&P 500)"}


def comma(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def comma_axes(fig):
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    for a in fig.axes:
        for axis in (a.xaxis, a.yaxis):
            if axis.get_scale() == "linear" and isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(FuncFormatter(lambda v, _: (f"{v:g}").replace(".", ",")))


# ================================================================ общие численные приёмы
def dfa_curve(x, scales=None):
    """Как common.dfa_alpha, но возвращает саму кривую F(n), не только показатель альфа."""
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    N = len(y)
    if scales is None:
        scales = np.unique(np.floor(np.logspace(np.log10(4), np.log10(max(N // 4, 5)), 10)).astype(int))
    ns, Fs = [], []
    for n in scales:
        m = N // n
        if m < 2:
            continue
        seg = y[: m * n].reshape(m, n)
        t = np.arange(n)
        res = [np.mean((s - np.polyval(np.polyfit(t, s, 1), t)) ** 2) for s in seg]
        ns.append(n); Fs.append(np.sqrt(np.mean(res)))
    ns, Fs = np.array(ns, float), np.array(Fs, float)
    alpha = float(np.polyfit(np.log(ns), np.log(Fs), 1)[0])
    return ns, Fs, alpha


def morlet_cwt(x, n_scales=40, w0=6.0):
    """Непрерывное вейвлет-преобразование (материнский вейвлет Морле) через БПФ — иллюстрация метода,
    без внешней зависимости от pywt/scipy.signal.cwt. Возвращает (псевдопериод в отсчётах, мощность)."""
    x = np.asarray(x, float); x = x - x.mean(); n = len(x)
    scales = np.geomspace(2.0, max(n / 4, 3.0), n_scales)
    X = np.fft.fft(x); freq = 2 * np.pi * np.fft.fftfreq(n)
    out = np.zeros((n_scales, n))
    for i, s in enumerate(scales):
        omega = freq * s
        psi = (np.pi ** -0.25) * np.exp(-0.5 * (omega - w0) ** 2) * (freq > 0) * np.sqrt(2 * np.pi * s)
        out[i] = np.abs(np.fft.ifft(X * np.conj(psi))) ** 2 / s
    period = 2 * np.pi * scales / w0
    return period, out


def load_domain_series():
    """Представительный ряд + опорное время (0 = момент экстремального события) для каждой области."""
    Rb = json.load(open(f"{RES}/brain/results.json")); Zb = np.load(f"{RES}/brain/indicators.npz")
    onset = 10396.445
    brain = dict(t=(np.arange(len(Zb["hi"])) - onset), x=Zb["hi"].astype(float), thr=float(Zb["thr_pot"]),
                 event_idx=Rb["detection"]["blind_pot"]["event_alarm"], unit="с", R=Rb)

    Ra = json.load(open(f"{RES}/aviation/results.json")); D = pd.read_csv(f"{RES}/aviation/detection_FD001.csv")
    his = np.load(f"{RES}/aviation/hi_FD001.npy", allow_pickle=True)
    base = np.load(f"{RES}/aviation/hi_pooled_base.npy")
    row = D.iloc[(D.life - D.life.median()).abs().sort_values().index[0]]
    hi = his[int(row.unit) - 1]; n = len(hi)
    thr, xi = C.pot_threshold(base)
    aviation = dict(t=np.arange(n) - n, x=hi.astype(float), thr=float(thr),
                    event_idx=int(row.det_gev) if not np.isnan(row.det_gev) else None, unit="циклов", R=Ra, life=n, xi=xi)

    Rf = json.load(open(f"{RES}/finance/results.json")); L = pd.read_csv(f"{RES}/finance/market_losses.csv", index_col=0, parse_dates=True).iloc[:, 0]
    det = Rf["detection"]; ed = pd.to_datetime(list(det["extreme_days"]))
    finance = dict(t=L.index, x=L.values.astype(float), thr=float(det["thr_pot"]), event_idx=None,
                  extreme_days=ed, unit="дн.", R=Rf)
    return {"brain": brain, "aviation": aviation, "finance": finance}


# ================================================================ G0: короткое объяснение задачи
def task_overview(S):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))

    a = ax[0]; d = S["brain"]
    a.plot(d["t"] / 3600, d["x"], color=BLUE, lw=0.4)
    a.axhline(d["thr"], color=ORANGE, lw=1, ls="--")
    a.scatter([d["t"][d["event_idx"]] / 3600], [d["x"][d["event_idx"]:d["event_idx"] + 8].max() * 1.4], marker="v", color=ORANGE, s=36, zorder=3)
    a.set_yscale("log"); a.set_xlabel("время записи, ч"); a.set_ylabel("индикатор состояния")
    a.set_title("а) Мозг: стерео-ЭЭГ, 100 контактов")
    a.text(0.02, 0.03, "экстремальное событие — приступ;\nиндикатор — Махаланобис по спектру 1–67 Гц;\nпорог — POT/GPD по фону 0–1 ч",
           transform=a.transAxes, fontsize=6.8, color=INK2, va="bottom", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))

    a = ax[1]; d = S["aviation"]
    a.plot(d["t"], d["x"], color=ORANGE, lw=0.9)
    a.axhline(d["thr"], color=BLUE, lw=1, ls="--")
    if d["event_idx"] is not None:
        ei = d["event_idx"]; a.scatter([ei - d["life"]], [d["x"][ei]], marker="v", color=BLUE, s=36, zorder=3)
    a.set_yscale("log"); a.set_xlabel("циклы до отказа"); a.set_ylabel("индикатор состояния")
    a.set_title("б) Авиадвигатель: C-MAPSS, 21 датчик")
    a.text(0.02, 0.03, "экстремальное событие — приближение к отказу\n(остаточный ресурс RUL → 0);\nпорог — POT/GPD по циклам 1–60",
           transform=a.transAxes, fontsize=6.8, color=INK2, va="bottom", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))

    a = ax[2]; d = S["finance"]
    a.plot(d["t"], d["x"] * 100, color=AQUA, lw=0.6)
    a.axhline(d["thr"] * 100, color=ORANGE, lw=1, ls="--")
    a.scatter(d["extreme_days"], pd.Series(d["x"], index=d["t"]).loc[d["extreme_days"]].values * 100, color=ORANGE, s=20, zorder=3)
    a.set_xlabel("торговый день"); a.set_ylabel("потери рыночной моды, % в день")
    a.set_title("в) Рынок: S&P 500, 470 акций")
    a.text(0.02, 0.97, "экстремальное событие — день сильного\nобщерыночного падения;\nпорог — POT/GPD по фону до 2015 г.",
           transform=a.transAxes, fontsize=6.8, color=INK2, va="top", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))

    ax[0].legend(handles=[Line2D([], [], color=ORANGE, ls="--", lw=1, label="POT/GPD-порог"),
                         Line2D([], [], marker="v", color=ORANGE, ls="", label="детекция")], fontsize=6.5, loc="upper right")
    ax[1].legend(handles=[Line2D([], [], color=BLUE, ls="--", lw=1, label="POT/GPD-порог"),
                         Line2D([], [], marker="v", color=BLUE, ls="", label="детекция")], fontsize=6.5, loc="upper right")
    ax[2].legend(handles=[Line2D([], [], color=ORANGE, ls="--", lw=1, label="POT/GPD-порог"),
                         Line2D([], [], marker="o", color=ORANGE, ls="", label="детекция")], fontsize=6.5, loc="lower right")
    fig.suptitle("Одна и та же задача в трёх областях: слепая EVT-детекция экстремального события по индикатору состояния", fontsize=9.5, y=1.02)
    comma_axes(fig); fig.tight_layout()
    f = f"{OUT}/g0_task-overview-evt-detection_{DATE}.png"; fig.savefig(f, dpi=200, bbox_inches="tight"); plt.close(fig); return f


# ================================================================ G1: вейвлет + DFA
def signal_wavelet_dfa(S):
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.2))
    segs = {
        "brain": (S["brain"]["t"][10096:10696], np.log(S["brain"]["x"][10096:10696]), "с (относительно приступа)"),
        "aviation": (S["aviation"]["t"], np.log(np.maximum(S["aviation"]["x"], 1e-6)), "циклов до отказа"),
        "finance": (np.arange(len(S["finance"]["x"]))[-300:] - len(S["finance"]["x"]), S["finance"]["x"][-300:] * 100, "дн. (последние 300 из 1258)"),
    }
    for k, dom in enumerate(["brain", "aviation", "finance"]):
        t, x, xlab = segs[dom]
        period, pw = morlet_cwt(x)
        a = ax[0, k]
        im = a.pcolormesh(np.asarray(t), period, np.sqrt(pw), cmap="magma", shading="auto")
        a.set_yscale("log"); a.set_xlabel(xlab); a.set_ylabel("псевдопериод, отсчётов")
        a.set_title(f"{'абв'[k]}) {DOM_LABEL[dom]}", fontsize=8.8)
        fig.colorbar(im, ax=a, fraction=0.05, pad=0.03).ax.tick_params(labelsize=6.5)
    for k, dom in enumerate(["brain", "aviation", "finance"]):
        ns, Fs, alpha = dfa_curve(S[dom]["x"])
        a = ax[1, k]
        a.plot(ns, Fs, "o-", color=DOM_COLOR[dom], ms=4, lw=1.2)
        fit = np.polyval(np.polyfit(np.log(ns), np.log(Fs), 1), np.log(ns))
        a.plot(ns, np.exp(fit), color=INK2, lw=0.9, ls="--")
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("масштаб n, отсчётов"); a.set_ylabel("F(n)")
        a.set_title(f"{'где'[k]}) {DOM_LABEL[dom]}: α = {comma(alpha)}", fontsize=8.8)
    fig.text(0.005, 0.75, "Вейвлет-\nразложение", fontsize=8, color=INK2, rotation=90, va="center", ha="center", fontweight="bold")
    fig.text(0.005, 0.28, "DFA", fontsize=8, color=INK2, rotation=90, va="center", ha="center", fontweight="bold")
    fig.suptitle("Один и тот же анализ сигнала на представительном ряду каждой области", fontsize=9.5, y=1.01)
    comma_axes(fig); fig.tight_layout(rect=[0.02, 0, 1, 1], w_pad=2.0)
    f = f"{OUT}/g1_signal-wavelet-dfa_{DATE}.png"; fig.savefig(f, dpi=200, bbox_inches="tight"); plt.close(fig); return f


# ================================================================ G2: структура графа объекта + ГНС/GAT + утечка
DOM_GRAPH_SPLIT = {"brain": ("structural", "shaft"), "aviation": ("physical", "group"), "finance": ("sector", "sector")}
GRAPH_LEGEND = [Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=INK2,
                       color="none", label="узел: канал / объект"),
                Line2D([], [], color=GRID, lw=1.4, label="ребро: предметная связь"),
                Line2D([], [], color=BLUE, lw=2.2, label="GAT: обучаемое внимание по ребру")]


def _graph_explanation(axis, text):
    """Apply the same legend and interpretation block to every domain graph."""
    axis.legend(handles=GRAPH_LEGEND, fontsize=5.7, loc="upper center",
                bbox_to_anchor=(0.5, -0.02), ncol=1, handletextpad=0.5)
    axis.text(0.5, -0.29, text, transform=axis.transAxes, ha="center", va="top",
              fontsize=6.2, color=INK2,
              bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.25"))


def _group_positions(nodes, group_of, group_order, col_gap=1.0, row_gap=0.8):
    """Позиции узлов графа: столбец — группа (модуль/ствол электрода/сектор) по group_order,
    внутри столбца узлы центрированы по вертикали. Тот же приём, что в src/make_figures.py
    (панель «а» рисунка 59), обобщённый на произвольный граф вместо только авиадвигателя."""
    xy = {}
    for k, g in enumerate(group_order):
        ids = [n for n in nodes if group_of[n] == g]
        for q, n in enumerate(ids):
            xy[n] = (k * col_gap, (q - (len(ids) - 1) / 2) * row_gap)
    return xy


def _draw_group_graph(a, xy, edges, highlight=None, edge_color=GRID, edge_lw=0.7, highlight_color=BLUE,
                      highlight_lw=1.6, node_color="white", node_size=200, node_lw=1.0, labels=None, fontsize=6.5):
    """Рисует граф (узлы — точки, рёбра — линии), где положение узла уже посчитано в xy.
    highlight — множество рёбер (пары id), которые нужно выделить другим цветом (например,
    рёбра, совпадающие с независимо построенным функциональным графом)."""
    a.grid(False); a.set_xticks([]); a.set_yticks([])
    for s in ("left", "bottom"):
        a.spines[s].set_visible(False)
    hi = highlight or set()
    for i, j in edges:
        hit = (i, j) in hi or (j, i) in hi
        c, lw, z = (highlight_color, highlight_lw, 2) if hit else (edge_color, edge_lw, 1)
        a.plot([xy[i][0], xy[j][0]], [xy[i][1], xy[j][1]], color=c, lw=lw, zorder=z)
    for n, (x, y) in xy.items():
        nc = node_color[n] if isinstance(node_color, dict) else node_color
        a.scatter([x], [y], s=node_size, color=nc, edgecolor=INK2, lw=node_lw, zorder=3)
        if labels is not None:
            a.text(x, y, labels[n], ha="center", va="center", fontsize=fontsize, zorder=4, color=INK)


def _brain_shaft_num(names):
    """Ствол электрода и номер контакта по имени («PM3» -> «PM», 3) — то же правило, что
    в src/brain_seeg_pipeline.py (load, structural_edges), пересчитано здесь по именам
    из results/brain/node_features.csv, без обращения к исходной записи ЭЭГ."""
    shaft = np.array([re.match(r"([A-Za-z]+'?)", n).group(1) for n in names])
    num = np.array([int(re.search(r"(\d+)$", n).group(1)) for n in names])
    return shaft, num


def _brain_structural_edges(shaft, num):
    E = []
    for s in np.unique(shaft):
        ids = sorted(np.where(shaft == s)[0], key=lambda i: num[i])
        E += [tuple(sorted((ids[k], ids[k + 1]))) for k in range(len(ids) - 1)]
    return sorted(E)


def object_structure_graphs(a_brain, a_aviation, a_finance, S):
    """Панели «а–в»: сам граф — структура объекта, которую ГНС получает на вход в каждой области,
    не только сводные метрики качества. Мозг и авиадвигатель показаны как настоящие графы датчиков/
    контактов (как в src/make_figures.py, рисунок 59а — тот же приём, что просил обобщить на все три
    примера); рынок — как секторный граф (полный граф внутри сектора, рёбер между секторами нет),
    показанный на масштабируемых глифах-«кликах», а не как нечитаемая масса из ~470 узлов."""
    # --- мозг: структурный граф контактов по стволам электродов (только внутри ствола)
    NF = pd.read_csv(f"{RES}/brain/node_features.csv")
    names = NF.name.values
    shaft, num = _brain_shaft_num(names)
    node_ids = list(range(len(names)))
    group_of = {i: shaft[i] for i in node_ids}
    group_order = sorted(np.unique(shaft), key=lambda s: (s.rstrip("'"), s.endswith("'")))
    xy = _group_positions(node_ids, group_of, group_order, col_gap=0.85, row_gap=0.55)
    E = _brain_structural_edges(shaft, num)
    _draw_group_graph(a_brain, xy, E, node_color="white", node_size=20, node_lw=0.45, edge_lw=0.7)
    top = max(y for _, y in xy.values())
    for k, g in enumerate(group_order):
        a_brain.text(k * 0.85, top + 0.55, g, ha="center", fontsize=6.5, color=INK2, fontweight="bold")
    a_brain.set_xlim(-0.6, (len(group_order) - 1) * 0.85 + 0.6); a_brain.set_ylim(-top - 0.6, top + 1.15)
    _graph_explanation(a_brain, "Контакты — узлы; соседство вдоль ствола — рёбра.\nGAT выделяет путь раннего распространения приступа.")
    a_brain.set_title("а) Мозг: граф контактов по стволам", fontsize=8.3)

    # --- авиадвигатель: тот же граф, что в src/make_figures.py (рис. 59а), пересчитан из results.json
    order = ["Fan", "LPC", "HPC", "Burner", "HPT", "LPT"]
    group_of = {s: A.MOD[s] for s in A.S}
    xy = _group_positions(A.S, group_of, order, col_gap=1.0, row_gap=0.8)
    Ra = S["aviation"]["R"]
    Ep = [tuple(e) for e in Ra["graphs"]["physical"]["edges"]]
    labels = {s: A.NAME[s] for s in A.S}
    _draw_group_graph(a_aviation, xy, Ep, node_size=280, node_lw=0.9, labels=labels, fontsize=5.8)
    for k, mname in enumerate(order):
        a_aviation.text(k, 2.55, mname, ha="center", fontsize=7, color=INK2, fontweight="bold")
    a_aviation.set_ylim(-3.9, 2.9); a_aviation.set_xlim(-0.6, 5.6)
    _graph_explanation(a_aviation, "Датчики — узлы; связь модулей тракта — рёбра.\nGAT взвешивает маршрут распространения деградации.")
    a_aviation.set_title("б) Авиадвигатель: граф датчиков по тракту", fontsize=8.3)

    # --- рынок: та же схема «группа-столбец, узлы, рёбра»; показывается до 6 акций сектора
    sc = S["finance"]["R"]["sector_counts"]
    order_f = sorted(sc, key=lambda s: -sc[s])
    nodes_f = [(sector, index) for sector in order_f for index in range(min(sc[sector], 6))]
    group_f = {node: node[0] for node in nodes_f}
    xy_f = _group_positions(nodes_f, group_f, order_f, col_gap=0.8, row_gap=0.42)
    edges_f = []
    for sector in order_f:
        ids = [node for node in nodes_f if node[0] == sector]
        edges_f.extend((ids[p], ids[q]) for p in range(len(ids)) for q in range(p + 1, len(ids)))
    _draw_group_graph(a_finance, xy_f, edges_f, node_color="white", node_size=22, node_lw=0.45, edge_lw=0.45)
    top_f = max(y for _, y in xy_f.values())
    for index, sector in enumerate(order_f):
        a_finance.text(index * 0.8, top_f + 0.45, f"S{index + 1}\nn={sc[sector]}",
                       ha="center", fontsize=5.5, color=INK2)
    a_finance.set_xlim(-0.5, (len(order_f) - 1) * 0.8 + 0.5); a_finance.set_ylim(-top_f - 0.5, top_f + 1.0)
    _graph_explanation(a_finance, "Акции — узлы; общий сектор GICS — рёбра.\nGAT выделяет секторное распространение рыночного шока.")
    a_finance.set_title("в) Рынок: секторный граф (клика GICS)", fontsize=8.3)


def graph_gnn_leakage(S):
    doms = ["brain", "aviation", "finance"]
    RUNS = {d: pd.read_csv(f"{RES}/{d}/gnn_runs.csv") for d in doms}
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 8.7), gridspec_kw={"wspace": 0.55, "hspace": 0.9})

    object_structure_graphs(ax[0, 0], ax[0, 1], ax[0, 2], S)

    a = ax[1, 0]
    models = [("gat", "ГНС GATv2"), ("mlp", "MLP"), ("lr", "Логрегрессия")]
    width = 0.25
    for j, (mdl, mlab) in enumerate(models):
        vals, errs = [], []
        for d in doms:
            g, split = DOM_GRAPH_SPLIT[d]
            graph = g if mdl == "gat" else "-"
            r = RUNS[d][(RUNS[d].model == mdl) & (RUNS[d].graph == graph) & (RUNS[d].split == split)].auroc
            vals.append(r.mean()); errs.append(r.std())
        x = np.arange(len(doms)) + (j - 1) * width
        a.bar(x, vals, width=width, yerr=errs, color=[BLUE, ORANGE, AQUA][j], alpha=0.85, capsize=3, label=mlab)
    a.axhline(0.5, color=INK2, lw=0.8, ls=":")
    a.set_xticks(range(len(doms))); a.set_xticklabels([DOM_LABEL[d].split(" (")[0] for d in doms], fontsize=8)
    a.set_ylabel("AUROC (протокол без утечки)"); a.legend(fontsize=7.5, loc="lower right")
    a.set_title("г) ГНС GATv2 против MLP и логрегрессии")

    a = ax[1, 1]
    for i, d in enumerate(doms):
        g, split = DOM_GRAPH_SPLIT[d]
        rand = RUNS[d][(RUNS[d].model == "gat") & (RUNS[d].graph == g) & (RUNS[d].split == "random")].auroc.mean()
        noleak = RUNS[d][(RUNS[d].model == "gat") & (RUNS[d].graph == g) & (RUNS[d].split == split)].auroc.mean()
        a.barh(i, rand - noleak, color=DOM_COLOR[d], alpha=0.85)
        a.text(rand - noleak + (0.002 if rand >= noleak else -0.002), i, comma(rand - noleak, 3),
              va="center", ha="left" if rand >= noleak else "right", fontsize=7.5, color=INK)
    a.axvline(0, color=INK2, lw=0.8); a.margins(x=0.3)
    a.set_yticks(range(len(doms))); a.set_yticklabels([DOM_LABEL[d].split(" (")[0] for d in doms], fontsize=8)
    a.set_xlabel("Δ AUROC: случайное − без утечки\n(ГНС GATv2)", fontsize=8)
    a.set_title("д) Утечка завышает качество")

    a = ax[1, 2]
    for i, d in enumerate(doms):
        g, split = DOM_GRAPH_SPLIT[d]
        gv = RUNS[d][(RUNS[d].model == "gat") & (RUNS[d].graph == g) & (RUNS[d].split == split)].auroc.mean()
        ev = RUNS[d][(RUNS[d].model == "gat") & (RUNS[d].graph == "empty") & (RUNS[d].split == split)].auroc.mean()
        a.barh(i, gv - ev, color=DOM_COLOR[d], alpha=0.85)
        a.text(gv - ev + (0.001 if gv >= ev else -0.001), i, comma(gv - ev, 3),
              va="center", ha="left" if gv >= ev else "right", fontsize=7.5, color=INK)
    a.axvline(0, color=INK2, lw=0.8); a.margins(x=0.3)
    a.set_yticks(range(len(doms))); a.set_yticklabels([DOM_LABEL[d].split(" (")[0] for d in doms], fontsize=8)
    a.set_xlabel("Δ AUROC: граф − без рёбер\n(ГНС GATv2, без утечки)", fontsize=8)
    a.set_title("е) Ценность графа")
    fig.suptitle("Структура графа объекта в трёх областях и роль этой структуры для ГНС", fontsize=9.5, y=1.0)
    comma_axes(fig); fig.tight_layout()
    f = f"{OUT}/g2_graph-gnn-gat-leakage_{DATE}.png"; fig.savefig(f, dpi=200, bbox_inches="tight"); plt.close(fig); return f


# ================================================================ G3: точность и матрица ошибок
def accuracy_confusion(S):
    try:
        repo = P.config()["biomedai_repo"]
        single = json.loads((repo / "gnn_model_result" / "attention_deep" / "sEEG-HFOs-8" / "gnn_model_result.json").read_text())
        cv_files = {"без DFA": "attention_deep_cv", "с DFA": "attention_deep_dfa_cv",
                    "без DFA, по стволу": "attention_deep_cv_shaft", "с DFA, по стволу": "attention_deep_dfa_cv_shaft"}
        cv = {lab: json.loads((repo / "gnn_model_result" / name / "sEEG-HFOs-8" / "gnn_cv_result.json").read_text()) for lab, name in cv_files.items()}
    except (FileNotFoundError, KeyError) as e:
        print(f"  блок E § 3.11 недоступен ({e}) — рисунок g3 пропущен; см. README.md, раздел 6")
        return None

    i = single["best_epoch"] - 1
    cm = np.array(single["val_confusion_matrix"]); acc = single["history"]["val_accuracy"][i]
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.8), gridspec_kw={"width_ratios": [0.8, 1.3]})

    a = ax[0]; a.grid(False)
    im = a.imshow(cm, cmap="Blues", vmin=0)
    for r in range(2):
        for c in range(2):
            a.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=13, color="white" if cm[r, c] > cm.max() / 2 else INK)
    a.set_xticks([0, 1]); a.set_xticklabels(["не ранний", "ранний"]); a.set_yticks([0, 1]); a.set_yticklabels(["не ранний", "ранний"])
    a.set_xlabel("предсказано"); a.set_ylabel("истина")
    a.set_title(f"а) Матрица ошибок: блок E § 3.11\n(attention_deep, доля верных {comma(acc, 3)})")

    a = ax[1]
    Rav = json.load(open(f"{RES}/aviation/results.json"))
    av_row = [g for g in Rav["gnn_summary"] if g["model"] == "gat" and g["graph"] == "physical" and g["split"] == "group"][0]
    labs = list(cv.keys()) + ["авиадвигатель: ГНС GATv2\n(macro-F1, для сравнения масштаба)"]
    means = [cv[k]["mean_val_macro_f1"] for k in cv] + [av_row["macro_f1_mean"]]
    stds = [cv[k]["std_val_macro_f1"] for k in cv] + [av_row["macro_f1_std"]]
    colors = [BLUE] * len(cv) + [ORANGE]
    y = np.arange(len(labs))[::-1]
    a.barh(y, means, xerr=stds, color=colors, alpha=0.85, capsize=3)
    for yy, m, s in zip(y, means, stds):
        a.text(m + s + 0.02, yy, comma(m), va="center", fontsize=7.5, color=INK)
    a.set_yticks(y); a.set_yticklabels(labs, fontsize=7.5); a.set_xlabel("macro-F1 (кросс-валидация)"); a.set_xlim(0, 1.15)
    a.axvline(0.5, color=INK2, lw=0.8, ls=":")
    a.set_title("б) Доля верных ответов вне фолда")
    comma_axes(fig); fig.tight_layout()
    f = f"{OUT}/g3_accuracy-confusion-matrix_{DATE}.png"; fig.savefig(f, dpi=200, bbox_inches="tight"); plt.close(fig); return f


SECT_RU = {"Consumer Discretionary": "Потреб. товары", "Consumer Staples": "Товары перв. необх.", "Energy": "Энергетика",
          "Financials": "Финансы", "Health Care": "Здравоохранение", "Industrials": "Промышленность",
          "Information Technology": "Инф. технологии", "Materials": "Сырьё и материалы", "Real Estate": "Недвижимость",
          "Utilities": "Коммун. услуги", "Telecommunication Services": "Телеком"}


# ================================================================ G4: каскад вовлечения + локализация
def cascade_localization(S):
    fig, ax = plt.subplots(2, 3, figsize=(11.5, 6.8))

    a = ax[0, 0]
    L = pd.read_csv(f"{RES}/brain/localization_contacts.csv").dropna(subset=["latency"]).sort_values("latency").head(10)
    a.barh(range(len(L)), L.latency.values[::-1], color=[BLUE if h == "R" else ORANGE for h in L.hem.values[::-1]])
    a.set_yticks(range(len(L))); a.set_yticklabels(L.name.values[::-1], fontsize=7.5)
    a.set_xlabel("латентность, с"); a.set_title("а) Мозг: 10 самых ранних контактов")

    a = ax[0, 1]
    D = pd.read_csv(f"{RES}/aviation/detection_FD001.csv").dropna(subset=["det_gev"])
    frac = np.sort(D.det_gev.values / D.life.values)
    a.step(frac, np.arange(1, len(frac) + 1) / len(frac), where="post", color=ORANGE, lw=1.4)
    a.set_xlabel("доля ресурса до срабатывания EVT-детектора", fontsize=8); a.set_ylabel("доля вовлечённых двигателей")
    a.set_title("б) Авиадвигатель: 100 двигателей")

    a = ax[0, 2]
    T = pd.read_csv(f"{RES}/finance/localization_sectors.csv")
    ep = T.loc[T.groupby("episode").footprint.transform("mean").idxmax(), "episode"]
    s = T[T.episode == ep].sort_values("footprint", ascending=False).head(8)
    a.barh(range(len(s)), s.footprint.values[::-1], color=AQUA)
    a.set_yticks(range(len(s))); a.set_yticklabels([SECT_RU.get(n, n) for n in s.sector.values[::-1]], fontsize=7)
    a.set_xlabel("доля акций сектора за EVT-порогом", fontsize=8); a.set_title(f"в) Рынок: эпизод {ep}")

    a = ax[1, 0]
    S1 = pd.read_csv(f"{RES}/brain/localization_shafts.csv")
    a.scatter(S1.footprint, S1.residual_z, s=40, color=BLUE, alpha=0.85)
    for _, r in S1.iterrows():
        a.annotate(r["shaft"], (r.footprint, r.residual_z), fontsize=6.5, color=INK2, xytext=(3, 3), textcoords="offset points")
    a.axhline(0, color=INK2, lw=0.7, ls=":"); a.set_xlabel("след события"); a.set_ylabel("остаток сверх моды, z")
    a.set_title("г) Мозг: локализация по стволам")

    a = ax[1, 1]
    Lc = pd.read_csv(f"{RES}/aviation/localization_FD003.csv")
    for src, c in [("HPC", BLUE), ("Fan", ORANGE)]:
        s = Lc[Lc.source_sig == src]
        a.scatter(s.z_BPR, s.z_Nf, s=16, color=c, alpha=0.8, label=src)
    a.set_xlabel("z-откл. BPR"); a.set_ylabel("z-откл. Nf"); a.legend(fontsize=7.5, loc="lower right")
    a.set_title("д) Авиадвигатель: источник (FD003)")

    a = ax[1, 2]; a.grid(False)
    eps = sorted(T.episode.unique()); sect = sorted(T.sector.unique())
    M = T.pivot(index="sector", columns="episode", values="footprint").loc[sect, eps]
    im = a.imshow(M.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    a.set_xticks(range(len(eps))); a.set_xticklabels(eps, rotation=30, fontsize=6.5)
    a.set_yticks(range(len(sect))); a.set_yticklabels([SECT_RU.get(s, s) for s in sect], fontsize=6)
    fig.colorbar(im, ax=a, fraction=0.05, pad=0.03).ax.tick_params(labelsize=6.5)
    a.set_title("е) Рынок: след по секторам × эпизодам")

    fig.suptitle("Каскад вовлечения и локализация источника — одна и та же схема анализа в трёх областях", fontsize=9.5, y=1.01)
    comma_axes(fig); fig.tight_layout(w_pad=2.2, h_pad=2.0)
    f = f"{OUT}/g4_cascade-localization_{DATE}.png"; fig.savefig(f, dpi=200, bbox_inches="tight"); plt.close(fig); return f


if __name__ == "__main__":
    which = sys.argv[1:] or ["g0", "g1", "g2", "g3", "g4"]
    S = load_domain_series()
    fn = {"g0": lambda: task_overview(S), "g1": lambda: signal_wavelet_dfa(S), "g2": lambda: graph_gnn_leakage(S),
          "g3": lambda: accuracy_confusion(S), "g4": lambda: cascade_localization(S)}
    for w in which:
        print(fn[w]())

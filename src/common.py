"""Общие функции конвейера «окна -> признаки -> граф -> ГНС -> EVT -> локализация -> leakage-safe проверка».

Модуль используется обоими небиомедицинскими примерами (авиационный двигатель C-MAPSS и фондовый
рынок S&P 500) и повторяет по смыслу элементы программных артефактов основной работы
(extreme_event_agent: GEV/POT-детекция; gnn_model: GATv2 + DropEdge; healthcare_gnn: параметризуемое
построение графа; группированная кросс-валидация как leakage-safe протокол).
"""
from __future__ import annotations
import numpy as np
from scipy import stats

# ---------------------------------------------------------------- статистика
def dfa_alpha(x, scales=None):
    """Показатель DFA-1 (Peng et al., 1994): F(n) ~ n^alpha."""
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    N = len(y)
    if scales is None:
        scales = np.unique(np.floor(np.logspace(np.log10(4), np.log10(N // 4), 8)).astype(int))
    F = []
    for n in scales:
        m = N // n
        if m < 2:
            continue
        seg = y[: m * n].reshape(m, n)
        t = np.arange(n)
        res = []
        for s in seg:
            p = np.polyfit(t, s, 1)
            res.append(np.mean((s - np.polyval(p, t)) ** 2))
        F.append((n, np.sqrt(np.mean(res))))
    F = np.array(F)
    return float(np.polyfit(np.log(F[:, 0]), np.log(F[:, 1]), 1)[0])


def rm_corr(subject, x, y):
    """Корреляция повторных измерений (Bakdash, Marusich, 2017): общая внутрисубъектная связь x и y."""
    import pandas as pd
    df = pd.DataFrame({"s": subject, "x": x, "y": y}).dropna()
    xc = df.x - df.groupby("s").x.transform("mean")
    yc = df.y - df.groupby("s").y.transform("mean")
    r = float(np.sum(xc * yc) / np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2)))
    n_sub = df.s.nunique()
    dof = len(df) - n_sub - 1
    t = r * np.sqrt(dof / (1 - r ** 2))
    p = float(2 * stats.t.sf(abs(t), dof))
    z = np.arctanh(r); se = 1 / np.sqrt(dof - 1)
    ci = (float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se)))
    return {"r_rm": r, "dof": int(dof), "p": p, "ci95": ci}


def gev_fit(x):
    """ML-оценка GEV с несколькими стартовыми точками (стандартный вызов scipy без стартовой точки
    на коротких выборках иногда сходится к локальному оптимуму с заведомо худшим правдоподобием)."""
    x = np.asarray(x, float)
    sc0 = x.std() * np.sqrt(6) / np.pi + 1e-12
    loc0 = x.mean() - 0.5772 * sc0
    best, best_nll = None, np.inf
    starts = [None] + [(c0, loc0, sc0) for c0 in (-0.3, -0.1, 0.0, 0.1, 0.3)]
    for st in starts:
        try:
            f = stats.genextreme.fit(x) if st is None else stats.genextreme.fit(x, st[0], loc=st[1], scale=st[2])
            nll = stats.genextreme.nnlf(f, x)
            if np.isfinite(nll) and nll < best_nll:
                best, best_nll = f, nll
        except Exception:
            pass
    return best


def gev_threshold(block_maxima, q=0.99):
    """Порог по GEV, подогнанной к блочным максимумам фона; возвращает (порог, параметр формы xi)."""
    c, loc, sc = gev_fit(block_maxima)
    return float(stats.genextreme.ppf(q, c, loc, sc)), float(-c)  # xi = -c в параметризации scipy


def pot_threshold(x, u_q=0.8, q=0.999):
    """Порог по методу превышений (POT/GPD): квантиль q распределения x через хвост выше квантиля u_q."""
    x = np.asarray(x, float)
    u = np.quantile(x, u_q)
    exc = x[x > u] - u
    best, best_nll = None, np.inf
    for st in [None, -0.2, 0.0, 0.2]:
        try:
            f = stats.genpareto.fit(exc, floc=0) if st is None else stats.genpareto.fit(exc, st, floc=0, scale=exc.mean())
            nll = stats.genpareto.nnlf(f, exc)
            if np.isfinite(nll) and nll < best_nll:
                best, best_nll = f, nll
        except Exception:
            pass
    xi, _, beta = best
    pu = (x > u).mean()
    return float(u + stats.genpareto.ppf(1 - (1 - q) / pu, xi, 0, beta)), float(xi)


def first_persistent(mask, start, k):
    """Первый индекс >= start, с которого mask истинна k раз подряд (правило устойчивости срабатывания)."""
    run = 0
    for i in range(start, len(mask)):
        run = run + 1 if mask[i] else 0
        if run == k:
            return i - k + 1
    return None


def jaccard(E1, E2):
    a = {tuple(sorted(e)) for e in E1}; b = {tuple(sorted(e)) for e in E2}
    return len(a & b) / max(1, len(a | b))

# ---------------------------------------------------------------- графы
def topk_edges(S, k, exclude_self=True):
    """Неориентированные рёбра top-k по матрице сходства S (симметризация объединением)."""
    S = np.array(S, float).copy()
    n = len(S)
    if exclude_self:
        np.fill_diagonal(S, -np.inf)
    E = set()
    for i in range(n):
        for j in np.argsort(-S[i])[:k]:
            E.add(tuple(sorted((i, int(j)))))
    return sorted(E)


def functional_edges(X, k=3):
    """Функциональный граф: top-k по |корреляции| временных рядов узлов (X: время x узлы)."""
    C = np.abs(np.corrcoef(X.T))
    C = np.nan_to_num(C)
    return topk_edges(C, k)


def knn_edges(F, k=3):
    """kNN-граф в пространстве признаков узлов (евклидова метрика после стандартизации)."""
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
    D = np.sqrt(((Fz[:, None, :] - Fz[None, :, :]) ** 2).sum(-1))
    return topk_edges(-D, k)


def edge_index_from(E, n):
    import torch
    if len(E) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    src = [i for i, j in E] + [j for i, j in E]
    dst = [j for i, j in E] + [i for i, j in E]
    return torch.tensor([src, dst], dtype=torch.long)

# ---------------------------------------------------------------- модели
def make_models():
    import torch
    from torch import nn
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
    from torch_geometric.utils import dropout_edge

    class GraphGAT(nn.Module):
        """GATv2 (Brody et al., 2022) + DropEdge (Rong et al., 2020): классификация графа целиком."""
        def __init__(self, f_in, hid=16, heads=2, p_edge=0.2):
            super().__init__()
            self.c1 = GATv2Conv(f_in, hid, heads=heads, add_self_loops=True)
            self.c2 = GATv2Conv(hid * heads, hid, heads=1, add_self_loops=True)
            self.out = nn.Linear(2 * hid, 1)
            self.p_edge = p_edge
        def forward(self, x, ei, batch, return_att=False):
            if self.training and self.p_edge > 0 and ei.size(1) > 0:
                ei, _ = dropout_edge(ei, p=self.p_edge)
            h = torch.nn.functional.elu(self.c1(x, ei))
            h = torch.nn.functional.elu(self.c2(h, ei))
            g = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], 1)
            return self.out(g).squeeze(-1)

    class NodeGAT(nn.Module):
        """GATv2 для классификации узлов (аналог SeizureGAT: роль узла в экстремальном событии)."""
        def __init__(self, f_in, hid=16, heads=2, p_edge=0.2, p_drop=0.2):
            super().__init__()
            self.c1 = GATv2Conv(f_in, hid, heads=heads, add_self_loops=True)
            self.c2 = GATv2Conv(hid * heads, 1, heads=1, add_self_loops=True)
            self.p_edge, self.p_drop = p_edge, p_drop
        def forward(self, x, ei):
            if self.training and self.p_edge > 0 and ei.size(1) > 0:
                ei, _ = dropout_edge(ei, p=self.p_edge)
            x = torch.nn.functional.dropout(x, self.p_drop, self.training)
            h = torch.nn.functional.elu(self.c1(x, ei))
            return self.c2(h, ei).squeeze(-1)

    class MLP(nn.Module):
        def __init__(self, f_in, hid=64):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(f_in, hid), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hid, 1))
        def forward(self, x):
            return self.net(x).squeeze(-1)

    return GraphGAT, NodeGAT, MLP


def metrics(y, p, thr=0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    y = np.asarray(y); p = np.asarray(p)
    out = {"auprc": float(average_precision_score(y, p)), "macro_f1": float(f1_score(y, p >= thr, average="macro"))}
    out["auroc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return out

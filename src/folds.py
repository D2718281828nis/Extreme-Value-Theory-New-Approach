"""Состав фолдов кросс-валидации, одинаковый на любой платформе.

sklearn.model_selection.GroupKFold упорядочивает группы по размеру через np.argsort без kind, и порядок групп
равного размера зависит от SIMD-сортировки платформы (x86 и arm64 дают разный состав фолдов). Здесь тот же
жадный алгоритм, но с детерминированным порядком: размер по убыванию, затем str(метка) по возрастанию.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def stable_group_kfold(groups, n_splits=5):
    """Как GroupKFold(n_splits).split(..., groups): каждая группа по порядку «размер ↓, str(метка) ↑» идёт
    в фолд с наименьшим текущим числом объектов (при равенстве — с меньшим номером).
    Возвращает [(train_idx, test_idx)] для фолдов 0..K-1, индексы по возрастанию."""
    groups = np.asarray(groups)
    labels, inv = np.unique(groups, return_inverse=True)
    if len(labels) < n_splits:
        raise ValueError(f"групп {len(labels)} меньше, чем фолдов {n_splits}")
    size = np.bincount(inv)
    order = sorted(range(len(labels)), key=lambda g: (-size[g], str(labels[g])))
    load = np.zeros(n_splits, dtype=np.int64); fold_of = np.empty(len(labels), dtype=np.int64)
    for g in order:
        f = int(np.argmin(load))                 # argmin возвращает первый минимум — фолд с меньшим номером
        load[f] += size[g]; fold_of[g] = f
    fold = fold_of[inv]
    idx = np.arange(len(groups))
    return [(idx[fold != k], idx[fold == k]) for k in range(n_splits)]


def random_kfold(n, n_splits, seed):
    """KFold(n_splits, shuffle=True, random_state=seed) — детерминирован и так; обёртка для единого интерфейса."""
    return list(KFold(n_splits, shuffle=True, random_state=seed).split(np.arange(n)))


def fold_table(ids, groups, folds, split, seed=None):
    """Таблица split, seed, id, group, fold для одного разбиения (seed — для случайного, иначе пусто)."""
    ids = np.asarray(ids); groups = np.asarray(groups) if groups is not None else np.full(len(ids), "")
    fold = np.full(len(ids), -1)
    for k, (_, te) in enumerate(folds):
        fold[te] = k
    assert (fold >= 0).all(), "каждый объект должен попасть в тест ровно одного фолда"
    return pd.DataFrame({"split": split, "seed": "" if seed is None else int(seed), "id": ids.astype(str),
                         "group": groups.astype(str), "fold": fold})


def save_fold_map(path, table):
    """Пишет таблицу фолдов в CSV (\\n, без индекса) и возвращает MD5 файла."""
    if isinstance(table, (list, tuple)):
        table = pd.concat(table, ignore_index=True)
    table.to_csv(path, index=False, lineterminator="\n")
    return hashlib.md5(open(path, "rb").read()).hexdigest()

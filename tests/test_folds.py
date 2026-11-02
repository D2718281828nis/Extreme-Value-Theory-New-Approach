"""Тесты src/folds.py: состав фолдов не зависит от порядка строк, номеров групп и платформы."""
import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import folds as FO


def label_to_fold(groups, folds):
    out = {}
    for k, (_, te) in enumerate(folds):
        for g in np.unique(np.asarray(groups)[te]):
            assert g not in out
            out[g] = k
    return out


def tied_groups():
    rng = np.random.default_rng(0)
    sizes = [10, 10, 10, 8, 8, 6, 6, 6, 5, 3, 3, 2]
    labels = [f"G{i:02d}" for i in range(len(sizes))]
    g = np.repeat(labels, sizes)
    return g[rng.permutation(len(g))]


def test_invariant_to_row_order_and_group_renumbering():
    g = tied_groups()
    base = label_to_fold(g, FO.stable_group_kfold(g))
    perm = np.random.default_rng(1).permutation(len(g))
    assert label_to_fold(g[perm], FO.stable_group_kfold(g[perm])) == base
    # перенумерация: внутренние коды групп (порядок первого появления, тип массива) другие, метки те же
    first = g[np.argsort([-ord(s[-1]) for s in g], kind="stable")]
    assert label_to_fold(first, FO.stable_group_kfold(first)) == base
    obj = np.array(list(g), dtype=object)
    assert label_to_fold(obj, FO.stable_group_kfold(obj)) == base


def test_matches_groupkfold_without_ties():
    sizes = [20, 17, 15, 14, 12, 11, 9, 7, 5, 4, 2, 1]
    g = np.repeat(np.arange(len(sizes)), sizes)
    g = g[np.random.default_rng(2).permutation(len(g))]
    a = [tuple(te) for _, te in GroupKFold(5).split(g, groups=g)]
    b = [tuple(te) for _, te in FO.stable_group_kfold(g)]
    assert a == b


def test_each_group_in_one_test_fold_and_fold_sizes_match_groupkfold():
    g = tied_groups()
    F = FO.stable_group_kfold(g)
    l2f = label_to_fold(g, F)
    assert set(l2f) == set(g)
    allte = np.concatenate([te for _, te in F])
    assert sorted(allte) == list(range(len(g)))
    for tr, te in F:
        assert np.all(np.diff(tr) > 0) and np.all(np.diff(te) > 0)
        assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == len(g)
    ref = sorted(len(te) for _, te in GroupKFold(5).split(g, groups=g))
    assert sorted(len(te) for _, te in F) == ref


def test_too_few_groups_raises():
    with pytest.raises(ValueError):
        FO.stable_group_kfold(np.array(["a", "b", "a"]), 5)


def test_fold_table_md5_platform_independent(tmp_path):
    g = tied_groups()
    t1 = FO.fold_table(np.arange(len(g)), g, FO.stable_group_kfold(g), "group")
    t2 = FO.fold_table(np.arange(len(g)), None, FO.random_kfold(len(g), 5, 0), "random", seed=0)
    md5 = FO.save_fold_map(tmp_path / "folds.csv", [t1, t2])
    assert md5 == "811e0b6ed0ba4b983db82e00e837b79f", md5

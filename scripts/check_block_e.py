"""Сверка чисел блока E § 3.11 исследования (обучение ГНС на графе объекта) с результатами репозитория автора.

    python scripts/check_block_e.py                                   # сохранённые результаты gnn_model_result/
    python scripts/check_block_e.py --results gnn_model_result_rerun  # результаты собственного повторного прогона

Сами расчёты блока E выполнены кодом репозитория BioMedAI-sEEG-core-of-epilepsy (пакет gnn_model, коммит a283320),
а не кодом этой папки; команды повторного прогона — scripts/block_e_commands.sh. Здесь только читаются
JSON-файлы результатов и граф объекта и сравниваются с числами, приведёнными в тексте.
Таблица по пяти затравкам (1, 2, 3, 7, 11) в репозитории как файлы не сохранена — она приведена в README
репозитория; её можно получить повторным прогоном с --seed (см. block_e_commands.sh).
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

# Числа из текста исследования (§ 3.11, блок E): параметры, ошибка валидации, доля верных ответов, матрица ошибок
SINGLE = {
    "baseline_overfit": dict(params=226, val_loss=6.29, val_acc=0.933, cm=[[1, 1], [1, 27]], at="final"),
    "regularized":      dict(params=114, val_loss=0.66, val_acc=0.333, cm=[[1, 1], [19, 9]], at="best"),
    "attention":        dict(params=54, val_loss=0.79, val_acc=0.867, cm=[[1, 1], [3, 25]], at="best"),
    "attention_deep":   dict(params=1562, val_loss=1.02, val_acc=0.933, cm=[[1, 1], [1, 27]], at="best"),
}
CV = {
    "attention_deep_cv":           dict(f1=(0.55, 0.06), oof=[[4, 1], [21, 71]], shaft=False, params=1562),
    "attention_deep_dfa_cv":       dict(f1=(0.84, 0.22), oof=[[4, 1], [4, 88]], shaft=False, params=1658),
    "attention_deep_cv_shaft":     dict(f1=(0.50, 0.13), oof=[[4, 1], [19, 73]], shaft=True, params=1562),
    "attention_deep_dfa_cv_shaft": dict(f1=(0.48, 0.02), oof=[[0, 5], [2, 90]], shaft=True, params=1658),
}
DFA_TEXT = {"earliest": (1.22, 0.05, 5), "later_recruited": (1.15, 0.10, 92), "p": 0.014}
SEEDS = [1, 2, 3, 7, 11]
SEEDS_TEXT = {"shaft": ([4, 4, 1, 4, 4], [8, 9, 16, 19, 17]), "dfa_shaft": ([4, 3, 2, 0, 3], [19, 7, 10, 2, 21])}
EDF_NAME = "sEEG-HFOs-8"


def mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def read_graphml(path: Path):
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    root = ET.parse(path).getroot()
    keys = {k.get("id"): k.get("attr.name") for k in root.findall("g:key", ns)}
    nodes = []
    for n in root.iter("{http://graphml.graphdrawing.org/xmlns}node"):
        nodes.append({keys[d.get("key")]: d.text for d in n.findall("g:data", ns)})
    n_edges = sum(1 for _ in root.iter("{http://graphml.graphdrawing.org/xmlns}edge"))
    return nodes, n_edges


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None, help="путь к BioMedAI-sEEG-core-of-epilepsy (по умолчанию из config/paths.json)")
    ap.add_argument("--results", default="gnn_model_result", help="папка результатов внутри репозитория")
    a = ap.parse_args()
    repo = Path(a.repo) if a.repo else P.config()["biomedai_repo"]
    res = repo / a.results
    if not res.exists():
        print(f"Нет папки {res}. Укажите --repo или путь в config/paths.json → biomedai_repo.")
        return 1
    bad = 0

    print("Граф объекта (object_model_result/sEEG-HFOs-8/)")
    for name in ["object_model_graph.graphml", "object_model_graph_dfa.graphml"]:
        f = repo / "object_model_result" / EDF_NAME / name
        if f.exists():
            nodes, ne = read_graphml(f)
            ch = [n for n in nodes if n.get("role") in ("earliest", "later_recruited")]
            early = sum(n["role"] == "earliest" for n in ch)
            print(f"  {name}: узлов {len(nodes)}, из них классифицируемых {len(ch)} (ранних {early}), рёбер {ne}")
    f = repo / "object_model_result" / EDF_NAME / "object_model_graph_dfa.graphml"
    if f.exists():
        import numpy as np
        from scipy import stats
        nodes, _ = read_graphml(f)
        v = {r: np.array([float(n["dfa_alpha"]) for n in nodes if n.get("role") == r and n.get("dfa_alpha")]) for r in DFA_TEXT if r != "p"}
        p = stats.mannwhitneyu(v["earliest"], v["later_recruited"]).pvalue
        print("  показатель DFA по ролям (ср. ± ст. откл., n):")
        for r in ("earliest", "later_recruited"):
            m, s, n = DFA_TEXT[r]
            got = (round(v[r].mean(), 2), round(v[r].std(), 2), len(v[r]))  # ст. откл. по генеральной совокупности, как в README
            ok = got == (m, s, n)
            bad += not ok
            print(f"    {mark(ok)} {r}: {got[0]} ± {got[1]} (n={got[2]}); в тексте {m} ± {s} (n={n})")
        ok = round(p, 3) == DFA_TEXT["p"]
        bad += not ok
        print(f"    {mark(ok)} критерий Манна–Уитни p = {p:.4f}; в тексте {DFA_TEXT['p']}")

    print("\nОдиночное разбиение 67/30 (затравка 7)")
    for name, e in SINGLE.items():
        f = res / name / EDF_NAME / "gnn_model_result.json"
        if not f.exists():
            print(f"  – {name}: нет файла {f}")
            bad += 1
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        h = d["history"]
        i = len(h["val_loss"]) - 1 if e["at"] == "final" or not d.get("best_epoch") else d["best_epoch"] - 1
        got = dict(params=d["trainable_parameters"], val_loss=round(h["val_loss"][i], 2),
                   val_acc=round(h["val_accuracy"][i], 3), cm=d["val_confusion_matrix"])
        ok = got["params"] == e["params"] and got["val_loss"] == e["val_loss"] and got["val_acc"] == e["val_acc"] and got["cm"] == e["cm"]
        bad += not ok
        print(f"  {mark(ok)} {name}: параметров {got['params']}, ошибка валидации {got['val_loss']} "
              f"({'последняя эпоха' if e['at'] == 'final' else 'лучшая эпоха ' + str(d.get('best_epoch'))}), "
              f"доля верных {got['val_acc']}, матрица {got['cm']}")

    print("\nПятикратная кросс-валидация (затравка 7)")
    for name, e in CV.items():
        f = res / name / EDF_NAME / "gnn_cv_result.json"
        if not f.exists():
            print(f"  – {name}: нет файла {f}")
            bad += 1
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        f1 = (round(d["mean_val_macro_f1"], 2), round(d["std_val_macro_f1"], 2))
        ok = f1 == e["f1"] and d["out_of_fold_confusion_matrix"] == e["oof"] and bool(d.get("group_by_shaft", False)) == e["shaft"] \
            and d["trainable_parameters"] == e["params"]
        bad += not ok
        print(f"  {mark(ok)} {name}: macro-F1 {f1[0]} ± {f1[1]}, матрица вне фолда {d['out_of_fold_confusion_matrix']}, "
              f"группировка по стволу: {bool(d.get('group_by_shaft', False))}, параметров {d['trainable_parameters']}")

    seeds_dir = res / "seeds"
    print("\nПять затравок, группировка по стволу (полнота по ранним из 5 / ложные срабатывания на поздних)")
    if not seeds_dir.exists():
        print(f"  – файлов нет ({seeds_dir}); в тексте — таблица из README репозитория. Получить: scripts/block_e_commands.sh, шаг 4")
    else:
        for tag, feat in [("shaft", "без DFA"), ("dfa_shaft", "с DFA")]:
            rec, fp = [], []
            for s in SEEDS:
                f = seeds_dir / f"{tag}_seed{s}" / EDF_NAME / "gnn_cv_result.json"
                if f.exists():
                    m = json.loads(f.read_text(encoding="utf-8"))["out_of_fold_confusion_matrix"]
                    rec.append(m[0][0]); fp.append(m[1][0])
                else:
                    rec.append(None); fp.append(None)
            ok = rec == SEEDS_TEXT[tag][0] and fp == SEEDS_TEXT[tag][1]
            bad += not ok
            print(f"  {mark(ok)} {feat}: полнота {rec}, ложные {fp}; в тексте {SEEDS_TEXT[tag][0]}, {SEEDS_TEXT[tag][1]}")

    print("\nИТОГ: все числа блока E совпадают с результатами." if not bad else f"\nИТОГ: расхождений — {bad}.")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())

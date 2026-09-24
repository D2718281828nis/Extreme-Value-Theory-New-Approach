"""Сверка results/<пример>/results.json с reference_results/<пример>/results.json.

    python scripts/compare_results.py                 # все три примера
    python scripts/compare_results.py brain           # один пример
    python scripts/compare_results.py --rtol 1e-4     # мягче допуск

Числа делятся на две группы.
* Детерминированные (EVT-детекция, гипотеза о предвестнике, графы, каскад, локализация без ГНС) — должны совпасть
  в пределах допуска на любом компьютере.
* Зависящие от обучения ГНС (ключи с «gnn», «graph_gain»): PyTorch не гарантирует побитовой воспроизводимости между
  разными процессорами, ОС и версиями библиотек (в том числе BLAS: OpenBLAS / Apple Accelerate). Для них печатается
  расхождение, но код выхода не меняется; содержательные выводы главы 7 (знак и порядок эффектов) нужно сверять по сути.
Код выхода 1 — если расходятся детерминированные числа или отсутствует файл.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

IGNORE = {"runtime_s"}


def flatten(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, f"{p}.{k}" if p else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flatten(v, f"{p}[{i}]")
    else:
        yield p, o


def gnn_dependent(path: str) -> bool:
    s = path.lower()
    return "gnn" in s or s.startswith("graph_gain")


def close(a, b, rtol, atol) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= atol + rtol * abs(b)


def compare(name: str, rtol: float, atol: float) -> bool:
    new_f, ref_f = P.RESULTS / name / "results.json", P.REFERENCE / name / "results.json"
    print(f"\n=== {name} ===")
    if not new_f.exists():
        print(f"  нет {new_f} — расчёт ещё не выполнен")
        return False
    new = dict(flatten(json.loads(new_f.read_text(encoding="utf-8"))))
    ref = dict(flatten(json.loads(ref_f.read_text(encoding="utf-8"))))
    keys = [k for k in ref if k.split(".")[0].split("[")[0] not in IGNORE]
    det_bad, gnn_bad, missing = [], [], []
    n_det = n_gnn = 0
    for k in keys:
        is_gnn = gnn_dependent(k)
        n_gnn += is_gnn
        n_det += not is_gnn
        if k not in new:
            missing.append(k)
            continue
        if not close(new[k], ref[k], rtol, atol):
            (gnn_bad if is_gnn else det_bad).append((k, ref[k], new[k]))
    extra = [k for k in new if k not in ref and k.split(".")[0] not in IGNORE]
    print(f"  детерминированные числа: {n_det - len(det_bad) - len([m for m in missing if not gnn_dependent(m)])} из {n_det} совпали")
    print(f"  числа, зависящие от обучения ГНС: {n_gnn - len(gnn_bad) - len([m for m in missing if gnn_dependent(m)])} из {n_gnn} совпали")
    for title, rows in [("РАСХОЖДЕНИЯ (детерминированные)", det_bad), ("расхождения ГНС (допустимы между платформами)", gnn_bad)]:
        if rows:
            print(f"  {title}:")
            for k, r, n in rows[:40]:
                print(f"    {k}: эталон {r!r} → сейчас {n!r}")
            if len(rows) > 40:
                print(f"    … и ещё {len(rows) - 40}")
    if missing:
        print(f"  нет в новом результате: {missing[:10]}{' …' if len(missing) > 10 else ''}")
    if extra:
        print(f"  новые ключи, которых нет в эталоне: {extra[:10]}")
    return not det_bad and not [m for m in missing if not gnn_dependent(m)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", default=["brain", "aviation", "finance"])
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    a = ap.parse_args()
    ok = all([compare(w, a.rtol, a.atol) for w in a.which])
    print("\nИТОГ: детерминированные результаты совпадают с эталоном." if ok else "\nИТОГ: есть расхождения — см. выше.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Сверка эталона этого репозитория с эталоном расчётного ядра диссертации (dissertation-calculations, R1).

    python scripts/check_sync_with_r1.py --r1 ../dissertation-calculations          # сверка
    python scripts/check_sync_with_r1.py --r1 ../dissertation-calculations --write  # сверка + reference_results/R1_SYNC.json

Здесь те же три конвейера, что в R1, плюс собственные дополнения (игрушечный набор мозга, окклюзия всех модулей
двигателя), поэтому файлы не копируются, а сравниваются:
* MD5 folds.csv — должен совпасть побайтно (состав фолдов задан src/folds.py);
* каждый ключ results.json эталона R1 должен быть и здесь и совпасть с допуском compare_results.py; собственные
  ключи этого репозитория (которых нет в R1) не проверяются.
Ключи из ALLOWED — известные отличия собственных дополнений; они печатаются, но не считаются ошибкой.
Код выхода 1 — при любом другом расхождении. R1_SYNC.json фиксирует результат сверки для CI (tests/test_sync.py).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_results import IGNORE, IGNORE_PATHS, close, flatten  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ["brain", "aviation", "finance"]
ALLOWED: dict[str, tuple[str, ...]] = {}
SYNCED_CODE = ["src/common.py", "src/folds.py", "src/aviation_false_alarm_bootstrap.py", "tests/test_folds.py", "config/systems.json"]
EXTRA_FILES = {"aviation": ["false_alarm_bootstrap.json"]}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def git_head(repo: Path) -> str:
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return head + (" + незакоммиченные изменения" if dirty else "")
    except OSError:
        return "неизвестно"


def check(r1: Path, rtol: float, atol: float) -> tuple[bool, dict]:
    ok, report = True, {}
    for s in SYSTEMS:
        mine, ref = ROOT / "reference_results" / s, r1 / "reference_results" / s
        fm, fr = md5(mine / "folds.csv"), md5(ref / "folds.csv")
        a = dict(flatten(json.loads((mine / "results.json").read_text(encoding="utf-8"))))
        b = dict(flatten(json.loads((ref / "results.json").read_text(encoding="utf-8"))))
        keys = [k for k in b if k.split(".")[0].split("[")[0] not in IGNORE and k not in IGNORE_PATHS]
        bad = [k for k in keys if k not in a or not close(a[k], b[k], rtol, atol)]
        allowed = [k for k in bad if k.startswith(ALLOWED.get(s, ())) and ALLOWED.get(s)]
        bad = [k for k in bad if k not in allowed]
        extra = {f: md5(mine / f) == md5(ref / f) for f in EXTRA_FILES.get(s, [])}
        sys_ok = fm == fr and not bad and all(extra.values())
        ok &= sys_ok
        print(f"[{s}] folds.csv {'совпадает' if fm == fr else 'ОТЛИЧАЕТСЯ'}; ключей R1 {len(keys)}, совпали {len(keys) - len(bad) - len(allowed)}"
              + (f", известные отличия {len(allowed)}" if allowed else "") + (f", РАСХОЖДЕНИЙ {len(bad)}" if bad else ""))
        for k in bad[:20]:
            print(f"    {k}: R1 {b.get(k)!r} — здесь {a.get(k, 'нет')!r}")
        for k in allowed:
            print(f"    (допустимо) {k}: R1 {b.get(k)!r} — здесь {a.get(k)!r}")
        for f, same in extra.items():
            print(f"    {f}: {'совпадает' if same else 'ОТЛИЧАЕТСЯ'}")
        report[s] = {"folds_md5": fm, "r1_keys": len(keys), "equal": len(keys) - len(bad) - len(allowed),
                     "allowed_differences": allowed, "extra_files_equal": extra}
    code = {f: md5(ROOT / f) == md5(r1 / f) for f in SYNCED_CODE}
    for f, same in code.items():
        print(f"[код] {f}: {'совпадает' if same else 'ОТЛИЧАЕТСЯ'}")
        ok &= same
    report["code_md5"] = {f: md5(ROOT / f) for f in SYNCED_CODE}
    return ok, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", type=Path, default=ROOT.parent / "dissertation-calculations")
    ap.add_argument("--rtol", type=float, default=1e-6); ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--write", action="store_true", help="записать reference_results/R1_SYNC.json (только при полном совпадении)")
    a = ap.parse_args()
    ok, report = check(a.r1.resolve(), a.rtol, a.atol)
    print("ИТОГ:", "эталон совпадает с R1" if ok else "есть расхождения с R1")
    if ok and a.write:
        out = {"checked": dt.date.today().isoformat(), "r1": "dissertation-calculations", "r1_commit": git_head(a.r1),
               "rtol": a.rtol, "atol": a.atol, **report}
        (ROOT / "reference_results" / "R1_SYNC.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("записан reference_results/R1_SYNC.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

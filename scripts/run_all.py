"""Полный повтор расчётов главы 7 (три примера) и сверка с результатами, по которым написан текст исследования.

    python scripts/run_all.py                      # проверка данных → мозг → двигатель → рынок → рисунки → сверка
    python scripts/run_all.py --only brain         # один шаг: brain | aviation | finance | figures | compare
    python scripts/run_all.py --only aviation --jobs 1   # пример 2 в одном процессе (по умолчанию — в двух)
    python scripts/run_all.py --quick              # быстрая проверка, что окружение работает (1 затравка, мало эпох)

Время на двух ядрах CPU: пример 1 ≈ 2 мин, пример 2 ≈ 45 мин, пример 3 ≈ 15 мин.
Каждый расчётный процесс получает OMP_NUM_THREADS=1 (как при исходном расчёте; см. --threads).
Расчётные скрипты в src/ не изменены по сравнению с disser-text/cases/code (MD5 — в README.md).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from importlib import metadata
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

PKGS = ["numpy", "scipy", "pandas", "scikit-learn", "torch", "torch_geometric", "matplotlib"]
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
ENV = os.environ.copy()   # окружение дочерних процессов; число потоков задаётся в main() (--threads)

# Ожидаемое время шагов, с (README, раздел «Время счёта», 2 ядра CPU) — только для масштаба индикатора
# прогресса. Сам расчёт идёт как раньше; если шаг длится дольше оценки, бар останавливается на 98%
# заполнения своей доли и не выходит за неё, пока шаг не завершится по-настоящему.
FULL_ESTIMATE = {"brain": 110, "aviation_gnn": 1440, "aviation_loc": 600, "aviation_final": 60,
                 "aviation_serial": 2700, "finance": 900, "figures": 20, "compare": 5}
QUICK_ESTIMATE = {"brain": 60, "aviation_serial": 300, "finance": 120}


def log(msg: str) -> None:
    """Печать, безопасная при активном progress-баре (не ломает его строку)."""
    tqdm.write(msg) if tqdm is not None else print(msg)


def estimate_total(steps: list[str], quick: bool, jobs: int) -> float:
    e = QUICK_ESTIMATE if quick else FULL_ESTIMATE
    total = 0.0
    for s in steps:
        if s == "brain":
            total += e["brain"]
        elif s == "aviation":
            total += e["aviation_serial"] if (quick or jobs < 2) else e["aviation_gnn"] + e["aviation_loc"] + e["aviation_final"]
        elif s == "finance":
            total += e["finance"]
        elif s == "figures":
            total += 0 if quick else e["figures"]
        elif s == "compare":
            total += 0 if quick else e["compare"]
    return total


class ProgressBar:
    """Индикатор прогресса полного расчёта. Доля внутри каждого шага — по прошедшему времени
    относительно оценки из FULL_ESTIMATE/QUICK_ESTIMATE, а не по факту (шаги — дочерние процессы,
    заглянуть внутрь нельзя). На границе шага бар всегда доводится ровно до конца отведённой ему доли."""

    def __init__(self, total: float):
        self.reported = 0.0
        self.total = max(total, 1e-9)
        self.bar = tqdm(total=self.total, unit="с", dynamic_ncols=True,
                        bar_format="{l_bar}{bar}| ~{n_fmt}/{total_fmt}с [{elapsed}<{remaining}]") if tqdm is not None else None
        self._t0 = time.time()

    def _advance_to(self, value: float) -> None:
        delta = min(value, self.total) - self.reported
        if delta <= 0:
            return
        self.reported += delta
        if self.bar is not None:
            self.bar.update(delta)
        else:
            frac = self.reported / self.total
            width = 30; filled = int(width * frac)
            sys.stdout.write("\r[" + "#" * filled + "-" * (width - filled) + f"] {frac * 100:5.1f}%  {time.time() - self._t0:5.0f} с")
            sys.stdout.flush()

    @contextmanager
    def step(self, desc: str, estimate: float):
        if self.bar is not None:
            self.bar.set_description(desc, refresh=True)
        else:
            print(f"\n{desc}")
        start = self.reported
        t0 = time.time()
        stop = threading.Event()

        def tick():
            while not stop.wait(0.3):
                frac = min((time.time() - t0) / estimate, 0.98) if estimate > 0 else 0.98
                self._advance_to(start + frac * estimate)

        th = threading.Thread(target=tick, daemon=True); th.start()
        try:
            yield
        finally:
            stop.set(); th.join()
            self._advance_to(start + estimate)  # шаг завершён — доля закрывается независимо от факта

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
        elif self.total > 0:
            print()


def run(args: list[str], logfile: Path) -> None:
    """Запуск скрипта из src/ (рабочая папка — src, как при исходном расчёте) с записью вывода в журнал."""
    log("  $ python " + " ".join(args))
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"\n$ python {' '.join(args)}   [{datetime.now():%Y-%m-%d %H:%M:%S}]\n")
        f.flush()
        subprocess.run([sys.executable, *args], cwd=P.SRC, stdout=f, stderr=subprocess.STDOUT, check=True, env=ENV)


def run_parallel(arg_lists: list[list[str]], logs: list[Path]) -> None:
    """Два процесса ГНС примера 2 одновременно (как при исходном расчёте)."""
    procs = []
    for args, logfile in zip(arg_lists, logs):
        log("  $ python " + " ".join(args) + "   (параллельно)")
        logfile.parent.mkdir(parents=True, exist_ok=True)
        f = open(logfile, "a", encoding="utf-8")
        procs.append((subprocess.Popen([sys.executable, *args], cwd=P.SRC, stdout=f, stderr=subprocess.STDOUT, env=ENV), f))
    bad = 0
    for p, f in procs:
        bad += p.wait() != 0
        f.close()
    if bad:
        raise SystemExit(f"{bad} параллельных процесса завершились с ошибкой — см. журналы в {logs[0].parent}")


def environment() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()}
    for p in PKGS:
        try:
            env[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            env[p] = None
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["brain", "aviation", "finance", "figures", "compare"], action="append",
                    help="выполнить только указанные шаги (можно несколько раз)")
    ap.add_argument("--jobs", type=int, default=2, help="процессов для ГНС примера 2 (1 или 2)")
    ap.add_argument("--quick", action="store_true", help="быстрая проверка окружения в results_quick/ (числа не сравниваются)")
    ap.add_argument("--threads", type=int, default=1,
                    help="потоков PyTorch/OpenMP на процесс (по умолчанию 1, как при исходном расчёте: так числа ГНС "
                         "совпадают с исследованием побитово; при 2 потоках они расходятся в 3–4-м знаке, а два параллельных "
                         "процесса по 2 потока на 2 ядрах замедляют обучение ГНС примерно в 15 раз)")
    a = ap.parse_args()
    for var in THREAD_VARS:
        ENV[var] = str(a.threads)
    steps = a.only or ["brain", "aviation", "finance", "figures", "compare"]
    out_root = P.ROOT / ("results_quick" if a.quick else "results")
    out_root.mkdir(exist_ok=True)
    logs = out_root / "logs"
    q = {"brain": ["--seeds", "0", "--epochs", "5"], "aviation": ["--seeds", "0", "--epochs", "1"],
         "finance": ["--seeds", "0", "--epochs", "5"]} if a.quick else {"brain": [], "aviation": [], "finance": []}

    # проверка данных (без скачивания)
    need = {"brain": "seeg", "aviation": "cmapss", "finance": "sp500"}
    for s in steps:
        if s in need:
            r = subprocess.run([sys.executable, str(P.ROOT / "scripts" / "fetch_data.py"), "--only", need[s], "--check"])
            if r.returncode:
                raise SystemExit("Нет данных. Запустите: python scripts/fetch_data.py  (см. README.md, раздел «Данные»)")

    info_path = out_root / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    info["environment"] = environment() | {"threads_per_process": a.threads}
    info.setdefault("steps", {})

    e = QUICK_ESTIMATE if a.quick else FULL_ESTIMATE
    pb = ProgressBar(estimate_total(steps, a.quick, a.jobs))

    for s in steps:
        t0 = time.time()
        log(f"\n=== {s} ===")
        if s == "brain":
            with pb.step("пример 1: мозг (§ 7.2)", e["brain"]):
                run(["brain_seeg_pipeline.py", "--data", str(P.find_seeg_derived()), "--out", str(out_root / "brain"), *q["brain"]],
                    logs / "brain.log")
        elif s == "aviation":
            base = ["aviation_cmapss_pipeline.py", "--data", str(P.cmapss_dir()), "--out", str(out_root / "aviation")]
            if a.quick or a.jobs < 2:
                with pb.step("пример 2: двигатель (§ 7.3)", e["aviation_serial"]):
                    run([*base, *q["aviation"]], logs / "aviation.log")
            else:
                # как при исходном расчёте: ГНС затравки 0 и 1–2 в двух процессах → локализация → сборка results.json
                with pb.step("пример 2: двигатель — ГНС (2 процесса)", e["aviation_gnn"]):
                    run_parallel([[*base, "--stage", "gnn", "--seeds", "0"], [*base, "--stage", "gnn", "--seeds", "1", "2"]],
                                 [logs / "aviation_gnn_seed0.log", logs / "aviation_gnn_seed1-2.log"])
                with pb.step("пример 2: двигатель — локализация", e["aviation_loc"]):
                    run([*base, "--stage", "loc"], logs / "aviation_loc.log")
                with pb.step("пример 2: двигатель — сборка результатов", e["aviation_final"]):
                    run([*base, "--stage", "final"], logs / "aviation_final.log")
        elif s == "finance":
            with pb.step("пример 3: рынок (§ 7.4)", e["finance"]):
                run(["finance_sp500_pipeline.py", "--data", str(P.sp500_dir()), "--out", str(out_root / "finance"), *q["finance"]],
                    logs / "finance.log")
        elif s == "figures":
            if a.quick:
                log("  пропущено в режиме --quick (рисунки строятся по results/)")
                continue
            with pb.step("рисунки 57–62 и сводные рисунки методологии", e["figures"]):
                run(["make_figures.py"], logs / "figures.log")
                run(["make_generality_figures.py"], logs / "generality_figures.log")
        elif s == "compare":
            if a.quick:
                log("  пропущено в режиме --quick")
                continue
            with pb.step("сверка с эталоном", e["compare"]):
                r = subprocess.run([sys.executable, str(P.ROOT / "scripts" / "compare_results.py")])
            info["steps"]["compare_exit_code"] = r.returncode
        info["steps"][s] = {"finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "seconds": round(time.time() - t0, 1)}
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"  готово за {time.time() - t0:.0f} с")
    pb.close()
    print(f"\nРезультаты: {out_root}\nЖурналы: {logs}\nОкружение и время: {info_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

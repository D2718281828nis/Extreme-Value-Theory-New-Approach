"""Пути проекта и к данным: config/paths.json + переменные окружения (DISSER_*) + файл .env в корне проекта."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # корень проекта dissertation-calculations
SRC = ROOT / "src"                                     # расчётный код главы 7 (без изменений)
RESULTS = ROOT / "results"                             # сюда пишут расчёты; make_figures.py читает именно ../results
REFERENCE = ROOT / "reference_results"                 # результаты, по которым написан текст диссертации

# Контрольные суммы входных данных (те же, что проверяют расчётные скрипты)
MD5 = {
    "train_FD001.txt": "259f340bac32ce6fa8894815600fa757",
    "train_FD003.txt": "81298e81977a53aa268ad500ac42b2d7",
    "all_stocks_5yr.csv": "6d2f3f2529cf6d8b443c8f4beee5638b",
    "constituents_financials_2018-02-08.csv": "8e571d9a5791c6355cd5ef8455d56920",
    "sEEG-HFOs-8.edf": "c83e85316860502327f6b7ed7ccb15d2",
    # производный набор, построенный на компьютере автора 23.09.2026; при повторном извлечении на другой
    # версии NumPy MD5 может отличаться — расчёт проверяет не его, а записанный внутрь MD5 исходного EDF
    "sEEG-HFOs-8_cases_derived.npz": "6e768cf1cc3a664ac287fc836d19ed62",
}


def _load_dotenv() -> None:
    """Минимальный разбор .env (KEY=VALUE) без внешних зависимостей; уже заданные переменные не перезаписываются."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _resolve(p: str | Path) -> Path:
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else (ROOT / p).resolve()


def config() -> dict:
    _load_dotenv()
    cfg = json.loads((ROOT / "config" / "paths.json").read_text(encoding="utf-8"))
    derived = cfg["seeg_derived"]
    if os.environ.get("DISSER_SEEG_DERIVED"):
        derived = [os.environ["DISSER_SEEG_DERIVED"]] + list(derived)
    return {
        "data_dir": _resolve(os.environ.get("DISSER_DATA_DIR", cfg["data_dir"])),
        "seeg_edf": _resolve(os.environ.get("DISSER_SEEG_EDF", cfg["seeg_edf"])),
        "seeg_derived": [_resolve(x) for x in derived],
        "biomedai_repo": _resolve(os.environ.get("DISSER_BIOMEDAI_REPO", cfg["biomedai_repo"])),
    }


def cmapss_dir() -> Path:
    return config()["data_dir"] / "CMAPSSData"


def sp500_dir() -> Path:
    return config()["data_dir"]


def seeg_derived_target() -> Path:
    """Куда пишется производный набор, если его приходится строить заново."""
    return config()["data_dir"] / "sEEG" / "sEEG-HFOs-8_cases_derived.npz"


def find_seeg_derived() -> Path | None:
    for p in config()["seeg_derived"]:
        if p.exists():
            return p
    return None


def md5sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

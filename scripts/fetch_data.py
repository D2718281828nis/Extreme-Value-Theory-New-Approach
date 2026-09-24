"""Подготовка данных для трёх примеров главы 7 с проверкой контрольных сумм MD5.

    python scripts/fetch_data.py                 # всё: C-MAPSS, S&P 500, стерео-ЭЭГ
    python scripts/fetch_data.py --only cmapss   # только авиационный двигатель
    python scripts/fetch_data.py --only sp500    # только фондовый рынок
    python scripts/fetch_data.py --only seeg     # только стерео-ЭЭГ (производный набор)
    python scripts/fetch_data.py --only seeg --download-edf   # скачать EDF 743 МБ с Zenodo, если его нет локально
    python scripts/fetch_data.py --check         # ничего не скачивать, только проверить наличие и MD5

Откуда берутся данные
---------------------
* C-MAPSS (NASA). Первоисточник — Zenodo, DOI 10.5281/zenodo.15346912 (CMAPSSData.zip); Kaggle behrad3d/nasa-cmaps.
  Скачиваются только train_FD001.txt и train_FD003.txt с побайтных копий в трёх независимых GitHub-репозиториях,
  закреплённых на конкретных коммитах; если первое зеркало недоступно, берётся следующее.
* S&P 500. Первоисточник — Kaggle camnugent/sandp500 (all_stocks_5yr.csv), три побайтных зеркала на GitHub.
  Секторы GICS — репозиторий datasets/s-and-p-500-companies-financials, коммит c9f83a9 от 08.02.2018.
* Стерео-ЭЭГ. Первоисточник — Zenodo, DOI 10.5281/zenodo.21967993 (sEEG-HFOs-8.edf, CC BY 4.0).
  Расчёт идёт по компактному производному набору (.npz). Если он уже есть (config/paths.json), ничего не делается;
  иначе он строится из локального EDF скриптом src/brain_extract_edf.py; EDF можно скачать флагом --download-edf.
Kaggle требует учётной записи, поэтому по умолчанию используются открытые зеркала; файл, скачанный с Kaggle вручную,
тоже подойдёт — положите его в папку data/ (или data/CMAPSSData/), скрипт проверит MD5.
"""
from __future__ import annotations

import argparse
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

RAW = "https://raw.githubusercontent.com"
MIRRORS = {
    "train_FD001.txt": [
        f"{RAW}/LahiruJayasinghe/RUL-Net/165933e9b370cb64703132b65e52ae4a4db7b29e/CMAPSSData/train_FD001.txt",
        f"{RAW}/hankroark/Turbofan-Engine-Degradation/ffde9492639af9c02ef6ab6607f8029dc8302591/CMAPSSData/train_FD001.txt",
        f"{RAW}/AndreaPi/Analysis-of-NASA-Turbofan-Degradation-Data/ed6ad87adcf94a302f9a12fd9282c98e0ecb5556/data/train_FD001.txt",
    ],
    "train_FD003.txt": [
        f"{RAW}/LahiruJayasinghe/RUL-Net/165933e9b370cb64703132b65e52ae4a4db7b29e/CMAPSSData/train_FD003.txt",
        f"{RAW}/hankroark/Turbofan-Engine-Degradation/ffde9492639af9c02ef6ab6607f8029dc8302591/CMAPSSData/train_FD003.txt",
        f"{RAW}/AndreaPi/Analysis-of-NASA-Turbofan-Degradation-Data/ed6ad87adcf94a302f9a12fd9282c98e0ecb5556/data/train_FD003.txt",
    ],
    "all_stocks_5yr.csv": [
        f"{RAW}/plotly/datasets/0c447c47b757ad74edecab31f0d72f849d2e67c2/all_stocks_5yr.csv",
        f"{RAW}/rishabhzn200/Stock-Data-Visualization-Using-Bokeh/66665f3c5920a9c6051225aa88958b4ad188ccba/all_stocks_5yr.csv",
        f"{RAW}/MainakRepositor/Datasets/f20fd12b065e2aa8d4ee436e984d275655b2de50/all_stocks_5yr.csv",
    ],
    "constituents_financials_2018-02-08.csv": [
        f"{RAW}/datasets/s-and-p-500-companies-financials/c9f83a9cdf04f59b060538b8d510c108d565781e/data/constituents-financials.csv",
    ],
}
# Прямая ссылка на файл записи Zenodo по стандартной схеме records/<id>/files/<имя>.
# Если Zenodo изменит адрес, скачайте файл вручную со страницы https://doi.org/10.5281/zenodo.21967993
ZENODO_EDF = "https://zenodo.org/records/21967993/files/sEEG-HFOs-8.edf?download=1"


def _ssl_context():
    try:
        import certifi  # на macOS python.org без certifi часто не видит корневые сертификаты
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "dissertation-calculations/1.0"})
    with urllib.request.urlopen(req, context=_ssl_context(), timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total > 50 << 20:
                print(f"\r    {done >> 20} / {total >> 20} МБ", end="", flush=True)
        if total > 50 << 20:
            print()
    tmp.replace(dest)


def ensure(name: str, dest_dir: Path, check_only: bool) -> bool:
    dest = dest_dir / name
    if dest.exists():
        got = P.md5sum(dest)
        if got == P.MD5[name]:
            print(f"  [ok]  {dest}  (MD5 совпадает)")
            return True
        print(f"  [!!]  {dest}: MD5 {got} ≠ {P.MD5[name]} — файл будет скачан заново")
        if check_only:
            return False
    elif check_only:
        print(f"  [нет] {dest}")
        return False
    dest_dir.mkdir(parents=True, exist_ok=True)
    for url in MIRRORS[name]:
        print(f"  скачиваю {name} ← {url.split('/')[3]}/{url.split('/')[4]}")
        try:
            download(url, dest)
        except Exception as e:  # сеть, 404, SSL
            print(f"    не удалось: {e}")
            continue
        if P.md5sum(dest) == P.MD5[name]:
            print(f"  [ok]  {dest}")
            return True
        print("    MD5 не совпал, пробую следующее зеркало")
    print(f"  [!!]  {name}: ни одно зеркало не дало файл с правильным MD5")
    return False


def _edf_md5_inside(npz: Path) -> str | None:
    import numpy as np
    try:
        with np.load(npz) as d:
            return str(d["edf_md5"])
    except Exception:
        return None


def seeg(check_only: bool, download_edf: bool) -> bool:
    cfg = P.config()
    found = P.find_seeg_derived()
    if found is not None:
        inside = _edf_md5_inside(found)
        if inside == P.MD5["sEEG-HFOs-8.edf"]:
            same = P.md5sum(found) == P.MD5["sEEG-HFOs-8_cases_derived.npz"]
            print(f"  [ok]  {found}\n        построен из EDF с правильным MD5; "
                  + ("файл побайтно совпадает с набором автора" if same else "побайтно отличается от набора автора (другая версия NumPy?) — расчёт допустим"))
            return True
        print(f"  [!!]  {found}: внутри записан MD5 исходного EDF {inside} — это не тот файл")
    if check_only:
        print("  [нет] производный набор стерео-ЭЭГ не найден (см. config/paths.json → seeg_derived)")
        return False
    edf = cfg["seeg_edf"]
    if not edf.exists():
        if not download_edf:
            print(f"  [нет] EDF не найден: {edf}\n"
                  "        Варианты: (1) укажите путь в config/paths.json → seeg_edf или в .env (DISSER_SEEG_EDF);\n"
                  "                  (2) запустите с флагом --download-edf (743 МБ с Zenodo);\n"
                  "                  (3) скачайте вручную: https://doi.org/10.5281/zenodo.21967993")
            return False
        edf = cfg["data_dir"] / "sEEG" / "sEEG-HFOs-8.edf"
        edf.parent.mkdir(parents=True, exist_ok=True)
        print(f"  скачиваю EDF с Zenodo (743 МБ) → {edf}")
        download(ZENODO_EDF, edf)
    print(f"  проверяю MD5 EDF {edf} …")
    got = P.md5sum(edf)
    if got != P.MD5["sEEG-HFOs-8.edf"]:
        print(f"  [!!]  MD5 EDF {got} ≠ {P.MD5['sEEG-HFOs-8.edf']} (значение со страницы Zenodo)")
        return False
    out = P.seeg_derived_target()
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"  строю производный набор → {out} (1–3 мин)")
    subprocess.run([sys.executable, str(P.SRC / "brain_extract_edf.py"), str(edf), str(out)], check=True)
    return _edf_md5_inside(out) == P.MD5["sEEG-HFOs-8.edf"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["cmapss", "sp500", "seeg", "all"], default="all")
    ap.add_argument("--check", action="store_true", help="только проверить, ничего не скачивать")
    ap.add_argument("--download-edf", action="store_true", help="скачать sEEG-HFOs-8.edf (743 МБ) с Zenodo, если его нет")
    a = ap.parse_args()
    ok = True
    if a.only in ("cmapss", "all"):
        print("Пример 2 — авиационный двигатель (NASA C-MAPSS)")
        for n in ("train_FD001.txt", "train_FD003.txt"):
            ok &= ensure(n, P.cmapss_dir(), a.check)
    if a.only in ("sp500", "all"):
        print("Пример 3 — фондовый рынок (S&P 500)")
        for n in ("all_stocks_5yr.csv", "constituents_financials_2018-02-08.csv"):
            ok &= ensure(n, P.sp500_dir(), a.check)
    if a.only in ("seeg", "all"):
        print("Пример 1 — головной мозг (стерео-ЭЭГ)")
        ok &= seeg(a.check, a.download_edf)
    print("\nГОТОВО: все данные на месте." if ok else "\nЕСТЬ ПРОБЛЕМЫ — см. сообщения выше.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

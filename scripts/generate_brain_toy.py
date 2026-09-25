"""Создать CSV-описание или развернуть синтетический набор для brain_seeg_pipeline.py."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "sEEG" / "brain_toy.csv"
SEED = 20260924
CONTACTS = [
    ("EEG R1", 1, 10360, 28, 260), ("EEG R2", 0, 0, 0, 0),
    ("EEG FD1", 1, 10366, 29, 248), ("EEG FD2", 1, 10372, 30, 236),
    ("EEG PA'5", 1, 10380, 31, 224), ("EEG PA'6", 0, 0, 0, 0),
    ("EEG PM3", 1, 10388, 32, 212), ("EEG PM4", 0, 0, 0, 0),
    ("EEG CC8", 1, 10397, 33, 200), ("EEG CC9", 0, 0, 0, 0),
    ("EEG A'1", 1, 10405, 34, 188), ("EEG A'2", 0, 0, 0, 0),
    ("EEG B1", 1, 10414, 35, 176), ("EEG B2", 0, 0, 0, 0),
    ("EEG C'1", 0, 0, 0, 0), ("EEG C'2", 0, 0, 0, 0),
    ("EEG D1", 0, 0, 0, 0), ("EEG D2", 0, 0, 0, 0),
    ("EEG E'1", 0, 0, 0, 0), ("EEG E'2", 0, 0, 0, 0),
]
FIELDS = ["name", "involved", "event_start_s", "frequency_hz", "amplitude_uv", "seed"]


def write_csv(out: Path) -> None:
    """Write the small, reviewable source-of-truth dataset."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for name, involved, start, frequency, amplitude in CONTACTS:
            writer.writerow(dict(name=name, involved=involved, event_start_s=start,
                                 frequency_hz=frequency, amplitude_uv=amplitude, seed=SEED))
    print(f"Создан {out} ({len(CONTACTS)} контактов)")


def build_dataset(path: Path) -> dict[str, np.ndarray]:
    """Deterministically expand the CSV parameters into pipeline-compatible arrays."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or set(rows[0]) != set(FIELDS):
        raise ValueError(f"Некорректная схема toy CSV; ожидаются поля: {', '.join(FIELDS)}")
    seeds = {int(row["seed"]) for row in rows}
    if len(seeds) != 1:
        raise ValueError("Все строки toy CSV должны иметь одинаковую затравку")
    rng = np.random.default_rng(seeds.pop())
    time_s, channels, fs = 10_550, len(rows), 256
    freqs = np.array([1, 3, 6, 10, 16, 24, 36, 48, 60, 67], dtype=np.float32)
    common = rng.normal(0, 0.055, (time_s, 1, 1)).astype(np.float32)
    logspec = rng.normal(-0.25, 0.12, (time_s, channels, len(freqs))).astype(np.float32)
    logspec += common
    logspec += np.linspace(0.15, -0.35, len(freqs), dtype=np.float32)[None, None, :]
    high = freqs >= 13
    for channel, row in enumerate(rows):
        if int(row["involved"]):
            rank = sum(int(prior["involved"]) for prior in rows[:channel])
            logspec[int(row["event_start_s"]):10452, channel, high] += 1.05 - 0.035 * rank
    logspec[10397:10430, :, high] += 0.65
    broadband = np.log10(np.maximum(np.power(10.0, logspec).sum(axis=2), 1e-8))
    linelen = np.exp(broadband).astype(np.float32)
    std_detr = (0.55 * linelen + rng.uniform(0.05, 0.12, broadband.shape)).astype(np.float32)
    ac1 = np.clip(0.65 + rng.normal(0, 0.035, broadband.shape), -0.99, 0.99).astype(np.float32)
    event_start, event_stop = 10000, 10550
    t = np.arange((event_stop - event_start) * fs, dtype=np.float32) / fs + event_start
    raw = rng.normal(0, 45, (channels, len(t))).astype(np.float32)
    for channel, row in enumerate(rows):
        if int(row["involved"]):
            active = (t >= int(row["event_start_s"])) & (t < 10452)
            raw[channel, active] += float(row["amplitude_uv"]) * np.sin(
                2 * np.pi * float(row["frequency_hz"]) * t[active]
            )
    raw = np.clip(np.rint(raw), -32768, 32767).astype(np.int16)
    return dict(dataset_kind=np.array("synthetic_toy"), names=np.array([r["name"] for r in rows]),
                fs=np.array(fs), crop=np.array(time_s), event_raw_start=np.array(event_start),
                base=np.array([0, 3600]), freqs=freqs, logspec=logspec.astype(np.float16),
                linelen=linelen, std_detr=std_detr, ac1=ac1, raw_event=raw,
                gain=np.ones(channels, dtype=np.float32), offs=np.zeros(channels, dtype=np.float32),
                corr_base=np.corrcoef(broadband[:3600], rowvar=False).astype(np.float32),
                ann_onset=np.array([10396.445]), ann_text=np.array(["СИНТЕТИЧЕСКОЕ СОБЫТИЕ"]),
                edf_md5=np.array("synthetic-no-edf"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    write_csv(args.out.resolve())


if __name__ == "__main__":
    main()

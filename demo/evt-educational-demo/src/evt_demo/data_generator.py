"""Воспроизводимая генерация AR(1)-ряда с внедрёнными событиями."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def _validate(n_points: int, phi: float, sigma: float, n_events: int,
              magnitude: float, duration: int) -> None:
    if n_points < 100: raise ValueError("n_points должен быть не меньше 100")
    if not -1 < phi < 1: raise ValueError("Для стационарности требуется abs(phi) < 1")
    if sigma <= 0: raise ValueError("sigma должен быть положительным")
    if n_events < 0 or duration <= 0 or magnitude <= 0:
        raise ValueError("Число событий неотрицательно, длительность и амплитуда положительны")
    warmup = max(20, duration)
    if n_events * duration > n_points - warmup:
        raise ValueError("Недостаточно точек для непересекающихся событий")


def generate_time_series_with_extreme_events(
    n_points: int = 2_000, phi: float = 0.8, sigma: float = 1.0,
    n_extreme_events: int = 3, extreme_magnitude: float = 8.0,
    extreme_duration: int = 3, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Сгенерировать стационарный AR(1)-ряд и маску внедрённых событий."""
    _validate(n_points, phi, sigma, n_extreme_events, extreme_magnitude, extreme_duration)
    rng = np.random.default_rng(seed)
    stationary_std = sigma / np.sqrt(1 - phi**2)
    values = np.empty(n_points, dtype=float)
    values[0] = rng.normal(0, stationary_std)
    innovations = rng.normal(0, sigma, n_points - 1)
    for index in range(1, n_points):
        values[index] = phi * values[index - 1] + innovations[index - 1]
    mask = np.zeros(n_points, dtype=bool)
    if n_extreme_events:
        warmup = max(20, extreme_duration)
        candidates = np.arange(warmup, n_points - extreme_duration + 1)
        starts: list[int] = []
        while len(starts) < n_extreme_events:
            if not len(candidates): raise ValueError("Не удалось разместить события без пересечений")
            start = int(rng.choice(candidates))
            starts.append(start)
            candidates = candidates[np.abs(candidates - start) >= extreme_duration]
        shape = np.sin(np.linspace(0, np.pi, extreme_duration + 2)[1:-1])
        for start in sorted(starts):
            values[start:start + extreme_duration] += extreme_magnitude * stationary_std * shape
            mask[start:start + extreme_duration] = True
    return values, mask


def save_to_csv(time_series: np.ndarray, extreme_mask: np.ndarray, path: str | Path) -> None:
    """Сохранить ряд в CSV с колонками timestamp, value и is_extreme."""
    values, mask = np.asarray(time_series), np.asarray(extreme_mask)
    if values.ndim != 1 or mask.ndim != 1 or len(values) != len(mask):
        raise ValueError("time_series и extreme_mask должны быть одномерными и равной длины")
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": np.arange(len(values)), "value": values,
                  "is_extreme": mask.astype(np.int8)}).to_csv(target, index=False, float_format="%.17g")

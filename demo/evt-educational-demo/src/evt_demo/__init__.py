"""Учебные инструменты для анализа экстремальных значений."""
from .ar_model import AutoregressiveModel
from .data_generator import generate_time_series_with_extreme_events, save_to_csv
from .evt_analysis import EVTAnalyzer, GEVFit, GPDFit

__all__ = ["AutoregressiveModel", "EVTAnalyzer", "GEVFit", "GPDFit",
           "generate_time_series_with_extreme_events", "save_to_csv"]

"""Two-component EVT detection and graph-attention localization."""

from .ar_model import AutoregressiveModel
from .data_generator import (
    GraphTimeSeries,
    generate_graph_time_series,
    generate_time_series_with_extreme_events,
    save_to_csv,
)
from .detection import DetectionResult, EVTEventDetector
from .evt_analysis import EVTAnalyzer
from .graph_model import CrossValidationResult, GraphSample, cross_validate_gat
from .pipeline import PipelineResult, prepare_sample, run_pipeline

__all__ = [
    "GraphTimeSeries",
    "generate_graph_time_series",
    "generate_time_series_with_extreme_events",
    "save_to_csv",
    "EVTAnalyzer",
    "AutoregressiveModel",
    "DetectionResult",
    "EVTEventDetector",
    "GraphSample",
    "CrossValidationResult",
    "cross_validate_gat",
    "PipelineResult",
    "prepare_sample",
    "run_pipeline",
]

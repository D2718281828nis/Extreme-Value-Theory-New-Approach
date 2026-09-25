"""Two-component EVT detection and graph-attention localization."""

from .data_generator import GraphTimeSeries, generate_graph_time_series
from .detection import DetectionResult, EVTEventDetector
from .graph_model import CrossValidationResult, GraphSample, cross_validate_gat
from .pipeline import PipelineResult, prepare_sample, run_pipeline

__all__ = [
    "GraphTimeSeries",
    "generate_graph_time_series",
    "DetectionResult",
    "EVTEventDetector",
    "GraphSample",
    "CrossValidationResult",
    "cross_validate_gat",
    "PipelineResult",
    "prepare_sample",
    "run_pipeline",
]

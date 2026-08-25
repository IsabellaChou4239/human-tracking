"""ROS bag based RGB-D operator localisation pipeline."""

from .config import PipelineConfig
from .pipeline import OperatorPipeline

__all__ = ["PipelineConfig", "OperatorPipeline"]

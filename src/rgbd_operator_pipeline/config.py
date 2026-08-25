"""Configuration objects; paths are supplied at runtime, never hard-coded."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float = 385.273
    fy: float = 384.892
    cx: float = 318.535
    cy: float = 241.057
    depth_scale: float = 0.001


@dataclass(frozen=True)
class PipelineConfig:
    bag_path: Path
    detector_model: Path
    pose_model: Path
    output_csv: Path = Path("outputs/operator_3d_position.csv")
    display_scale: float = 1.0
    show_window: bool = True
    intrinsics: CameraIntrinsics = CameraIntrinsics()

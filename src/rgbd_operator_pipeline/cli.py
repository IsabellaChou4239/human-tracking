"""Command-line entry point for a complete processing run."""
from __future__ import annotations

import argparse
from pathlib import Path
from .bag_runner import run_bag
from .config import PipelineConfig
from .pipeline import OperatorPipeline
from .visualization import LiveVisualizer, StopDisplay
from .vision import PersonDetector, PoseEstimator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track an operator in a ROS RGB-D bag and export their 3-D centre.")
    parser.add_argument("bag", type=Path); parser.add_argument("--detector", type=Path, required=True); parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/operator_3d_position.csv"))
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--display", dest="show_window", action="store_true", help="show the live visualisation (default)")
    display.add_argument("--no-display", dest="show_window", action="store_false", help="run without a visualisation window")
    parser.set_defaults(show_window=True)
    parser.add_argument("--display-scale", type=float, default=1.0, help="live display scale; default: 1.0 (original resolution)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(args.bag, args.detector, args.pose, args.output, display_scale=args.display_scale, show_window=args.show_window)
    pipeline = OperatorPipeline(config, PersonDetector(str(args.detector)), PoseEstimator(str(args.pose)))
    visualizer = LiveVisualizer(config.display_scale) if config.show_window else None
    try:
        try: print(f"Processed {run_bag(args.bag, pipeline, visualizer.show if visualizer else None)} RGB frames; trajectory: {args.output}")
        except StopDisplay: print(f"Stopped at frame {pipeline.frame}; trajectory: {args.output}")
    finally:
        pipeline.close()
        if visualizer: visualizer.close()


if __name__ == "__main__": main()

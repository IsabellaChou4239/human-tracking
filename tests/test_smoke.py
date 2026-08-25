"""Smoke test: verifies the dependency-light orchestration and CSV output."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from rgbd_operator_pipeline.config import PipelineConfig
from rgbd_operator_pipeline.pipeline import OperatorPipeline
from rgbd_operator_pipeline.cli import build_parser


class FakeDetector:
    def detect(self, image):
        return [{"confidence": .99, "box": (100, 100, 220, 300), "center": (160, 200)}], {"inference": 1.0}, 1.0


class FakePoseEstimator:
    def estimate(self, image, box):
        points = [{"id": point_id, "x": 160, "y": 200, "confidence": .9} for point_id in (5, 6, 11, 12)]
        return points, {"inference": 1.0}, 1.0


class PipelineSmokeTest(unittest.TestCase):
    def test_cli_displays_by_default_and_can_be_disabled(self):
        args = build_parser().parse_args(["demo.bag", "--detector", "detector.pt", "--pose", "pose.pt"])
        self.assertTrue(args.show_window)
        self.assertEqual(args.display_scale, 1.0)
        args = build_parser().parse_args(["demo.bag", "--detector", "detector.pt", "--pose", "pose.pt", "--no-display"])
        self.assertFalse(args.show_window)

    def test_one_rgbd_frame_creates_a_3d_trajectory_row(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trajectory.csv"
            pipeline = OperatorPipeline(PipelineConfig(Path("demo.bag"), Path("detector.pt"), Path("pose.pt"), output, show_window=False), FakeDetector(), FakePoseEstimator())
            try:
                pipeline.accept_depth(np.full((480, 640), 1500, dtype=np.uint16))
                result = pipeline.accept_color(np.zeros((480, 640, 3), dtype=np.uint8), 12.5)
            finally: pipeline.close()
            self.assertIsNotNone(result.center)
            with output.open(newline="") as file: rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1); self.assertEqual(rows[0]["timestamp"], "12.5")


if __name__ == "__main__": unittest.main()

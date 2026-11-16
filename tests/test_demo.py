"""Executable demo and complete batch/streaming pipeline regression tests."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from one2n_inference.config import load_config
from one2n_inference.demo import run_demo
from _fixtures import forecast_filename

ROOT = Path(__file__).resolve().parents[1]


class DemoTests(unittest.TestCase):
    def test_both_schedules_and_off_interval_final_output(self):
        config = load_config(ROOT / "configs/demo.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = replace(
                config,
                paths=replace(config.paths, output=root),
                forecast=replace(config.forecast, num_timesteps=5, output_step_hours=6),
            )
            summary = json.loads(run_demo(configured).read_text(encoding="utf-8"))
            self.assertEqual(summary["result"], "PASS")
            self.assertEqual(summary["updates"], configured.forecast.num_timesteps - 1)
            self.assertEqual(summary["output_files_per_schedule"], 2)
            expected = [
                forecast_filename(configured.forecast.output_step_hours),
                forecast_filename(
                    (configured.forecast.num_timesteps - 1) * configured.forecast.input_step_hours
                ),
            ]
            for schedule in ("batch", "streaming"):
                self.assertEqual(
                    sorted(
                        path.name
                        for path in (root / schedule / config.forecast.start_time).glob(
                            "output_one2n_*.nc"
                        )
                    ),
                    expected,
                )
            self.assertTrue((root / "inputs/initial_pressure.nc").is_file())
            self.assertTrue((root / "artifacts/random_model.pt").is_file())

    def test_documented_root_demo_command_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [
                sys.executable,
                str(ROOT / "demo.py"),
                "--config",
                str(ROOT / "configs/demo.yaml"),
                "--output",
                str(root / "result"),
            ]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("PASS", result.stdout)
            self.assertEqual(
                json.loads((root / "result/summary.json").read_text())["result"], "PASS"
            )

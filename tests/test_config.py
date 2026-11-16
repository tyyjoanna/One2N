"""YAML validation, portable paths, and lightweight command-line checks."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from one2n_inference import __version__
from one2n_inference.config import config_from_dict, load_config, missing_inputs
from _fixtures import EPOCH, INPUT_STEP_HOURS, time_string

ROOT = Path(__file__).resolve().parents[1]


def demo_settings():
    return yaml.safe_load((ROOT / "configs/demo.yaml").read_text(encoding="utf-8"))


class ConfigTests(unittest.TestCase):
    def test_paths_resolve_from_yaml_and_base_dir(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_dir = root / "configs"
            config_dir.mkdir()
            source = config_dir / "run.yaml"
            raw = demo_settings()
            raw.pop("synthetic")
            names = list(raw["paths"])
            raw["paths"] = {name: f"input/{name}" for name in names if name != "base_dir"}
            raw["paths"]["base_dir"] = ".."
            source.write_text(yaml.safe_dump(raw), encoding="utf-8")
            config = load_config(source)
            self.assertEqual(config.paths.one2n_checkpoint, root / "input/one2n_checkpoint")
            self.assertEqual(config.output_dir, root / "input/output" / config.forecast.start_time)
            self.assertEqual(len(missing_inputs(config)), len(raw["paths"]) - 2)
            run = subprocess.run(
                [sys.executable, "-m", "one2n_inference", "check", "--config", str(source)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run.returncode, 1, run.stderr)
            result = json.loads(run.stdout)
            self.assertEqual(len(result["missing_inputs"]), len(missing_inputs(config)))
            self.assertEqual(result["configuration"]["paths"]["output"], str(config.paths.output))

    def test_forecast_geometry_and_runtime_validation(self):
        invalid = (
            ("forecast", "num_timesteps", 1),
            ("forecast", "num_timesteps", True),
            ("forecast", "input_step_hours", 1.5),
            ("forecast", "output_step_hours", 3),
            ("forecast", "start_time", EPOCH[:6] + "3200"),
            ("forecast", "start_time", int(EPOCH)),
            ("forecast", "schedule", "unknown"),
            ("runtime", "epsilon", 0),
            ("runtime", "epsilon", float("nan")),
            ("runtime", "coordinate_tolerance", -1),
            ("runtime", "jax_platform", "tpu"),
            ("model", "trained_step_hours", 1),
            ("model", "in_chans", 1),
            ("model", "num_heads", [3, 3, 3, 3]),
            ("model", "qk_scale", None),
            ("data", "pressure_levels_hpa", [100]),
            ("data", "fine_grid_shape", [33, 32]),
        )
        for section, name, value in invalid:
            with (
                self.subTest(section=section, name=name, value=value),
                self.assertRaises(ValueError),
            ):
                raw = demo_settings()
                raw[section][name] = value
                config_from_dict(raw, base_dir=ROOT / "configs")
        raw = demo_settings()
        raw["forecast"]["output_step_hours"] = 2 * raw["forecast"]["input_step_hours"]
        forecast = config_from_dict(raw, base_dir=ROOT / "configs").forecast
        final_step = forecast.num_timesteps - 1
        self.assertEqual(
            [step for step in range(1, final_step + 1) if forecast.should_save(step)],
            [2, final_step],
        )

    def test_no_implicit_experiment_defaults_or_unknown_keys(self):
        raw = demo_settings()
        raw["forecats"] = raw.pop("forecast")
        with self.assertRaisesRegex(ValueError, "Unknown configuration"):
            config_from_dict(raw)
        for section in ("forecast", "runtime", "model", "neuralgcm", "data"):
            raw = demo_settings()
            del raw[section]
            with self.subTest(section=section), self.assertRaisesRegex(ValueError, section):
                config_from_dict(raw)
        raw = demo_settings()
        del raw["neuralgcm"]["seed"]
        with self.assertRaisesRegex(ValueError, "seed"):
            config_from_dict(raw)
        raw = demo_settings()
        raw["model"]["embedding_typo"] = raw["model"].pop("embed_dim")
        with self.assertRaisesRegex(ValueError, "embedding_typo"):
            config_from_dict(raw)

    def test_serialized_worker_roundtrip_and_synthetic_inputs(self):
        config = load_config(ROOT / "configs/demo.yaml")
        restored = config_from_dict(json.loads(json.dumps(config.to_dict())))
        self.assertEqual(restored, config)
        self.assertEqual(missing_inputs(config), [])
        self.assertIsNone(config.model.to_kwargs()["qk_scale"])

    def test_real_profile_requires_explicit_initialization_time(self):
        source = ROOT / "configs/inference.yaml"
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        self.assertNotIn("start_time", raw["forecast"])
        with self.assertRaisesRegex(ValueError, "--start-time"):
            load_config(source)
        explicit_time = time_string(INPUT_STEP_HOURS)
        configured = load_config(source, start_time=explicit_time)
        self.assertEqual(configured.forecast.start_time, explicit_time)
        self.assertEqual(configured.output_dir.name, explicit_time)
        raw["forecast"]["start_time"] = EPOCH
        self.assertEqual(config_from_dict(raw).forecast.start_time, EPOCH)
        command = [
            sys.executable,
            "-m",
            "one2n_inference",
            "check",
            "--config",
            str(source),
            "--start-time",
            explicit_time,
        ]
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual(
            json.loads(run.stdout)["configuration"]["forecast"]["start_time"], explicit_time
        )

    def test_invalid_yaml_and_old_runtime_format_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "run.yaml"
            source.write_text("forecast: [broken", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid YAML"):
                load_config(source)
            source.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "YAML mapping"):
                load_config(source)
            old = source.with_suffix(".toml")
            old.write_text("[forecast]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "yaml"):
                load_config(old)

    def test_import_and_help_do_not_initialize_numerical_runtimes(self):
        code = (
            "import sys; import one2n_inference.cli; "
            "assert 'torch' not in sys.modules; assert 'jax' not in sys.modules; "
            "one2n_inference.cli.main(['--version'])"
        )
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn(__version__, run.stdout)

"""Full package pipeline with real tiny weights, random inputs, and worker IPC.

Only the upstream NeuralGCM model is a controlled test double. Data loading,
regridding, normalization, checkpoint loading, Swin inference, and writes are real.
"""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import xarray as xr

from one2n_inference.constants import VARIABLES
from one2n_inference.inference.pipeline import run_prediction
from one2n_inference.models.checkpoint import create_model

# unittest discovery adds tests/ to the module search path.
from test_neuralgcm import _make_fixture, _small_config
from _fixtures import EPOCH, YEAR


@unittest.skipUnless(importlib.util.find_spec("dinosaur"), "Install the neuralgcm extra")
class CompletePipelineTests(unittest.TestCase):
    def test_real_worker_and_swin_are_interleaved_and_match_batch_outputs(self):
        os.environ["JAX_PLATFORMS"] = "cpu"
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            units, event_file = _make_fixture(root)
            reference = _small_config(root, "batch")
            rng = np.random.default_rng(reference.neuralgcm.seed)
            # Randomize the physical initialization while retaining real source-file schemas.
            for filename in sorted((root / YEAR).glob(f"*{EPOCH}.nc")):
                with xr.open_dataset(filename, engine="netcdf4") as source:
                    randomized = source.load()
                for name in randomized.data_vars:
                    randomized[name].values += rng.normal(
                        scale=0.1, size=randomized[name].shape
                    ).astype(randomized[name].dtype)
                randomized.to_netcdf(filename, engine="netcdf4")
            paths = replace(
                reference.paths,
                one2n_checkpoint=root / "small_model.pt",
                stats_coarse=root / "coarse_stats.pt",
                stats_initial=root / "residual_stats.pt",
                stats_target=root / "residual_stats.pt",
            )
            reference = replace(reference, paths=paths)
            torch.manual_seed(reference.neuralgcm.seed)
            torch.save(
                {"model_state_dict": create_model(reference.model).state_dict()},
                paths.one2n_checkpoint,
            )
            shape = (len(VARIABLES), len(reference.data.pressure_levels_hpa))
            for filename in (paths.stats_coarse, paths.stats_initial):
                means = torch.tensor(rng.normal(size=shape), dtype=torch.float64)
                variances = torch.tensor(rng.uniform(0.5, 2, size=shape), dtype=torch.float64)
                torch.save({"means": means, "variances": variances}, filename)
            # Add only the upstream test-double directory. Installed-wheel imports are
            # retained when this suite runs against a built package outside src/.
            pythonpath = os.pathsep.join(
                filter(
                    None,
                    (
                        str(root / "upstream_test_double"),
                        os.environ.get("PYTHONPATH", ""),
                    ),
                )
            )
            products = {}
            with patch.dict(os.environ, {"PYTHONPATH": pythonpath}):
                products["batch"] = run_prediction(reference)
                expected = []
                for filename in products["batch"]:
                    with xr.open_dataset(filename, engine="netcdf4") as saved:
                        expected.append(saved.load())
                event_file.write_text("", encoding="utf-8")
                streamed = replace(
                    reference,
                    forecast=replace(reference.forecast, schedule="streaming"),
                    paths=replace(reference.paths, output=root / "streaming"),
                )
                with paths.neuralgcm_checkpoint.open("wb") as handle:
                    pickle.dump(
                        {
                            "events": str(event_file),
                            "required_output_dir": str(streamed.output_dir),
                            "start_time": streamed.forecast.start_time,
                        },
                        handle,
                    )

                def record_actual_correction(index):
                    events = [json.loads(line) for line in event_file.read_text().splitlines()]
                    self.assertEqual(events[-1]["kind"], "emit")
                    self.assertEqual(events[-1]["index"], index)
                    with event_file.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"kind": "correction", "index": index}) + "\n")

                products["streaming"] = run_prediction(streamed, on_update=record_actual_correction)
            self.assertEqual(
                [path.name for path in products["streaming"]],
                [path.name for path in products["batch"]],
            )
            self.assertEqual(len(expected), reference.forecast.num_timesteps - 1)
            for index, filename in enumerate(products["streaming"]):
                with xr.open_dataset(filename, engine="netcdf4") as actual:
                    xr.testing.assert_allclose(actual, expected[index], atol=0, rtol=0)
                    self.assertEqual(list(actual.data_vars), list(VARIABLES))
                    self.assertEqual(
                        [actual[name].attrs["units"] for name in VARIABLES], list(units)
                    )
                    self.assertGreaterEqual(float(actual.specific_humidity.min()), 0)
                    self.assertTrue(all(np.isfinite(actual[name]).all() for name in VARIABLES))
            events = [json.loads(line) for line in event_file.read_text().splitlines()]
            expected_order = [("encode", None), ("initial_decode", 0)]
            for index in range(1, reference.forecast.num_timesteps):
                expected_order.extend((("emit", index), ("correction", index)))
            self.assertEqual(
                [(event["kind"], event.get("index")) for event in events], expected_order
            )
            record = json.loads(
                (streamed.output_dir / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["forecast"]["schedule"], "streaming")


if __name__ == "__main__":
    unittest.main()

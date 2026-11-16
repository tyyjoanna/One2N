"""Actual pipeline with synthetic baseline inputs, a real small model, and real regridding."""

import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import xarray as xr

from types import SimpleNamespace
from one2n_inference.config import Paths, load_config
from one2n_inference.constants import VARIABLES
from one2n_inference.data.statistics import Statistics
from one2n_inference.inference.packing import TilePacker
from one2n_inference.inference import pipeline
from one2n_inference.models import create_model
from _fixtures import INPUT_STEP_HOURS, forecast_filename, timestamp


@unittest.skipUnless(importlib.util.find_spec("dinosaur"), "requires the neuralgcm extra")
class PipelineTests(unittest.TestCase):
    def test_real_recurrence_reconstruction_metadata_and_final_save(self):
        from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils
        from one2n_inference.neuralgcm.client import PreparedInputs

        torch.manual_seed(19)
        torch.set_num_threads(1)
        native = spherical_harmonic.Grid(
            latitude_nodes=16, longitude_nodes=16, latitude_spacing="gauss"
        )
        fine = spherical_harmonic.Grid(
            latitude_nodes=32, longitude_nodes=32, latitude_spacing="gauss"
        )
        times = timestamp() + np.arange(4) * np.timedelta64(INPUT_STEP_HOURS, "h")
        units = ("m s-1", "m s-1", "K", "kg kg-1", "m2 s-2")
        baseline = xr.Dataset(
            {
                name: xr.DataArray(
                    np.broadcast_to(
                        np.asarray(constant + np.arange(4) * 0.1, dtype=np.float32)[
                            :, None, None, None
                        ],
                        (4, 5, 16, 16),
                    ).copy(),
                    dims=("time", "level", "latitude", "longitude"),
                    attrs={"units": unit},
                )
                for name, constant, unit in zip(VARIABLES, (10, 20, 250, -100, 1000), units)
            },
            coords={
                "time": times,
                "level": [100, 300, 500, 700, 900],
                "latitude": np.rad2deg(native.latitudes)[::-1],
                "longitude": np.rad2deg(native.longitudes),
            },
        )
        baseline.level.attrs["units"] = "hPa"
        initial = {
            name: np.random.default_rng(i).normal(size=(1, 5, 32, 32)).astype(np.float32)
            for i, name in enumerate(VARIABLES)
        }
        coarse = {name: baseline[name].diff("time").values for name in VARIABLES}
        profile = load_config(Path(__file__).resolve().parents[1] / "configs/demo.yaml")
        model = create_model(profile.model).eval()
        reference_model = copy.deepcopy(model)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            paths = Paths(
                folder / "model.pt",
                folder / "neuralgcm.pkl",
                folder / "low.pt",
                folder / "initial.pt",
                folder / "target.pt",
                folder,
                folder,
                folder,
                folder / "results",
            )
            config = replace(
                profile,
                paths=paths,
                forecast=replace(profile.forecast, num_timesteps=4, output_step_hours=4),
                runtime=replace(profile.runtime, device="cpu"),
                synthetic=None,
            )
            torch.save({"model_state_dict": model.state_dict()}, paths.one2n_checkpoint)
            stats = []
            for index, filename in enumerate(
                (paths.stats_coarse, paths.stats_initial, paths.stats_target)
            ):
                mean, variance = torch.full((5, 5), index * 0.1, dtype=torch.float64), torch.full(
                    (5, 5), 1 + index * 0.5, dtype=torch.float64
                )
                torch.save({"means": mean, "variances": variance}, filename)
                stats.append(
                    Statistics(mean[None, :, :, None, None], variance[None, :, :, None, None])
                )
            state = stats[1].normalize(pipeline.stack_fields(initial), config.runtime.epsilon)
            increments = stats[0].normalize(pipeline.stack_fields(coarse), config.runtime.epsilon)
            packer = TilePacker(increments[:1].shape, state.shape)
            expected = {}
            with torch.inference_mode():
                for step in range(3):
                    state = packer.unpack(
                        reference_model(packer.model_input(increments[step : step + 1], state))
                    )
                    residual = stats[2].denormalize(state).numpy()
                    direct = xarray_utils.regrid(
                        baseline.isel(time=[step + 1]).sortby("latitude"),
                        horizontal_interpolation.ConservativeRegridder(native, fine, skipna=True),
                    )
                    direct = direct.transpose("time", "level", "latitude", "longitude").sortby(
                        "latitude", ascending=False
                    )
                    for index, name in enumerate(VARIABLES):
                        values = direct[name].values + residual[:, index]
                        if name == "specific_humidity":
                            values = np.maximum(values, 0)
                        direct[name] = direct[name].copy(data=values)
                    expected[step + 1] = direct

            def prepare(settings):
                target = (
                    settings.output_dir / f"output_neuralgcm_{settings.forecast.start_time}_4.nc"
                )
                baseline.to_netcdf(target, engine="netcdf4")
                return PreparedInputs(fine, coarse.copy(), initial.copy(), target)

            with (patch.object(pipeline, "load_model", wraps=pipeline.load_model) as loaded,):
                products = pipeline.run_prediction(
                    config, baseline_provider=SimpleNamespace(prepare_inputs=prepare)
                )
            loaded.assert_called_once()
            self.assertEqual(
                [p.name for p in products],
                [forecast_filename(step * INPUT_STEP_HOURS) for step in (2, 3)],
            )
            for step, filename in zip((2, 3), products):
                with xr.open_dataset(filename, engine="netcdf4") as saved:
                    xr.testing.assert_allclose(saved, expected[step], atol=0, rtol=0)
                    self.assertEqual(list(saved.data_vars), list(VARIABLES))
                    self.assertEqual(saved.level.attrs["units"], "hPa")
                    self.assertEqual(
                        [saved[name].attrs["units"] for name in VARIABLES], list(units)
                    )
                    self.assertEqual(np.count_nonzero(saved.specific_humidity), 0)
            record = json.loads((config.output_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(record["forecast"]["num_timesteps"], 4)
            self.assertEqual(record["paths"]["output"], str(paths.output))

"""Controlled upstream integration: real files, remapping, IPC, and continuous state."""

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
import xarray as xr

from one2n_inference.constants import VARIABLES
from one2n_inference.data.io import parse_timestamp
from one2n_inference.neuralgcm.client import BaselineSession, prepare_inputs
from one2n_inference.neuralgcm.rollout import run_model
from _fixtures import DAY, EPOCH, MONTH, YEAR

HAS_UPSTREAM = importlib.util.find_spec("dinosaur") and importlib.util.find_spec("jax")


def _make_fixture(root):
    """Create a small complete initialization and a controlled upstream test module."""
    from dinosaur import spherical_harmonic

    (root / YEAR).mkdir()
    (root / "upstream_test_double").mkdir()
    grid = spherical_harmonic.Grid(latitude_nodes=8, longitude_nodes=16, latitude_spacing="gauss")
    latitude = np.rad2deg(grid.latitudes)[::-1]
    longitude = np.rad2deg(grid.longitudes)
    coordinates = {
        "time": [parse_timestamp(EPOCH)],
        "level": [100.0, 500.0],
        "latitude": latitude,
        "longitude": longitude,
    }
    fields = (
        ("U", "128_131_u"),
        ("V", "128_132_v"),
        ("T", "128_130_t"),
        ("Q", "128_133_q"),
        ("Z", "128_129_z"),
    )
    units = ("m s-1", "m s-1", "K", "kg kg-1", "m2 s-2")
    for index, (name, token) in enumerate(fields):
        data = xr.Dataset(
            {
                name: (
                    ("time", "level", "latitude", "longitude"),
                    np.full((1, 2, 8, 16), 10 * index, dtype=np.float32),
                )
            },
            coords=coordinates,
        )
        data[name].attrs["units"] = units[index]
        data.level.attrs["units"] = "hPa"
        data.to_netcdf(root / YEAR / f"{token}_{EPOCH}.nc", engine="netcdf4")
    for name, token in (("CI", "128_031_ci"), ("SSTK", "128_034_sstk")):
        xr.Dataset(
            {name: (("time", "latitude", "longitude"), np.ones((1, 8, 16)))},
            coords={key: value for key, value in coordinates.items() if key != "level"},
        ).to_netcdf(root / YEAR / f"{token}_{DAY}.nc", engine="netcdf4")
    xr.Dataset(
        {
            name: (
                ("valid_time", "pressure_level", "latitude", "longitude"),
                np.zeros((1, 2, 8, 16), dtype=np.float32),
            )
            for name in ("ciwc", "clwc")
        },
        coords={
            "valid_time": coordinates["time"],
            "pressure_level": coordinates["level"],
            "latitude": latitude,
            "longitude": longitude,
        },
    ).to_netcdf(root / f"{MONTH}.nc", engine="netcdf4")
    event_path = root / "events.jsonl"
    with (root / "model.pkl").open("wb") as handle:
        pickle.dump({"events": str(event_path)}, handle)
    (root / "upstream_test_double" / "neuralgcm.py").write_text(
        CONTROLLED_NEURALGCM, encoding="utf-8"
    )
    return units, event_path


def _small_config(root, schedule):
    from one2n_inference.config import config_from_dict, load_config

    reference = load_config(
        Path(__file__).resolve().parents[1] / "configs/inference.yaml", start_time=EPOCH
    )
    raw = reference.to_dict()
    raw["paths"].update(
        {
            "neuralgcm_checkpoint": str(root / "model.pkl"),
            "hres_pressure": str(root),
            "hres_surface": str(root),
            "era5_cloud": str(root),
            "output": str(root / schedule),
        }
    )
    raw["forecast"].update(
        {
            "start_time": EPOCH,
            "num_timesteps": 5,
            "input_step_hours": 2,
            "output_step_hours": 2,
            "schedule": schedule,
        }
    )
    raw["model"].update(
        {
            "img_size": [2, 8, 16],
            "patch_size": [1, 2, 2],
            "in_chans": 10,
            "embed_dim": 8,
            "depths": [2, 2, 2, 2],
            "num_heads": [2, 2, 2, 2],
            "window_size": [2, 2, 2],
        }
    )
    raw["data"].update(
        {
            "pressure_levels_hpa": [100.0, 500.0],
            "fine_grid_shape": [8, 16],
            "latitude_spacing": "gauss",
        }
    )
    raw["neuralgcm"].update({"seed": 17, "rollout_window": 2})
    return config_from_dict(raw, base_dir=root)


@unittest.skipUnless(HAS_UPSTREAM, "Install the neuralgcm extra")
class NeuralGCMTests(unittest.TestCase):
    def test_windowed_state_continuity_seed_and_persisted_forcing(self):
        os.environ["JAX_PLATFORMS"] = "cpu"
        import jax

        calls = []

        class ControlledModel:
            def inputs_from_xarray(self, data):
                return "inputs"

            def forcings_from_xarray(self, data):
                return "persisted"

            def encode(self, inputs, forcing, key):
                np.testing.assert_array_equal(jax.random.key_data(key), [0, 7])
                assert inputs == "inputs" and forcing == "persisted"
                return 0

            def unroll(self, state, forcing, steps, timedelta, start_with_input):
                assert forcing == "persisted" and timedelta == np.timedelta64(2, "h")
                assert start_with_input
                calls.append(steps)
                return state + steps, {"x": np.arange(state, state + steps)}

            def data_to_xarray(self, values, times):
                return xr.Dataset({"x": ("time", values["x"])}, coords={"time": times})

        initial = xr.Dataset(coords={"time": [parse_timestamp(EPOCH)]})
        result = run_model(ControlledModel(), initial, 2, 61, window=13, seed=7)
        np.testing.assert_array_equal(result.x, np.arange(61))
        np.testing.assert_array_equal(result.time, 2 * np.arange(61))
        self.assertEqual(calls, [13, 13, 13, 13, 9])

    def test_batch_and_streaming_match_and_interleave_without_reencoding(self):
        os.environ["JAX_PLATFORMS"] = "cpu"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            units, events = _make_fixture(root)
            batch = _small_config(root, "batch")
            source_root = Path(__file__).resolve().parents[1] / "src"
            pythonpath = os.pathsep.join((str(root / "upstream_test_double"), str(source_root)))
            with patch.dict(os.environ, {"PYTHONPATH": pythonpath}):
                prepared = prepare_inputs(batch)
                batch_events = [json.loads(line) for line in events.read_text().splitlines()]
                self.assertEqual(sum(event["kind"] == "encode" for event in batch_events), 1)
                self.assertEqual(
                    [event["count"] for event in batch_events if event["kind"] == "emit"], [2, 2, 1]
                )
                with xr.open_dataset(prepared.baseline_path) as saved:
                    expected_baseline = saved.load()
                events.write_text("", encoding="utf-8")
                stream = _small_config(root, "streaming")
                actual_steps = []
                with BaselineSession(stream) as session:
                    self.assertEqual(session.grid.latitude_nodes, prepared.grid.latitude_nodes)
                    for name in VARIABLES:
                        np.testing.assert_array_equal(session.initial[name], prepared.initial[name])
                    np.testing.assert_array_equal(
                        session.initial_baseline.time.values, expected_baseline.time.values[:1]
                    )
                    for step in session.iter_steps():
                        # The worker is idle here: its next advance waits for the next iterator request.
                        current_events = [
                            json.loads(line) for line in events.read_text().splitlines()
                        ]
                        self.assertEqual(current_events[-1]["kind"], "emit")
                        self.assertEqual(current_events[-1]["index"], step.index)
                        with events.open("a", encoding="utf-8") as handle:
                            handle.write(
                                json.dumps({"kind": "correction", "index": step.index}) + "\n"
                            )
                        for name in VARIABLES:
                            np.testing.assert_array_equal(
                                step.increment[name],
                                prepared.coarse[name][step.index - 1 : step.index],
                            )
                        actual_steps.append(step.index)
                self.assertEqual(actual_steps, [1, 2, 3, 4])
                with xr.open_dataset(session.baseline_path) as actual:
                    xr.testing.assert_allclose(actual, expected_baseline, atol=0, rtol=0)
                    self.assertEqual(
                        actual.attrs["completed_states"], stream.forecast.num_timesteps
                    )
                    self.assertEqual(actual.level.attrs["units"], "hPa")
                    for name, unit in zip(VARIABLES, units):
                        self.assertEqual(actual[name].attrs["units"], unit)
                actual_events = [json.loads(line) for line in events.read_text().splitlines()]
                self.assertEqual(sum(event["kind"] == "encode" for event in actual_events), 1)
                self.assertEqual(
                    [event["index"] for event in actual_events if event["kind"] == "emit"],
                    [1, 2, 3, 4],
                )
                self.assertTrue(
                    all(
                        not event["start_with_input"]
                        for event in actual_events
                        if event["kind"] == "emit"
                    )
                )
                order = [(event["kind"], event.get("index")) for event in actual_events]
                self.assertEqual(
                    order,
                    [
                        ("encode", None),
                        ("initial_decode", 0),
                        ("emit", 1),
                        ("correction", 1),
                        ("emit", 2),
                        ("correction", 2),
                        ("emit", 3),
                        ("correction", 3),
                        ("emit", 4),
                        ("correction", 4),
                    ],
                )

    def test_early_close_releases_worker_and_marks_partial_trajectory(self):
        os.environ["JAX_PLATFORMS"] = "cpu"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _make_fixture(root)
            config = _small_config(root, "streaming")
            pythonpath = os.pathsep.join(
                (
                    str(root / "upstream_test_double"),
                    str(Path(__file__).resolve().parents[1] / "src"),
                )
            )
            with patch.dict(os.environ, {"PYTHONPATH": pythonpath}):
                session = BaselineSession(config)
                with session:
                    process = session._process
                    self.assertEqual(next(session.iter_steps()).index, 1)
                self.assertIsNotNone(process.poll())
            with xr.open_dataset(session.baseline_path) as saved:
                self.assertEqual(saved.attrs["completed_states"], 2)
                self.assertEqual(saved.attrs["expected_states"], config.forecast.num_timesteps)


CONTROLLED_NEURALGCM = """from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import xarray as xr
from dinosaur import spherical_harmonic

class PressureLevelModel:
    @classmethod
    def from_checkpoint(cls, checkpoint):
        self = cls()
        self.events = Path(checkpoint["events"])
        self.required_output_dir = checkpoint.get("required_output_dir")
        self.start_time = checkpoint.get("start_time")
        self.grid = spherical_harmonic.Grid(latitude_nodes=8, longitude_nodes=16, latitude_spacing="gauss")
        self.data_coords = SimpleNamespace(horizontal=self.grid)
        return self

    def event(self, **values):
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values) + "\\n")

    def inputs_from_xarray(self, data): return data
    def forcings_from_xarray(self, data): return data
    def encode(self, inputs, forcing, key):
        import jax
        assert list(np.asarray(jax.random.key_data(key))) == [0, 17]
        self.event(kind="encode")
        return 0

    def fields(self, indices):
        names = ("u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity", "geopotential")
        return {name: np.broadcast_to((np.asarray(indices, dtype=np.float32) + 5 + i)[:, None, None, None],
                                     (len(indices), 2, 16, 8)).copy() for i, name in enumerate(names)}

    def decode(self, state, forcing):
        self.event(kind="initial_decode", index=state)
        return {name: values[0] for name, values in self.fields([state]).items()}

    def unroll(self, state, forcing, steps, timedelta, start_with_input):
        assert timedelta == np.timedelta64(2, "h")
        first = state if start_with_input else state + 1
        if self.required_output_dir is not None and not start_with_input and state:
            from datetime import datetime, timedelta as TimeDelta
            previous_time = datetime.strptime(self.start_time, "%Y%m%d%H") + TimeDelta(hours=state*int(timedelta / np.timedelta64(1, "h")))
            previous_file = Path(self.required_output_dir) / f"output_one2n_{previous_time:%Y%m%d%H}.nc"
            assert previous_file.is_file(), f"Native advance preceded residual output: {previous_file}"
        self.event(kind="emit", index=first, count=steps, start_with_input=start_with_input)
        return state + steps, self.fields(np.arange(first, first + steps))

    def data_to_xarray(self, values, times):
        return xr.Dataset({name: (("time", "level", "longitude", "latitude"), value) for name, value in values.items()},
                          coords={"time": times, "level": [100.0, 500.0],
                                  "longitude": np.rad2deg(self.grid.longitudes), "latitude": np.rad2deg(self.grid.latitudes)})
"""


if __name__ == "__main__":
    unittest.main()

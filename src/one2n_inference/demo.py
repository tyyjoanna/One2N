"""Random-data integration example using the actual configured inference pipeline."""

from dataclasses import replace
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import xarray as xr

from .config import RunConfig
from .constants import VARIABLES
from .data.io import parse_timestamp
from .data.regridding import to_target_grid
from .inference.pipeline import run_prediction
from .models.checkpoint import create_model
from .neuralgcm.client import BaselineStep, PreparedInputs

VARIABLE_UNITS = {
    "u_component_of_wind": "m s-1",
    "v_component_of_wind": "m s-1",
    "temperature": "K",
    "specific_humidity": "kg kg-1",
    "geopotential": "m2 s-2",
}


class SyntheticBaselineProvider:
    """Controlled random native states; this is not a NeuralGCM forecast model."""

    def __init__(self, config: RunConfig):
        from dinosaur import spherical_harmonic

        self.events = []
        self.config = config
        synthetic = config.synthetic
        rng = np.random.default_rng(synthetic.seed)
        native_shape = config.model.img_size
        native_grid = spherical_harmonic.Grid(
            latitude_nodes=native_shape[1],
            longitude_nodes=native_shape[2],
            latitude_spacing=config.data.latitude_spacing,
        )
        self.grid = spherical_harmonic.Grid(
            latitude_nodes=config.data.fine_grid_shape[0],
            longitude_nodes=config.data.fine_grid_shape[1],
            latitude_spacing=config.data.latitude_spacing,
        )
        times = parse_timestamp(config.forecast.start_time) + np.arange(
            config.forecast.num_timesteps
        ) * np.timedelta64(config.forecast.input_step_hours, "h")
        native_values = rng.normal(
            synthetic.mean,
            synthetic.noise_std,
            size=(config.forecast.num_timesteps, len(VARIABLES), *native_shape),
        ).astype(np.float32)
        self.trajectory = self._dataset(native_values, times, native_grid)
        fine_values = rng.normal(
            synthetic.mean,
            synthetic.noise_std,
            size=(
                len(times[:1]),
                len(VARIABLES),
                len(config.data.pressure_levels_hpa),
                *config.data.fine_grid_shape,
            ),
        ).astype(np.float32)
        self.initial_pressure = self._dataset(fine_values, times[:1], self.grid)
        mapped = to_target_grid(self.trajectory.isel(time=[0]), self.grid)
        mapped = mapped.transpose("time", "level", "latitude", "longitude").sortby(
            "latitude", ascending=False
        )
        mapped = mapped.assign_coords(
            latitude=self.initial_pressure.latitude, longitude=self.initial_pressure.longitude
        )
        residual = self.initial_pressure - mapped
        self.initial = {name: residual[name].values for name in VARIABLES}

    def _dataset(self, values, times, grid):
        return xr.Dataset(
            {
                name: xr.DataArray(
                    values[:, index],
                    dims=("time", "level", "latitude", "longitude"),
                    attrs={"units": VARIABLE_UNITS[name]},
                )
                for index, name in enumerate(VARIABLES)
            },
            coords={
                "time": times,
                "level": np.asarray(self.config.data.pressure_levels_hpa),
                "latitude": np.rad2deg(grid.latitudes)[::-1],
                "longitude": np.rad2deg(grid.longitudes),
            },
            attrs={"description": "Random synthetic fields; no forecast skill implied"},
        ).assign_coords(
            level=xr.DataArray(
                np.asarray(self.config.data.pressure_levels_hpa),
                dims="level",
                attrs={"units": "hPa"},
            )
        )

    def _baseline_path(self, config):
        return config.output_dir / (
            f"output_neuralgcm_{config.forecast.start_time}_{config.forecast.num_timesteps}.nc"
        )

    def prepare_inputs(self, config):
        self.events.extend(f"baseline:{index}" for index in range(config.forecast.num_timesteps))
        path = self._baseline_path(config)
        self.trajectory.to_netcdf(path, engine="netcdf4")
        differences = self.trajectory.diff("time")
        return PreparedInputs(
            self.grid, {name: differences[name].values for name in VARIABLES}, self.initial, path
        )

    def BaselineSession(self, config):
        return SyntheticSession(self, config)


class SyntheticSession:
    """Expose each synthetic state only when the pipeline requests the next one."""

    def __init__(self, provider, config):
        self.provider, self.config = provider, config
        self.grid, self.initial = provider.grid, provider.initial
        self.initial_baseline = provider.trajectory.isel(time=[0])
        self.baseline_path = provider._baseline_path(config)

    def __enter__(self):
        self.provider.events.append("baseline:0")
        return self

    def iter_steps(self):
        trajectory = self.provider.trajectory
        for index in range(1, self.config.forecast.num_timesteps):
            if index > 1 and self.config.forecast.should_save(index - 1):
                previous_time = np.datetime_as_string(trajectory.time.values[index - 1], unit="h")
                previous_time = previous_time.replace("-", "").replace("T", "")
                if not (self.config.output_dir / f"output_one2n_{previous_time}.nc").is_file():
                    raise AssertionError("The previous output must be written before advancing")
            self.provider.events.append(f"baseline:{index}")
            increment = {
                name: trajectory[name].values[index : index + 1]
                - trajectory[name].values[index - 1 : index]
                for name in VARIABLES
            }
            yield BaselineStep(index, increment, trajectory.isel(time=[index]))
        trajectory.to_netcdf(self.baseline_path, engine="netcdf4")

    def __exit__(self, error_type, error, traceback):
        return False


def run_demo(config: RunConfig) -> Path:
    """Generate local artifacts and compare both complete inference schedules."""
    if config.synthetic is None:
        raise ValueError("The random-data demo needs explicit synthetic settings")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    started = time.perf_counter()
    torch.manual_seed(config.synthetic.seed)
    torch.set_num_threads(config.synthetic.torch_threads)
    root = config.paths.output
    artifacts, inputs = root / "artifacts", root / "inputs"
    artifacts.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)
    model_path, stats_path = artifacts / "random_model.pt", artifacts / "random_stats.pt"
    model = create_model(config.model)
    torch.save(
        {"model_state_dict": model.state_dict(), "description": "Random demo weights"}, model_path
    )
    del model
    shape = (len(VARIABLES), len(config.data.pressure_levels_hpa))
    torch.save(
        {
            "means": torch.full(shape, config.synthetic.mean, dtype=torch.float32),
            "variances": torch.full(shape, config.synthetic.variance, dtype=torch.float32),
        },
        stats_path,
    )
    base = replace(
        config,
        paths=replace(
            config.paths,
            one2n_checkpoint=model_path,
            stats_coarse=stats_path,
            stats_initial=stats_path,
            stats_target=stats_path,
            hres_pressure=inputs,
            hres_surface=inputs,
            era5_cloud=inputs,
        ),
        runtime=replace(config.runtime, device="cpu", jax_platform="cpu"),
    )
    traces, products = {}, {}
    for schedule in ("batch", "streaming"):
        run_config = replace(
            base,
            paths=replace(base.paths, output=root / schedule),
            forecast=replace(base.forecast, schedule=schedule),
        )
        provider = SyntheticBaselineProvider(run_config)
        if schedule == "batch":
            provider.initial_pressure.to_netcdf(inputs / "initial_pressure.nc", engine="netcdf4")
        products[schedule] = run_prediction(
            run_config,
            baseline_provider=provider,
            on_update=lambda index, events=provider.events: events.append(f"residual:{index}"),
        )
        traces[schedule] = provider.events
    expected_stream = ["baseline:0"]
    for index in range(1, config.forecast.num_timesteps):
        expected_stream.extend((f"baseline:{index}", f"residual:{index}"))
    if traces["streaming"] != expected_stream:
        raise AssertionError("Streaming must alternate baseline advancement and residual updates")
    expected_batch = [f"baseline:{index}" for index in range(config.forecast.num_timesteps)]
    expected_batch += [f"residual:{index}" for index in range(1, config.forecast.num_timesteps)]
    if traces["batch"] != expected_batch:
        raise AssertionError("Batch mode must prepare the trajectory before correcting it")
    if [p.name for p in products["batch"]] != [p.name for p in products["streaming"]]:
        raise AssertionError("Both schedules must publish the same valid times")
    for batch, streaming in zip(products["batch"], products["streaming"]):
        with xr.open_dataset(batch) as left, xr.open_dataset(streaming) as right:
            xr.testing.assert_allclose(
                left, right, atol=config.synthetic.atol, rtol=config.synthetic.rtol
            )
            if not all(np.isfinite(left[name]).all() for name in VARIABLES):
                raise AssertionError("Synthetic forecast contains non-finite values")
            if (left.specific_humidity < 0).any():
                raise AssertionError("Saved humidity must be nonnegative")
    summary = {
        "result": "PASS",
        "scope": "Random synthetic baseline and weights; no weather skill test",
        "updates": config.forecast.num_timesteps - 1,
        "output_files_per_schedule": len(products["batch"]),
        "schedules": traces,
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"PASS: random-data pipeline, both schedules, output equality, and NetCDF checks.\n{summary_path}"
    )
    print(f"Elapsed after imports: {summary['elapsed_seconds']:.2f} s")
    return summary_path

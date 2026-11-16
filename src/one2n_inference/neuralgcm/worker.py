"""Isolated NeuralGCM batch or persistent streaming worker."""

import argparse
import contextlib
import json
import logging
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import xarray as xr

from ..constants import VARIABLES
from ..data.io import load_hres_pressure, load_hres_surface, load_era5_cloud, parse_timestamp
from ..data.regridding import interpolate_pressure, grid_from_dataset, to_model_grid, to_target_grid
from .rollout import StreamingRollout, run_model


def _initialize(config):
    os.environ["JAX_PLATFORMS"] = config.runtime.jax_platform
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import neuralgcm

    with config.paths.neuralgcm_checkpoint.open("rb") as handle:
        model = neuralgcm.PressureLevelModel.from_checkpoint(pickle.load(handle))
    timestamp = config.forecast.start_time
    pressure = load_hres_pressure(config.paths.hres_pressure, timestamp)
    surface = load_hres_surface(config.paths.hres_surface, timestamp)
    auxiliary = load_era5_cloud(config.paths.era5_cloud, timestamp)
    if not np.array_equal(pressure.level.values, config.data.pressure_levels_hpa):
        raise ValueError("HRES pressure levels/order do not match data.pressure_levels_hpa")
    shape = (pressure.sizes["latitude"], pressure.sizes["longitude"])
    if shape != tuple(config.data.fine_grid_shape):
        raise ValueError("HRES latitude/longitude shape does not match data.fine_grid_shape")
    grid = grid_from_dataset(pressure)
    if grid.latitude_spacing != config.data.latitude_spacing:
        raise ValueError("HRES latitude spacing does not match data.latitude_spacing")
    pressure_interp = interpolate_pressure(pressure, auxiliary.level.values)
    model_grid = model.data_coords.horizontal
    native_shape = (model_grid.latitude_nodes, model_grid.longitude_nodes)
    if native_shape != tuple(config.model.img_size[1:]):
        raise ValueError("NeuralGCM native grid does not match model.img_size")
    initial = xr.merge(
        [
            to_model_grid(pressure_interp, model_grid),
            to_model_grid(surface, model_grid),
            to_model_grid(auxiliary, model_grid),
        ]
    )
    return model, initial, pressure


def _canonicalize(decoded, config, levels, attributes, level_attributes):
    baseline = decoded[list(VARIABLES)].sel(level=levels)
    lead_hours = baseline.time.values.astype("timedelta64[h]")
    baseline = baseline.assign_coords(time=parse_timestamp(config.forecast.start_time) + lead_hours)
    baseline = baseline.transpose("time", "level", "latitude", "longitude").sortby(
        "latitude", ascending=False
    )
    for name in VARIABLES:
        baseline[name].attrs = attributes[name].copy()
    baseline.level.attrs = level_attributes.copy()
    return baseline


def _initial_residual(baseline, pressure, config):
    fine = to_target_grid(baseline.isel(time=[0]), grid_from_dataset(pressure))
    fine = fine.transpose("time", "level", "latitude", "longitude").sortby(
        "latitude", ascending=False
    )
    for name in ("latitude", "longitude"):
        if not np.allclose(
            pressure[name].values,
            fine[name].values,
            atol=config.runtime.coordinate_tolerance,
            rtol=0,
        ):
            raise ValueError(f"Regridded {name} does not match the HRES grid")
    fine = fine.assign_coords(latitude=pressure.latitude, longitude=pressure.longitude)
    return pressure - fine


def _baseline_path(config):
    return config.output_dir / (
        f"output_neuralgcm_{config.forecast.start_time}_{config.forecast.num_timesteps}.nc"
    )


def prepare_baseline(config, residual_path):
    """Generate the complete native trajectory before the residual model runs."""
    model, initial, pressure = _initialize(config)
    decoded = run_model(
        model,
        initial,
        config.forecast.input_step_hours,
        config.forecast.num_timesteps,
        window=config.neuralgcm.rollout_window,
        seed=config.neuralgcm.seed,
    )
    baseline = _canonicalize(
        decoded,
        config,
        pressure.level.values,
        {name: pressure[name].attrs for name in VARIABLES},
        pressure.level.attrs,
    )
    _initial_residual(baseline, pressure, config).to_netcdf(residual_path, engine="netcdf4")
    baseline.attrs["expected_states"] = config.forecast.num_timesteps
    baseline.attrs["completed_states"] = baseline.sizes["time"]
    baseline.to_netcdf(_baseline_path(config), engine="netcdf4")


def _append_state(filename, state, expected_states):
    """Append one state without retaining the complete streaming trajectory."""
    import netCDF4

    if not filename.exists():
        state = state.copy()
        state.attrs["expected_states"] = expected_states
        state.attrs["completed_states"] = state.sizes["time"]
        origin = pd.Timestamp(state.time.values[0]).strftime("%Y-%m-%d %H:%M:%S")
        encoding = {
            "time": {
                "units": f"hours since {origin}",
                "dtype": "int64",
                "calendar": "proleptic_gregorian",
            }
        }
        state.to_netcdf(filename, engine="netcdf4", unlimited_dims=["time"], encoding=encoding)
        return
    with netCDF4.Dataset(filename, "a") as destination:
        index = len(destination.dimensions["time"])
        time_variable = destination.variables["time"]
        valid_time = pd.Timestamp(state.time.values[0]).to_pydatetime()
        time_variable[index] = netCDF4.date2num(
            valid_time,
            units=time_variable.units,
            calendar=getattr(time_variable, "calendar", "standard"),
        )
        for name in VARIABLES:
            destination.variables[name][index] = state[name].values[0]
        destination.setncattr("completed_states", index + state.sizes["time"])


def serve_stream(config, work_dir, protocol):
    """Publish P0, then advance only after each parent request."""
    model, initial, pressure = _initialize(config)
    rollout = StreamingRollout(
        model, initial, config.forecast.input_step_hours, config.neuralgcm.seed
    )
    levels = pressure.level.values.copy()
    attributes = {name: pressure[name].attrs.copy() for name in VARIABLES}
    level_attributes = pressure.level.attrs.copy()
    state = _canonicalize(rollout.initialization(), config, levels, attributes, level_attributes)
    residual_path = work_dir / "initial_residual.nc"
    _initial_residual(state, pressure, config).to_netcdf(residual_path, engine="netcdf4")
    del initial, pressure
    state_path = work_dir / "native_state.nc"
    trajectory = _baseline_path(config)
    if trajectory.exists():
        trajectory.unlink()
    state.to_netcdf(state_path, engine="netcdf4")
    _append_state(trajectory, state, config.forecast.num_timesteps)
    _respond(
        protocol,
        status="ready",
        state_path=str(state_path),
        residual_path=str(residual_path),
        index=0,
    )
    for line in sys.stdin:
        command = json.loads(line).get("command")
        if command == "close":
            _respond(protocol, status="closed")
            return
        if command != "advance":
            raise ValueError(f"Unknown streaming command: {command}")
        if rollout.index >= config.forecast.num_timesteps - 1:
            _respond(protocol, status="complete")
            continue
        state = _canonicalize(rollout.advance(), config, levels, attributes, level_attributes)
        state.to_netcdf(state_path, engine="netcdf4")
        _append_state(trajectory, state, config.forecast.num_timesteps)
        _respond(protocol, status="state", state_path=str(state_path), index=rollout.index)


def _respond(protocol, **message):
    protocol.write(json.dumps(message) + "\n")
    protocol.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="JSON request written by the client")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    from ..config import config_from_dict

    config = config_from_dict(request["config"], base_dir=args.request.parent)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = sys.stdout
    # Upstream printing must never corrupt the JSON-line protocol.
    with contextlib.redirect_stdout(sys.stderr):
        if args.stream:
            serve_stream(config, Path(request["work_dir"]), protocol)
        else:
            prepare_baseline(config, Path(request["residual_path"]))


if __name__ == "__main__":
    main()

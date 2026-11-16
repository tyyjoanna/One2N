"""Read the original HRES and ERA5 initialization file conventions."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..constants import VARIABLES


def parse_timestamp(value: str) -> np.datetime64:
    """Parse an initialization in UTC using the original YYYYMMDDHH format."""
    return np.datetime64(pd.to_datetime(value, format="%Y%m%d%H"), "ns")


def _matching_file(directory: str | Path, timestamp: str, identifier: str) -> Path:
    matches = sorted(Path(directory).glob(f"*{timestamp}*.nc"))
    matches = [path for path in matches if identifier in path.name]
    if not matches:
        raise FileNotFoundError(f"No {identifier} file for {timestamp} in {directory}")
    return matches[0]


def load_hres_pressure(directory: str | Path, target_time: str) -> xr.Dataset:
    """Read the five HRES pressure fields at initialization only."""
    fields = (
        ("U", "128_131_u"),
        ("V", "128_132_v"),
        ("T", "128_130_t"),
        ("Q", "128_133_q"),
        ("Z", "128_129_z"),
    )
    result = {}
    for name, (source, identifier) in zip(VARIABLES, fields):
        path = _matching_file(Path(directory) / target_time[:4], target_time, identifier)
        with xr.open_dataset(path, engine="netcdf4") as data:
            result[name] = data[source].sel(time=[parse_timestamp(target_time)]).load()
    return xr.Dataset(result).sortby("latitude", ascending=False)


def load_hres_surface(directory: str | Path, target_time: str) -> xr.Dataset:
    """Read sea-ice cover and sea-surface temperature at initialization."""
    fields = (
        ("CI", "128_031_ci", "sea_ice_cover"),
        ("SSTK", "128_034_sstk", "sea_surface_temperature"),
    )
    result = {}
    for source, identifier, name in fields:
        path = _matching_file(Path(directory) / target_time[:4], target_time[:8], identifier)
        with xr.open_dataset(path, engine="netcdf4") as data:
            result[name] = data[source].sel(time=[parse_timestamp(target_time)]).load()
    return xr.Dataset(result)


def load_era5_cloud(directory: str | Path, target_time: str) -> xr.Dataset:
    """Read initial cloud ice/liquid fields and the ERA5 pressure levels."""
    names = {
        "ciwc": "specific_cloud_ice_water_content",
        "clwc": "specific_cloud_liquid_water_content",
    }
    path = Path(directory) / f"{target_time[:6]}.nc"
    with xr.open_dataset(path, engine="netcdf4") as data:
        result = data[list(names)].sel(valid_time=[parse_timestamp(target_time)]).load()
    return (
        result.rename(names)
        .rename({"valid_time": "time", "pressure_level": "level"})
        .sortby("level")
    )

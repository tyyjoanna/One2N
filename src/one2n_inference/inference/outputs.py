"""Physical forecast reconstruction and NetCDF serialization."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from ..constants import VARIABLES


def reconstruct_forecast(residual: torch.Tensor, baseline: xr.Dataset, target_grid) -> xr.Dataset:
    from ..data.regridding import to_target_grid

    forecast = to_target_grid(baseline, target_grid)
    forecast = forecast.transpose("time", "level", "latitude", "longitude")
    forecast = forecast.sortby("latitude", ascending=False).copy()
    for index, name in enumerate(VARIABLES):
        values = forecast[name].values + residual[:, index].numpy()
        if name == "specific_humidity":
            values = np.maximum(values, 0)
        forecast[name] = forecast[name].copy(data=values)
        forecast[name].attrs = baseline[name].attrs.copy()
    forecast.level.attrs = baseline.level.attrs.copy()
    return forecast


def save_forecast(forecast: xr.Dataset, output_dir: Path) -> Path:
    valid_time = pd.Timestamp(forecast.time.values[0]).strftime("%Y%m%d%H")
    filename = output_dir / f"output_one2n_{valid_time}.nc"
    forecast.to_netcdf(filename, engine="netcdf4")
    return filename

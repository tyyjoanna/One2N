"""Log-pressure interpolation and conservative global-grid remapping."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from dinosaur import spherical_harmonic


def interpolate_pressure(data: xr.Dataset, target_levels: np.ndarray) -> xr.Dataset:
    """Interpolate linearly in log pressure, including reference extrapolation."""
    logged = data.assign_coords(level=np.log(data.level.values))
    result = logged.interp(
        level=np.log(target_levels), method="linear", kwargs={"fill_value": "extrapolate"}
    ).astype("float32")
    return result.assign_coords(level=target_levels)


def grid_from_dataset(data: xr.Dataset) -> spherical_harmonic.Grid:
    """Build a Dinosaur grid from increasing global latitude/longitude nodes."""
    from dinosaur import spherical_harmonic, xarray_utils

    latitude = np.sort(data.latitude.values)
    longitude = np.sort(data.longitude.values)
    return spherical_harmonic.Grid(
        latitude_nodes=data.sizes["latitude"],
        longitude_nodes=data.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(longitude),
    )


def _regrid_horizontal(
    data: xr.Dataset,
    source_grid: spherical_harmonic.Grid,
    target_grid: spherical_harmonic.Grid,
    fill_missing: bool = False,
) -> xr.Dataset:
    from dinosaur import horizontal_interpolation, xarray_utils

    # Dinosaur grid nodes run south to north; arrange the field values likewise.
    data = data.sortby("latitude").sortby("longitude")
    regridder = horizontal_interpolation.ConservativeRegridder(
        source_grid, target_grid, skipna=True
    )
    result = xarray_utils.regrid(data, regridder)
    if fill_missing:
        result = xarray_utils.fill_nan_with_nearest(result)
    for name in result.data_vars:
        if not np.isfinite(result[name].values).all():
            raise ValueError(f"Conservative regridding produced non-finite values in {name}.")
    return result


def to_model_grid(data: xr.Dataset, model_grid: spherical_harmonic.Grid) -> xr.Dataset:
    """Conservatively remap and fill the fields used to initialize NeuralGCM."""
    return _regrid_horizontal(data, grid_from_dataset(data), model_grid, fill_missing=True)


def to_target_grid(model_data: xr.Dataset, target_grid: spherical_harmonic.Grid) -> xr.Dataset:
    """Regrid a native NeuralGCM field to the HRES grid conservatively."""
    return _regrid_horizontal(model_data, grid_from_dataset(model_data), target_grid)

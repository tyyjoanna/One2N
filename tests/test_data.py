"""Input-file and interpolation contracts using generated meteorological fields."""

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import xarray as xr

from one2n_inference.constants import VARIABLES
from one2n_inference.data.io import (
    load_era5_cloud,
    load_hres_pressure,
    load_hres_surface,
    parse_timestamp,
)
from one2n_inference.data.regridding import (
    grid_from_dataset,
    interpolate_pressure,
    to_model_grid,
    to_target_grid,
)
from _fixtures import DAY, EPOCH, INPUT_STEP_HOURS, MONTH, YEAR, time_string


class InputTests(unittest.TestCase):
    def test_log_pressure_profile_and_extrapolation(self):
        levels = np.array([100.0, 500.0, 1000.0])
        data = xr.Dataset(
            {
                "u": (
                    ("time", "level", "latitude", "longitude"),
                    (3 * np.log(levels) + 2)[None, :, None, None],
                )
            },
            coords={
                "time": [parse_timestamp(EPOCH)],
                "level": levels,
                "latitude": [0.0],
                "longitude": [0.0],
            },
        )
        target = np.array([50.0, 200.0, 900.0])
        result = interpolate_pressure(data, target)
        np.testing.assert_allclose(result.u.values.flatten(), 3 * np.log(target) + 2, rtol=1e-7)
        np.testing.assert_array_equal(result.level.values, target)
        self.assertEqual(result.u.dtype, np.float32)

    def test_original_netcdf_layout_selects_only_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / YEAR).mkdir()
            times = [parse_timestamp(EPOCH), parse_timestamp(time_string(INPUT_STEP_HOURS))]
            coordinates = {
                "time": times,
                "level": [100.0, 500.0],
                "latitude": [45.0, -45.0],
                "longitude": [0.0, 180.0],
            }
            names = (
                ("U", "128_131_u"),
                ("V", "128_132_v"),
                ("T", "128_130_t"),
                ("Q", "128_133_q"),
                ("Z", "128_129_z"),
            )
            for index, (name, token) in enumerate(names):
                values = np.full((2, 2, 2, 2), index, dtype=np.float32)
                values[1] = index + 100
                xr.Dataset(
                    {name: (("time", "level", "latitude", "longitude"), values)}, coords=coordinates
                ).to_netcdf(root / YEAR / f"{token}_{EPOCH}.nc")
            for name, token in (("CI", "128_031_ci"), ("SSTK", "128_034_sstk")):
                xr.Dataset(
                    {name: (("time", "latitude", "longitude"), np.ones((2, 2, 2)))},
                    coords={k: v for k, v in coordinates.items() if k != "level"},
                ).to_netcdf(root / YEAR / f"{token}_{DAY}.nc")
            xr.Dataset(
                {
                    name: (
                        ("valid_time", "pressure_level", "latitude", "longitude"),
                        np.ones((2, 2, 2, 2)),
                    )
                    for name in ("ciwc", "clwc")
                },
                coords={
                    "valid_time": times,
                    "pressure_level": [500.0, 100.0],
                    "latitude": coordinates["latitude"],
                    "longitude": coordinates["longitude"],
                },
            ).to_netcdf(root / f"{MONTH}.nc")

            pressure = load_hres_pressure(root, EPOCH)
            self.assertEqual(tuple(pressure.data_vars), VARIABLES)
            self.assertEqual(pressure.sizes["time"], 1)
            for index, name in enumerate(VARIABLES):
                np.testing.assert_array_equal(pressure[name].values, index)
            surface = load_hres_surface(root, EPOCH)
            self.assertEqual(surface.sizes["time"], 1)
            self.assertEqual(set(surface), {"sea_ice_cover", "sea_surface_temperature"})
            auxiliary = load_era5_cloud(root, EPOCH)
            self.assertEqual(auxiliary.sizes["time"], 1)
            np.testing.assert_array_equal(auxiliary.level, [100.0, 500.0])
            self.assertEqual(
                set(auxiliary),
                {"specific_cloud_ice_water_content", "specific_cloud_liquid_water_content"},
            )
            with self.assertRaises(FileNotFoundError):
                load_hres_pressure(root, time_string(24))


@unittest.skipUnless(importlib.util.find_spec("dinosaur"), "Install the neuralgcm extra")
class ConservativeRegriddingTests(unittest.TestCase):
    def test_grid_direction_constants_and_reference_operator(self):
        os.environ["JAX_PLATFORMS"] = "cpu"
        from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

        native = spherical_harmonic.Grid(
            latitude_nodes=8, longitude_nodes=16, latitude_spacing="gauss"
        )
        for spacing in ("gauss", "equiangular", "equiangular_with_poles"):
            with self.subTest(spacing=spacing):
                source = spherical_harmonic.Grid(
                    latitude_nodes=16, longitude_nodes=32, latitude_spacing=spacing
                )
                latitudes = np.rad2deg(source.latitudes)
                data = xr.Dataset(
                    {
                        "constant": (
                            ("latitude", "longitude"),
                            np.ones((16, 32), dtype=np.float32),
                        ),
                        "hemisphere": (
                            ("latitude", "longitude"),
                            np.broadcast_to(
                                np.sin(np.deg2rad(latitudes))[:, None], (16, 32)
                            ).copy(),
                        ),
                    },
                    coords={"latitude": latitudes, "longitude": np.rad2deg(source.longitudes)},
                ).sortby("latitude", ascending=False)
                grid = grid_from_dataset(data)
                self.assertEqual(grid.latitude_spacing, spacing)
                coarse = to_model_grid(data, native)
                np.testing.assert_allclose(coarse.constant, 1, atol=2e-6, rtol=0)
                np.testing.assert_array_equal(
                    np.sign(coarse.hemisphere.mean("longitude")), np.sign(coarse.latitude)
                )
                reference = xarray_utils.regrid(
                    data,
                    horizontal_interpolation.ConservativeRegridder(source, native, skipna=True),
                )
                xr.testing.assert_allclose(coarse, reference)
                fine = to_target_grid(coarse, grid)
                np.testing.assert_allclose(fine.constant, 1, atol=3e-6, rtol=0)
                self.assertTrue(all(np.isfinite(field.values).all() for field in fine.values()))


if __name__ == "__main__":
    unittest.main()

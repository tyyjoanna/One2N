# Data and artifacts

[← README](../README.md) · [Architecture](architecture.md) · [Reproducibility](reproducibility.md)

The real-data pipeline requires initialization-time fields, matching One2N weights and statistics, and a NeuralGCM checkpoint. 

## Input file convention

All timestamps are **UTC**, and the exact initialization must exist. The real-data profile has no default initialization date: pass `--start-time` to `predict` and `check`, or set `forecast.start_time` in YAML to a time present in your data. The demo's synthetic epoch is not used as a fallback for real forecasts.

| Source | Files selected by the loader | Dimensions and coordinates |
| --- | --- | --- |
| HRES pressure | `hres_pressure/YYYY/`, filenames containing `YYYYMMDDHH` and the identifier below | `time, level, latitude, longitude` |
| HRES surface | `hres_surface/YYYY/`, filenames containing `YYYYMMDD` and the identifier below | `time, latitude, longitude` |
| ERA5 clouds | `era5_cloud/YYYYMM.nc` | `valid_time, pressure_level, latitude, longitude` |

Timestamp and identifier may occur in either filename order. The first sorted match is read, following the original loader. Keep one intended match per field and initialization.

### HRES fields [NCAR HRES dataset d113001](https://doi.org/10.5065/D68050ZV)

| File variable | Filename identifier | Internal name | Required unit |
| --- | --- | --- | --- |
| `U` | `128_131_u` | `u_component_of_wind` | m s⁻¹ |
| `V` | `128_132_v` | `v_component_of_wind` | m s⁻¹ |
| `T` | `128_130_t` | `temperature` | K |
| `Q` | `128_133_q` | `specific_humidity` | kg kg⁻¹ |
| `Z` | `128_129_z` | `geopotential` | m² s⁻² |
| `CI` | `128_031_ci` | `sea_ice_cover` | Fraction, 0–1 |
| `SSTK` | `128_034_sstk` | `sea_surface_temperature` | K |

### ERA5 auxiliary fields [ERA5 pressure-level data](https://doi.org/10.24381/cds.bd0915c6)

The monthly file contains `ciwc` and `clwc`, in kg kg⁻¹. They are renamed to `specific_cloud_ice_water_content` and `specific_cloud_liquid_water_content`. Coordinates `valid_time` and `pressure_level` become `time` and `level`, with pressure sorted ascending. Their levels must match NeuralGCM's input grid. HRES pressure fields are interpolated to these levels linearly in log pressure with extrapolation.

## Grid and pressure contract

- Pressure coordinates use **hPa**.
- Global grids must be supported by Dinosaur: Gaussian, equiangular, or equiangular with poles. Longitude uses `[0, 360)` without a duplicated periodic endpoint.
- Reference YAML specifies expected fine-grid dimensions and pressure levels. Reconfiguration must match the checkpoint and statistics.
- Dinosaur remapping uses south-to-north source latitude; One2N tensors and forecasts use north-to-south latitude.
- Target levels are `1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 925, 950, 1000` hPa. [Preprint, Supplementary Table S1](https://arxiv.org/html/2611.12249v1)

The reference input shape is `(batch, channel, level, latitude, longitude) = (1, 505, 25, 256, 512)`. Five coarse channels precede 500 fine-residual channels, representing five variables and a 10 × 10 spatial tile array. Model values reside in YAML; these dimensions describe the research architecture, not interchangeable settings for the same weights.

## Checkpoints and model configuration

`paths.one2n_checkpoint` identifies a trusted `.pt` mapping with `model_state_dict`. Loading is strict: all names and shapes must match YAML's `model` section. Inactive persistent parameters needed by the original checkpoint are retained. A different architecture requires corresponding trained weights.

The small `configs/demo.yaml` model uses random weights, which do not substitute for trained weights. `paths.neuralgcm_checkpoint` identifies the pretrained pickle; the reference is `neural_gcm_dynamic_forcing_deterministic_0_7_deg.pkl` with NeuralGCM 1.1.2. Obtain it from the [official project](https://github.com/neuralgcm/neuralgcm) under its terms. Load only the intended research artifacts.

## Normalization statistics

Each `.pt` file contains `means` and `variances`.

| Path setting | Quantity |
| --- | --- |
| `stats_coarse` | Native increments `P[t+1] − P[t]` |
| `stats_initial` | Initial residual `HRES(t0) − regrid(decoded P0)` |
| `stats_target` | Predicted fine residual for physical output reconstruction |

Channel order is zonal wind, meridional wind, temperature, specific humidity, and geopotential. Statistics broadcast to `(time, variable, level, latitude, longitude)`; `(variable, level)` arrays expand along singleton time/latitude/longitude axes. Grid-resolved statistics need the exact tensor grid and descending latitude. Stored arithmetic precision is preserved until the final float32 cast.

Reference initial and target paths identify the same `real_residual_stats.pt`. Retain the assignments associated with training. Normalization is `(x − mean) / sqrt(variance + epsilon)`; physical output scaling follows the original source, `x * sqrt(variance) + mean`. Predictions feed back directly without an additional normalization transform.

## Time and output conventions

`num_timesteps` includes decoded initialization. Final lead time is `(num_timesteps − 1) × input_step_hours`. A residual update occurs at every coupling interval, even with less frequent saving. `output_step_hours` must be a multiple of `input_step_hours`; the final update is always saved.

# Software architecture and numerical contract

[← README](../README.md) · [Data and artifacts](data.md) · [Reproducibility](reproducibility.md)

Scientific transformations are separate from orchestration. Model definitions contain tensor operations; data modules read physical arrays; the baseline adapter owns upstream state; the pipeline coordinates them. YAML supplies experiment-dependent values.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| Root `demo.py`, `predict.py` | Visible demonstration and prediction entry points |
| `cli.py`, `__main__.py` | Console/module commands |
| `config.py` | Load YAML, validate settings, resolve paths, serialize configuration |
| `constants.py` | Canonical atmospheric variable order |
| `models/swin_unet.py` | Checkpoint-compatible three-dimensional Swin-UNet |
| `models/checkpoint.py` | Construct the configured network and load all parameters strictly |
| `data/io.py` | Read HRES and ERA5 initialization fields |
| `data/regridding.py` | Log-pressure interpolation and conservative remapping |
| `data/statistics.py` | Load normalization statistics and physical output scaling |
| `neuralgcm/rollout.py` | Encode once and continuously carry latent state |
| `neuralgcm/worker.py` | Isolated baseline process, persistent for streaming |
| `neuralgcm/client.py` | Baseline requests and physical array transfer |
| `inference/packing.py` | Spatial-to-channel folding and inverse recovery |
| `inference/engine.py` | Normalized residual updates |
| `inference/outputs.py` | Physical reconstruction, humidity constraint, NetCDF serialization |
| `inference/pipeline.py` | Preparation, statistics, model loading, scheduling, and outputs |
| Package `demo.py` | Generate random inputs and compare complete small-data runs |

The installed package is used by both user commands and baseline subprocesses. Worker execution does not require adding the source directory to `sys.path`. Relative data paths are anchored to YAML through `paths.base_dir`.

## Two execution schedules

Batch retains the original execution order: prepare the baseline, then run One2N. `neuralgcm.rollout_window` controls integration batches without restarting state. Streaming interleaves one baseline update and one residual update, keeping the encoded state in a persistent process. The parent requests another baseline state only after the current residual update and due output finish. Initialization is never repeated at each step.

Streaming decodes `P0` directly without advancing. Each later request uses one upstream unroll interval with `start_with_input=False`, returning the newly advanced state without computing a future decoded state. Batch mode retains the original windowed `start_with_input=True` route. Controlled-model tests verify their state times and requested advance counts.

## Scientific invariants

Let `P0 … P[N−1]` be the baseline, including decoded initialization, and `G` conservative fine-grid remapping.

```text
R0       = HRES(t0) − G(P0)
S0       = normalize_initial(R0)
Ct       = normalize_coarse(P[t+1] − P[t])
S[t+1]   = recover(fine_channels(One2N(concat(Ct, pack(St)))))
R[t+1]   = denormalize_target(S[t+1])
Y[t+1]   = G(P[t+1]) + R[t+1]
```

- **State counting:** initialization is included; One2N makes one fewer updates than `num_timesteps`.
- **Initial residual:** subtract remapped decoded `P0`, not the pre-encoding initialization.
- **Preprocessing:** log-pressure interpolation includes original extrapolation; horizontal remapping is conservative with `skipna=True`. Upstream initialization uses nearest filling as needed; reverse remapping does not silently fill invalid outputs.
- **Forcing/state:** persist initialization-time surface forcing and carry the encoded state continuously. Seeds and window sizes are configuration values.
- **Variable/tile order:** zonal wind, meridional wind, temperature, specific humidity, geopotential; then latitude tile and longitude tile within each variable.
- **Replacement feedback:** discard coarse output channels and directly replace the fine normalized state. Do not add the prediction to the preceding state or normalize it again.
- **Precision:** network tensors use float32 and original manual attention. Statistics preserve stored precision until the final conversion.
- **Physical scaling:** preserve epsilon in normalization and no epsilon in denormalization, as in the supplied source. This source-code convention differs from the epsilon-inclusive scaling in [preprint Eq. 12](https://arxiv.org/html/2611.12249v1); no silent numerical change is made here.
- **Output constraint:** clip humidity only in reconstructed physical forecasts, without changing recurrent states.

Output subsampling never omits residual updates. Input intervals must match `model.trained_step_hours`; a different interval requires a matching trained checkpoint, not merely a YAML edit.

## Mapping from the original code

| Original responsibility | Package implementation |
| --- | --- |
| `Swin_Unet.py` network | `models/swin_unet.py` |
| Architecture and checkpoint loading | Configured factory in `models/checkpoint.py` |
| `dataset.py` readers/interpolation | `data/io.py`, `data/regridding.py` |
| `dataset.py` upstream forecast | `neuralgcm/rollout.py`, `worker.py`, `client.py` |
| Driver statistics and tile layout | `data/statistics.py`, `inference/packing.py` |
| Driver recurrence and saving | `inference/engine.py`, `outputs.py`, `pipeline.py` |
| Machine-specific arguments | YAML and command overrides |

The active network in `Swin_Unet_model_union.py` is equivalent to the standalone model; training and inactive experimental code are omitted. Persistent checkpoint entries remain, including inactive forward-path parameters. Debug barriers and unused fused branches are removed. The unreachable final-save condition and lost unit metadata are corrected explicitly.

## Extension boundaries

Place new readers in `data/io.py`, output formats in `inference/outputs.py`, and baseline adapters in `neuralgcm/`. A different remapper, pressure grid, transform, attention backend, or coupling interval requires independent comparison. Synthetic examples validate execution paths, not atmospheric skill or manuscript evaluation.

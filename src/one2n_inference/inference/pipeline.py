"""Batch and stepwise schedules for the same residual correction algorithm."""

from collections.abc import Callable
import json
import logging
import os
from pathlib import Path
import torch
import xarray as xr

from ..config import RunConfig
from ..constants import VARIABLES
from ..data.statistics import load_statistics
from ..models.checkpoint import load_model
from ..neuralgcm import client as neuralgcm_client
from .engine import InferenceEngine
from .outputs import reconstruct_forecast, save_forecast

logger = logging.getLogger(__name__)


def stack_fields(fields: dict) -> torch.Tensor:
    return torch.stack(
        [torch.tensor(fields[name], dtype=torch.float32) for name in VARIABLES], dim=1
    )


def _device(config):
    if config.runtime.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.runtime.device)


def _initialize(config, initial_fields):
    initial_stats = load_statistics(config.paths.stats_initial)
    initial = initial_stats.normalize(stack_fields(initial_fields), config.runtime.epsilon)
    if config.paths.stats_initial == config.paths.stats_target:
        target_stats = initial_stats
    else:
        del initial_stats
        target_stats = load_statistics(config.paths.stats_target)
    model = load_model(config.paths.one2n_checkpoint, _device(config), config.model)
    engine = InferenceEngine(model)
    engine.initialize(initial)
    return engine, target_stats


def _publish(config, index, state, baseline, grid, target_stats, products):
    if config.forecast.should_save(index):
        residual = target_stats.denormalize(state.cpu())
        forecast = reconstruct_forecast(residual, baseline, grid)
        product = save_forecast(forecast, config.output_dir)
        products.append(product)
        logger.info("Saved update %s: %s", index, product)


def run_prediction(
    config: RunConfig, *, baseline_provider=None, on_update: Callable[[int], None] | None = None
) -> list[Path]:
    """Run a forecast, optionally using an explicit test trajectory provider.

    The prediction CLI uses NeuralGCM. The random-data demo supplies its
    documented synthetic provider while exercising this same pipeline.
    """
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    provider = neuralgcm_client if baseline_provider is None else baseline_provider
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    products = []
    low_stats = load_statistics(config.paths.stats_coarse)
    if config.forecast.schedule == "batch":
        prepared = provider.prepare_inputs(config)
        engine, target_stats = _initialize(config, prepared.initial)
        coarse = low_stats.normalize(stack_fields(prepared.coarse), config.runtime.epsilon)
        grid, baseline_path = prepared.grid, prepared.baseline_path
        del prepared, low_stats
        if coarse.shape[0] != config.forecast.num_timesteps - 1:
            raise ValueError("Baseline must supply num_timesteps - 1 native increments")
        with xr.open_dataset(baseline_path, engine="netcdf4") as baseline:
            for index, increment in enumerate(coarse.split(1), start=1):
                state = engine.step(increment)
                if on_update is not None:
                    on_update(index)
                _publish(
                    config, index, state, baseline.isel(time=[index]), grid, target_stats, products
                )
    else:
        with provider.BaselineSession(config) as session:
            engine, target_stats = _initialize(config, session.initial)
            completed = 0
            for native_step in session.iter_steps():
                if native_step.index != completed + 1:
                    raise ValueError("Streaming baseline steps must be consecutive")
                coarse = low_stats.normalize(
                    stack_fields(native_step.increment), config.runtime.epsilon
                )
                state = engine.step(coarse)
                if on_update is not None:
                    on_update(native_step.index)
                _publish(
                    config,
                    native_step.index,
                    state,
                    native_step.baseline,
                    session.grid,
                    target_stats,
                    products,
                )
                completed = native_step.index
            if completed != config.forecast.num_timesteps - 1:
                raise ValueError("Streaming baseline ended before the configured horizon")
    return products

"""Batch preparation and request-driven persistent NeuralGCM sessions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from ..constants import VARIABLES
from ..data.regridding import grid_from_dataset

if TYPE_CHECKING:
    from dinosaur import spherical_harmonic
    from ..config import RunConfig

WORKER_SHUTDOWN_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class PreparedInputs:
    grid: spherical_harmonic.Grid
    coarse: dict[str, np.ndarray]
    initial: dict[str, np.ndarray]
    baseline_path: Path


@dataclass(frozen=True)
class BaselineStep:
    index: int
    increment: dict[str, np.ndarray]
    baseline: xr.Dataset


def _parent_backend():
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _read_state(filename):
    with xr.open_dataset(filename, engine="netcdf4") as data:
        return data.load()


def baseline_path(config):
    return config.output_dir / (
        f"output_neuralgcm_{config.forecast.start_time}_{config.forecast.num_timesteps}.nc"
    )


def prepare_inputs(config: RunConfig) -> PreparedInputs:
    """Finish NeuralGCM's full trajectory before returning residual-model inputs."""
    _parent_backend()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="one2n-") as folder:
        folder = Path(folder)
        residual_path = folder / "initial_residual.nc"
        request_path = folder / "request.json"
        request_path.write_text(
            json.dumps(
                {
                    "config": config.to_dict(),
                    "residual_path": str(residual_path),
                }
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [sys.executable, "-m", "one2n_inference.neuralgcm.worker", str(request_path)],
            check=True,
        )
        initial = _read_state(residual_path)
    trajectory = baseline_path(config)
    with xr.open_dataset(trajectory, engine="netcdf4") as data:
        increments = data[list(VARIABLES)].diff("time").load()
    return PreparedInputs(
        grid_from_dataset(initial),
        {name: increments[name].values for name in VARIABLES},
        {name: initial[name].values for name in VARIABLES},
        trajectory,
    )


class BaselineSession:
    """Stream states from a single continuously integrated NeuralGCM process.

    A new native advance occurs only when the caller requests the next item.
    Residual prediction and output can run between consecutive native advances.
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self.baseline_path = baseline_path(config)
        self._temporary = None
        self._process = None
        self._previous = None

    def __enter__(self):
        _parent_backend()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix="one2n-stream-")
        folder = Path(self._temporary.name)
        request = folder / "request.json"
        request.write_text(
            json.dumps(
                {
                    "config": self.config.to_dict(),
                    "work_dir": str(folder),
                }
            ),
            encoding="utf-8",
        )
        self._process = subprocess.Popen(
            [sys.executable, "-m", "one2n_inference.neuralgcm.worker", str(request), "--stream"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        try:
            message = self._receive("ready")
            residual = _read_state(message["residual_path"])
            self.initial = {name: residual[name].values for name in VARIABLES}
            self.grid = grid_from_dataset(residual)
            self.initial_baseline = _read_state(message["state_path"])
            self._previous = self.initial_baseline
            return self
        except BaseException:
            self.close()
            raise

    def _receive(self, expected):
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("NeuralGCM streaming worker exited before returning a state")
        message = json.loads(line)
        if message.get("status") != expected:
            raise RuntimeError(f"Unexpected NeuralGCM worker response: {message}")
        return message

    def iter_steps(self):
        for index in range(1, self.config.forecast.num_timesteps):
            self._process.stdin.write(json.dumps({"command": "advance"}) + "\n")
            self._process.stdin.flush()
            message = self._receive("state")
            if message["index"] != index:
                raise RuntimeError("NeuralGCM streaming states arrived out of sequence")
            current = _read_state(message["state_path"])
            increments = {
                name: current[name].values - self._previous[name].values for name in VARIABLES
            }
            self._previous = current
            yield BaselineStep(index, increments, current)

    def close(self):
        if self._process is not None:
            if self._process.poll() is None:
                try:
                    self._process.stdin.write(json.dumps({"command": "close"}) + "\n")
                    self._process.stdin.flush()
                    self._process.wait(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    self._process.terminate()
                    self._process.wait()
            if self._process.stdin is not None:
                self._process.stdin.close()
            if self._process.stdout is not None:
                self._process.stdout.close()
            self._process = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

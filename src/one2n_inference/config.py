"""Validated YAML settings for model geometry, forecast scheduling, and paths."""

from dataclasses import asdict, dataclass
from datetime import datetime
import math
from pathlib import Path
import re

import yaml

from .constants import VARIABLES


def _integer(name, value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(name, value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")


def _shape(name, values, length):
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} must contain {length} integer dimensions")
    for value in values:
        _integer(name, value, 1)
    return tuple(values)


@dataclass(frozen=True)
class Paths:
    one2n_checkpoint: Path
    neuralgcm_checkpoint: Path
    stats_coarse: Path
    stats_initial: Path
    stats_target: Path
    hres_pressure: Path
    hres_surface: Path
    era5_cloud: Path
    output: Path


@dataclass(frozen=True)
class Forecast:
    start_time: str
    num_timesteps: int
    input_step_hours: int
    output_step_hours: int
    schedule: str

    def __post_init__(self):
        if not isinstance(self.start_time, str) or not re.fullmatch(r"[0-9]{10}", self.start_time):
            raise ValueError("forecast.start_time must be a quoted UTC YYYYMMDDHH string")
        datetime.strptime(self.start_time, "%Y%m%d%H")
        _integer("forecast.num_timesteps", self.num_timesteps, 2)
        _integer("forecast.input_step_hours", self.input_step_hours, 1)
        _integer("forecast.output_step_hours", self.output_step_hours, 1)
        if self.output_step_hours % self.input_step_hours:
            raise ValueError("forecast.output_step_hours must be a multiple of input_step_hours")
        if self.schedule not in ("batch", "streaming"):
            raise ValueError("forecast.schedule must be batch or streaming")

    @property
    def save_interval(self):
        return self.output_step_hours // self.input_step_hours

    def should_save(self, step: int) -> bool:
        """One-based correction updates; always save the final update."""
        return step % self.save_interval == 0 or step == self.num_timesteps - 1


@dataclass(frozen=True)
class Runtime:
    device: str
    jax_platform: str
    epsilon: float
    coordinate_tolerance: float

    def __post_init__(self):
        if not isinstance(self.device, str) or not re.fullmatch(
            r"auto|cpu|cuda(?::[0-9]+)?", self.device
        ):
            raise ValueError("runtime.device must be auto, cpu, cuda, or cuda:N")
        if self.jax_platform not in ("cpu", "cuda"):
            raise ValueError("runtime.jax_platform must be cpu or cuda")
        _number("runtime.epsilon", self.epsilon, positive=True)
        _number("runtime.coordinate_tolerance", self.coordinate_tolerance, positive=True)


@dataclass(frozen=True)
class ModelConfig:
    img_size: tuple[int, ...]
    patch_size: tuple[int, ...]
    in_chans: int
    embed_dim: int
    depths: tuple[int, ...]
    updown_sizes: tuple[tuple[int, ...], ...]
    num_heads: tuple[int, ...]
    window_size: tuple[int, ...]
    mlp_ratio: float
    qkv_bias: bool
    qk_scale: float | str
    drop_rate: float
    attn_drop_rate: float
    drop_path_rate: float
    ape: bool
    patch_norm: bool
    trained_step_hours: int

    def __post_init__(self):
        for name in ("img_size", "patch_size", "window_size"):
            object.__setattr__(self, name, _shape(f"model.{name}", getattr(self, name), 3))
        for name in ("in_chans", "embed_dim", "trained_step_hours"):
            _integer(f"model.{name}", getattr(self, name), 1)
        if not isinstance(self.depths, (list, tuple)) or not self.depths or len(self.depths) % 2:
            raise ValueError("model.depths must contain an even number of encoder/decoder stages")
        object.__setattr__(self, "depths", _shape("model.depths", self.depths, len(self.depths)))
        object.__setattr__(
            self, "num_heads", _shape("model.num_heads", self.num_heads, len(self.depths))
        )
        if (
            not isinstance(self.updown_sizes, (list, tuple))
            or len(self.updown_sizes) != len(self.depths) // 2
        ):
            raise ValueError("model.updown_sizes must have one factor per encoder stage")
        object.__setattr__(
            self,
            "updown_sizes",
            tuple(
                _shape("model.updown_sizes", factor, len(self.img_size))
                for factor in self.updown_sizes
            ),
        )
        for name in ("qkv_bias", "ape", "patch_norm"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"model.{name} must be true or false")
        _number("model.mlp_ratio", self.mlp_ratio, positive=True)
        if self.qk_scale != "auto":
            _number("model.qk_scale", self.qk_scale, positive=True)
        for name in ("drop_rate", "attn_drop_rate", "drop_path_rate"):
            _number(f"model.{name}", getattr(self, name))
            if getattr(self, name) >= 1:
                raise ValueError(f"model.{name} must be less than one")
        if any(size % patch for size, patch in zip(self.img_size, self.patch_size)):
            raise ValueError("model.img_size must be divisible by patch_size")
        resolution = tuple(size // patch for size, patch in zip(self.img_size, self.patch_size))
        stage_resolutions = []
        for stage in range(len(self.depths) // 2):
            stage_resolutions.append(resolution)
            if stage < len(self.depths) // 2 - 1:
                factor = self.updown_sizes[stage]
                if any(size % stride for size, stride in zip(resolution, factor)):
                    raise ValueError("Encoder resolution must be divisible by its up/down factor")
                resolution = tuple(size // stride for size, stride in zip(resolution, factor))
        resolutions = stage_resolutions + list(reversed(stage_resolutions))
        for stage, (shape, heads) in enumerate(zip(resolutions, self.num_heads)):
            if any(size % window for size, window in zip(shape, self.window_size)):
                raise ValueError("Every encoder/decoder stage must be divisible by window_size")
            dimension = self.embed_dim * 2 ** min(stage, len(self.depths) - stage - 1)
            if dimension % heads:
                raise ValueError("Every model stage embedding must be divisible by num_heads")

    def to_kwargs(self) -> dict:
        result = asdict(self)
        result.pop("trained_step_hours")
        if result["qk_scale"] == "auto":
            result["qk_scale"] = None
        return result


@dataclass(frozen=True)
class NeuralGCMConfig:
    seed: int
    rollout_window: int

    def __post_init__(self):
        _integer("neuralgcm.seed", self.seed, 0)
        _integer("neuralgcm.rollout_window", self.rollout_window, 1)


@dataclass(frozen=True)
class DataConfig:
    pressure_levels_hpa: tuple[float, ...]
    fine_grid_shape: tuple[int, ...]
    latitude_spacing: str

    def __post_init__(self):
        if not isinstance(self.pressure_levels_hpa, (list, tuple)) or not self.pressure_levels_hpa:
            raise ValueError("data.pressure_levels_hpa must be a nonempty ordered list")
        for level in self.pressure_levels_hpa:
            _number("data.pressure_levels_hpa", level, positive=True)
        if len(set(self.pressure_levels_hpa)) != len(self.pressure_levels_hpa):
            raise ValueError("data.pressure_levels_hpa must not contain duplicate levels")
        object.__setattr__(self, "pressure_levels_hpa", tuple(self.pressure_levels_hpa))
        object.__setattr__(
            self, "fine_grid_shape", _shape("data.fine_grid_shape", self.fine_grid_shape, 2)
        )
        if self.latitude_spacing not in ("gauss", "equiangular", "equiangular_with_poles"):
            raise ValueError(
                "data.latitude_spacing must be gauss, equiangular, or equiangular_with_poles"
            )


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int
    noise_std: float
    mean: float
    variance: float
    torch_threads: int
    atol: float
    rtol: float

    def __post_init__(self):
        _integer("synthetic.seed", self.seed, 0)
        _integer("synthetic.torch_threads", self.torch_threads, 1)
        _number("synthetic.noise_std", self.noise_std, positive=True)
        _number("synthetic.variance", self.variance, positive=True)
        if (
            isinstance(self.mean, bool)
            or not isinstance(self.mean, (int, float))
            or not math.isfinite(self.mean)
        ):
            raise ValueError("synthetic.mean must be a finite number")
        _number("synthetic.atol", self.atol)
        _number("synthetic.rtol", self.rtol)


@dataclass(frozen=True)
class RunConfig:
    paths: Paths
    forecast: Forecast
    runtime: Runtime
    model: ModelConfig
    neuralgcm: NeuralGCMConfig
    data: DataConfig
    synthetic: SyntheticConfig | None = None

    def __post_init__(self):
        if self.forecast.input_step_hours != self.model.trained_step_hours:
            raise ValueError(
                "forecast.input_step_hours must match model.trained_step_hours for the checkpoint"
            )
        if len(self.data.pressure_levels_hpa) != self.model.img_size[0]:
            raise ValueError("The number of pressure levels must match model.img_size[0]")
        coarse_horizontal = self.model.img_size[1:]
        if any(fine % coarse for fine, coarse in zip(self.data.fine_grid_shape, coarse_horizontal)):
            raise ValueError("data.fine_grid_shape must be divisible by the model horizontal shape")
        latitude_tiles, longitude_tiles = (
            fine // coarse for fine, coarse in zip(self.data.fine_grid_shape, coarse_horizontal)
        )
        expected_channels = len(VARIABLES) * (1 + latitude_tiles * longitude_tiles)
        if self.model.in_chans != expected_channels:
            raise ValueError("model.in_chans does not match the variable and fine-grid tile count")

    @property
    def output_dir(self) -> Path:
        return self.paths.output / self.forecast.start_time

    def to_dict(self) -> dict:
        result = asdict(self)
        result["paths"] = {name: str(value) for name, value in result["paths"].items()}
        return result


def config_from_dict(raw: dict, base_dir: str | Path = Path(".")) -> RunConfig:
    """Validate a YAML mapping or an absolute-path job serialized for a worker."""
    if not isinstance(raw, dict):
        raise ValueError("The configuration must be a YAML mapping")
    sections = {"paths", "forecast", "runtime", "model", "neuralgcm", "data", "synthetic"}
    unknown = set(raw) - sections
    if unknown:
        raise ValueError(f"Unknown configuration sections: {', '.join(sorted(unknown))}")
    for name in sections - {"synthetic"}:
        if not isinstance(raw.get(name), dict):
            raise ValueError(f"{name} must be an explicitly populated YAML mapping")
    if "start_time" not in raw["forecast"]:
        raise ValueError("Set forecast.start_time in YAML or supply --start-time (UTC YYYYMMDDHH)")
    paths = dict(raw["paths"])
    base_value = paths.pop("base_dir", ".")
    if not isinstance(base_value, str) or not base_value.strip():
        raise ValueError("paths.base_dir must be a nonempty path string")
    base = Path(base_value).expanduser()
    if not base.is_absolute():
        base = Path(base_dir) / base
    base = base.resolve()
    resolved = {}
    for name, value in paths.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{name} must be a nonempty path string")
        path = Path(value).expanduser()
        resolved[name] = (path if path.is_absolute() else base / path).resolve()
    try:
        synthetic = raw.get("synthetic")
        if synthetic is not None and not isinstance(synthetic, dict):
            raise ValueError("synthetic must be a mapping when provided")
        return RunConfig(
            paths=Paths(**resolved),
            forecast=Forecast(**raw["forecast"]),
            runtime=Runtime(**raw["runtime"]),
            model=ModelConfig(**raw["model"]),
            neuralgcm=NeuralGCMConfig(**raw["neuralgcm"]),
            data=DataConfig(**raw["data"]),
            synthetic=SyntheticConfig(**synthetic) if synthetic is not None else None,
        )
    except TypeError as error:
        raise ValueError(f"Invalid configuration fields: {error}") from error


def load_config(filename: str | Path, *, start_time: str | None = None) -> RunConfig:
    """Load YAML and an optional explicit initialization-time override.

    paths.base_dir is relative to the configuration file. A real forecast has
    no default date: provide forecast.start_time or the command-line override.
    """
    source = Path(filename).expanduser().resolve()
    if source.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError("Use a .yaml or .yml runtime configuration file")
    try:
        with source.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML configuration: {error}") from error
    if start_time is not None:
        if not isinstance(raw, dict) or not isinstance(raw.get("forecast"), dict):
            raise ValueError("forecast must be an explicitly populated YAML mapping")
        raw = dict(raw)
        raw["forecast"] = dict(raw["forecast"], start_time=start_time)
    return config_from_dict(raw, base_dir=source.parent)


def missing_inputs(config: RunConfig) -> list[Path]:
    """Check required files/directories without opening meteorological arrays."""
    if config.synthetic is not None:
        return []
    files = (
        "one2n_checkpoint",
        "neuralgcm_checkpoint",
        "stats_coarse",
        "stats_initial",
        "stats_target",
    )
    directories = ("hres_pressure", "hres_surface", "era5_cloud")
    missing = [
        getattr(config.paths, name) for name in files if not getattr(config.paths, name).is_file()
    ]
    missing += [
        getattr(config.paths, name)
        for name in directories
        if not getattr(config.paths, name).is_dir()
    ]
    return list(dict.fromkeys(missing))

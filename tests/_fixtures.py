"""Shared synthetic clock; all test dates derive from the demo YAML profile."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEMO_SETTINGS = yaml.safe_load((ROOT / "configs/demo.yaml").read_text(encoding="utf-8"))
EPOCH = DEMO_SETTINGS["forecast"]["start_time"]
INPUT_STEP_HOURS = DEMO_SETTINGS["forecast"]["input_step_hours"]
YEAR, MONTH, DAY = EPOCH[:4], EPOCH[:6], EPOCH[:8]


def time_string(hours=0):
    value = datetime.strptime(EPOCH, "%Y%m%d%H") + timedelta(hours=hours)
    return value.strftime("%Y%m%d%H")


def timestamp(hours=0):
    value = datetime.strptime(EPOCH, "%Y%m%d%H") + timedelta(hours=hours)
    return np.datetime64(value, "ns")


def forecast_filename(hours):
    return f"output_one2n_{time_string(hours)}.nc"

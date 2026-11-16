"""Commands for YAML-configured forecasts and the random-data demonstration."""

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import time
from . import __version__
from .config import load_config, missing_inputs


def build_parser():
    parser = argparse.ArgumentParser(prog="one2n", description="One2N atmospheric inference")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Run a configured real-data forecast")
    predict.add_argument("--config", type=Path, required=True, help="YAML configuration")
    predict.add_argument("--start-time", help="UTC YYYYMMDDHH")
    predict.add_argument("--num-timesteps", type=int, help="States including initialization")
    predict.add_argument("--input-step-hours", type=int, help="NeuralGCM/One2N coupling interval")
    predict.add_argument("--output-step-hours", type=int, help="Saved forecast interval")
    predict.add_argument("--schedule", choices=("batch", "streaming"))
    predict.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    demo = commands.add_parser("demo", help="Test both schedules with random synthetic fields")
    demo.add_argument("--config", type=Path, required=True, help="Synthetic YAML profile")
    demo.add_argument("--output", type=Path, help="Override paths.output")
    check = commands.add_parser("check", help="Validate YAML and required input paths")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--start-time", help="UTC YYYYMMDDHH; overrides forecast.start_time")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        config = load_config(args.config, start_time=getattr(args, "start_time", None))
        if args.command == "predict":
            updates = {
                name: value
                for name in (
                    "num_timesteps",
                    "input_step_hours",
                    "output_step_hours",
                    "schedule",
                )
                if (value := getattr(args, name)) is not None
            }
            config = replace(config, forecast=replace(config.forecast, **updates))
            if args.device is not None:
                config = replace(config, runtime=replace(config.runtime, device=args.device))
        elif args.command == "demo" and args.output is not None:
            config = replace(config, paths=replace(config.paths, output=args.output.resolve()))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.command == "demo":
        if config.synthetic is None:
            parser.error("The demo requires a profile with an explicit synthetic section")
        from .demo import run_demo

        run_demo(config)
        return 0
    if args.command == "predict" and config.synthetic is not None:
        parser.error("Use demo for a synthetic profile; predict requires real research inputs")
    missing = missing_inputs(config)
    if args.command == "check":
        print(
            json.dumps(
                {"configuration": config.to_dict(), "missing_inputs": [str(p) for p in missing]},
                indent=2,
            )
        )
        return 1 if missing else 0
    if missing:
        parser.error("Required inputs are missing:\n" + "\n".join(str(p) for p in missing))
    from .inference.pipeline import run_prediction

    started = time.perf_counter()
    products = run_prediction(config)
    logging.info("Completed: %s outputs in %.1f s", len(products), time.perf_counter() - started)
    return 0

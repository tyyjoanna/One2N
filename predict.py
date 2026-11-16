"""Run a configured One2N forecast from the repository root."""

import sys

from one2n_inference.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["predict", *sys.argv[1:]]))

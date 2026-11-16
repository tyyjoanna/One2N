"""Run the configured random-data demonstration from the repository root."""

import sys

from one2n_inference.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))

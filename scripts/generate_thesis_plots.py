#!/usr/bin/env python3
"""
Generate thesis plots from locally downloaded W&B data.

- Finds the latest wandb_data_*_deduped directory by default
- Saves plots into plots/thesis_figures/<timestamp>
"""

import argparse
import datetime
import glob
import os
import sys


def find_latest_wandb_dir(base_dir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(base_dir, "wandb_data_*_deduped")))
    if not candidates:
        raise FileNotFoundError("No 'wandb_data_*_deduped' directory found. Run download_wandb_data.py first.")
    return candidates[-1]


def make_output_dir(base_dir: str, output_dir: str | None) -> str:
    if output_dir:
        out = output_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(base_dir, "plots", "thesis_figures", timestamp)
    os.makedirs(out, exist_ok=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate thesis plots from W&B dump")
    parser.add_argument("--data-dir", dest="data_dir", default=None, help="Path to wandb_data_*_deduped directory")
    parser.add_argument("--output-dir", dest="output_dir", default=None, help="Directory to save plots into")
    parser.add_argument("--comparison-bound", dest="comparison_bound", type=float, default=0.05, help="Comparison safety bound for plots")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    data_dir = args.data_dir or find_latest_wandb_dir(base_dir)
    output_dir = make_output_dir(base_dir, args.output_dir)

    print(f"Using data_dir={data_dir}")
    print(f"Saving plots to={output_dir}")

    # Ensure repository root is importable
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    try:
        import thesis_plotting_functions as t
    except Exception as exc:
        print(f"Failed to import thesis_plotting_functions: {exc}", file=sys.stderr)
        return 2

    try:
        t.generate_all_plots(data_dir=data_dir, output_dir=output_dir, comparison_bound=args.comparison_bound)
    except Exception as exc:
        print(f"Plot generation failed: {exc}", file=sys.stderr)
        return 3

    # Emit a machine-readable line for callers
    print(f"OUTPUT_DIR={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



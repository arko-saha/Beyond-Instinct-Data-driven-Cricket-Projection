"""
CLI entry point for the cricket data preprocessing pipeline.

Usage
-----
.. code-block:: bash

    # Preprocessing only (clean + reorder):
    python -m src.preprocessing.pipeline --input data/merged.csv --output data/preprocessed.csv

    # Preprocessing + feature engineering:
    python -m src.preprocessing.pipeline --input data/merged.csv --output data/preprocessed.csv --features

The pipeline orchestrates:
1. Load raw CSV and standardize column names.
2. Parse ball numbers into ``completed_over`` and ``ball_no``.
3. Clean the data (reorder, remove super overs, add wicket indicator).
4. (Optional) Engineer features: cumulative stats, CRR, RRR, rolling form.
5. Save the result to a CSV.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .loader import load_match_data
from .ball_parser import separate_ball_numbers
from .cleaner import clean
from .features import engineer_features


def run_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    with_features: bool = False,
) -> None:
    """Execute the preprocessing pipeline end-to-end.

    Parameters
    ----------
    input_path : str or Path
        Path to the raw ball-by-ball CSV.
    output_path : str or Path
        Destination path for the processed CSV.
    with_features : bool
        If ``True``, also run the feature-engineering stage.
    """
    start = time.time()

    print(f"[1/{'4' if with_features else '3'}] Loading data from {input_path}…")
    df = load_match_data(input_path)
    print(f"      → {len(df):,} rows, {len(df.columns)} columns loaded.")

    print(f"[2/{'4' if with_features else '3'}] Parsing ball numbers…")
    df = separate_ball_numbers(df)

    print(f"[3/{'4' if with_features else '3'}] Cleaning data…")
    df = clean(df)
    print(f"      → {len(df):,} rows after cleaning.")

    if with_features:
        print("[4/4] Engineering features…")
        df = engineer_features(df)
        print(f"      → {len(df.columns)} columns after feature engineering.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start
    print(f"\n✔ Saved {len(df):,} rows to {output_path}  ({elapsed:.1f}s)")


def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess Cricsheet ball-by-ball CSV data.",
    )
    parser.add_argument(
        "--input",
        default="data/merged.csv",
        help="Path to input CSV (default: data/merged.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/preprocessed.csv",
        help="Path to output CSV (default: data/preprocessed.csv)",
    )
    parser.add_argument(
        "--features",
        action="store_true",
        help="Also run the feature-engineering stage.",
    )

    args = parser.parse_args()
    run_pipeline(args.input, args.output, with_features=args.features)


if __name__ == "__main__":
    main()

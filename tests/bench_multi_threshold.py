#!/usr/bin/env python3
"""
Benchmarks for multi-threshold cubical zigzag persistence.
==========================================================

Run with::

    python3 tests/bench_multi_threshold.py

Outputs a CSV summary to stdout and optionally to a file
(``--output bench_results.csv``).

Benchmarks
----------
1. **precompute_vs_naive** — amortised precomputation vs. per-threshold
   cell extraction.
2. **scaling_thresholds** — wall time vs. number of thresholds (fixed grid).
3. **scaling_frames** — wall time vs. number of frames (fixed grid, thresholds).
4. **2d_vs_3d** — compare 2-D and 3-D grids of similar cell count.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from typing import List, Tuple

import numpy as np

# ── ensure zztop importable when run from repo root ──
sys.path.insert(0, ".")

from zztop.builders.cubical_builder import (
    _grid_cells_direct,
    _precompute_cell_filtrations_direct,
    precompute_cell_filtrations,
    threshold_cells,
)
from zztop.builders.multi_threshold import run_cubical_zigzag_multi_threshold


# ======================================================================
# Timing helper
# ======================================================================

def _timeit(fn, *args, repeats=3, **kwargs):
    """Return (mean_seconds, std_seconds) over *repeats* runs."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


# ======================================================================
# Benchmarks
# ======================================================================

def bench_precompute_vs_naive(
    shape: Tuple[int, ...] = (10, 10),
    n_thresholds: int = 20,
    seed: int = 0,
) -> List[dict]:
    """Compare precompute+threshold against per-threshold naive extraction."""
    rng = np.random.default_rng(seed)
    frame = rng.standard_normal(shape)

    thresholds = np.linspace(-2, 2, n_thresholds)

    # Naive: call _grid_cells_direct once per threshold
    def naive():
        for a in thresholds:
            _grid_cells_direct(frame, a)

    # Precompute: one precompute, then threshold
    def precompute():
        filt = _precompute_cell_filtrations_direct(frame)
        for a in thresholds:
            threshold_cells(filt, a)

    t_naive, s_naive = _timeit(naive, repeats=3)
    t_pre, s_pre = _timeit(precompute, repeats=3)

    return [
        {
            "benchmark": "precompute_vs_naive",
            "variant": "naive",
            "shape": str(shape),
            "n_thresholds": n_thresholds,
            "mean_s": f"{t_naive:.4f}",
            "std_s": f"{s_naive:.4f}",
        },
        {
            "benchmark": "precompute_vs_naive",
            "variant": "precompute",
            "shape": str(shape),
            "n_thresholds": n_thresholds,
            "mean_s": f"{t_pre:.4f}",
            "std_s": f"{s_pre:.4f}",
        },
    ]


def bench_scaling_thresholds(
    shape: Tuple[int, ...] = (8, 8),
    n_frames: int = 5,
    threshold_counts: Tuple[int, ...] = (1, 5, 10, 20, 50),
    seed: int = 1,
) -> List[dict]:
    """Wall time vs. number of thresholds."""
    rng = np.random.default_rng(seed)
    grid = rng.standard_normal(shape + (n_frames,))
    rows = []
    for m in threshold_counts:
        def run(m=m):
            run_cubical_zigzag_multi_threshold(
                grid, thresholds=m, backend="python",
                n_jobs=1, use_gudhi=False,
            )
        t, s = _timeit(run, repeats=2)
        rows.append({
            "benchmark": "scaling_thresholds",
            "shape": str(shape),
            "n_frames": n_frames,
            "n_thresholds": m,
            "mean_s": f"{t:.4f}",
            "std_s": f"{s:.4f}",
        })
    return rows


def bench_scaling_frames(
    shape: Tuple[int, ...] = (8, 8),
    n_thresholds: int = 5,
    frame_counts: Tuple[int, ...] = (2, 5, 10, 20),
    seed: int = 2,
) -> List[dict]:
    """Wall time vs. number of frames."""
    rng = np.random.default_rng(seed)
    max_frames = max(frame_counts)
    grid_full = rng.standard_normal(shape + (max_frames,))
    rows = []
    for T in frame_counts:
        slicing = tuple([slice(None)] * len(shape) + [slice(T)])
        grid_sub = grid_full[slicing]

        def run(g=grid_sub):
            run_cubical_zigzag_multi_threshold(
                g, thresholds=n_thresholds, backend="python",
                n_jobs=1, use_gudhi=False,
            )
        t, s = _timeit(run, repeats=2)
        rows.append({
            "benchmark": "scaling_frames",
            "shape": str(shape),
            "n_frames": T,
            "n_thresholds": n_thresholds,
            "mean_s": f"{t:.4f}",
            "std_s": f"{s:.4f}",
        })
    return rows


def bench_2d_vs_3d(seed: int = 3) -> List[dict]:
    """Compare similar-cell-count 2-D vs 3-D grids."""
    rng = np.random.default_rng(seed)
    n_frames = 5
    n_thresholds = 5

    # 2D: 15×15 → 225 vertices
    grid_2d = rng.standard_normal((15, 15, n_frames))
    # 3D: 6×6×6 → 216 vertices (similar)
    grid_3d = rng.standard_normal((6, 6, 6, n_frames))

    rows = []
    for label, grid in [("2d_15x15", grid_2d), ("3d_6x6x6", grid_3d)]:
        def run(g=grid):
            run_cubical_zigzag_multi_threshold(
                g, thresholds=n_thresholds, backend="python",
                n_jobs=1, use_gudhi=False,
            )
        t, s = _timeit(run, repeats=2)
        rows.append({
            "benchmark": "2d_vs_3d",
            "variant": label,
            "n_frames": n_frames,
            "n_thresholds": n_thresholds,
            "mean_s": f"{t:.4f}",
            "std_s": f"{s:.4f}",
        })
    return rows


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-threshold zigzag benchmarks")
    parser.add_argument("--output", "-o", default=None, help="CSV output file")
    args = parser.parse_args()

    print("Running multi-threshold zigzag benchmarks…", file=sys.stderr)

    all_rows: List[dict] = []
    all_rows.extend(bench_precompute_vs_naive())
    all_rows.extend(bench_scaling_thresholds())
    all_rows.extend(bench_scaling_frames())
    all_rows.extend(bench_2d_vs_3d())

    # Determine all columns
    columns = []
    for r in all_rows:
        for k in r:
            if k not in columns:
                columns.append(k)

    # Write CSV
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    for r in all_rows:
        writer.writerow(r)

    csv_text = buf.getvalue()
    print(csv_text)

    if args.output:
        with open(args.output, "w") as f:
            f.write(csv_text)
        print(f"Results written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

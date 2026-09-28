"""
Threshold selection strategies for multi-threshold zigzag persistence.
=====================================================================

Given precomputed cell filtration values across multiple frames, choose
a set of threshold levels that span the topologically interesting range
of the data.  This avoids hand-tuning and adapts to the empirical
distribution of cell filtration values.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np

# Type alias — identical to what cubical_builder uses
CellFiltrations = Dict  # CubicalCell → float, but kept generic here


# ======================================================================
# Public API
# ======================================================================

def select_thresholds(
    cell_filtrations_per_frame: Sequence[CellFiltrations],
    n_thresholds: int = 20,
    strategy: str = "percentile",
    *,
    thresholds: Optional[np.ndarray] = None,
    percentile_lo: float = 2.0,
    percentile_hi: float = 98.0,
) -> np.ndarray:
    """Choose threshold values from cell filtration data.

    Parameters
    ----------
    cell_filtrations_per_frame : sequence of dict
        One dict per frame, mapping cells to their filtration values.
        Typically produced by :func:`precompute_cell_filtrations`.
    n_thresholds : int
        Number of thresholds to return (ignored for ``strategy='custom'``).
    strategy : str
        Selection strategy:

        ``'percentile'``
            Evenly spaced in percentile space (default).  Concentrates
            thresholds where cell values are dense.
        ``'uniform'``
            Evenly spaced in value space between the global min and max.
        ``'histogram'``
            Place thresholds at bin edges of an equal-count histogram.
        ``'custom'``
            Pass through the *thresholds* keyword directly.
    thresholds : ndarray, optional
        Required when ``strategy='custom'``.  Returned as-is.
    percentile_lo, percentile_hi : float
        Lower/upper percentile bounds for the ``'percentile'`` strategy
        (default 2–98, avoiding extreme tails).

    Returns
    -------
    thresholds : ndarray of shape ``(n_thresholds,)``
        Sorted threshold values.

    Raises
    ------
    ValueError
        If *strategy* is unknown or ``'custom'`` without *thresholds*.
    """
    if strategy == "custom":
        if thresholds is None:
            raise ValueError("strategy='custom' requires the `thresholds` kwarg.")
        return np.sort(np.asarray(thresholds, dtype=np.float64))

    values = _pool_finite_values(cell_filtrations_per_frame)
    if len(values) == 0:
        return np.zeros(n_thresholds, dtype=np.float64)

    if strategy == "percentile":
        return _thresholds_percentile(
            values, n_thresholds, percentile_lo, percentile_hi,
        )
    elif strategy == "uniform":
        return _thresholds_uniform(values, n_thresholds)
    elif strategy == "histogram":
        return _thresholds_histogram(values, n_thresholds)
    else:
        raise ValueError(
            f"Unknown threshold strategy {strategy!r}. "
            "Choose from 'percentile', 'uniform', 'histogram', 'custom'."
        )


# ======================================================================
# Individual strategies
# ======================================================================

def _pool_finite_values(
    cell_filtrations_per_frame: Sequence[CellFiltrations],
) -> np.ndarray:
    """Collect all finite cell filtration values across frames."""
    vals: List[float] = []
    for frame_dict in cell_filtrations_per_frame:
        for v in frame_dict.values():
            if np.isfinite(v):
                vals.append(v)
    arr = np.array(vals, dtype=np.float64)
    return arr


def _thresholds_percentile(
    values: np.ndarray,
    n: int,
    lo: float = 2.0,
    hi: float = 98.0,
) -> np.ndarray:
    """Thresholds at evenly-spaced percentiles."""
    pcts = np.linspace(lo, hi, n)
    return np.unique(np.percentile(values, pcts))


def _thresholds_uniform(values: np.ndarray, n: int) -> np.ndarray:
    """Thresholds uniformly spaced between min and max."""
    vmin, vmax = values.min(), values.max()
    if vmin == vmax:
        return np.array([vmin], dtype=np.float64)
    return np.linspace(vmin, vmax, n)


def _thresholds_histogram(values: np.ndarray, n: int) -> np.ndarray:
    """Thresholds at equal-count histogram bin edges."""
    # Use n+1 quantiles to get n bins; edges are the thresholds.
    pcts = np.linspace(0, 100, n + 1)
    edges = np.unique(np.percentile(values, pcts))
    return edges

"""
Multi-threshold cubical zigzag persistence.
===========================================

This module computes zigzag persistence **at multiple threshold levels**
simultaneously for time-varying grid data.  The core idea:

1. **Precompute** cell filtration values once per frame (expensive).
2. **Threshold cheaply** for every level by a simple dict-comprehension.
3. **Run zigzag** at each threshold, parallelised over thresholds with
   `joblib`.

The result is a :class:`MultiThresholdResult` that stores one zigzag
barcode per threshold and provides convenience accessors for the Betti
surface :math:`\\beta_k(a, t)`, per-threshold diagrams, etc.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from zztop.complex.cubical import CubicalCell
from zztop.zigzag.engine import ZigzagEngine
from zztop.zigzag.filtration import ZigzagFiltration

from zztop.builders.cubical_builder import (
    _cubical_boundary,
    _cubical_dim,
    precompute_cell_filtrations,
    threshold_cells,
)
from zztop.builders.threshold import select_thresholds

logger = logging.getLogger(__name__)


# ======================================================================
# Result container
# ======================================================================

class MultiThresholdResult:
    """Container for multi-threshold zigzag persistence results.

    Stores one zigzag barcode per threshold and exposes convenience
    methods for downstream analysis and vectorisation.

    Parameters
    ----------
    thresholds : ndarray of shape ``(m,)``
        Sorted threshold values.
    bars : dict of float → list of (dim, birth_frame, death_frame)
        Zigzag barcode for each threshold.
    grid_shape : tuple of int
        Spatial dimensions of the input grid (excluding the frame axis).
    n_frames : int
        Number of time frames.

    Attributes
    ----------
    thresholds : ndarray
    bars : dict
    grid_shape : tuple
    n_frames : int
    """

    def __init__(
        self,
        thresholds: np.ndarray,
        bars: Dict[float, List[Tuple[int, int, int]]],
        grid_shape: Tuple[int, ...],
        n_frames: int,
    ) -> None:
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.bars = bars
        self.grid_shape = grid_shape
        self.n_frames = n_frames

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def threshold_slice(
        self, threshold: float,
    ) -> List[Tuple[int, int, int]]:
        """Return the barcode at the given *threshold*.

        Parameters
        ----------
        threshold : float
            Must be one of ``self.thresholds``.

        Returns
        -------
        list of (dim, birth_frame, death_frame)
        """
        key = float(threshold)
        if key not in self.bars:
            # Allow small floating-point mismatch
            idx = int(np.argmin(np.abs(self.thresholds - key)))
            key = float(self.thresholds[idx])
        return self.bars[key]

    def diagrams(
        self, threshold: float,
    ) -> Dict[int, np.ndarray]:
        """Per-dimension birth/death arrays at a single threshold.

        Returns
        -------
        dict of int → ndarray of shape ``(n_bars, 2)``
        """
        bars = self.threshold_slice(threshold)
        return _bars_to_diagram_dict(bars)

    def all_diagrams(self) -> Dict[float, Dict[int, np.ndarray]]:
        """Diagrams for every threshold.

        Returns
        -------
        dict of float → dict of int → ndarray
        """
        return {
            float(a): self.diagrams(a) for a in self.thresholds
        }

    def betti_surface(self, dim: int = 0) -> np.ndarray:
        r"""Compute the Betti surface :math:`\beta_k(a, t)`.

        For each threshold :math:`a_i` and frame :math:`t_j`, count the
        number of zigzag bars of homological dimension *dim* that are
        alive at frame :math:`t_j`.

        Parameters
        ----------
        dim : int
            Homological dimension (default 0).

        Returns
        -------
        surface : ndarray of shape ``(n_thresholds, n_frames)``
            ``surface[i, j]`` = :math:`\beta_k(a_i, t_j)`.
        """
        m = len(self.thresholds)
        T = self.n_frames
        surface = np.zeros((m, T), dtype=np.int64)
        for i, a in enumerate(self.thresholds):
            for d, b, de in self.bars[float(a)]:
                if d != dim:
                    continue
                for t in range(b, de):
                    if 0 <= t < T:
                        surface[i, t] += 1
        return surface

    def frame_betti(
        self, frame: int, dim: int = 0,
    ) -> np.ndarray:
        r"""Betti number as a function of threshold at a fixed frame.

        Parameters
        ----------
        frame : int
            0-based frame index.
        dim : int
            Homological dimension.

        Returns
        -------
        betti : ndarray of shape ``(n_thresholds,)``
        """
        return self.betti_surface(dim)[:, frame]

    def total_persistence(
        self, dim: int = 0,
    ) -> np.ndarray:
        """Total persistence at each threshold.

        Parameters
        ----------
        dim : int
            Homological dimension.

        Returns
        -------
        totals : ndarray of shape ``(n_thresholds,)``
        """
        out = np.zeros(len(self.thresholds), dtype=np.float64)
        for i, a in enumerate(self.thresholds):
            for d, b, de in self.bars[float(a)]:
                if d == dim and de > b:
                    out[i] += de - b
        return out

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.thresholds)

    def __iter__(self):
        for a in self.thresholds:
            yield float(a), self.bars[float(a)]

    def __repr__(self) -> str:
        return (
            f"MultiThresholdResult("
            f"n_thresholds={len(self.thresholds)}, "
            f"n_frames={self.n_frames}, "
            f"grid_shape={self.grid_shape})"
        )


# ======================================================================
# Internal helpers
# ======================================================================

def _bars_to_diagram_dict(
    bars: List[Tuple[int, int, int]],
) -> Dict[int, np.ndarray]:
    """Convert ``(dim, birth, death)`` tuples to per-dim arrays."""
    by_dim: Dict[int, list] = {}
    for dim, b, d in bars:
        by_dim.setdefault(dim, []).append([b, d])
    return {
        dim: np.array(pairs, dtype=np.float64).reshape(-1, 2)
        for dim, pairs in sorted(by_dim.items())
    }


def _run_single_zigzag(
    frames: List[Set[CubicalCell]],
    backend: str,
) -> List[Tuple[int, int, int]]:
    """Build filtration + run zigzag for one threshold.

    This is a module-level function (not a closure) so that it is
    picklable by ``joblib`` / ``multiprocessing``.
    """
    filt = ZigzagFiltration(
        frames=frames,
        cell_dim_fn=_cubical_dim,
        cell_boundary_fn=_cubical_boundary,
    )
    engine = ZigzagEngine(backend=backend)
    layer_bars = engine.run(filt)
    # layer → frame mapping (same as run_cubical_zigzag)
    return [(dim, b // 2, d // 2) for dim, b, d in layer_bars]


# ======================================================================
# Builder + runner
# ======================================================================

def build_cubical_zigzag_multi_threshold(
    grid_data: np.ndarray,
    thresholds: Optional[Union[np.ndarray, int]] = 20,
    threshold_strategy: str = "percentile",
    use_gudhi: bool = True,
    **threshold_kwargs,
) -> Tuple[List[List[Set[CubicalCell]]], np.ndarray]:
    """Build per-threshold frame lists from time-varying grid data.

    This performs the **expensive** step (cell enumeration + filtration
    value computation) once per frame, then derives active-cell sets for
    each threshold cheaply.

    Parameters
    ----------
    grid_data : ndarray of shape ``(N1, …, Nd, n_frames)``
        Grid values over time.  Last axis is the frame/time axis.
    thresholds : ndarray, int, or None
        If an int, auto-select that many thresholds via *threshold_strategy*.
        If an ndarray, use those values directly.
        If ``None``, equivalent to ``20``.
    threshold_strategy : str
        Strategy for :func:`~zztop.builders.threshold.select_thresholds`.
    use_gudhi : bool
        Use GUDHI for cell extraction (default True).
    **threshold_kwargs
        Extra keyword arguments forwarded to
        :func:`~zztop.builders.threshold.select_thresholds`
        (e.g. ``percentile_lo``, ``percentile_hi``).

    Returns
    -------
    all_frames : list of list of set of CubicalCell
        ``all_frames[i][t]`` is the set of active cells at threshold
        ``resolved_thresholds[i]`` and frame ``t``.
    resolved_thresholds : ndarray of shape ``(m,)``
    """
    ndim = grid_data.ndim - 1
    n_frames = grid_data.shape[-1]

    # 1. Precompute cell filtrations for every frame (expensive, once)
    logger.info(
        "Precomputing cell filtrations for %d frames (use_gudhi=%s)…",
        n_frames, use_gudhi,
    )
    cached_filts: List[Dict[CubicalCell, float]] = []
    for t in range(n_frames):
        slicing = tuple([slice(None)] * ndim + [t])
        frame_grid = grid_data[slicing]
        cached_filts.append(precompute_cell_filtrations(frame_grid, use_gudhi))

    # 2. Resolve thresholds
    if thresholds is None:
        thresholds = 20
    if isinstance(thresholds, (int, np.integer)):
        resolved = select_thresholds(
            cached_filts,
            n_thresholds=int(thresholds),
            strategy=threshold_strategy,
            **threshold_kwargs,
        )
    else:
        resolved = select_thresholds(
            cached_filts,
            strategy="custom",
            thresholds=np.asarray(thresholds, dtype=np.float64),
        )
    logger.info("Using %d thresholds in [%.4g, %.4g].",
                len(resolved), resolved[0], resolved[-1])

    # 3. Build active-cell frames for every threshold (cheap)
    all_frames: List[List[Set[CubicalCell]]] = []
    for a in resolved:
        frames_a: List[Set[CubicalCell]] = [
            threshold_cells(cached_filts[t], float(a))
            for t in range(n_frames)
        ]
        all_frames.append(frames_a)

    return all_frames, resolved


def run_cubical_zigzag_multi_threshold(
    grid_data: np.ndarray,
    thresholds: Optional[Union[np.ndarray, int]] = 20,
    threshold_strategy: str = "percentile",
    use_gudhi: bool = True,
    backend: str = "auto",
    n_jobs: int = -1,
    verbose: int = 0,
    **threshold_kwargs,
) -> MultiThresholdResult:
    """Compute zigzag persistence at multiple threshold levels.

    This is the main entry point for the multi-threshold zigzag
    pipeline.  It:

    1. Precomputes cell filtration values once per frame.
    2. Selects thresholds adaptively (or from user-supplied values).
    3. Runs zigzag persistence in parallel across thresholds (joblib).

    Parameters
    ----------
    grid_data : ndarray of shape ``(N1, …, Nd, n_frames)``
        Grid values over time.  Last axis is the frame/time axis.
    thresholds : ndarray, int, or None
        If an int, auto-select that many thresholds.
        If an ndarray, use those values directly.
    threshold_strategy : str
        ``'percentile'`` (default), ``'uniform'``, ``'histogram'``,
        or ``'custom'``.
    use_gudhi : bool
        Use GUDHI for cell extraction (default True).
    backend : str
        Zigzag engine backend: ``'auto'`` or ``'python'`` (the same engine).
    n_jobs : int
        Number of parallel jobs for zigzag computation.
        ``-1`` = all cores (default).  ``1`` = sequential.
    verbose : int
        Verbosity level for joblib.
    **threshold_kwargs
        Forwarded to :func:`~zztop.builders.threshold.select_thresholds`.

    Returns
    -------
    MultiThresholdResult
        Object containing one barcode per threshold plus convenience
        accessors (Betti surface, diagrams, etc.).

    Examples
    --------
    >>> import numpy as np
    >>> from zztop import run_cubical_zigzag_multi_threshold
    >>> data = np.random.default_rng(42).standard_normal((8, 8, 10))
    >>> result = run_cubical_zigzag_multi_threshold(
    ...     data, thresholds=5, backend="python", n_jobs=1,
    ... )
    >>> result
    MultiThresholdResult(n_thresholds=5, n_frames=10, grid_shape=(8, 8))
    >>> surface = result.betti_surface(dim=0)  # shape (5, 10)
    """
    ndim = grid_data.ndim - 1
    n_frames = grid_data.shape[-1]
    grid_shape = grid_data.shape[:ndim]

    all_frames, resolved = build_cubical_zigzag_multi_threshold(
        grid_data,
        thresholds=thresholds,
        threshold_strategy=threshold_strategy,
        use_gudhi=use_gudhi,
        **threshold_kwargs,
    )

    # Run zigzag at each threshold (embarrassingly parallel)
    m = len(resolved)
    if m == 0:
        return MultiThresholdResult(
            thresholds=resolved,
            bars={},
            grid_shape=grid_shape,
            n_frames=n_frames,
        )

    if n_jobs == 1 or m == 1:
        # Sequential path — avoids joblib overhead for small workloads
        all_bars = [
            _run_single_zigzag(all_frames[i], backend)
            for i in range(m)
        ]
    else:
        from joblib import Parallel, delayed
        all_bars = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_run_single_zigzag)(all_frames[i], backend)
            for i in range(m)
        )

    bars_dict: Dict[float, List[Tuple[int, int, int]]] = {
        float(resolved[i]): all_bars[i] for i in range(m)
    }

    return MultiThresholdResult(
        thresholds=resolved,
        bars=bars_dict,
        grid_shape=grid_shape,
        n_frames=n_frames,
    )

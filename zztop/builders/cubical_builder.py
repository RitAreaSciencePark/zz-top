"""
High-level builder for **cubical** zigzag persistence.

Replaces ``zigzag_from_cubical_complex()`` from the original ``zigzag.py``,
but operates **directly on cubical cells** instead of converting to
simplicial complexes.

Given an *n*-D grid of values evolving over frames, this module:

1. For each frame, determines which cubical cells are *active* (filtration
   value below a threshold) using GUDHI's ``CubicalComplex``.
2. Builds the closure of active cells as :class:`CubicalCell` objects.
3. Constructs a :class:`~zztop.zigzag.filtration.ZigzagFiltration`.
4. Runs the zigzag engine and returns the barcode.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from zztop.complex.cubical import CubicalCell
from zztop.zigzag.filtration import ZigzagFiltration
from zztop.zigzag.engine import ZigzagEngine


# ======================================================================
# Module-level callables (picklable, for joblib serialization)
# ======================================================================

def _cubical_dim(cell: CubicalCell) -> int:
    """Return the dimension of a cubical cell (picklable)."""
    return cell.dim


def _cubical_boundary(cell: CubicalCell) -> list:
    """Return the mod-2 boundary of a cubical cell (picklable)."""
    return cell.boundary()


# ======================================================================
# Grid → cubical cells  (precomputation + thresholding)
# ======================================================================

def _cell_corner_values(grid: np.ndarray, intervals) -> List[float]:
    """Get the grid values at all corner vertices of a cubical cell."""
    ndim = len(intervals)
    corners = []
    nd_axes = [i for i, (lo, hi) in enumerate(intervals) if lo != hi]
    n_nd = len(nd_axes)
    for bits in range(1 << n_nd):
        idx = []
        j = 0
        for i, (lo, hi) in enumerate(intervals):
            if lo != hi:
                idx.append(hi if (bits & (1 << j)) else lo)
                j += 1
            else:
                idx.append(lo)
        corners.append(grid[tuple(idx)])
    return corners


# -- Precompute cell filtration values ------------------------------------

def _precompute_cell_filtrations_gudhi(
    grid_frame: np.ndarray,
) -> Dict[CubicalCell, float]:
    """Precompute filtration values for every cell using GUDHI.

    Calls ``gd.CubicalComplex(vertices=-grid_frame).all_cells()`` **once**
    and parses the expanded array to produce a mapping from every
    :class:`CubicalCell` to its filtration value.

    Vertices are assigned ``-inf`` so they are always active at any
    finite threshold.

    Parameters
    ----------
    grid_frame : ndarray of shape (N1, …, Nd)
        Grid values for a single frame.

    Returns
    -------
    cell_filtrations : dict of CubicalCell → float
    """
    import gudhi as gd

    cc = gd.CubicalComplex(vertices=-grid_frame)
    all_cells = cc.all_cells()
    expanded_shape = all_cells.shape  # (2*N1-1, 2*N2-1, …)

    cell_filt: Dict[CubicalCell, float] = {}
    for idx in np.ndindex(*expanded_shape):
        value = float(all_cells[idx])
        intervals = []
        cell_dim = 0
        for axis, coord in enumerate(idx):
            grid_coord = coord // 2
            if coord % 2 == 0:
                intervals.append((grid_coord, grid_coord))
            else:
                intervals.append((grid_coord, grid_coord + 1))
                cell_dim += 1
        cell = CubicalCell(intervals)
        if cell_dim == 0:
            cell_filt[cell] = -np.inf  # vertices always active
        else:
            cell_filt[cell] = value
    return cell_filt


def _precompute_cell_filtrations_direct(
    grid_frame: np.ndarray,
) -> Dict[CubicalCell, float]:
    """Precompute filtration values for every cell — pure Python.

    Enumerates all cubical cells from dimensions 0 to *ndim* and
    computes the lower-star filtration value for each
    (``-min(corner values)`` = ``max(-grid[corners])``).

    Vertices are assigned ``-inf``.

    Parameters
    ----------
    grid_frame : ndarray
        Grid values for a single frame.

    Returns
    -------
    cell_filtrations : dict of CubicalCell → float
    """
    ndim = grid_frame.ndim
    shape = grid_frame.shape
    cell_filt: Dict[CubicalCell, float] = {}

    # Vertices (dimension 0) — always active
    for idx in np.ndindex(*shape):
        cell = CubicalCell(tuple((c, c) for c in idx))
        cell_filt[cell] = -np.inf

    # Higher-dimensional cells
    for k in range(1, ndim + 1):
        for nd_axes in combinations(range(ndim), k):
            ranges_list = []
            for axis in range(ndim):
                if axis in nd_axes:
                    ranges_list.append(range(shape[axis] - 1))
                else:
                    ranges_list.append(range(shape[axis]))

            for pos in np.ndindex(*[len(r) for r in ranges_list]):
                coords = [list(ranges_list[a])[pos[a]] for a in range(ndim)]
                intervals = []
                for axis in range(ndim):
                    c = coords[axis]
                    if axis in nd_axes:
                        intervals.append((c, c + 1))
                    else:
                        intervals.append((c, c))

                corners = _cell_corner_values(grid_frame, intervals)
                cell_value = -np.min(corners) if len(corners) > 0 else 0.0
                cell = CubicalCell(intervals)
                cell_filt[cell] = cell_value

    return cell_filt


def precompute_cell_filtrations(
    grid_frame: np.ndarray,
    use_gudhi: bool = True,
) -> Dict[CubicalCell, float]:
    """Precompute filtration values for all cubical cells in a frame.

    This is the amortisation primitive for multi-threshold zigzag:
    call once per frame, then use :func:`threshold_cells` cheaply for
    any number of thresholds.

    Parameters
    ----------
    grid_frame : ndarray
        Spatial grid for a single time frame.
    use_gudhi : bool
        If True use GUDHI; otherwise pure-Python.

    Returns
    -------
    dict of CubicalCell → float
    """
    if use_gudhi:
        return _precompute_cell_filtrations_gudhi(grid_frame)
    return _precompute_cell_filtrations_direct(grid_frame)


def threshold_cells(
    cell_filtrations: Dict[CubicalCell, float],
    threshold: float,
) -> Set[CubicalCell]:
    """Select active cells from precomputed filtration values.

    Under the lower-star convention (GUDHI default), the filtration
    value of a cell equals the max of its vertex filtration values.
    Because vertices are assigned ``-inf``, they are always included.
    A higher-dimensional cell is active when its value < *threshold*.

    Since cell filtration values are monotone w.r.t. the face relation
    (``filt(face) ≤ filt(coface)``), the resulting set is automatically
    closed under taking faces — **no explicit closure computation is
    needed**.

    Parameters
    ----------
    cell_filtrations : dict
        Mapping from :class:`CubicalCell` to its filtration value,
        as returned by :func:`precompute_cell_filtrations`.
    threshold : float
        Activation threshold.

    Returns
    -------
    active : set of CubicalCell
    """
    return {c for c, v in cell_filtrations.items() if v < threshold}


# -- Legacy wrappers (delegate to precompute → threshold) -----------------

def _grid_cells_from_gudhi(grid_frame: np.ndarray, threshold: float):
    """Extract active cubical cells using GUDHI.

    Delegates to :func:`precompute_cell_filtrations` +
    :func:`threshold_cells` for backward-compatibility.
    """
    filt = _precompute_cell_filtrations_gudhi(grid_frame)
    return threshold_cells(filt, threshold)


def _grid_cells_direct(grid_frame: np.ndarray, threshold: float):
    """Extract active cubical cells — pure Python fallback.

    Delegates to :func:`precompute_cell_filtrations` +
    :func:`threshold_cells` for backward-compatibility.
    """
    filt = _precompute_cell_filtrations_direct(grid_frame)
    return threshold_cells(filt, threshold)


# ======================================================================
# Builder functions
# ======================================================================

def build_cubical_zigzag(
    grid_data: np.ndarray,
    threshold: float = 0.0,
    use_gudhi: bool = True,
) -> ZigzagFiltration:
    """Build a :class:`ZigzagFiltration` from time-varying grid data.

    Parameters
    ----------
    grid_data : ndarray of shape (N1, ..., Nd, n_frames)
        Grid values over time.  The **last** axis is the frame/time axis.
    threshold : float
        Cells with (negated) filtration value below this are active.
    use_gudhi : bool
        If True (default), use GUDHI ``CubicalComplex`` for cell extraction.
        If False, use a pure-Python fallback.

    Returns
    -------
    ZigzagFiltration
    """
    ndim = grid_data.ndim - 1  # spatial dimensions
    n_frames = grid_data.shape[-1]

    extract_fn = _grid_cells_from_gudhi if use_gudhi else _grid_cells_direct

    frames: List[Set[CubicalCell]] = []
    for t in range(n_frames):
        # Extract the spatial grid for this frame
        slicing = tuple([slice(None)] * ndim + [t])
        frame_grid = grid_data[slicing]
        active_cells = extract_fn(frame_grid, threshold)
        frames.append(active_cells)

    return ZigzagFiltration(
        frames=frames,
        cell_dim_fn=_cubical_dim,
        cell_boundary_fn=_cubical_boundary,
    )


def run_cubical_zigzag(
    grid_data: np.ndarray,
    threshold: float = 0.0,
    use_gudhi: bool = True,
    backend: str = "auto",
    output_file: Optional[str] = None,
) -> List[Tuple[int, int, int]]:
    """One-call convenience: build cubical filtration and compute barcode.

    Parameters
    ----------
    grid_data : ndarray of shape (N1, ..., Nd, n_frames)
        Grid values over time.
    threshold : float
        Activation threshold for cells.
    use_gudhi : bool
        Use GUDHI for cell extraction (the standard GUDHI release; the
        optional GPU extension is not needed here).
    backend : str
        Zigzag engine backend: ``'auto'`` or ``'python'`` (the same engine).
        (This selects the zigzag engine, not GUDHI's cell extraction.)
    output_file : str, optional
        CSV path to write the barcode to.

    Returns
    -------
    bars : list of (dimension, birth_frame, death_frame)
        The persistence barcode.  Birth/death are 0-based **frame**
        indices (i.e. indices into the last axis of *grid_data*).
    """
    filt = build_cubical_zigzag(grid_data, threshold=threshold, use_gudhi=use_gudhi)
    engine = ZigzagEngine(backend=backend)
    # engine.run returns 1-based layer indices
    layer_bars = engine.run(filt, output_file=output_file)

    # Convert 1-based zigzag-step indices → 0-based frame indices.
    # Steps: K0(1), K0∩K1(2), K1(3), K1∩K2(4), ..., K_{N-1}(2N-1)
    # Following the convention in Gardinazzi et al. (2025),
    # intersection steps are shifted forward to the next frame:
    #   frame = step // 2   (with 1-based step indexing)
    # This gives: K0→0, K0∩K1→1, K1→1, K1∩K2→2, K2→2, ...
    frame_bars = [
        (dim, b // 2, d // 2) for dim, b, d in layer_bars
    ]
    return frame_bars

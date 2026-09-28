"""
Persistence diagram normalisation utilities.

Convert the heterogeneous output formats produced by various zztop builders
into a canonical representation that every vectorizer can consume.

Canonical format
----------------
``Dict[int, np.ndarray]``  —  mapping ``dim → (n_bars, 2)`` float64 array
where each row is ``(birth, death)``.  Infinite bars are **dropped** by
default (``drop_inf=True``).

Accepted inputs
~~~~~~~~~~~~~~~
* **Zigzag barcode** — ``List[Tuple[int, int, int]]`` with ``(dim, birth, death)``
  as produced by :func:`run_simplicial_zigzag` and :func:`run_cubical_zigzag`.
* **GPU per-frame dict** — ``Dict[int, np.ndarray]`` as produced by
  :func:`run_cubical_persistence_gpu` (a single frame's diagram).
* **GPU per-frame dict list** — ``List[Dict[int, np.ndarray]]``
  (batch of frames from GPU persistence).
* **Raw ndarray** — ``(n_bars, 3)`` with columns ``(dim, birth, death)``
  or ``(n_bars, 2)`` assumed to be a single-dimension diagram (dim 0).
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
#  Type aliases
# ---------------------------------------------------------------------------

ZigzagBarcode = List[Tuple[int, int, int]]
DiagramDict = Dict[int, np.ndarray]
DiagramInput = Union[
    ZigzagBarcode,
    DiagramDict,
    List[DiagramDict],
    np.ndarray,
    "Sequence[Tuple[int, float, float]]",
]


# ---------------------------------------------------------------------------
#  Public helpers
# ---------------------------------------------------------------------------

def normalize_diagram(
    diagram: DiagramInput,
    *,
    drop_inf: bool = True,
    dtype: type = np.float64,
) -> DiagramDict:
    """Convert *any* zztop diagram output to the canonical ``{dim: ndarray}`` form.

    Parameters
    ----------
    diagram : various
        See module docstring for accepted types.
    drop_inf : bool
        If ``True`` (default), remove bars whose death value is ``np.inf``.
    dtype : numpy dtype
        All coordinates are cast to this dtype.

    Returns
    -------
    dict mapping ``int → ndarray(n_bars, 2)``
    """
    if isinstance(diagram, np.ndarray):
        return _from_ndarray(diagram, drop_inf=drop_inf, dtype=dtype)
    if isinstance(diagram, dict):
        return _from_dict(diagram, drop_inf=drop_inf, dtype=dtype)
    if isinstance(diagram, (list, tuple)):
        if len(diagram) == 0:
            return {}
        first = diagram[0]
        # list of dicts → pick first frame (caller should normalize per-frame)
        if isinstance(first, dict):
            return _from_dict(first, drop_inf=drop_inf, dtype=dtype)
        # list of tuples → zigzag barcode
        if isinstance(first, (list, tuple)) and len(first) >= 3:
            return _from_zigzag(diagram, drop_inf=drop_inf, dtype=dtype)  # type: ignore[arg-type]
    raise TypeError(
        f"Cannot normalise diagram of type {type(diagram).__name__}. "
        "Expected ndarray, dict, list-of-tuples, or list-of-dicts."
    )


def normalize_diagrams(
    diagrams: Union[DiagramInput, List[DiagramInput]],
    *,
    drop_inf: bool = True,
    dtype: type = np.float64,
) -> List[DiagramDict]:
    """Normalise a *batch* of diagrams.

    If *diagrams* is a ``List[Dict[int, ndarray]]`` (GPU batch result) each
    dict is normalised independently.  Otherwise the single input is wrapped
    in a length-1 list.

    Parameters
    ----------
    diagrams : one diagram or list thereof
    drop_inf, dtype : forwarded to :func:`normalize_diagram`.

    Returns
    -------
    list of ``{dim: ndarray(n_bars, 2)}``
    """
    # GPU batch output: List[Dict[int, ndarray]]
    if isinstance(diagrams, list) and len(diagrams) > 0 and isinstance(diagrams[0], dict):
        return [normalize_diagram(d, drop_inf=drop_inf, dtype=dtype) for d in diagrams]
    # Single diagram
    return [normalize_diagram(diagrams, drop_inf=drop_inf, dtype=dtype)]


# ---------------------------------------------------------------------------
#  Internal converters
# ---------------------------------------------------------------------------

def _from_zigzag(
    bars: ZigzagBarcode,
    *,
    drop_inf: bool,
    dtype: type,
) -> DiagramDict:
    """Convert ``[(dim, birth, death), ...]`` → ``{dim: ndarray}``."""
    from collections import defaultdict

    buckets: Dict[int, list] = defaultdict(list)
    for dim, b, d in bars:
        if drop_inf and (d == np.inf or d == float("inf")):
            continue
        buckets[int(dim)].append((float(b), float(d)))

    out: DiagramDict = {}
    for dim in sorted(buckets):
        arr = np.array(buckets[dim], dtype=dtype)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        out[dim] = arr
    return out


def _from_dict(
    d: DiagramDict,
    *,
    drop_inf: bool,
    dtype: type,
) -> DiagramDict:
    """Canonicalise a ``{dim: ndarray}`` dict (cast dtype, filter inf)."""
    out: DiagramDict = {}
    for dim, arr in d.items():
        arr = np.asarray(arr, dtype=dtype)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        if arr.size == 0:
            out[int(dim)] = arr.reshape(0, 2)
            continue
        if drop_inf:
            finite_mask = np.isfinite(arr[:, 1])
            arr = arr[finite_mask]
        out[int(dim)] = arr
    return out


def _from_ndarray(
    arr: np.ndarray,
    *,
    drop_inf: bool,
    dtype: type,
) -> DiagramDict:
    """Convert an ndarray diagram to canonical dict form.

    * ``(n, 3)`` → interpret columns as ``(dim, birth, death)``.
    * ``(n, 2)`` → assume single-dimension (dim=0) ``(birth, death)``.
    """
    arr = np.asarray(arr, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {arr.shape}")
    if arr.shape[1] == 3:
        return _from_zigzag(
            [(int(row[0]), row[1], row[2]) for row in arr],
            drop_inf=drop_inf,
            dtype=dtype,
        )
    if arr.shape[1] == 2:
        if drop_inf:
            arr = arr[np.isfinite(arr[:, 1])]
        return {0: arr}
    raise ValueError(f"Expected array with 2 or 3 columns, got {arr.shape[1]}")


def persistence_values(dgm: DiagramDict, dim: int) -> np.ndarray:
    """Return ``death - birth`` array for dimension *dim*."""
    if dim not in dgm:
        return np.empty(0, dtype=np.float64)
    bd = dgm[dim]
    return bd[:, 1] - bd[:, 0]


def birth_values(dgm: DiagramDict, dim: int) -> np.ndarray:
    if dim not in dgm:
        return np.empty(0, dtype=np.float64)
    return dgm[dim][:, 0]


def death_values(dgm: DiagramDict, dim: int) -> np.ndarray:
    if dim not in dgm:
        return np.empty(0, dtype=np.float64)
    return dgm[dim][:, 1]


def midlife_values(dgm: DiagramDict, dim: int) -> np.ndarray:
    """``(birth + death) / 2``."""
    if dim not in dgm:
        return np.empty(0, dtype=np.float64)
    bd = dgm[dim]
    return (bd[:, 0] + bd[:, 1]) / 2

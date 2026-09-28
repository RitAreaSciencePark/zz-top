"""
High-level builder for **mesh** zigzag persistence.

Tracks the zigzag persistent homology of a scalar field that evolves in time
on a **fixed** triangulated surface mesh (e.g. a FreeSurfer cortical surface).
Each mesh vertex carries a time series of values (neural activity); at each
frame we materialise the sub-complex of the mesh spanned by the *co-active*
region and follow how its topology reorganises over time.

Construction (edge co-activity, instantaneous endpoint model)
-------------------------------------------------------------
Let ``M`` be the fixed ambient complex — the mesh vertices, its unique
undirected edges, and its triangular faces.  For a frame ``t`` with per-vertex
values ``x[:, t]``:

* an **edge** ``(i, j)`` is present iff ``reduce(x[i, t], x[j, t]) > threshold``
  (``reduce`` defaults to ``min``);
* a **triangle** ``(i, j, k)`` is present iff all three of its edges are
  present;
* a **vertex** ``v`` is present iff ``x[v, t] > threshold`` *or* ``v`` is an
  endpoint of a present edge.

The vertex rule keeps every frame a **closed** complex for *any* reducer, and
for ``reduce=min`` it reproduces exactly the excursion-set (superlevel)
induced sub-complex of the mesh on ``{v : x[v, t] > threshold}`` — isolated
supra-threshold vertices included.  Set ``include_isolated_vertices=False`` to
drop the "own value" clause and keep only edge-incident vertices (a pure
edge-cut complex).

Temporal hysteresis (optional)
------------------------------
Passing ``threshold_high`` turns the single cut into a **two-threshold
(Schmitt-trigger) gate along time**: a cell switches on when its (reduced)
signal first exceeds ``threshold_high`` and stays on until the signal falls
to/below ``threshold`` (now the low/hold level).  This suppresses the
frame-to-frame flicker of cells whose value dithers around a single threshold —
the dominant source of spurious single-frame features on smooth, oversampled
fields (e.g. HRF-smoothed fMRI) — and lengthens genuine feature lifetimes
without lowering the entry bar.  ``threshold_high=None`` (default) is the
original single-threshold behaviour.  Frame closure is preserved for any
reducer (edge-incident vertices are always included).

Each frame is a ``Set[Simplex]``; the list of frames is handed **directly** to
:class:`~zztop.zigzag.filtration.ZigzagFiltration`, which builds the
intersection layers and the insertion/deletion stream itself.

See ``docs/mesh_zigzag_design.md`` for the full mathematical justification.

Note
----
Frames scale as ``O(mesh_size × n_frames)`` in both time and memory (every
active vertex/edge/face is materialised as a :class:`Simplex` for every
frame).  Restricting the mesh with ``vertex_mask`` (e.g. to a single ROI or a
hemisphere without the medial wall) is the intended lever for keeping the
problem tractable on large cortical surfaces.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Set, Tuple, Union

import numpy as np

from zztop.complex.simplicial import Simplex
from zztop.zigzag.filtration import ZigzagFiltration
from zztop.zigzag.engine import ZigzagEngine


# ======================================================================
# Module-level callables (picklable, for joblib serialization)
# ======================================================================

def _simplex_dim(s: Simplex) -> int:
    """Return the dimension of a simplex (picklable)."""
    return s.dimension


def _simplex_boundary(s: Simplex) -> List[Simplex]:
    """Return the mod-2 boundary (codimension-1 faces) of a simplex."""
    return s.faces()


# ======================================================================
# Co-activity reducers
# ======================================================================

Reducer = Callable[[np.ndarray, np.ndarray], np.ndarray]

_COACTIVITY_REDUCERS: dict = {
    "min": np.minimum,
    "max": np.maximum,
    "mean": lambda a, b: 0.5 * (a + b),
    "product": lambda a, b: a * b,
}


def _resolve_coactivity(coactivity: Union[str, Reducer]) -> Reducer:
    """Resolve *coactivity* to an elementwise reducer ``(a, b) -> ndarray``.

    Parameters
    ----------
    coactivity : str or callable
        One of ``"min"``, ``"mean"``, ``"product"``, ``"max"``, or an
        arbitrary callable applied elementwise to two endpoint-value arrays.

    Returns
    -------
    reducer : callable
    """
    if callable(coactivity):
        return coactivity
    try:
        return _COACTIVITY_REDUCERS[coactivity]
    except KeyError:
        raise ValueError(
            f"Unknown coactivity reducer {coactivity!r}; choose from "
            f"{sorted(_COACTIVITY_REDUCERS)} or pass a callable(a, b) -> ndarray"
        ) from None


# ======================================================================
# Mesh geometry → unique edges
# ======================================================================

def _unique_edges(
    faces: np.ndarray, n_vertices: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive the unique undirected edge set of a triangulated mesh.

    Parameters
    ----------
    faces : ndarray of shape (F, 3)
        Triangle vertex indices.
    n_vertices : int
        Number of mesh vertices (used as the encoding base; must exceed the
        maximum vertex index in *faces*).

    Returns
    -------
    edges : ndarray of shape (E, 2)
        Sorted (``i < j``) unique undirected edges.
    face_edges : ndarray of shape (F, 3)
        For each face, the row indices into *edges* of its three edges
        ``(v0, v1)``, ``(v1, v2)``, ``(v0, v2)``.
    """
    faces = np.asarray(faces, dtype=np.int64)
    F = faces.shape[0]
    if F == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, 3), dtype=np.int64),
        )

    # Stack the three edges of every face: (v0,v1), (v1,v2), (v0,v2).
    stacked = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], axis=0
    )
    stacked.sort(axis=1)  # undirected: smaller index first

    # Encode each undirected edge as a single integer key for a robust,
    # version-independent unique/inverse (avoids np.unique axis quirks).
    base = np.int64(n_vertices)
    keys = stacked[:, 0].astype(np.int64) * base + stacked[:, 1].astype(np.int64)
    uniq_keys, inverse = np.unique(keys, return_inverse=True)

    edges = np.stack([uniq_keys // base, uniq_keys % base], axis=1).astype(np.int64)
    face_edges = np.asarray(inverse).reshape(3, F).T  # (F, 3)
    return edges, face_edges


# ======================================================================
# Temporal hysteresis (two-threshold presence gate)
# ======================================================================

def _hysteresis_gate(
    signal: np.ndarray, low: float, high: float, *, strict: bool = True
) -> np.ndarray:
    """Two-threshold (hysteresis) temporal gate over the time axis.

    For each row (cell) the boolean on-state switches *on* when the signal
    first exceeds ``high`` and stays on until the signal falls to/below
    ``low``::

        on[:, 0] = signal[:, 0] > high
        on[:, t] = (signal[:, t] > high) | (on[:, t-1] & (signal[:, t] > low))

    A Schmitt-trigger / Canny-style double threshold applied along time: it
    seeds each active run at the strict ``high`` level and holds it through the
    looser ``low`` level, suppressing frame-to-frame flicker of cells whose
    value dithers around a single cut.  With ``strict=False`` the comparisons
    use ``>=``.  Requires ``high >= low``.

    Parameters
    ----------
    signal : ndarray of shape (N, T)
        Per-cell time series (edge reduced-values or vertex own-values).
    low, high : float
        Hold and enter thresholds; ``high >= low``.
    strict : bool, optional
        Strict ``>`` (default) or non-strict ``>=`` comparisons.

    Returns
    -------
    on : ndarray of bool, shape (N, T)
    """
    if signal.shape[1] == 0:
        return np.empty(signal.shape, dtype=bool)
    hi = (signal > high) if strict else (signal >= high)
    lo = (signal > low) if strict else (signal >= low)
    on = np.empty(signal.shape, dtype=bool)
    prev = hi[:, 0].copy()
    on[:, 0] = prev
    for t in range(1, signal.shape[1]):
        prev = hi[:, t] | (prev & lo[:, t])
        on[:, t] = prev
    return on


# ======================================================================
# Frame materialisation
# ======================================================================

def _mesh_frames(
    faces,
    activity,
    threshold: float,
    *,
    threshold_high: Optional[float] = None,
    coactivity: Union[str, Reducer] = "min",
    include_isolated_vertices: bool = True,
    vertex_mask=None,
    strict: bool = True,
) -> List[Set[Simplex]]:
    """Materialise the per-frame co-activity sub-complexes as simplex sets.

    See :func:`build_mesh_zigzag` for parameter semantics.  This helper does
    the vectorised work and returns the raw list of frames; the builder wraps
    them in a :class:`ZigzagFiltration`.

    Returns
    -------
    frames : list of set of Simplex
        ``frames[t]`` is the closed simplicial complex active at frame *t*.
    """
    faces = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
    activity = np.asarray(activity, dtype=float)

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(
            f"faces must have shape (F, 3); got {faces.shape}"
        )
    if activity.ndim != 2:
        raise ValueError(
            f"activity must have shape (V, T); got {activity.shape}"
        )

    V, T = activity.shape
    if faces.size:
        if int(faces.max()) >= V:
            raise ValueError(
                f"max vertex index in faces ({int(faces.max())}) must be < "
                f"activity.shape[0] ({V})"
            )
        if int(faces.min()) < 0:
            raise ValueError("faces contain negative vertex indices")

    if threshold_high is not None and threshold_high < threshold:
        raise ValueError(
            f"threshold_high ({threshold_high}) must be >= threshold "
            f"({threshold}): 'threshold' is the low/hold level and "
            f"'threshold_high' the high/enter level of the hysteresis gate."
        )

    reducer = _resolve_coactivity(coactivity)

    # Optional vertex mask (True = keep the vertex; False = drop it and any
    # incident edge/face from the ambient mesh entirely).
    vkeep: Optional[np.ndarray] = None
    if vertex_mask is not None:
        vkeep = np.asarray(vertex_mask, dtype=bool).reshape(-1)
        if vkeep.shape[0] != V:
            raise ValueError(
                f"vertex_mask must have length V={V}; got {vkeep.shape[0]}"
            )

    edges, face_edges = _unique_edges(faces, V)
    Ei = edges[:, 0]
    Ej = edges[:, 1]

    # ---- edge presence: W = reduce(x[Ei, :], x[Ej, :]) computed once -------
    if edges.shape[0]:
        W = reducer(activity[Ei, :], activity[Ej, :])  # (E, T)
        W = np.asarray(W, dtype=float)
    else:
        W = np.empty((0, T), dtype=float)
    if threshold_high is None:
        edge_present = (W > threshold) if strict else (W >= threshold)
    else:
        edge_present = _hysteresis_gate(W, threshold, threshold_high, strict=strict)

    # Drop masked-out edges before deriving faces / vertex incidence.
    if vkeep is not None and edges.shape[0]:
        edge_keep = vkeep[Ei] & vkeep[Ej]
        edge_present = edge_present & edge_keep[:, None]

    # ---- triangle presence: boolean AND over the three face edges ---------
    if faces.shape[0]:
        tri_present = (
            edge_present[face_edges[:, 0], :]
            & edge_present[face_edges[:, 1], :]
            & edge_present[face_edges[:, 2], :]
        )
    else:
        tri_present = np.empty((0, T), dtype=bool)

    # ---- vertex presence ---------------------------------------------------
    vertex_incident = np.zeros((V, T), dtype=bool)
    if edges.shape[0]:
        np.logical_or.at(vertex_incident, Ei, edge_present)
        np.logical_or.at(vertex_incident, Ej, edge_present)

    if include_isolated_vertices:
        if threshold_high is None:
            vertex_own = (activity > threshold) if strict else (activity >= threshold)
        else:
            vertex_own = _hysteresis_gate(
                activity, threshold, threshold_high, strict=strict
            )
        vertex_present = vertex_own | vertex_incident
    else:
        vertex_present = vertex_incident
    if vkeep is not None:
        vertex_present = vertex_present & vkeep[:, None]

    # ---- assemble frames ---------------------------------------------------
    frames: List[Set[Simplex]] = []
    for t in range(T):
        frame: Set[Simplex] = set()
        for v in np.nonzero(vertex_present[:, t])[0]:
            frame.add(Simplex((int(v),)))
        for e in np.nonzero(edge_present[:, t])[0]:
            frame.add(Simplex((int(edges[e, 0]), int(edges[e, 1]))))
        for f in np.nonzero(tri_present[:, t])[0]:
            frame.add(
                Simplex(
                    (int(faces[f, 0]), int(faces[f, 1]), int(faces[f, 2]))
                )
            )
        frames.append(frame)
    return frames


# ======================================================================
# Builder functions
# ======================================================================

def build_mesh_zigzag(
    faces,
    activity,
    threshold: float,
    *,
    threshold_high: Optional[float] = None,
    coactivity: Union[str, Reducer] = "min",
    include_isolated_vertices: bool = True,
    vertex_mask=None,
    strict: bool = True,
) -> ZigzagFiltration:
    """Build a :class:`ZigzagFiltration` from a scalar field on a fixed mesh.

    A fixed triangulated surface mesh ``M`` (vertices, unique undirected edges,
    triangular faces) carries a per-vertex time series.  For each frame the
    co-active sub-complex of ``M`` is materialised (see the module docstring
    for the exact edge/triangle/vertex rules), and the resulting list of frames
    is passed directly to :class:`ZigzagFiltration`.

    Parameters
    ----------
    faces : (F, 3) int array-like
        Triangle vertex indices.  The unique undirected edge set is derived
        from these faces.  The maximum index must be ``< activity.shape[0]``.
    activity : (V, T) float array-like
        ``activity[v, t]`` is the value of vertex *v* at frame *t*.
    threshold : float
        Activation threshold.  An edge is present iff its reduced endpoint
        value is ``> threshold`` (when ``strict``) or ``>= threshold``
        (when ``strict=False``).  With ``threshold_high`` set this is instead
        the **low/hold** level of a hysteresis gate.
    threshold_high : float, optional
        Enables **temporal hysteresis** presence.  When given (must be
        ``>= threshold``), each cell's presence is gated by a Schmitt trigger
        along time: it switches on when the (reduced) signal first exceeds
        ``threshold_high`` and stays on until the signal falls to/below
        ``threshold`` (the hold level).  This suppresses frame-to-frame flicker
        of cells whose value dithers around a single cut, lengthening feature
        lifetimes without lowering the entry bar.  ``None`` (default) uses the
        original single-threshold presence.  Applied to both the edge reduced
        signal and the vertex own-value signal; frame closure is preserved.
    coactivity : {"min", "mean", "product", "max"} or callable, optional
        The reducer combining the two endpoint values of an edge.  ``"min"``
        (default) means "both endpoints active", giving the excursion-set
        induced sub-complex.  A callable receives two elementwise-aligned
        ``ndarray`` s and must return an ``ndarray``.
    include_isolated_vertices : bool, optional
        If True (default), a vertex is present when its own value crosses the
        threshold even without any incident edge (isolated supra-threshold
        vertices are kept).  If False, only edge-incident vertices are kept.
    vertex_mask : (V,) bool array-like, optional
        Restricts the ambient mesh: ``True`` keeps the vertex, ``False`` drops
        it — along with every incident edge and face — from all frames.  Use
        it to exclude the medial wall or restrict to an ROI.
    strict : bool, optional
        Use strict ``>`` comparisons (default) or non-strict ``>=``.  Applied
        consistently to both the edge reduction and the vertex "own value"
        clause.

    Returns
    -------
    ZigzagFiltration

    Examples
    --------
    >>> filt = build_mesh_zigzag(faces, activity, threshold=0.5)  # doctest: +SKIP
    """
    frames = _mesh_frames(
        faces,
        activity,
        threshold,
        threshold_high=threshold_high,
        coactivity=coactivity,
        include_isolated_vertices=include_isolated_vertices,
        vertex_mask=vertex_mask,
        strict=strict,
    )
    return ZigzagFiltration(
        frames=frames,
        cell_dim_fn=_simplex_dim,
        cell_boundary_fn=_simplex_boundary,
    )


def run_mesh_zigzag(
    faces,
    activity,
    threshold: float,
    *,
    threshold_high: Optional[float] = None,
    coactivity: Union[str, Reducer] = "min",
    include_isolated_vertices: bool = True,
    vertex_mask=None,
    strict: bool = True,
    backend: str = "auto",
    output_file: Optional[str] = None,
) -> List[Tuple[int, int, int]]:
    """One-call convenience: build the mesh filtration and compute the barcode.

    Parameters
    ----------
    faces, activity, threshold, threshold_high, coactivity, \
    include_isolated_vertices, vertex_mask, strict
        See :func:`build_mesh_zigzag`.
    backend : str, optional
        Zigzag engine backend: ``"auto"`` (default) or
        ``"python"``.
    output_file : str, optional
        If given, write the barcode as CSV to this path.

    Returns
    -------
    bars : list of (dimension, birth, death)
        The persistence barcode.  Birth/death use the 1-based **zigzag-layer**
        numbering of :meth:`ZigzagEngine.run`:
        ``1 = K0``, ``2 = K0∩K1``, ``3 = K1``, …, ``2N-1 = K_{N-1}``.

    Examples
    --------
    >>> bars = run_mesh_zigzag(faces, activity, threshold=0.5, coactivity="min")  # doctest: +SKIP
    """
    filt = build_mesh_zigzag(
        faces,
        activity,
        threshold,
        threshold_high=threshold_high,
        coactivity=coactivity,
        include_isolated_vertices=include_isolated_vertices,
        vertex_mask=vertex_mask,
        strict=strict,
    )
    engine = ZigzagEngine(backend=backend)
    return engine.run(filt, output_file=output_file)


# ======================================================================
# Optional FreeSurfer geometry loader (lazy import; never required)
# ======================================================================

def load_freesurfer_surface(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a FreeSurfer surface geometry file (``lh.white`` etc.).

    This is an *optional* convenience helper.  It lazily imports ``nibabel``
    and is never needed by the core mesh-zigzag pipeline, which operates on a
    plain ``faces`` array.

    Parameters
    ----------
    path : str
        Path to a FreeSurfer surface geometry file.

    Returns
    -------
    coords : ndarray of shape (V, 3)
        Vertex coordinates (unused by the pipeline; returned for convenience).
    faces : ndarray of shape (F, 3)
        Triangle vertex indices, ready to pass to :func:`build_mesh_zigzag`.
    """
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "load_freesurfer_surface requires nibabel "
            "(`pip install nibabel`)."
        ) from exc

    coords, faces = nib.freesurfer.read_geometry(path)
    return np.asarray(coords), np.asarray(faces, dtype=np.int64)

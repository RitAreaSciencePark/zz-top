"""
GPU-accelerated **static** cubical persistence for batched frames.

This module provides a bridge between zztop and the GPU cubical persistence
backend available in the `GUDHI GPU fork
<https://github.com/matteobiagetti/gudhi-devel>`_.  It computes **standard
(non-zigzag) persistence diagrams** independently for each frame in a batch,
optionally using GPU acceleration when available.

For *zigzag* persistence across frames, use :func:`run_cubical_zigzag`
instead — that path is inherently sequential and is handled by the zztop
zigzag engine.

Usage example::

    from zztop import run_cubical_persistence_gpu

    # grid_data: (N1, ..., Nd, n_frames)
    diagrams = run_cubical_persistence_gpu(grid_data, min_persistence=0.0)
    # diagrams[t] = {dim: ndarray of shape (n_bars, 2)}
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Sequence, Union

import numpy as np


# ======================================================================
#  GPU / sklearn backend detection
# ======================================================================

def _detect_gpu_backend():
    """Detect available GPU cubical persistence backends.

    Returns
    -------
    info : dict
        ``'available'``: bool — True if the GPU extension is importable.
        ``'backend_tag'``: str — ``'cuda'``, ``'cpu-fallback'``, or ``'none'``.
        ``'batched_top_cells_fn'``: callable or None — batched top-cells GPU function.
        ``'batched_vertices_fn'``: callable or None — batched vertices GPU function.
        ``'single_top_cells_fn'``: callable or None
        ``'single_vertices_fn'``: callable or None
        ``'sklearn_available'``: bool — True if ``gudhi.sklearn.cubical_persistence`` is importable.
    """
    info = dict(
        available=False,
        backend_tag="none",
        # 2D functions
        batched_top_cells_fn=None,
        batched_vertices_fn=None,
        single_top_cells_fn=None,
        single_vertices_fn=None,
        # 3D functions
        batched_top_cells_3d_fn=None,
        batched_vertices_3d_fn=None,
        single_top_cells_3d_fn=None,
        single_vertices_3d_fn=None,
        sklearn_available=False,
    )
    try:
        module = importlib.import_module("gudhi._pers_cub_low_dim_gpu_ext")
        info["available"] = True
        info["backend_tag"] = getattr(module, "__backend__", "unknown")
        info["batched_top_cells_fn"] = getattr(
            module,
            "_persistence_on_rectangles_from_top_cells_gpu_batched",
            None,
        )
        info["batched_vertices_fn"] = getattr(
            module,
            "_persistence_on_rectangles_from_vertices_gpu_batched",
            None,
        )
        info["single_top_cells_fn"] = getattr(
            module,
            "_persistence_on_rectangle_from_top_cells_gpu",
            None,
        )
        info["single_vertices_fn"] = getattr(
            module,
            "_persistence_on_rectangle_from_vertices_gpu",
            None,
        )
        # 3D functions
        info["batched_top_cells_3d_fn"] = getattr(
            module,
            "_persistence_on_boxes_from_top_cells_gpu_batched",
            None,
        )
        info["batched_vertices_3d_fn"] = getattr(
            module,
            "_persistence_on_boxes_from_vertices_gpu_batched",
            None,
        )
        info["single_top_cells_3d_fn"] = getattr(
            module,
            "_persistence_on_box_from_top_cells_gpu",
            None,
        )
        info["single_vertices_3d_fn"] = getattr(
            module,
            "_persistence_on_box_from_vertices_gpu",
            None,
        )
    except Exception:
        pass

    try:
        importlib.import_module("gudhi.sklearn.cubical_persistence")
        info["sklearn_available"] = True
    except Exception:
        pass

    return info


_GPU_INFO = None


def gpu_info() -> dict:
    """Return a dict describing the available GPU cubical backend.

    Cached after first call.
    """
    global _GPU_INFO
    if _GPU_INFO is None:
        _GPU_INFO = _detect_gpu_backend()
    return _GPU_INFO


def gpu_available() -> bool:
    """Return True if the GPU cubical persistence extension can be imported."""
    return gpu_info()["available"]


# ======================================================================
#  Per-frame standard persistence (GPU-aware, batched)
# ======================================================================

def _persistence_per_frame_gudhi(
    frame: np.ndarray,
    homology_coeff_field: int = 11,
    min_persistence: float = 0.0,
    max_dimension: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Compute standard cubical persistence on a single frame via GUDHI.

    Parameters
    ----------
    frame : ndarray of shape (N1, ..., Nd)
        Filtration values (vertices / lower-star convention).
    homology_coeff_field : int
        Prime coefficient field for the persistence computation.
    min_persistence : float
        Minimum persistence to report.
    max_dimension : int or None
        If given, only return dimensions 0..max_dimension.

    Returns
    -------
    diagrams : dict mapping int → ndarray of shape (n_bars, 2)
    """
    import gudhi as gd

    cc = gd.CubicalComplex(vertices=frame)
    cc.compute_persistence(
        homology_coeff_field=homology_coeff_field,
        min_persistence=min_persistence,
    )
    ndim = frame.ndim
    max_d = max_dimension if max_dimension is not None else ndim
    diagrams: Dict[int, np.ndarray] = {}
    for d in range(max_d + 1):
        intervals = cc.persistence_intervals_in_dimension(d)
        if len(intervals) == 0:
            diagrams[d] = np.empty((0, 2))
        else:
            diagrams[d] = np.asarray(intervals)
    return diagrams


def _persistence_2d_gpu_batched(
    frames: np.ndarray,
    min_persistence: float = 0.0,
    input_type: str = "vertices",
) -> List[Dict[int, np.ndarray]]:
    """Compute 2D cubical persistence for a batch of frames using GPU path.

    Parameters
    ----------
    frames : ndarray of shape (n_frames, H, W)
        Batch of 2D grids.
    min_persistence : float
        Minimum persistence threshold.
    input_type : str
        ``'vertices'`` or ``'top_dimensional_cells'``.

    Returns
    -------
    List of dicts, one per frame, mapping dim → ndarray(n, 2).
    """
    info = gpu_info()

    if input_type == "vertices" and info["batched_vertices_fn"] is not None:
        raw = info["batched_vertices_fn"](frames, float(min_persistence))
    elif input_type == "top_dimensional_cells" and info["batched_top_cells_fn"] is not None:
        raw = info["batched_top_cells_fn"](frames, float(min_persistence))
    elif input_type == "vertices" and info["single_vertices_fn"] is not None:
        raw = [
            info["single_vertices_fn"](frames[i], float(min_persistence))
            for i in range(frames.shape[0])
        ]
    elif input_type == "top_dimensional_cells" and info["single_top_cells_fn"] is not None:
        raw = [
            info["single_top_cells_fn"](frames[i], float(min_persistence))
            for i in range(frames.shape[0])
        ]
    else:
        raise RuntimeError(
            f"No GPU function available for input_type={input_type!r}. "
            f"GPU info: {info}"
        )

    # Raw output is list of [H0_array, H1_array] (2D case: only H0, H1)
    results: List[Dict[int, np.ndarray]] = []
    for per_frame in raw:
        dgm: Dict[int, np.ndarray] = {}
        for dim_idx, arr in enumerate(per_frame):
            arr = np.asarray(arr)
            if arr.ndim != 2 or arr.shape[1] != 2:
                arr = arr.reshape(-1, 2) if arr.size > 0 else np.empty((0, 2))
            dgm[dim_idx] = arr
        results.append(dgm)
    return results


def _persistence_3d_gpu_batched(
    frames: np.ndarray,
    min_persistence: float = 0.0,
    input_type: str = "vertices",
) -> List[Dict[int, np.ndarray]]:
    """Compute 3D cubical persistence for a batch of frames using GPU path.

    Parameters
    ----------
    frames : ndarray of shape (n_frames, H, W, D)
        Batch of 3D grids.
    min_persistence : float
        Minimum persistence threshold.
    input_type : str
        ``'vertices'`` or ``'top_dimensional_cells'``.

    Returns
    -------
    List of dicts, one per frame, mapping dim -> ndarray(n, 2).
    """
    info = gpu_info()

    if input_type == "vertices" and info["batched_vertices_3d_fn"] is not None:
        raw = info["batched_vertices_3d_fn"](frames, float(min_persistence))
    elif input_type == "top_dimensional_cells" and info["batched_top_cells_3d_fn"] is not None:
        raw = info["batched_top_cells_3d_fn"](frames, float(min_persistence))
    elif input_type == "vertices" and info["single_vertices_3d_fn"] is not None:
        raw = [
            info["single_vertices_3d_fn"](frames[i], float(min_persistence))
            for i in range(frames.shape[0])
        ]
    elif input_type == "top_dimensional_cells" and info["single_top_cells_3d_fn"] is not None:
        raw = [
            info["single_top_cells_3d_fn"](frames[i], float(min_persistence))
            for i in range(frames.shape[0])
        ]
    else:
        raise RuntimeError(
            f"No GPU function available for 3D input_type={input_type!r}. "
            f"GPU info: {info}"
        )

    # Raw output is list of [H0, H1, H2] (3D case)
    results: List[Dict[int, np.ndarray]] = []
    for per_frame in raw:
        dgm: Dict[int, np.ndarray] = {}
        for dim_idx, arr in enumerate(per_frame):
            arr = np.asarray(arr)
            if arr.ndim != 2 or arr.shape[1] != 2:
                arr = arr.reshape(-1, 2) if arr.size > 0 else np.empty((0, 2))
            dgm[dim_idx] = arr
        results.append(dgm)
    return results


def _gpu_supports_3d() -> bool:
    """Return True if the GPU extension exposes 3D persistence functions."""
    info = gpu_info()
    return (
        info["single_top_cells_3d_fn"] is not None
        or info["single_vertices_3d_fn"] is not None
    )


def run_cubical_persistence_gpu(
    grid_data: np.ndarray,
    *,
    min_persistence: float = 0.0,
    max_dimension: Optional[int] = None,
    homology_coeff_field: int = 11,
    backend: str = "auto",
    input_type: str = "vertices",
    negate: bool = True,
) -> List[Dict[int, np.ndarray]]:
    """Compute **standard** cubical persistence per frame, GPU-accelerated when possible.

    This is *not* zigzag persistence — it computes an independent persistence
    diagram for each time frame.  Useful in learning pipelines where per-frame
    topological features are needed.

    Parameters
    ----------
    grid_data : ndarray of shape ``(N1, ..., Nd, n_frames)``
        Grid values over time.  Last axis = frame/time axis.
    min_persistence : float
        Minimum bar length to keep (strictly greater).
    max_dimension : int or None
        Cap on homology dimension.  Default = spatial dimension.
    homology_coeff_field : int
        Coefficient field (prime).  Only used for the GUDHI fallback.
    backend : str
        ``'auto'`` — use GPU if available **and** the batch has more than one
        frame (for a single frame CPU is faster).  Spatial dim must be 2 or 3.
        ``'gpu'``  — force GPU even for a single frame; raise if unavailable
        or unsupported dimension.
        ``'cpu'``  — always use GUDHI ``CubicalComplex``.
    input_type : str
        ``'vertices'`` (default) or ``'top_dimensional_cells'``.
    negate : bool
        If True (default), negate the grid values before passing to GUDHI/GPU
        (matching the ``-grid`` convention used in standard lower-star
        cubical persistence).  Set to False if values are already negated.

    Returns
    -------
    diagrams : list of dict
        ``diagrams[t][dim]`` is an ``(n_bars, 2)`` ndarray with birth/death
        filtration values for frame *t* in dimension *dim*.

    Raises
    ------
    RuntimeError
        If ``backend='gpu'`` but GPU extension is not available, or if spatial
        dimension is not supported by the installed GPU extension.
    """
    ndim = grid_data.ndim - 1  # spatial dimensions
    n_frames = grid_data.shape[-1]
    if max_dimension is None:
        max_dimension = ndim

    # Decide backend
    # GPU is beneficial for batched computation (n_frames > 1).  For a single
    # frame the kernel-launch overhead dominates and CPU is faster, so
    # backend="auto" only selects GPU when there are multiple frames.
    use_gpu = False
    if backend == "gpu":
        if not gpu_available():
            raise RuntimeError(
                "GPU backend requested but gudhi._pers_cub_low_dim_gpu_ext "
                "is not available.  Install the GPU-enabled GUDHI fork."
            )
        if ndim == 2:
            use_gpu = True
        elif ndim == 3 and _gpu_supports_3d():
            use_gpu = True
        else:
            raise RuntimeError(
                f"GPU cubical persistence not available for {ndim}D spatial data. "
                f"Supported: 2D" + (", 3D" if _gpu_supports_3d() else "") + "."
            )
    elif backend == "auto":
        if gpu_available() and n_frames > 1:
            if ndim == 2:
                use_gpu = True
            elif ndim == 3 and _gpu_supports_3d():
                use_gpu = True
    elif backend != "cpu":
        raise ValueError(f"Unknown backend {backend!r}; expected 'auto', 'gpu', or 'cpu'.")

    # Extract per-frame slices
    slicing_prefix = tuple([slice(None)] * ndim)

    if negate:
        grid_data = -grid_data

    if use_gpu:
        # Stack frames into (n_frames, N1, ..., Nd)
        frames_batch = np.stack(
            [grid_data[slicing_prefix + (t,)] for t in range(n_frames)],
            axis=0,
        )
        if ndim == 2:
            return _persistence_2d_gpu_batched(
                frames_batch,
                min_persistence=min_persistence,
                input_type=input_type,
            )
        elif ndim == 3:
            return _persistence_3d_gpu_batched(
                frames_batch,
                min_persistence=min_persistence,
                input_type=input_type,
            )
        else:
            raise RuntimeError(f"GPU path reached with unsupported ndim={ndim}")

    # CPU fallback via standard GUDHI CubicalComplex
    #
    # For 3D + input_type="vertices": the GPU kernel only supports
    # top-dimensional cells, so the GPU path internally converts via
    # _vertices_to_top_cells_3d (max of 8 voxel corners).  To maintain
    # CPU/GPU parity we apply the same conversion here.  (The 2D GPU
    # kernel uses _vertices_to_embedded_top_cells_2d which is equivalent
    # to CubicalComplex(vertices=...), so no special handling is needed.)
    convert_3d_vertices = (
        ndim == 3 and input_type == "vertices" and gpu_available()
    )
    if convert_3d_vertices:
        try:
            from gudhi._pers_cub_low_dim_gpu_ext import (
                _vertices_to_top_cells_3d,
            )
        except ImportError:
            convert_3d_vertices = False

    results: List[Dict[int, np.ndarray]] = []
    for t in range(n_frames):
        frame = grid_data[slicing_prefix + (t,)]
        if convert_3d_vertices:
            import gudhi as gd

            # _vertices_to_top_cells_3d requires C-contiguous float64/float32
            frame_c = np.ascontiguousarray(frame, dtype=np.float64)
            top_cells = np.asarray(_vertices_to_top_cells_3d(frame_c))
            cc = gd.CubicalComplex(top_dimensional_cells=top_cells)
            cc.compute_persistence(
                homology_coeff_field=homology_coeff_field,
                min_persistence=min_persistence,
            )
            max_d = max_dimension if max_dimension is not None else ndim
            dgm: Dict[int, np.ndarray] = {}
            for d in range(max_d + 1):
                intervals = cc.persistence_intervals_in_dimension(d)
                dgm[d] = (
                    np.asarray(intervals)
                    if len(intervals) > 0
                    else np.empty((0, 2))
                )
            results.append(dgm)
        else:
            dgm = _persistence_per_frame_gudhi(
                frame,
                homology_coeff_field=homology_coeff_field,
                min_persistence=min_persistence,
                max_dimension=max_dimension,
            )
            results.append(dgm)
    return results


def run_cubical_persistence_sklearn(
    grid_data: np.ndarray,
    *,
    homology_dimensions: Union[int, Sequence[int]] = (0, 1),
    min_persistence: float = 0.0,
    backend: str = "auto",
    input_type: str = "vertices",
    negate: bool = True,
    n_jobs: Optional[int] = None,
) -> list:
    """Compute per-frame cubical persistence via GUDHI's sklearn interface.

    This is a thin convenience wrapper around
    :class:`gudhi.sklearn.cubical_persistence.CubicalPersistence` — useful
    when the GPU fork is installed and you want the familiar sklearn
    ``fit_transform`` pipeline with GPU dispatch.

    Parameters
    ----------
    grid_data : ndarray of shape ``(N1, ..., Nd, n_frames)``
        Last axis = frame/time.
    homology_dimensions : int or sequence of int
        Which homology dimensions to return.
    min_persistence : float
        Minimum bar length.
    backend : str
        Passed directly to ``CubicalPersistence(backend=...)``.
    input_type : str
        ``'vertices'`` or ``'top_dimensional_cells'``.
    negate : bool
        Negate grid values.
    n_jobs : int or None
        Parallelism for the sklearn transformer.

    Returns
    -------
    diagrams : list
        One entry per frame; format depends on ``homology_dimensions``.
    """
    from gudhi.sklearn.cubical_persistence import CubicalPersistence
    import inspect

    ndim = grid_data.ndim - 1
    n_frames = grid_data.shape[-1]
    slicing_prefix = tuple([slice(None)] * ndim)

    if negate:
        grid_data = -grid_data

    # Build list of per-frame grids (what sklearn expects as X)
    X = [grid_data[slicing_prefix + (t,)] for t in range(n_frames)]

    # The 'backend' parameter is only available in the GPU fork.
    # Standard GUDHI does not accept it, so we only pass it when
    # the constructor supports it.
    init_params = inspect.signature(CubicalPersistence.__init__).parameters
    kwargs = dict(
        homology_dimensions=homology_dimensions,
        input_type=input_type,
        min_persistence=min_persistence,
        n_jobs=n_jobs,
    )
    if "backend" in init_params:
        kwargs["backend"] = backend

    cp = CubicalPersistence(**kwargs)
    return cp.fit_transform(X)

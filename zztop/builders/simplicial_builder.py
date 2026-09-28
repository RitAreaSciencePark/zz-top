"""
High-level builder for **simplicial** zigzag persistence.

Replaces the ``ZIGZAG`` class from the original ``zigzagllms.py``.
Given a sequence of point clouds (one per layer/frame), this module:

1. Builds a k-nearest-neighbour graph for each point cloud.
2. Expands it to a GUDHI ``SimplexTree`` up to a given dimension.
3. Collects the union of all simplices, per-frame membership.
4. Constructs a :class:`~zztop.zigzag.filtration.ZigzagFiltration`.
5. Runs the zigzag engine and returns the barcode.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from zztop.complex.simplicial import Simplex
from zztop.zigzag.filtration import ZigzagFiltration
from zztop.zigzag.engine import ZigzagEngine


def build_simplicial_zigzag(
    point_clouds: Sequence[np.ndarray],
    knn: int = 10,
    dim: int = 2,
    labels: Optional[Sequence[np.ndarray]] = None,
    backend: str = "auto",
) -> ZigzagFiltration:
    """Build a :class:`ZigzagFiltration` from a sequence of point clouds.

    For each point cloud, a k-nearest-neighbour graph is computed using
    scikit-learn, then expanded into a GUDHI ``SimplexTree`` up to the
    requested dimension.

    Parameters
    ----------
    point_clouds : sequence of ndarray, shape (n_points, n_features)
        One point cloud per frame/layer.
    knn : int
        Number of nearest neighbours for the connectivity graph.
    dim : int
        Maximum simplex dimension for the flag expansion.
    labels : sequence of ndarray, optional
        Per-point labels.  If provided, edges connect only points with
        *different* labels (cross-label connectivity).

    Returns
    -------
    ZigzagFiltration
    """
    import sklearn.neighbors
    import gudhi as gd

    frames: List[Set[Simplex]] = []

    for i, cloud in enumerate(point_clouds):
        G = sklearn.neighbors.kneighbors_graph(cloud, n_neighbors=knn)
        G_arr = G.toarray()
        S = gd.SimplexTree()

        # Add vertices
        for k in range(len(cloud)):
            S.insert([k])

        # Add edges
        ind = np.array(np.where(G_arr == 1)).T
        for k in range(len(ind)):
            u, v = ind[k]
            if labels is not None:
                if labels[i][u] != labels[i][v]:
                    S.insert(list(ind[k]))
            else:
                S.insert(list(ind[k]))

        # Flag expansion
        S.expansion(dim)

        frame_set: Set[Simplex] = set()
        for s_filtration_pair in S.get_skeleton(dim):
            simplex = Simplex(s_filtration_pair[0])
            frame_set.add(simplex)
        frames.append(frame_set)

    return ZigzagFiltration(
        frames=frames,
        cell_dim_fn=lambda s: s.dimension,
        cell_boundary_fn=lambda s: s.faces(),
    )


def run_simplicial_zigzag(
    point_clouds: Sequence[np.ndarray],
    knn: int = 10,
    dim: int = 2,
    labels: Optional[Sequence[np.ndarray]] = None,
    backend: str = "auto",
    output_file: Optional[str] = None,
) -> List[Tuple[int, int, int]]:
    """One-call convenience: build filtration and compute barcode.

    Parameters
    ----------
    point_clouds : sequence of ndarray
        One point cloud per frame/layer.
    knn : int
        Number of neighbours for k-NN graph.
    dim : int
        Maximum simplex dimension.
    labels : sequence of ndarray, optional
        Per-point labels for cross-label connectivity.
    backend : str
        ``'auto'`` or ``'python'`` (the same engine).
    output_file : str, optional
        CSV path to write the barcode to.

    Returns
    -------
    bars : list of (dimension, birth, death)
    """
    filt = build_simplicial_zigzag(
        point_clouds, knn=knn, dim=dim, labels=labels, backend=backend
    )
    engine = ZigzagEngine(backend=backend)
    return engine.run(filt, output_file=output_file)

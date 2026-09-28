"""Front-ends: parameters to per-frame simplex values.

A front-end is the map ``G`` in ``theta -> V``, where ``V`` has shape
``(n_frames, n_cells)`` in the canonical cell order.  Only two are provided,
and the core never learns which one produced its input.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from zztop.complex.simplicial import Simplex

from .events import cell_order

__all__ = ["lower_star", "passthrough", "validate_face_condition", "vertex_order"]


def vertex_order(complex_) -> List[Simplex]:
    """The 0-cells of *complex_* in canonical order.

    They occupy the first ``n_vertices`` positions of
    :func:`~zztop.zigzagdiff.events.cell_order`, so ``theta``'s columns line up
    with them directly.
    """
    return [c for c in cell_order(complex_) if c.dimension == 0]


def _vertex_columns(complex_) -> Tuple[List[Simplex], Dict[int, List[List[int]]]]:
    """Group cells by dimension, mapping each to its vertex column indices."""
    order = cell_order(complex_)
    pos = {int(next(iter(v))): i for i, v in enumerate(vertex_order(complex_))}
    by_dim: Dict[int, List[List[int]]] = {}
    for cell in order:
        by_dim.setdefault(cell.dimension, []).append([pos[v] for v in cell])
    return order, by_dim


def lower_star(theta: torch.Tensor, complex_) -> torch.Tensor:
    """Induce per-simplex values from vertex values by the minimum rule.

    ``G(theta)(sigma)_i = min over vertices p of sigma of theta(p)_i``.

    The draft calls this the *lower-star* map, and the name is kept, but note
    the convention in force here is an **upper**-level set: a simplex is
    present iff its value exceeds the threshold.  Taking the minimum over
    vertices is what makes the face condition hold automatically -- a face
    minimises over a subset, so it can only have a larger-or-equal value.

    Backward sends gradient to the argmin vertex; at ties torch's ``amin``
    picks a valid subgradient element (see the package docstring).

    Parameters
    ----------
    theta : Tensor, shape (n_frames, n_vertices)
    complex_ : SimplicialComplex

    Returns
    -------
    Tensor, shape (n_frames, n_cells)
    """
    if theta.ndim != 2:
        raise ValueError(f"theta must be 2-D (n_frames, n_vertices), got {theta.shape}")
    order, by_dim = _vertex_columns(complex_)
    n_vertices = sum(1 for c in order if c.dimension == 0)
    if theta.shape[1] != n_vertices:
        raise ValueError(
            f"theta has {theta.shape[1]} vertex columns, complex has {n_vertices}"
        )

    # Cells grouped by ascending dimension, each group in sorted order, is
    # exactly the canonical order -- so the pieces concatenate into place.
    pieces = []
    for dim in sorted(by_dim):
        cols = torch.as_tensor(np.array(by_dim[dim], dtype=np.int64), device=theta.device)
        pieces.append(theta[:, cols].amin(dim=-1))
    return torch.cat(pieces, dim=1)


def _face_pairs(complex_) -> np.ndarray:
    """Array of ``(face_column, coface_column)`` index pairs."""
    order = cell_order(complex_)
    index_of = {cell: i for i, cell in enumerate(order)}
    pairs = [
        (index_of[face], i) for i, cell in enumerate(order) for face in cell.faces()
    ]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def validate_face_condition(values: torch.Tensor, complex_) -> None:
    """Check that every proper face carries a value at least as large.

    The specification states the strict condition ``v(face) > v(coface)``; the
    check enforces the weak form ``>=`` because :func:`lower_star` produces
    equality generically (an edge's value *is* one endpoint's value).
    Equality is handled downstream by simultaneity batching, so it is
    admissible -- a strict violation is not.
    """
    pairs = _face_pairs(complex_)
    if pairs.size == 0:
        return
    face_col = torch.as_tensor(pairs[:, 0], device=values.device)
    coface_col = torch.as_tensor(pairs[:, 1], device=values.device)
    bad = (values[:, face_col] < values[:, coface_col]).nonzero()
    if bad.numel():
        frame, which = (int(x) for x in bad[0])
        order = cell_order(complex_)
        face = order[int(pairs[which, 0])]
        coface = order[int(pairs[which, 1])]
        raise ValueError(
            f"face condition violated at frame {frame}: face {face} has value "
            f"{float(values[frame, pairs[which, 0]])!r} < coface {coface} value "
            f"{float(values[frame, pairs[which, 1]])!r}"
        )


def passthrough(values: torch.Tensor, complex_, validate: bool = True) -> torch.Tensor:
    """Accept per-simplex values directly; the caller owns the face condition."""
    if values.ndim != 2:
        raise ValueError(f"values must be 2-D (n_frames, n_cells), got {values.shape}")
    n_cells = len(cell_order(complex_))
    if values.shape[1] != n_cells:
        raise ValueError(
            f"values has {values.shape[1]} columns, complex has {n_cells} cells"
        )
    if validate:
        validate_face_condition(values, complex_)
    return values

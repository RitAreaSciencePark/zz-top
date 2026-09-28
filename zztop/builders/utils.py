"""
Shared utility functions for the zigzag pipeline builders.
"""

from __future__ import annotations

import itertools
from typing import List, Set

import numpy as np


def ranges(indices):
    """Group consecutive 0-based indices into ``(start+1, end+2)`` ranges.

    This is the ``ranges()`` function from the original ``fclaux.py``.
    It converts sorted 0-based frame indices where a cell is present into
    (1-based, half-open) zigzag time intervals used by the Dionysus-style
    ``times`` encoding.

    Parameters
    ----------
    indices : array-like of int
        Sorted 0-based indices.

    Yields
    ------
    (int, int)
        ``(start, end)`` with 1-based inclusive start and exclusive end.

    Examples
    --------
    >>> list(ranges([0, 1, 2, 5, 6]))
    [(1, 4), (6, 8)]
    """
    for _, grp in itertools.groupby(
        enumerate(indices), lambda pair: pair[1] - pair[0]
    ):
        grp = list(grp)
        yield grp[0][1] + 1, grp[-1][1] + 2


def build_intersection_layers(
    frame_cell_ids: List[Set[int]],
) -> List[Set[int]]:
    """Build zigzag layers with intersections between consecutive frames.

    Given frame membership sets (each a set of cell IDs), produce the
    interleaved layer sequence::

        K_0 ↔ (K_0 ∩ K_1) ↔ K_1 ↔ (K_1 ∩ K_2) ↔ K_2 ↔ …

    Parameters
    ----------
    frame_cell_ids : list of set of int
        ``frame_cell_ids[t]`` is the set of canonical cell IDs present at
        frame *t*.

    Returns
    -------
    layers : list of set of int
    """
    n = len(frame_cell_ids)
    if n == 0:
        return []
    if n == 1:
        return [frame_cell_ids[0]]

    layers: List[Set[int]] = []
    for i in range(n):
        layers.append(frame_cell_ids[i])
        if i < n - 1:
            layers.append(frame_cell_ids[i] & frame_cell_ids[i + 1])
    return layers


def build_appearance_matrix(
    layers: List[Set[int]], n_cells: int
) -> np.ndarray:
    """Build the binary appearance matrix.

    Parameters
    ----------
    layers : list of set of int
        Each set contains 0-based cell IDs present in that layer.
    n_cells : int
        Total number of unique cells.

    Returns
    -------
    A : ndarray of shape (n_layers, n_cells), dtype int8
    """
    A = np.zeros((len(layers), n_cells), dtype=np.int8)
    for k, layer in enumerate(layers):
        for cid in layer:
            A[k, cid] = 1
    return A


def compute_times_from_appearance(A: np.ndarray) -> List[List[int]]:
    """Compute Dionysus-style ``times`` from an appearance matrix.

    Parameters
    ----------
    A : ndarray of shape (n_layers, n_cells)
        Binary appearance matrix.

    Returns
    -------
    times : list of list of int
        ``times[j]`` is the flattened ranges list for cell *j*.
    """
    times: List[List[int]] = []
    for j in range(A.shape[1]):
        present = np.where(A[:, j] == 1)[0]
        flat: List[int] = []
        for a, b in ranges(present):
            flat.append(a)
            flat.append(b)
        times.append(flat)
    return times

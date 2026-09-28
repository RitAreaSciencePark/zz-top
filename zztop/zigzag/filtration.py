"""
Zigzag filtration: manages the sequence of complexes and produces the
operation list consumed by the zigzag backends.

The filtration is *complex-type agnostic* — it works with any hashable cell
objects (simplices, cubical cells, or anything else) and delegates boundary
computation to a :class:`~zztop.complex.boundary.CellComplex` when needed.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from typing import (
    Dict,
    Hashable,
    List,
    Sequence,
    Set,
    Tuple,
)


class ZigzagFiltration:
    """Encode a zigzag sequence of complexes and convert it to a
    simplex-wise (cell-wise) insertion/deletion stream.

    Parameters
    ----------
    frames : list of sets (or list-like) of hashable cells
        ``frames[t]`` is the collection of cells present at time *t*.
    cell_dim_fn : callable(cell) -> int
        Returns the dimension of a cell.
    cell_boundary_fn : callable(cell) -> list[cell]
        Returns the mod-2 boundary (list of boundary cells) of a cell.
    use_intersections : bool
        If *True* (default), interleave intersection layers between
        consecutive frames:  K0 ↔ (K0∩K1) ↔ K1 ↔ (K1∩K2) ↔ K2 ↔ …
        This is the standard zigzag construction.

    Attributes
    ----------
    cell_to_id : dict
        Mapping from each unique cell to a canonical integer ID.
    n_cells : int
        Total number of unique cells across all frames.
    n_layers : int
        Number of layers (frames + intersections) in the zigzag.
    """

    def __init__(
        self,
        frames: Sequence[Set[Hashable]],
        cell_dim_fn,
        cell_boundary_fn,
        use_intersections: bool = True,
    ) -> None:
        self._frames = [set(f) for f in frames]
        self._dim_fn = cell_dim_fn
        self._bdry_fn = cell_boundary_fn
        self._use_intersections = use_intersections

        # Build global cell catalogue
        self.cell_to_id: Dict[Hashable, int] = OrderedDict()
        self._cells: List[Hashable] = []
        for frame in self._frames:
            for cell in frame:
                if cell not in self.cell_to_id:
                    self.cell_to_id[cell] = len(self._cells)
                    self._cells.append(cell)

        self.n_cells = len(self._cells)
        self._layers = self._build_layers()
        self.n_layers = len(self._layers)

    # ------------------------------------------------------------------
    # Layer construction
    # ------------------------------------------------------------------

    def _build_layers(self) -> List[Set[int]]:
        """Convert frames to layers (with optional intersections)."""
        n = len(self._frames)
        id_frames = [
            {self.cell_to_id[c] for c in f} for f in self._frames
        ]
        if not self._use_intersections or n <= 1:
            return id_frames

        layers: List[Set[int]] = []
        for i in range(n):
            layers.append(id_frames[i])
            if i < n - 1:
                layers.append(id_frames[i] & id_frames[i + 1])
        return layers

    # ------------------------------------------------------------------
    # Appearance matrix & ranges
    # ------------------------------------------------------------------

    def appearance_matrix(self) -> "numpy.ndarray":
        """Return a (n_layers × n_cells) binary matrix.

        ``A[k, j] == 1`` iff cell *j* is present in layer *k*.
        """
        import numpy as np

        A = np.zeros((self.n_layers, self.n_cells), dtype=np.int8)
        for k, layer in enumerate(self._layers):
            for cid in layer:
                A[k, cid] = 1
        return A

    @staticmethod
    def _ranges(indices):
        """Group consecutive indices into ``(start+1, end+2)`` ranges.

        This is the ``ranges()`` function from the original fclaux helper,
        which converts 0-based frame indices where a cell is present into
        (1-based, half-open) zigzag time intervals.
        """
        for _, grp in itertools.groupby(
            enumerate(indices), lambda pair: pair[1] - pair[0]
        ):
            grp = list(grp)
            yield grp[0][1] + 1, grp[-1][1] + 2

    def compute_times(self) -> List[List[int]]:
        """Compute Dionysus-style ``times`` list for every cell.

        Returns
        -------
        times : list of list of int
            ``times[j]`` is a flat list of time values for cell *j*.
            Even positions (0, 2, …) are appearance times; odd positions
            (1, 3, …) are disappearance times.
        """
        import numpy as np

        A = self.appearance_matrix()
        times: List[List[int]] = []
        for j in range(self.n_cells):
            present = np.where(A[:, j] == 1)[0]
            ranges_list = list(self._ranges(present))
            flat = []
            for r in ranges_list:
                flat.extend(r)
            times.append(flat)
        return times

    # ------------------------------------------------------------------
    # Operation sequence (for fast-zigzag backend)
    # ------------------------------------------------------------------

    def to_operations(self) -> List[Tuple[str, Hashable]]:
        """Convert the zigzag filtration to a cell-wise insertion / deletion
        sequence suitable for the fast-zigzag algorithm.

        Returns
        -------
        ops : list of (op, cell)
            ``op`` is ``'i'`` (insert) or ``'d'`` (delete).  Insertions go
            from lowest to highest dimension; deletions from highest to
            lowest, so that the closure property is maintained.
        """
        import numpy as np

        times = self.compute_times()

        # Build (time, op, cell_id) triples — same logic as the original
        # compute_zigzag in zigzag.py
        triples: List[Tuple[int, str, int]] = []
        for cell_id in range(self.n_cells):
            t = times[cell_id]
            for pos in range(len(t)):
                if pos % 2 == 0:
                    triples.append((t[pos], "i", cell_id))
                else:
                    triples.append((t[pos], "d", cell_id))

        dim_of = [self._dim_fn(c) for c in self._cells]

        def sort_key(triple):
            time, op, cid = triple
            d = dim_of[cid]
            # insertion: lower dim first  → +d
            # deletion:  higher dim first → -d
            return (time, d if op == "i" else -d)

        triples.sort(key=sort_key)

        return [(op, self._cells[cid]) for (_, op, cid) in triples]

    def to_simplicial_operations(self) -> List[Tuple[str, List[int]]]:
        """Convert to ``(op, sorted_vertex_list)`` format for the C++
        simplicial backend.  Only valid when cells are
        :class:`~zztop.complex.simplicial.Simplex` objects.
        """
        ops = self.to_operations()
        result: List[Tuple[str, List[int]]] = []
        for op, cell in ops:
            result.append((op, list(cell.vertices)))
        return result

    def to_abstract_operations(self) -> Tuple[
        List[Tuple[str, int, List[int]]], Dict[Hashable, int]
    ]:
        """Convert to ``(op, dim, boundary_filt_indices)`` format for the
        C++ abstract backend.

        Returns
        -------
        ops : list of (op_char, dim, boundary_indices)
            For insertions: ``boundary_indices`` is a sorted list of the
            filtration-operation indices of the cell's boundary cells.
            For deletions: ``boundary_indices`` is ``[insertion_index]``
            pointing to the operation index where this cell was inserted.
        cell_op_map : dict
            Mapping from cell → operation index of its most recent insertion.
        """
        raw_ops = self.to_operations()
        result: List[Tuple[str, int, List[int]]] = []
        # Map cell → operation index of its current insertion
        cell_insert_idx: Dict[Hashable, int] = {}

        for i, (op, cell) in enumerate(raw_ops):
            d = self._dim_fn(cell)
            if op == "i":
                # Compute boundary → find their insertion op indices
                bdry_cells = self._bdry_fn(cell)
                bdry_indices = sorted(
                    cell_insert_idx[bc] for bc in bdry_cells
                )
                result.append(("i", d, bdry_indices))
                cell_insert_idx[cell] = i
            else:
                # Deletion: pass the insertion op index so C++ can look up
                ins_idx = cell_insert_idx.pop(cell, -1)
                result.append(("d", d, [ins_idx] if ins_idx >= 0 else []))
        return result, cell_insert_idx

    # ------------------------------------------------------------------
    # Operation ↔ layer mapping
    # ------------------------------------------------------------------

    def operation_to_layer_map(self) -> List[int]:
        """Map each operation index to its 1-based layer index.

        The zigzag algorithm works on a flat sequence of cell insertions
        and deletions.  Each operation belongs to a specific layer
        (frame or intersection) in the zigzag.  This method returns the
        layer assignment, replicating the same sorting used in
        :meth:`to_operations`.

        Returns
        -------
        layer_map : list of int
            ``layer_map[i]`` is the 1-based layer index for operation *i*.
            Length equals the total number of operations.
        """
        import numpy as np

        times = self.compute_times()

        triples: List[Tuple[int, str, int]] = []
        for cell_id in range(self.n_cells):
            t = times[cell_id]
            for pos in range(len(t)):
                if pos % 2 == 0:
                    triples.append((t[pos], "i", cell_id))
                else:
                    triples.append((t[pos], "d", cell_id))

        dim_of = [self._dim_fn(c) for c in self._cells]

        def sort_key(triple):
            time, op, cid = triple
            d = dim_of[cid]
            return (time, d if op == "i" else -d)

        triples.sort(key=sort_key)
        return [time for (time, _, _) in triples]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def cell(self, cell_id: int) -> Hashable:
        """Return the cell object for a given canonical ID."""
        return self._cells[cell_id]

    @property
    def layers(self) -> List[Set[int]]:
        return self._layers

    @property
    def n_frames(self) -> int:
        return len(self._frames)

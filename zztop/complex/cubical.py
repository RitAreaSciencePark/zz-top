"""
Cubical complex data structures.

A :class:`CubicalCell` is an *n*-dimensional elementary interval product,
identified by a tuple of ``(min, max)`` coordinate pairs.  A
:class:`CubicalComplex` stores a set of cubical cells with the closure
property.

This representation works for **arbitrary dimension** (2-D grids, 3-D grids,
etc.).

Encoding convention
-------------------
Each axis contributes an interval ``(a, a)`` (degenerate / vertex-type) or
``(a, a+1)`` (non-degenerate / edge-type in that direction).  The
**dimension** of a cell equals the number of non-degenerate intervals.

Example — a 2-D grid cell labelling::

    vertex (i,j)   → ((i,i), (j,j))          dim 0
    horiz edge     → ((i,i), (j,j+1))        dim 1 (non-deg in y)
    vert  edge     → ((i,i+1), (j,j))        dim 1 (non-deg in x)
    square         → ((i,i+1), (j,j+1))      dim 2
"""

from __future__ import annotations

from typing import Dict, Hashable, Iterable, List, Sequence, Tuple

from zztop.complex.boundary import CellComplex

# Type alias — each interval is (lo, hi) with lo <= hi, lo == hi for degenerate.
Interval = Tuple[int, int]
CellKey = Tuple[Interval, ...]


class CubicalCell:
    """An elementary cube (interval product) in an *n*-dimensional grid.

    Parameters
    ----------
    intervals : sequence of (int, int)
        One ``(lo, hi)`` pair per axis.  Degenerate axes have ``lo == hi``.

    Examples
    --------
    >>> c = CubicalCell(((0, 1), (2, 2)))
    >>> c.dim
    1
    >>> c.boundary()             # doctest: +NORMALIZE_WHITESPACE
    [CubicalCell(((0, 0), (2, 2))), CubicalCell(((1, 1), (2, 2)))]
    """

    __slots__ = ("_intervals",)

    def __init__(self, intervals: Sequence[Tuple[int, int]]) -> None:
        self._intervals: CellKey = tuple(
            (int(lo), int(hi)) for lo, hi in intervals
        )

    # -- properties --------------------------------------------------------

    @property
    def intervals(self) -> CellKey:
        return self._intervals

    @property
    def dim(self) -> int:
        return sum(1 for lo, hi in self._intervals if lo != hi)

    @property
    def ndim(self) -> int:
        """Ambient dimension (number of axes)."""
        return len(self._intervals)

    @property
    def non_degenerate_axes(self) -> List[int]:
        """Indices of axes where the interval is non-degenerate."""
        return [i for i, (lo, hi) in enumerate(self._intervals) if lo != hi]

    # -- boundary ----------------------------------------------------------

    def boundary(self) -> List["CubicalCell"]:
        r"""Mod-2 boundary: for each non-degenerate axis, collapse to the
        lower face and the upper face.

        Returns a list of :math:`2 k` cells of dimension :math:`k - 1` where
        :math:`k = \dim(\text{self})`.  The ordering is: for each non-degenerate
        axis (in order), lower face then upper face.
        """
        faces: List[CubicalCell] = []
        for axis in self.non_degenerate_axes:
            lo, hi = self._intervals[axis]
            # lower face: collapse interval to its lower endpoint
            lower = list(self._intervals)
            lower[axis] = (lo, lo)
            faces.append(CubicalCell(lower))
            # upper face: collapse interval to its upper endpoint
            upper = list(self._intervals)
            upper[axis] = (hi, hi)
            faces.append(CubicalCell(upper))
        return faces

    def closure(self) -> List["CubicalCell"]:
        """Return *all* faces of this cell (every sub-interval product),
        including the cell itself."""
        nd_axes = self.non_degenerate_axes
        n = len(nd_axes)
        cells: List[CubicalCell] = []
        # Iterate over all 2^n subsets of non-degenerate axes to collapse
        for mask in range(1 << n):
            for choices in _product_choices(n, mask, nd_axes, self._intervals):
                cells.append(CubicalCell(choices))
        return cells

    # -- Python protocols --------------------------------------------------

    def __hash__(self) -> int:
        return hash(self._intervals)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CubicalCell):
            return NotImplemented
        return self._intervals == other._intervals

    def __lt__(self, other: "CubicalCell") -> bool:
        if self.dim != other.dim:
            return self.dim < other.dim
        return self._intervals < other._intervals

    def __repr__(self) -> str:
        return f"CubicalCell({self._intervals})"


def _product_choices(n_nd, mask, nd_axes, intervals):
    """Generate all interval-tuple variants for the given bitmask.

    For each non-degenerate axis flagged in *mask*, we collapse it to either
    its lower or upper endpoint (two choices).  Axes not in *mask* stay as-is.
    This yields 2^(popcount(mask)) cell tuples.
    """
    # axes flagged in mask are collapsed; we need to enumerate lo/hi choices.
    collapse_axes = [nd_axes[i] for i in range(n_nd) if mask & (1 << i)]
    n_collapse = len(collapse_axes)

    for bits in range(1 << n_collapse):
        ivs = list(intervals)
        for j, axis in enumerate(collapse_axes):
            lo, hi = intervals[axis]
            if bits & (1 << j):
                ivs[axis] = (hi, hi)
            else:
                ivs[axis] = (lo, lo)
        yield tuple(ivs)


# ---------------------------------------------------------------------------
# CubicalComplex
# ---------------------------------------------------------------------------

class CubicalComplex(CellComplex):
    """A cubical complex stored as a set of :class:`CubicalCell` objects.

    The complex is *closed*: adding a cell automatically adds all its faces.

    Parameters
    ----------
    cells : iterable of CubicalCell, optional
        Initial cells to insert (closure is computed automatically).
    """

    def __init__(self, cells: Iterable[CubicalCell] = ()) -> None:
        self._cells: set = set()
        for c in cells:
            self.add(c)

    # -- mutators ----------------------------------------------------------

    def add(self, cell: CubicalCell) -> None:
        """Insert *cell* and all its faces into the complex."""
        if cell in self._cells:
            return
        for face in cell.closure():
            self._cells.add(face)

    def remove(self, cell: CubicalCell) -> None:
        """Remove *cell* (does **not** remove faces)."""
        self._cells.discard(cell)

    # -- CellComplex interface ---------------------------------------------

    def dimension(self, cell: Hashable) -> int:
        if not isinstance(cell, CubicalCell):
            raise TypeError(f"Expected CubicalCell, got {type(cell)}")
        return cell.dim

    def cells_by_dimension(self) -> Dict[int, List[CubicalCell]]:
        result: Dict[int, List[CubicalCell]] = {}
        for c in self._cells:
            result.setdefault(c.dim, []).append(c)
        for v in result.values():
            v.sort()
        return result

    def all_cells(self) -> Iterable[CubicalCell]:
        return iter(self._cells)

    def boundary(self, cell: Hashable) -> List[CubicalCell]:
        if not isinstance(cell, CubicalCell):
            raise TypeError(f"Expected CubicalCell, got {type(cell)}")
        return cell.boundary()

    # -- convenience -------------------------------------------------------

    def __contains__(self, item) -> bool:
        return item in self._cells

    def __len__(self) -> int:
        return len(self._cells)

    def __repr__(self) -> str:
        return f"CubicalComplex({len(self._cells)} cells)"

    # -- class methods for grid construction -------------------------------

    @classmethod
    def from_grid_shape(cls, shape: Tuple[int, ...]) -> "CubicalComplex":
        """Create the *full* cubical complex for a grid of the given shape.

        Parameters
        ----------
        shape : tuple of int
            Number of vertices along each axis, e.g. ``(3, 4)`` for a
            3×4 grid (2 × 3 = 6 squares).

        Returns
        -------
        CubicalComplex
        """
        ndim = len(shape)
        cc = cls()
        # Generate the top-dimensional cubes and let closure fill the rest.
        for idx in _grid_top_cubes(shape):
            cc.add(CubicalCell(idx))
        return cc


def _grid_top_cubes(shape):
    """Yield the top-dimensional cubes (n-cubes) for a grid of given shape."""
    ndim = len(shape)
    # Top cube at grid position (i0, i1, ...) has intervals
    # ((i0, i0+1), (i1, i1+1), ...)
    # where each ik ranges from 0 to shape[k]-2 (need at least 2 vertices
    # along each axis to form an edge).
    ranges_ = [range(max(s - 1, 0)) for s in shape]
    for combo in _ndindex(ranges_):
        yield tuple((c, c + 1) for c in combo)


def _ndindex(ranges_):
    """Cartesian product over a list of range objects (arbitrary dim)."""
    if not ranges_:
        yield ()
        return
    for val in ranges_[0]:
        for rest in _ndindex(ranges_[1:]):
            yield (val,) + rest

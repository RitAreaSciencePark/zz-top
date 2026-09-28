"""
Simplicial complex data structures.

A :class:`Simplex` is an immutable, hashable object wrapping a sorted tuple
of integer vertex indices.  A :class:`SimplicialComplex` stores a set of
simplices and ensures the *closure property* (every face of a simplex in the
complex is also in the complex).
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Hashable, Iterable, Iterator, List, Tuple

from zztop.complex.boundary import CellComplex


class Simplex:
    """An abstract simplex represented by a sorted tuple of vertex indices.

    Parameters
    ----------
    vertices : iterable of int
        Vertex indices (need not be sorted on input).

    Examples
    --------
    >>> s = Simplex([2, 0, 1])
    >>> s.vertices
    (0, 1, 2)
    >>> s.dimension
    2
    >>> list(s.faces())
    [Simplex((1, 2)), Simplex((0, 2)), Simplex((0, 1))]
    """

    __slots__ = ("_vertices",)

    def __init__(self, vertices: Iterable[int]) -> None:
        self._vertices: Tuple[int, ...] = tuple(sorted(vertices))

    # -- properties --------------------------------------------------------

    @property
    def vertices(self) -> Tuple[int, ...]:
        return self._vertices

    @property
    def dimension(self) -> int:
        return len(self._vertices) - 1

    # -- boundary / faces --------------------------------------------------

    def faces(self) -> List["Simplex"]:
        r"""Return the :math:`(\dim-1)`-dimensional faces (mod-2 boundary).

        For a *k*-simplex ``[v0, ..., vk]`` the faces are obtained by
        removing each vertex in turn.
        """
        if self.dimension <= 0:
            return []
        verts = self._vertices
        return [Simplex(verts[:i] + verts[i + 1:]) for i in range(len(verts))]

    def closure(self) -> List["Simplex"]:
        """Return *all* faces of every dimension (including this simplex)."""
        result: List[Simplex] = []
        verts = self._vertices
        for k in range(1, len(verts) + 1):
            for combo in combinations(verts, k):
                result.append(Simplex(combo))
        return result

    # -- Python protocols --------------------------------------------------

    def __hash__(self) -> int:
        return hash(self._vertices)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Simplex):
            return NotImplemented
        return self._vertices == other._vertices

    def __lt__(self, other: "Simplex") -> bool:
        if self.dimension != other.dimension:
            return self.dimension < other.dimension
        return self._vertices < other._vertices

    def __repr__(self) -> str:
        return f"Simplex({self._vertices})"

    def __len__(self) -> int:
        return len(self._vertices)

    def __iter__(self) -> Iterator[int]:
        return iter(self._vertices)

    def __contains__(self, v: int) -> bool:
        return v in self._vertices


class SimplicialComplex(CellComplex):
    """An abstract simplicial complex stored as a set of :class:`Simplex`
    objects.

    The complex is automatically *closed*: adding a simplex also adds all
    of its faces.

    Parameters
    ----------
    simplices : iterable of Simplex or list-of-int, optional
        Initial simplices to insert.

    Examples
    --------
    >>> sc = SimplicialComplex([[0, 1, 2]])
    >>> sorted(sc.all_cells())                       # doctest: +NORMALIZE_WHITESPACE
    [Simplex((0,)), Simplex((1,)), Simplex((2,)),
     Simplex((0, 1)), Simplex((0, 2)), Simplex((1, 2)),
     Simplex((0, 1, 2))]
    """

    def __init__(self, simplices: Iterable = ()) -> None:
        self._cells: set = set()
        for s in simplices:
            self.add(s)

    # -- mutators ----------------------------------------------------------

    def add(self, simplex) -> None:
        """Insert *simplex* (and all its faces) into the complex."""
        if not isinstance(simplex, Simplex):
            simplex = Simplex(simplex)
        if simplex in self._cells:
            return
        # Insert the simplex and its full closure
        for face in simplex.closure():
            self._cells.add(face)

    def remove(self, simplex) -> None:
        """Remove *simplex* from the complex (does **not** remove faces)."""
        if not isinstance(simplex, Simplex):
            simplex = Simplex(simplex)
        self._cells.discard(simplex)

    # -- CellComplex interface ---------------------------------------------

    def dimension(self, cell: Hashable) -> int:
        if not isinstance(cell, Simplex):
            cell = Simplex(cell)
        return cell.dimension

    def cells_by_dimension(self) -> Dict[int, List[Simplex]]:
        result: Dict[int, List[Simplex]] = {}
        for s in self._cells:
            result.setdefault(s.dimension, []).append(s)
        for v in result.values():
            v.sort()
        return result

    def all_cells(self) -> Iterable[Simplex]:
        return iter(self._cells)

    def boundary(self, cell: Hashable) -> List[Simplex]:
        if not isinstance(cell, Simplex):
            cell = Simplex(cell)
        return cell.faces()

    # -- convenience -------------------------------------------------------

    def __contains__(self, item) -> bool:
        if not isinstance(item, Simplex):
            item = Simplex(item)
        return item in self._cells

    def __len__(self) -> int:
        return len(self._cells)

    def __repr__(self) -> str:
        return f"SimplicialComplex({len(self._cells)} cells)"

"""
Abstract base for cell complexes used by the zigzag engine.

Both :class:`~zztop.complex.simplicial.SimplicialComplex` and
:class:`~zztop.complex.cubical.CubicalComplex` satisfy this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Hashable, Iterable, List


class CellComplex(ABC):
    """Protocol that any cell complex must satisfy for use with the zigzag
    engine.

    A *cell* is any hashable Python object.  The complex must be able to
    report the dimension of a cell, enumerate cells by dimension, and compute
    the mod-2 boundary chain (list of boundary cells).
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def dimension(self, cell: Hashable) -> int:
        """Return the dimension of *cell*."""

    @abstractmethod
    def cells_by_dimension(self) -> Dict[int, List[Hashable]]:
        """Return ``{dim: [cell, ...]}`` for every dimension present."""

    @abstractmethod
    def all_cells(self) -> Iterable[Hashable]:
        """Iterate over every cell (in any order)."""

    @abstractmethod
    def boundary(self, cell: Hashable) -> List[Hashable]:
        """Return the mod-2 boundary of *cell* as a list of cells.

        Vertices (0-cells) must return ``[]``.
        """

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def max_dimension(self) -> int:
        cbd = self.cells_by_dimension()
        return max(cbd.keys()) if cbd else -1

    def num_cells(self) -> int:
        return sum(len(v) for v in self.cells_by_dimension().values())

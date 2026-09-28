"""Parity: native cubical zigzag vs old-style cubical→simplicial conversion.

Compares **persistence diagrams** (per-dimension Betti curves and total
persistence) between the two approaches.

Per-frame Betti numbers must match because a cubical complex and its
triangulation are homotopy-equivalent.  Individual bar endpoints may differ
due to different cell insertion orders and the extra diagonal edges
introduced by triangulation.

See docs/cubical_zigzag_design.md for the full mathematical justification.
"""

import pytest
import numpy as np

from conftest import (
    bars_to_diagrams,
    diagram_betti_at,
    diagram_total_persistence,
)

from zztop.complex.simplicial import Simplex
from zztop.complex.cubical import CubicalCell, CubicalComplex
from zztop.zigzag.filtration import ZigzagFiltration
from zztop.zigzag.engine import ZigzagEngine


# ======================================================================
# Helpers: cubical → simplicial triangulation
# ======================================================================

def _cubical_to_simplicial_frame(cubical_cells, grid_shape):
    """Triangulate active cubical cells into simplices (2-D only)."""
    simplices = set()
    for cell in cubical_cells:
        if cell.dim == 0:
            simplices.add(Simplex([_vertex_index(cell, grid_shape)]))
        elif cell.dim == 1:
            lo = _vertex_index(_lower_vertex(cell), grid_shape)
            hi = _vertex_index(_upper_vertex(cell), grid_shape)
            simplices.add(Simplex([lo]))
            simplices.add(Simplex([hi]))
            simplices.add(Simplex([lo, hi]))
        elif cell.dim == 2:
            corners = _square_corners(cell, grid_shape)
            a, b, c, d = corners
            for tri in [(a, b, d), (a, c, d)]:
                t = Simplex(list(tri))
                for s in t.closure():
                    simplices.add(s)
    return simplices


def _vertex_index(cell, grid_shape):
    coords = tuple(lo for (lo, hi) in cell.intervals)
    idx, multiplier = 0, 1
    for i in range(len(coords) - 1, -1, -1):
        idx += coords[i] * multiplier
        multiplier *= grid_shape[i]
    return idx


def _lower_vertex(cell):
    return CubicalCell(tuple((lo, lo) for (lo, hi) in cell.intervals))


def _upper_vertex(cell):
    return CubicalCell(tuple((hi, hi) for (lo, hi) in cell.intervals))


def _square_corners(cell, grid_shape):
    nd_axes = [i for i, (lo, hi) in enumerate(cell.intervals) if lo != hi]
    assert len(nd_axes) == 2
    corners = []
    for bits in range(4):
        intervals, j = [], 0
        for i, (lo, hi) in enumerate(cell.intervals):
            if lo != hi:
                val = hi if (bits & (1 << j)) else lo
                intervals.append((val, val))
                j += 1
            else:
                intervals.append((lo, lo))
        corners.append(_vertex_index(CubicalCell(tuple(intervals)), grid_shape))
    return corners


# ======================================================================
# Test class
# ======================================================================

class TestCubicalVsSimplicialParity:
    """Compare persistence diagrams from native cubical vs
    triangulated-simplicial pipelines."""

    @staticmethod
    def _make_test_grid(shape=(4, 4), n_frames=5, seed=99):
        return np.random.RandomState(seed).randn(*shape, n_frames)

    def _run_cubical_native(self, grid_data):
        from zztop.builders.cubical_builder import _grid_cells_direct
        ndim = grid_data.ndim - 1
        frames = []
        for t in range(grid_data.shape[-1]):
            slicing = tuple([slice(None)] * ndim + [t])
            frames.append(_grid_cells_direct(grid_data[slicing], 0.0))
        filt = ZigzagFiltration(
            frames=frames,
            cell_dim_fn=lambda c: c.dim,
            cell_boundary_fn=lambda c: c.boundary(),
        )
        return bars_to_diagrams(ZigzagEngine(backend="python").run(filt)), filt

    def _run_simplicial_converted(self, grid_data):
        from zztop.builders.cubical_builder import _grid_cells_direct
        ndim = grid_data.ndim - 1
        grid_shape = grid_data.shape[:-1]
        frames = []
        for t in range(grid_data.shape[-1]):
            slicing = tuple([slice(None)] * ndim + [t])
            active_cub = _grid_cells_direct(grid_data[slicing], 0.0)
            frames.append(
                _cubical_to_simplicial_frame(active_cub, grid_shape)
            )
        filt = ZigzagFiltration(
            frames=frames,
            cell_dim_fn=lambda s: s.dimension,
            cell_boundary_fn=lambda s: s.faces(),
        )
        return bars_to_diagrams(ZigzagEngine(backend="python").run(filt)), filt

    # ------------------------------------------------------------------

    def test_betti_numbers_match_per_frame(self):
        """β_k at each original frame must agree between the two approaches."""
        grid = self._make_test_grid(shape=(3, 3), n_frames=3, seed=42)
        pd_c, filt_c = self._run_cubical_native(grid)
        pd_s, filt_s = self._run_simplicial_converted(grid)

        # Filter out zero-length bars (b == d): these are trivial features
        # from cell-level operations and differ between representations.
        def _nontrivial(dgms):
            return {
                dim: [(b, d) for b, d in pts if b != d]
                for dim, pts in dgms.items()
            }
        pd_c = _nontrivial(pd_c)
        pd_s = _nontrivial(pd_s)

        for dim in [0, 1]:
            for t in range(filt_c.n_frames):
                layer = 2 * t + 1  # 1-based layer for original frame t
                b_c = diagram_betti_at(pd_c, dim, layer)
                b_s = diagram_betti_at(pd_s, dim, layer)
                assert b_c == b_s, (
                    f"β_{dim} mismatch at frame {t} (layer {layer}): "
                    f"cubical={b_c}, simplicial={b_s}\n"
                    f"  cubical PD[{dim}] = {pd_c.get(dim, [])}\n"
                    f"  simplic PD[{dim}] = {pd_s.get(dim, [])}"
                )

    def test_total_persistence_comparable(self):
        """Total persistence must be in the same ballpark.

        When one side is 0 and the other is small, that is acceptable —
        triangulation adds diagonal edges that affect intersection layers.
        """
        grid = self._make_test_grid(shape=(3, 3), n_frames=3, seed=42)
        pd_c, _ = self._run_cubical_native(grid)
        pd_s, _ = self._run_simplicial_converted(grid)

        for dim in [0, 1]:
            tp_c = diagram_total_persistence(pd_c, dim)
            tp_s = diagram_total_persistence(pd_s, dim)

            if tp_c == 0 and tp_s == 0:
                continue
            n_layers = 2 * grid.shape[-1] - 1
            if tp_c == 0 or tp_s == 0:
                assert max(tp_c, tp_s) <= 2 * n_layers, (
                    f"dim={dim}: one side 0, other too large "
                    f"(cub={tp_c}, simp={tp_s})"
                )
                continue
            ratio = tp_c / tp_s
            assert 0.2 < ratio < 5.0, (
                f"dim={dim}: total persistence ratio {ratio:.2f} extreme "
                f"(cub={tp_c}, simp={tp_s})"
            )

    def test_h0_diagram_count_matches(self):
        """Number of H0 points must be the same (both start from the
        same vertex set, so connected-component births are identical)."""
        grid = self._make_test_grid(shape=(3, 3), n_frames=3, seed=42)
        pd_c, _ = self._run_cubical_native(grid)
        pd_s, _ = self._run_simplicial_converted(grid)

        n_c = len(pd_c.get(0, []))
        n_s = len(pd_s.get(0, []))
        assert n_c == n_s, (
            f"H0 point count: cubical={n_c}, simplicial={n_s}"
        )

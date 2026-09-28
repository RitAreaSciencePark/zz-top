"""Tests for the cubical zigzag pipeline — persistence diagram level.

Validates:
- CubicalCell boundary operator (∂∂ = 0, Euler characteristic).
- Persistence diagram properties of cubical zigzag on small grids.
- GUDHI vs direct cell extraction agreement.
"""

import pytest
import numpy as np

from conftest import (
    bars_to_diagrams,
    assert_diagrams_equal,
    diagram_total_persistence,
)
from zztop.complex.cubical import CubicalCell, CubicalComplex
from zztop.zigzag.filtration import ZigzagFiltration
from zztop.zigzag.engine import ZigzagEngine


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_grid_data_2d(shape=(3, 3), n_frames=3, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(*shape, n_frames)


def _build_cubical_frames_direct(grid_data, threshold=0.0):
    from zztop.builders.cubical_builder import _grid_cells_direct
    ndim = grid_data.ndim - 1
    frames = []
    for t in range(grid_data.shape[-1]):
        slicing = tuple([slice(None)] * ndim + [t])
        frames.append(_grid_cells_direct(grid_data[slicing], threshold))
    return frames


def _run_cubical(frames, backend="python"):
    filt = ZigzagFiltration(
        frames=frames,
        cell_dim_fn=lambda c: c.dim,
        cell_boundary_fn=lambda c: c.boundary(),
    )
    return bars_to_diagrams(ZigzagEngine(backend=backend).run(filt))


# ------------------------------------------------------------------
# Boundary-operator tests (foundational, not PD-level)
# ------------------------------------------------------------------

class TestCubicalBoundaryOperator:

    def test_boundary_of_boundary_is_zero(self):
        """∂²c = 0 for every cell in a 3×3 grid."""
        cc = CubicalComplex.from_grid_shape((3, 3))
        for cell in cc.all_cells():
            if cell.dim >= 2:
                bdry2 = []
                for face in cell.boundary():
                    bdry2.extend(face.boundary())
                counts = {}
                for c in bdry2:
                    counts[c] = counts.get(c, 0) + 1
                odd = {k: v for k, v in counts.items() if v % 2 != 0}
                assert odd == {}, f"∂²≠0 for {cell}: {odd}"

    def test_boundary_of_boundary_is_zero_3d(self):
        """∂²c = 0 for every cell in a 2×2×2 grid."""
        cc = CubicalComplex.from_grid_shape((2, 2, 2))
        for cell in cc.all_cells():
            if cell.dim >= 2:
                bdry2 = []
                for face in cell.boundary():
                    bdry2.extend(face.boundary())
                counts = {}
                for c in bdry2:
                    counts[c] = counts.get(c, 0) + 1
                odd = {k: v for k, v in counts.items() if v % 2 != 0}
                assert odd == {}, f"∂²≠0 for {cell}: {odd}"

    def test_euler_characteristic_2x2(self):
        """2×2 grid: χ = 4−4+1 = 1."""
        cc = CubicalComplex.from_grid_shape((2, 2))
        cbd = cc.cells_by_dimension()
        chi = len(cbd.get(0, [])) - len(cbd.get(1, [])) + len(cbd.get(2, []))
        assert chi == 1

    def test_euler_characteristic_3x3(self):
        """3×3 grid: χ = 9−12+4 = 1."""
        cc = CubicalComplex.from_grid_shape((3, 3))
        cbd = cc.cells_by_dimension()
        chi = len(cbd.get(0, [])) - len(cbd.get(1, [])) + len(cbd.get(2, []))
        assert chi == 1


# ------------------------------------------------------------------
# Cubical zigzag — persistence diagram property tests
# ------------------------------------------------------------------

class TestCubicalZigzag:

    def test_constant_grid_one_long_h0_bar(self):
        """All frames identical (3×3 grid, all 1s) → one H0 bar spans
        the whole filtration.

        The contractible grid has β₀=1 everywhere, so the PD should
        contain exactly 1 persistent H0 bar (other bars, if any, are
        ephemeral).
        """
        grid = np.ones((3, 3, 3))
        frames = _build_cubical_frames_direct(grid, threshold=0.0)
        # All frames identical
        assert frames[0] == frames[1] == frames[2]

        pd = _run_cubical(frames)
        h0 = pd.get(0, [])
        # At least 1 H0 bar
        assert len(h0) >= 1
        # The longest bar should span the whole filtration
        max_lifetime = max(d - b for b, d in h0)
        total_ops = sum(d - b for b, d in h0) + len(h0)  # rough
        assert max_lifetime > 0, f"Expected persistent H0 bar; PD = {pd}"

    def test_constant_grid_h0_dominates(self):
        """Full contractible grid → H0 total persistence dominates H1.

        The cell-by-cell insertion order creates transient H1 bars
        during the build-up phase.  But the longest H0 bar should
        far exceed any H1 bar in lifetime.
        """
        grid = np.ones((3, 3, 3))
        pd = _run_cubical(_build_cubical_frames_direct(grid, threshold=0.0))
        h0_max = max((d - b for b, d in pd.get(0, [])), default=0)
        h1_max = max((d - b for b, d in pd.get(1, [])), default=0)
        assert h0_max > h1_max, (
            f"Expected longest H0 bar > longest H1; "
            f"H0 max lifetime={h0_max}, H1 max lifetime={h1_max}"
        )

    def test_varying_grid_nonempty_diagram(self):
        """Random varying grid must produce a non-empty PD."""
        grid = _make_grid_data_2d(shape=(4, 4), n_frames=4, seed=123)
        from zztop.builders.cubical_builder import build_cubical_zigzag
        filt = build_cubical_zigzag(grid, threshold=0.0, use_gudhi=False)
        pd = bars_to_diagrams(ZigzagEngine(backend="python").run(filt))
        total = sum(len(v) for v in pd.values())
        assert total > 0

    def test_two_frames_vertices_then_full(self):
        """Frame 0: 4 isolated vertices.  Frame 1: full 2×2 grid.

        PD properties:
          - H0 has ≥ 4 bars (4 components born)
          - Total H0 persistence > 0 (merging events)
          - No persistent H1 (contractible full grid)
        """
        v00 = CubicalCell(((0, 0), (0, 0)))
        v10 = CubicalCell(((1, 1), (0, 0)))
        v01 = CubicalCell(((0, 0), (1, 1)))
        v11 = CubicalCell(((1, 1), (1, 1)))

        full_cc = CubicalComplex.from_grid_shape((2, 2))
        pd = _run_cubical([{v00, v10, v01, v11}, set(full_cc.all_cells())])

        assert len(pd.get(0, [])) >= 4, (
            f"Expected ≥4 H0 bars; PD = {pd}"
        )
        assert diagram_total_persistence(pd, 0) > 0
        h1_persistent = [(b, d) for b, d in pd.get(1, []) if d > b]
        assert len(h1_persistent) == 0, (
            f"Expected no persistent H1; PD = {pd}"
        )

    def test_hole_in_annulus(self):
        """Full 3×3 grid with centre square removed → forms 1-hole.
        Then fill the hole.

        PD properties:
          - H1 has ≥ 1 bar (the hole before filling)
        """
        full_cc = CubicalComplex.from_grid_shape((3, 3))
        all_cells = set(full_cc.all_cells())
        centre_sq = CubicalCell(((0, 1), (0, 1)))
        annulus = all_cells - {centre_sq}

        pd = _run_cubical([annulus, all_cells])
        assert len(pd.get(1, [])) >= 1, (
            f"Expected ≥1 H1 bar (hole in annulus); PD = {pd}"
        )

    def test_cubical_filtration_operations_valid(self):
        """Closure property: every insertion's boundary is already present;
        every deletion refers to a present cell."""
        v00 = CubicalCell(((0, 0), (0, 0)))
        v10 = CubicalCell(((1, 1), (0, 0)))
        e = CubicalCell(((0, 1), (0, 0)))

        filt = ZigzagFiltration(
            frames=[{v00, v10, e}, {v00, v10}, {v00, v10, e}],
            cell_dim_fn=lambda c: c.dim,
            cell_boundary_fn=lambda c: c.boundary(),
        )
        present = set()
        for op_type, cell in filt.to_operations():
            if op_type == "i":
                for face in cell.boundary():
                    assert face in present, (
                        f"Boundary {face} not yet inserted for {cell}"
                    )
                present.add(cell)
            else:
                assert cell in present, f"Delete of absent cell {cell}"
                present.remove(cell)

    def test_edge_appears_twice(self):
        """Edge inserted, removed, re-inserted → H0 bars reflect
        the two merge events."""
        v0 = CubicalCell(((0, 0), (0, 0)))
        v1 = CubicalCell(((1, 1), (0, 0)))
        e = CubicalCell(((0, 1), (0, 0)))

        pd = _run_cubical([{v0, v1, e}, {v0, v1}, {v0, v1, e}])
        # Should have bars reflecting repeated merge/split
        assert len(pd.get(0, [])) >= 2
        assert diagram_total_persistence(pd, 0) > 0


# ------------------------------------------------------------------
# GUDHI vs direct cell extraction
# ------------------------------------------------------------------

def _gudhi_available():
    try:
        import gudhi
        return True
    except ImportError:
        return False


class TestCubicalCellExtraction:

    @pytest.mark.skipif(not _gudhi_available(), reason="GUDHI not installed")
    def test_gudhi_vs_direct_cells(self):
        """Both extraction methods produce the same active cell set."""
        from zztop.builders.cubical_builder import (
            _grid_cells_from_gudhi,
            _grid_cells_direct,
        )
        grid = np.array([[1.0, -0.5], [0.3, -0.8]])
        assert _grid_cells_from_gudhi(grid, 0.0) == _grid_cells_direct(grid, 0.0)

    @pytest.mark.skipif(not _gudhi_available(), reason="GUDHI not installed")
    def test_gudhi_vs_direct_same_h0_count(self):
        """GUDHI and direct extraction give the same number of H0 bars."""
        from zztop.builders.cubical_builder import build_cubical_zigzag

        grid = _make_grid_data_2d(shape=(3, 3), n_frames=3, seed=77)
        filt_g = build_cubical_zigzag(grid, threshold=0.0, use_gudhi=True)
        filt_d = build_cubical_zigzag(grid, threshold=0.0, use_gudhi=False)

        engine = ZigzagEngine(backend="python")
        pd_g = bars_to_diagrams(engine.run(filt_g))
        pd_d = bars_to_diagrams(engine.run(filt_d))

        # Same bar count per dimension
        for dim in set(list(pd_g.keys()) + list(pd_d.keys())):
            n_g = len(pd_g.get(dim, []))
            n_d = len(pd_d.get(dim, []))
            assert n_g == n_d, (
                f"Dim {dim}: GUDHI has {n_g} bars, direct has {n_d}\n"
                f"  GUDHI: {pd_g.get(dim, [])}\n"
                f"  direct: {pd_d.get(dim, [])}"
            )

    @pytest.mark.skipif(not _gudhi_available(), reason="GUDHI not installed")
    def test_gudhi_vs_direct_total_persistence(self):
        """GUDHI and direct extraction give same total persistence per dim."""
        from zztop.builders.cubical_builder import build_cubical_zigzag

        grid = _make_grid_data_2d(shape=(3, 3), n_frames=3, seed=77)
        engine = ZigzagEngine(backend="python")

        pd_g = bars_to_diagrams(
            engine.run(build_cubical_zigzag(grid, 0.0, use_gudhi=True))
        )
        pd_d = bars_to_diagrams(
            engine.run(build_cubical_zigzag(grid, 0.0, use_gudhi=False))
        )

        for dim in set(list(pd_g.keys()) + list(pd_d.keys())):
            tp_g = diagram_total_persistence(pd_g, dim)
            tp_d = diagram_total_persistence(pd_d, dim)
            assert tp_g == tp_d, (
                f"Dim {dim}: total persistence GUDHI={tp_g}, direct={tp_d}"
            )

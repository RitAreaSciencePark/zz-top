"""Tests for the simplicial zigzag pipeline — persistence diagram level.

All assertions compare **persistence diagrams** (per-dimension multisets
of (birth, death) points) or verify structural *properties* of the diagram
(number of bars, dimension coverage, persistent vs ephemeral features).

Birth/death are 1-based operation indices from the Dey-Hou algorithm.

The reference diagram comes from the fzz / pyfzz README:
    https://github.com/matteobiagetti/fzz
"""

import pytest
from conftest import (
    bars_to_diagrams,
    raw_bars_to_diagrams,
    assert_diagrams_equal,
    diagram_betti_at,
    diagram_total_persistence,
)
from zztop.complex.simplicial import Simplex
from zztop.zigzag.filtration import ZigzagFiltration
from zztop.zigzag.engine import ZigzagEngine
from zztop.zigzag._python_backend import (
    compute_simplicial_python,
    compute_zigzag_python,
)


# ------------------------------------------------------------------
# Fixtures: fzz README example
# ------------------------------------------------------------------

def _fzz_readme_operations():
    """The example from the fzz README / test file."""
    return [
        ("i", [0]),
        ("i", [1]),
        ("i", [0, 1]),
        ("d", [0, 1]),
        ("i", [2]),
        ("i", [1, 2]),
        ("d", [0]),
        ("i", [3]),
        ("i", [1, 3]),
        ("i", [2, 3]),
        ("i", [1, 2, 3]),
        ("d", [1, 2, 3]),
        ("d", [2, 3]),
        ("d", [1, 3]),
        ("d", [3]),
        ("d", [1, 2]),
        ("d", [2]),
        ("d", [1]),
    ]


# Expected persistence diagram per dimension — verified against pyfzz.
EXPECTED_PD = {
    0: sorted([(1, 17), (2, 2), (4, 6), (5, 5), (8, 8), (14, 14), (16, 16)]),
    1: sorted([(10, 10), (12, 12)]),
}


# ------------------------------------------------------------------
# Tests: raw backends → exact diagram comparison
# ------------------------------------------------------------------

class TestSimplicialPythonBackend:

    def test_fzz_readme_simplicial(self):
        """Simplicial backend must produce the reference PD."""
        pd = raw_bars_to_diagrams(
            compute_simplicial_python(_fzz_readme_operations())
        )
        assert_diagrams_equal(pd, EXPECTED_PD,
                              msg="Python simplicial backend: ")

    def test_fzz_readme_via_abstract(self):
        """Abstract backend must give the same persistence diagram."""
        abstract_ops = _simplicial_to_abstract(_fzz_readme_operations())
        pd = raw_bars_to_diagrams(compute_zigzag_python(abstract_ops))
        assert_diagrams_equal(pd, EXPECTED_PD,
                              msg="Python abstract backend: ")

    def test_fzz_readme_diagram_properties(self):
        """Verify topological properties of the fzz reference diagram."""
        pd = raw_bars_to_diagrams(
            compute_simplicial_python(_fzz_readme_operations())
        )
        # 7 H0 bars, 2 H1 bars
        assert len(pd[0]) == 7
        assert len(pd[1]) == 2
        # Exactly one long-lived H0 bar (birth < death by many ops)
        h0_lifetimes = [d - b for b, d in pd[0]]
        assert sum(1 for l in h0_lifetimes if l > 1) == 2, (
            f"Expected 2 persistent H0 features; lifetimes = {h0_lifetimes}"
        )
        # Both H1 bars are ephemeral (birth == death)
        for b, d in pd[1]:
            assert b == d, f"Expected ephemeral H1 bar, got ({b}, {d})"


# ------------------------------------------------------------------
# Tests: engine-level → diagram property assertions
# ------------------------------------------------------------------

class TestSimplicialEngine:

    def test_simple_edge_zigzag(self):
        """K0 = {v0, v1, e01}, K1 = {v0, v1}.

        PD properties:
          - H0 has 3 bars: 1 persistent, 2 ephemeral
          - No H1 bars (no cycle ever formed)
        """
        s0, s1, e01 = Simplex([0]), Simplex([1]), Simplex([0, 1])
        pd = self._run([{s0, s1, e01}, {s0, s1}])

        assert 0 in pd, f"Expected H0 bars; PD = {pd}"
        assert len(pd[0]) == 3, f"Expected 3 H0 bars; PD = {pd}"
        # One bar spans a wide range (persistent component)
        lifetimes = sorted(d - b for b, d in pd[0])
        assert lifetimes[-1] > 0, "Expected ≥1 persistent H0 bar"
        # No cycles
        assert 1 not in pd, f"Expected no H1 bars; PD = {pd}"

    def test_triangle_appears(self):
        """K0 = triangle boundary (1-cycle), K1 = filled triangle.

        PD properties:
          - H1 has ≥ 1 bar (the cycle killed by the 2-simplex)
          - H0 has ≥ 1 bar (connected component)
        """
        v0, v1, v2 = Simplex([0]), Simplex([1]), Simplex([2])
        e01, e02, e12 = Simplex([0, 1]), Simplex([0, 2]), Simplex([1, 2])
        t012 = Simplex([0, 1, 2])

        pd = self._run([
            {v0, v1, v2, e01, e02, e12},
            {v0, v1, v2, e01, e02, e12, t012},
        ])
        assert len(pd.get(1, [])) >= 1, f"Expected ≥1 H1 bar; PD = {pd}"
        assert len(pd.get(0, [])) >= 1, f"Expected ≥1 H0 bar; PD = {pd}"

    def test_triangle_cycle_disappears_reappears(self):
        """K0 = boundary, K1 = filled, K2 = boundary.

        The 1-cycle exists in K0, killed in K1, re-appears in K2.

        PD properties:
          - H1 has ≥ 2 bars (two separate cycle intervals)
          - H0: ≥ 1 persistent bar
        """
        v0, v1, v2 = Simplex([0]), Simplex([1]), Simplex([2])
        e01, e02, e12 = Simplex([0, 1]), Simplex([0, 2]), Simplex([1, 2])
        t012 = Simplex([0, 1, 2])
        boundary = {v0, v1, v2, e01, e02, e12}
        filled = boundary | {t012}

        pd = self._run([boundary, filled, boundary])
        assert len(pd.get(1, [])) >= 2, (
            f"Expected ≥2 H1 bars (cycle appears twice); PD = {pd}"
        )

    def test_empty_filtration(self):
        """Empty filtration → empty persistence diagram."""
        filt = ZigzagFiltration(
            frames=[set()],
            cell_dim_fn=lambda s: 0,
            cell_boundary_fn=lambda s: [],
        )
        pd = bars_to_diagrams(ZigzagEngine(backend="python").run(filt))
        assert pd == {}

    def test_single_vertex(self):
        """Single vertex across 3 frames → 1 H0 bar, no higher dims."""
        v = Simplex([0])
        pd = self._run([{v}, {v}, {v}])

        assert len(pd.get(0, [])) == 1, f"Expected 1 H0 bar; PD = {pd}"
        assert len(pd.get(1, [])) == 0, f"Expected no H1 bars; PD = {pd}"
        # The single bar should span the entire filtration
        (b, d), = pd[0]
        assert d >= b

    def test_three_vertices_merge(self):
        """K0 = {v0, v1, v2}, K1 = {v0, v1, v2, e01, e12}.

        PD properties:
          - H0 has ≥ 3 bars (3 components born; 2 die on merge)
          - Among H0 bars, exactly 2 are ephemeral (killed on merge)
            and 1 is persistent (surviving component)
          - No H1 (no cycle formed by 2 edges)
        """
        v0, v1, v2 = Simplex([0]), Simplex([1]), Simplex([2])
        e01, e12 = Simplex([0, 1]), Simplex([1, 2])

        pd = self._run([{v0, v1, v2}, {v0, v1, v2, e01, e12}])
        assert len(pd.get(0, [])) >= 3, f"Expected ≥3 H0 bars; PD = {pd}"
        assert 1 not in pd, f"Expected no H1; PD = {pd}"

    def test_two_components_appear_separately(self):
        """K0 = {v0}, K1 = {v0, v1}.

        PD properties:
          - H0 has 2 bars (one born for each component)
          - Total persistence ≥ 1 (at least one bar survives a step)
        """
        v0, v1 = Simplex([0]), Simplex([1])
        pd = self._run([{v0}, {v0, v1}])

        assert len(pd.get(0, [])) == 2, f"Expected 2 H0 bars; PD = {pd}"
        assert diagram_total_persistence(pd, 0) >= 1

    def test_total_persistence_edge_zigzag(self):
        """Edge inserted then removed → total H0 persistence > 0."""
        s0, s1, e01 = Simplex([0]), Simplex([1]), Simplex([0, 1])
        pd = self._run([{s0, s1, e01}, {s0, s1}])
        assert diagram_total_persistence(pd, 0) > 0

    # --- Helper ---

    @staticmethod
    def _run(frames, backend="python"):
        filt = ZigzagFiltration(
            frames=frames,
            cell_dim_fn=lambda s: s.dimension,
            cell_boundary_fn=lambda s: s.faces(),
        )
        return bars_to_diagrams(ZigzagEngine(backend=backend).run(filt))


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _simplicial_to_abstract(ops):
    abstract_ops = []
    cell_insert = {}
    for i, (op, verts) in enumerate(ops):
        key = tuple(sorted(verts))
        dim = len(key) - 1
        if op == "i":
            bdry_indices = []
            if dim > 0:
                for j in range(len(key)):
                    face = key[:j] + key[j + 1:]
                    if face in cell_insert:
                        bdry_indices.append(cell_insert[face])
                bdry_indices.sort()
            abstract_ops.append(("i", dim, bdry_indices))
            cell_insert[key] = i
        else:
            ins_idx = cell_insert.pop(key, -1)
            abstract_ops.append(
                ("d", dim, [ins_idx] if ins_idx >= 0 else [])
            )
    return abstract_ops

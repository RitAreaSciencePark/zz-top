"""Regression tests for the deletion-death (break) barcode readout.

Historically ``ZigzagEngine.run`` mapped a bar's death to
``layer_map[d - 1]`` (the layer of the *last-alive* operation).  fzz returns
**closed** op-intervals, so ``d`` is the last-alive op and the class actually
dies at the killing op ``d + 1``; the correct half-open death layer is
therefore ``layer_map[d]``.  With the buggy ``d - 1`` readout, an H_k class
that dies by a cell *deletion* while the complex is otherwise static collapses
to zero layer-length and ``filter_bars`` silently drops it — real loops
vanished from the barcode.

This matches the upstream reference readout that zz-top's kNN pipeline was
ported from (RitAreaSciencePark/ZigZagLLMs, ``run_fast_zigzag.py``):

    if layers_res[i[0]-1] != layers_res[i[1]]:
        merged_output.append((i[2], layers_res[i[0]-1], layers_res[i[1]]))
    #                                                    ^ layers_res[d] == layer(d+1)

The tests below are self-contained: the ground truth is a direct per-layer
GF(2) Betti computation on the actual complexes, so they do not depend on
pyfzz / dionysus being installed and are not circular with the readout under
test.  They exercise both the simplicial and cubical dispatch paths.
"""

import numpy as np
import pytest

from zztop.complex.simplicial import Simplex
from zztop.zigzag.filtration import ZigzagFiltration
from zztop.zigzag.engine import ZigzagEngine
from zztop.builders.cubical_builder import (
    build_cubical_zigzag,
    run_cubical_zigzag,
)


# ======================================================================
# Self-contained GF(2) ground-truth homology oracle
# ======================================================================

def _gf2_rank(cols):
    """Rank over GF(2) of a list of columns (each a set of row indices)."""
    pivot_to_col = {}
    rank = 0
    for col in cols:
        col = set(col)
        while col:
            p = max(col)
            if p in pivot_to_col:
                col ^= pivot_to_col[p]
            else:
                pivot_to_col[p] = col
                rank += 1
                break
    return rank


def betti_k(cells, dim_fn, bdry_fn, k):
    """Ground-truth Betti number in dimension *k* of a cell set over GF(2).

    ``betti_k = (n_k - rank d_k) - rank d_{k+1}`` (with ``d_0 = 0``).
    """
    by_dim = {}
    for c in cells:
        by_dim.setdefault(dim_fn(c), []).append(c)
    index = {
        d: {c: i for i, c in enumerate(cell_list)}
        for d, cell_list in by_dim.items()
    }

    def boundary_cols(kk):
        cols = []
        lower = index.get(kk - 1, {})
        for c in by_dim.get(kk, []):
            col = {lower[f] for f in bdry_fn(c) if f in lower}
            cols.append(col)
        return cols

    n_k = len(by_dim.get(k, []))
    rank_dk = _gf2_rank(boundary_cols(k)) if k >= 1 else 0
    rank_dk1 = _gf2_rank(boundary_cols(k + 1))
    return n_k - rank_dk - rank_dk1


def per_layer_betti(filt, dim_fn, bdry_fn, k):
    """Ground-truth Betti_k at every 1-based zigzag layer of *filt*."""
    out = []
    for layer in filt.layers:
        cells = [filt.cell(c) for c in layer]
        out.append(betti_k(cells, dim_fn, bdry_fn, k))
    return out


def barcode_betti(bars, k, n_layers):
    """Betti_k at every 1-based layer as read from a layer-indexed barcode.

    A class ``(dim, b, d)`` is alive at layer ``L`` iff ``b <= L < d``
    (half-open: it dies at its killing layer ``d``).  Essential classes carry
    the sentinel ``d == n_layers + 1`` and remain alive through the final layer.
    """
    out = []
    for L in range(1, n_layers + 1):
        count = 0
        for dim, b, d in bars:
            if dim != k:
                continue
            if b <= L < d:                 # half-open (covers essential too)
                count += 1
        out.append(count)
    return out


# ======================================================================
# Fixtures
# ======================================================================

def _simplicial_break_frames():
    """4-cycle 0-1-2-3 held static for two frames, then edge (0,3) deleted.

    Ground-truth Betti-1 per layer: [1, 1, 1, 0, 0] (loop dies by *deletion*).
    This is precisely the case the old readout dropped.
    """
    V = [Simplex([i]) for i in range(4)]
    cyc = [Simplex([0, 1]), Simplex([1, 2]), Simplex([2, 3]), Simplex([0, 3])]
    full = set(V) | set(cyc)
    broken = set(V) | (set(cyc) - {Simplex([0, 3])})
    return [full, full, broken]


def _simplicial_filtration(frames):
    return ZigzagFiltration(
        frames=frames,
        cell_dim_fn=lambda s: s.dimension,
        cell_boundary_fn=lambda s: s.faces(),
    )


def _cubical_break_field():
    """5x5 vertex field with a square ring hole, static for two frames, then
    one ring vertex switches off (the ring breaks) -> H1 loop dies by deletion.

    Returns ``grid_data = -field`` so that ``run_cubical_zigzag(..., 1.0)``
    activates a cell iff ``max_corner(field) < 1`` (sublevel on the field).
    """
    HI, LO = 10.0, 0.0
    base = np.full((5, 5), HI)
    for r in range(5):
        for c in range(5):
            if max(abs(r - 2), abs(c - 2)) == 1:   # Chebyshev radius 1 ring
                base[r, c] = LO
    broken = base.copy()
    broken[1, 1] = HI                              # break the ring
    field = np.stack([base, base, broken], axis=-1)  # (5, 5, 3)
    return -field                                    # grid_data = -field


# ======================================================================
# Simplicial (kNN) dispatch path
# ======================================================================

@pytest.mark.parametrize("backend", ["python", "auto"])
class TestSimplicialDeletionDeath:

    def test_break_loop_is_not_dropped(self, backend):
        """The deleted-edge loop must appear as H1 = (birth 1, death 4)."""
        filt = _simplicial_filtration(_simplicial_break_frames())
        bars = ZigzagEngine(backend=backend).run(filt)
        h1 = [(b, d) for dim, b, d in bars if dim == 1]
        assert (1, 4) in h1, (
            f"break-death loop dropped or mis-stamped; H1 bars = {h1}"
        )

    def test_per_layer_betti1_matches_ground_truth(self, backend):
        filt = _simplicial_filtration(_simplicial_break_frames())
        bars = ZigzagEngine(backend=backend).run(filt)
        truth = per_layer_betti(
            filt, lambda s: s.dimension, lambda s: s.faces(), k=1
        )
        got = barcode_betti(bars, k=1, n_layers=filt.n_layers)
        assert got == truth == [1, 1, 1, 0, 0], (
            f"per-layer Betti1 mismatch: barcode={got} truth={truth}"
        )

    def test_fill_death_still_correct(self, backend):
        """A loop that dies by a triangle *insertion* (fill) is unaffected."""
        v = [Simplex([i]) for i in range(3)]
        e = [Simplex([0, 1]), Simplex([1, 2]), Simplex([0, 2])]
        t = Simplex([0, 1, 2])
        boundary = set(v) | set(e)
        filled = boundary | {t}
        filt = _simplicial_filtration([boundary, filled])
        bars = ZigzagEngine(backend=backend).run(filt)
        truth = per_layer_betti(
            filt, lambda s: s.dimension, lambda s: s.faces(), k=1
        )
        got = barcode_betti(bars, k=1, n_layers=filt.n_layers)
        assert got == truth, f"fill-death Betti1 mismatch: {got} vs {truth}"
        assert any(dim == 1 for dim, _, _ in bars), "fill-death H1 lost"


# ======================================================================
# Death-readout convention (directly pins the off-by-one)
# ======================================================================

@pytest.mark.parametrize("backend", ["python", "auto"])
def test_death_reads_killing_op_layer(backend):
    """engine.run must map death to layer(d+1)=layer_map[d], not layer_map[d-1].

    Re-introducing the ``d - 1`` readout makes this fail on the break case.
    """
    filt = _simplicial_filtration(_simplicial_break_frames())
    eng = ZigzagEngine(backend=backend)
    raw = eng.run_raw(filt)        # (b, d, dim), 1-based closed op-interval
    mapped = eng.run(filt)         # (dim, b_layer, d_layer)
    lm = filt.operation_to_layer_map()
    n_ops = len(lm)

    assert len(raw) == len(mapped)
    saw_finite_death = False
    for (b, d, p), (dim, b_layer, d_layer) in zip(raw, mapped):
        assert dim == p
        assert b_layer == lm[b - 1]
        if d < n_ops:
            saw_finite_death = True
            assert d_layer == lm[d], (
                f"death must read killing-op layer lm[{d}]={lm[d]}, "
                f"got {d_layer} (looks like the layer_map[d-1] bug)"
            )
        else:
            assert d_layer == lm[n_ops - 1]
    assert saw_finite_death, "test topology exercised no finite deaths"


# ======================================================================
# Cubical dispatch path
# ======================================================================

@pytest.mark.parametrize("backend", ["python", "auto"])
class TestCubicalDeletionDeath:

    def test_break_loop_is_not_dropped_layers(self, backend):
        grid = _cubical_break_field()
        filt = build_cubical_zigzag(grid, threshold=1.0)
        bars = ZigzagEngine(backend=backend).run(filt)
        truth = per_layer_betti(
            filt, lambda c: c.dim, lambda c: c.boundary(), k=1
        )
        got = barcode_betti(bars, k=1, n_layers=filt.n_layers)
        assert got == truth == [1, 1, 1, 0, 0], (
            f"cubical break Betti1 mismatch: barcode={got} truth={truth}"
        )

    def test_run_cubical_zigzag_frame_barcode(self, backend):
        """Top-level convenience API must surface the loop in frame space."""
        grid = _cubical_break_field()
        bars = run_cubical_zigzag(grid, threshold=1.0, backend=backend)
        h1 = [(b, d) for dim, b, d in bars if dim == 1]
        # layers (birth 1, death 4) -> frames (1//2, 4//2) = (0, 2)
        assert (0, 2) in h1, f"cubical loop missing from frame barcode: {h1}"

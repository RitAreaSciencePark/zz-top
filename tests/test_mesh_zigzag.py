"""Tests for the mesh zigzag pipeline — persistence diagram level.

A scalar field evolves in time on a **fixed** triangulated surface mesh; the
co-active sub-complex is tracked with zigzag persistence.  Assertions compare
per-dimension Betti numbers read off the persistence diagram, following
``tests/test_simplicial_zigzag.py`` and the ``tests/conftest.py`` helpers.

Conventions (mirroring ``tests/test_parity.py``):
- Zero-length bars (``birth == death`` in layer numbering) are cell-insertion
  artifacts and are filtered out before reading Betti numbers.
- The Betti number of the original **frame** ``t`` is read at the 1-based
  odd **layer** ``2*t + 1`` (``K0`` = 1, ``K1`` = 3, ``K2`` = 5, …); even
  layers are the intersection complexes.
"""

import numpy as np
import pytest

from conftest import (
    bars_to_diagrams,
    diagram_betti_at,
)
from zztop.complex.simplicial import Simplex
from zztop.zigzag.filtration import ZigzagFiltration
from zztop.zigzag.engine import ZigzagEngine
from zztop.builders.mesh_builder import (
    build_mesh_zigzag,
    run_mesh_zigzag,
    _mesh_frames,
    _unique_edges,
)


# ------------------------------------------------------------------
# Small hand-built meshes with known homology
# ------------------------------------------------------------------

# A "necklace" annulus: three triangles joined pairwise at single vertices,
# leaving a central triangular hole bounded by 0–2–4.  β0=1, β1=1, β2=0.
NECKLACE_FACES = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 0]])

# An octahedron: a triangulation of the 2-sphere S².  β0=1, β1=0, β2=1.
OCTAHEDRON_FACES = np.array(
    [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4],
     [0, 1, 5], [1, 2, 5], [2, 3, 5], [3, 0, 5]]
)

# A single filled triangle: contractible.  β0=1, β1=0, β2=0.
TRIANGLE_FACES = np.array([[0, 1, 2]])


# ------------------------------------------------------------------
# Diagram helpers (frame-Betti convention)
# ------------------------------------------------------------------

def _nontrivial(dgm):
    """Drop zero-length (``b == d``) bars from a per-dimension diagram."""
    return {
        dim: [(b, d) for b, d in pts if b != d]
        for dim, pts in dgm.items()
    }


def _frame_betti(bars, dim, frame):
    """β_dim at original *frame* (filtered diagram, odd zigzag layer)."""
    pd = _nontrivial(bars_to_diagrams(bars))
    return diagram_betti_at(pd, dim, 2 * frame + 1)


def _is_closed(frame):
    """True iff every face of every simplex in *frame* is also present."""
    return all(f in frame for s in frame for f in s.faces())


def _induced_subcomplex(faces, edges, active):
    """Independently build the mesh sub-complex induced on vertex set *active*."""
    active = set(int(v) for v in active)
    cx = {Simplex((v,)) for v in active}
    for i, j in edges:
        if int(i) in active and int(j) in active:
            cx.add(Simplex((int(i), int(j))))
    for f in faces:
        if all(int(v) in active for v in f):
            cx.add(Simplex(tuple(int(v) for v in f)))
    return cx


# ==================================================================
# 1. Closure invariant
# ==================================================================

class TestClosureInvariant:
    """Every produced frame must be a closed complex, for any reducer."""

    @pytest.mark.parametrize(
        "coactivity", ["min", "mean", "product", "max", lambda a, b: np.abs(a - b)]
    )
    @pytest.mark.parametrize("include_isolated_vertices", [True, False])
    @pytest.mark.parametrize("strict", [True, False])
    def test_frames_are_closed(self, coactivity, include_isolated_vertices, strict):
        rng = np.random.RandomState(1234)
        activity = rng.randn(6, 5)
        threshold = float(rng.uniform(-0.5, 0.5))
        frames = _mesh_frames(
            OCTAHEDRON_FACES,
            activity,
            threshold,
            coactivity=coactivity,
            include_isolated_vertices=include_isolated_vertices,
            strict=strict,
        )
        for t, frame in enumerate(frames):
            assert _is_closed(frame), (
                f"frame {t} is not closed under taking faces"
            )


# ==================================================================
# 2. Excursion-set equivalence
# ==================================================================

class TestExcursionEquivalence:
    """min-reduction with isolated vertices = superlevel induced sub-complex."""

    def test_matches_induced_subcomplex(self):
        rng = np.random.RandomState(7)
        activity = rng.randn(6, 4)
        threshold = 0.15
        edges, _ = _unique_edges(OCTAHEDRON_FACES, activity.shape[0])
        frames = _mesh_frames(
            OCTAHEDRON_FACES,
            activity,
            threshold,
            coactivity="min",
            include_isolated_vertices=True,
        )
        for t, frame in enumerate(frames):
            active = np.nonzero(activity[:, t] > threshold)[0]
            expected = _induced_subcomplex(OCTAHEDRON_FACES, edges, active)
            assert frame == expected, (
                f"frame {t} != induced subcomplex on {sorted(active)}"
            )


# ==================================================================
# 3. Filled triangle, constant high activity
# ==================================================================

class TestFilledTriangle:

    def test_disk_betti(self):
        """A filled triangle held active ⇒ β0=1, β1=0 (contractible disk)."""
        activity = np.ones((3, 3))
        bars = run_mesh_zigzag(TRIANGLE_FACES, activity, threshold=0.5)
        assert _frame_betti(bars, 0, 1) == 1
        assert _frame_betti(bars, 1, 1) == 0
        # The 2-simplex is filled: no persistent 2-cycle.
        assert _frame_betti(bars, 2, 1) == 0


# ==================================================================
# 4. Annulus born, then cut
# ==================================================================

class TestAnnulusCut:
    """A triangulated ring holds an H1 class that dies when the ring is cut."""

    def test_h1_born_then_dies(self):
        # Frames: [full, full, cut].  In the third frame vertex 0 (on the
        # hole boundary) drops below threshold, opening the loop into a disk.
        activity = np.ones((6, 3))
        activity[0, 2] = 0.0
        bars = run_mesh_zigzag(
            NECKLACE_FACES, activity, threshold=0.5, coactivity="min"
        )

        # Per-frame Betti: the hole is present at frames 0 and 1, gone at 2.
        assert _frame_betti(bars, 1, 0) == 1
        assert _frame_betti(bars, 1, 1) == 1
        assert _frame_betti(bars, 1, 2) == 0
        # The complex stays connected throughout.
        assert [_frame_betti(bars, 0, t) for t in range(3)] == [1, 1, 1]

        # Exactly one persistent H1 bar, born in K0 (layer 1) and dying at the
        # K1∩K2 intersection (layer 4), the first layer without the full ring.
        pd = _nontrivial(bars_to_diagrams(bars))
        assert pd.get(1) == [(1, 4)], f"unexpected H1 diagram: {pd.get(1)}"


# ==================================================================
# 5. Closed-surface sanity (static 2-sphere)
# ==================================================================

class TestClosedSurface:
    """A fully active octahedron is the S² a whole hemisphere looks like."""

    def test_sphere_betti(self):
        activity = np.ones((6, 2))
        bars = run_mesh_zigzag(OCTAHEDRON_FACES, activity, threshold=0.5)

        # β0 = 1, β1 = 0 from the persistent (non-trivial) diagram.
        assert _frame_betti(bars, 0, 0) == 1
        assert _frame_betti(bars, 1, 0) == 0

        # β2 = 1: the sphere carries one essential 2-dimensional class.  It is
        # born in K0 (layer 1) and, being essential, is reported with the
        # half-open sentinel death n_layers + 1 = 4 (two frames → three layers),
        # i.e. alive through the final layer.  (Before the deletion-death fix
        # the engine collapsed every essential/deletion class onto its birth
        # layer, so this read as a degenerate zero-length bar (1, 1).)
        pd_all = bars_to_diagrams(bars)
        assert pd_all.get(2) == [(1, 4)], (
            f"expected one essential H2 class at (1, 4); got {pd_all.get(2)}"
        )
        assert diagram_betti_at(pd_all, 2, 1) == 1


# ==================================================================

class TestVertexMask:

    def test_mask_removes_vertex_and_incident_simplices(self):
        activity = np.ones((6, 3))
        mask = np.ones(6, dtype=bool)
        mask[5] = False  # drop vertex 5 (and its incident edges/faces)

        masked = _mesh_frames(OCTAHEDRON_FACES, activity, 0.5, vertex_mask=mask)
        full = _mesh_frames(OCTAHEDRON_FACES, activity, 0.5)

        for t, frame in enumerate(masked):
            # No simplex in any frame touches the masked-out vertex.
            assert all(5 not in s.vertices for s in frame), (
                f"frame {t} still contains vertex 5"
            )
            # Everything else that does not touch vertex 5 is unchanged.
            expected = {s for s in full[t] if 5 not in s.vertices}
            assert frame == expected

    def test_masked_run_executes(self):
        """A masked run produces a valid barcode (smoke test)."""
        activity = np.ones((6, 2))
        mask = np.ones(6, dtype=bool)
        mask[5] = False
        bars = run_mesh_zigzag(OCTAHEDRON_FACES, activity, 0.5, vertex_mask=mask)
        assert isinstance(bars, list)


# ==================================================================
# Public one-call entry point (the documented DONE example)
# ==================================================================

def test_run_mesh_zigzag_smoke():
    faces = NECKLACE_FACES
    activity = np.ones((6, 3))
    bars = run_mesh_zigzag(faces, activity, threshold=0.5, coactivity="min")
    assert all(len(b) == 3 for b in bars)


# ==================================================================
# 8. Temporal hysteresis (two-threshold presence gate)
# ==================================================================

from zztop.builders.mesh_builder import _hysteresis_gate  # noqa: E402


class TestHysteresisGate:
    """Unit behaviour of the Schmitt-trigger temporal gate."""

    def test_seed_and_hold(self):
        # One cell: below, spike above high, dither in the band, drop below low.
        #            t: 0    1    2    3    4    5
        sig = np.array([[0.1, 0.9, 0.5, 0.5, 0.05, 0.9]])
        on = _hysteresis_gate(sig, low=0.4, high=0.8, strict=True)
        # t0 below high -> off; t1 crosses high -> on; t2,t3 in band, held on;
        # t4 below low -> off; t5 crosses high -> on again.
        assert on.tolist() == [[False, True, True, True, False, True]]

    def test_never_seeds_without_crossing_high(self):
        sig = np.array([[0.5, 0.6, 0.7, 0.5]])  # in band but never > high
        on = _hysteresis_gate(sig, low=0.4, high=0.8)
        assert not on.any()

    def test_reduces_to_single_threshold_when_low_equals_high(self):
        rng = np.random.RandomState(0)
        sig = rng.randn(4, 10)
        thr = 0.2
        on = _hysteresis_gate(sig, low=thr, high=thr, strict=True)
        # low == high: a cell is on at t iff it is above thr at t (seed) OR was
        # on and still above thr -> exactly (sig > thr) pointwise.
        assert np.array_equal(on, sig > thr)


class TestHysteresisFrames:
    """Hysteresis in _mesh_frames / builders."""

    def test_none_matches_single_threshold(self):
        rng = np.random.RandomState(3)
        activity = rng.randn(6, 5)
        a = _mesh_frames(OCTAHEDRON_FACES, activity, 0.1, threshold_high=None)
        b = _mesh_frames(OCTAHEDRON_FACES, activity, 0.1)
        assert a == b

    @pytest.mark.parametrize(
        "coactivity", ["min", "mean", "product", "max", lambda a, b: np.abs(a - b)]
    )
    @pytest.mark.parametrize("include_isolated_vertices", [True, False])
    def test_frames_are_closed(self, coactivity, include_isolated_vertices):
        rng = np.random.RandomState(99)
        activity = rng.randn(6, 8)
        frames = _mesh_frames(
            OCTAHEDRON_FACES, activity, -0.3, threshold_high=0.3,
            coactivity=coactivity,
            include_isolated_vertices=include_isolated_vertices,
        )
        for t, frame in enumerate(frames):
            assert _is_closed(frame), f"frame {t} not closed under hysteresis"

    def test_high_below_low_raises(self):
        activity = np.ones((6, 3))
        with pytest.raises(ValueError, match="threshold_high"):
            _mesh_frames(OCTAHEDRON_FACES, activity, 0.8, threshold_high=0.4)

    def test_hysteresis_holds_loop_through_a_dip(self):
        """A hole-boundary vertex dips into the band: a single high cut opens
        the loop, hysteresis holds it closed."""
        # Necklace hole is bounded by 0-2-4; drop vertex 0 at frame 1 only.
        activity = np.ones((6, 3))
        activity[0, 1] = 0.6
        hi, lo = 0.8, 0.4

        # Single threshold at the HIGH level: loop opens at frame 1.
        bars_single = run_mesh_zigzag(NECKLACE_FACES, activity, threshold=hi,
                                      coactivity="min")
        assert [_frame_betti(bars_single, 1, t) for t in range(3)] == [1, 0, 1]

        # Hysteresis (enter hi, hold lo): vertex 0 stays on -> loop persists.
        bars_hyst = run_mesh_zigzag(NECKLACE_FACES, activity, threshold=lo,
                                    threshold_high=hi, coactivity="min")
        assert [_frame_betti(bars_hyst, 1, t) for t in range(3)] == [1, 1, 1]

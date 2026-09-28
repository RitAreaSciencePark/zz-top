"""Regression tests for the cone-dimension bug in the abstract-cell (cubical) path.

The Dey-Hou construction cones every deleted cell.  ``compute_zigzag_python``
(and, before its removal in 0.8.0, the C++ backend) used to set the cone's
dimension from its chain length, ``|chain| - 1``.  That is right for a simplex
(a k-simplex has k+1 faces, so the cone of it has k+2 entries and dimension
k+1) but wrong for a cube: a k-cube has 2k faces, so the cone over a square was
labelled 4 instead of 3 and the cone over a cube 6 instead of 4.  Two symptoms:

* The former C++ backend grouped columns by dimension in its twist reduction,
  so in a 3-D cubical complex the true 3-dimensional columns (cubes, label 3,
  and square cones, label 4) were split across passes and H1/H2 bars were
  paired incorrectly.  The Python reduction is a plain left-to-right reduction
  and was not affected by that part.
* The dimension of an open-closed interval is read from its birth column; for
  square cones this reported genuine H2 classes as "H3" (which ``max_dim=2``
  pipelines then discarded).

2-D cubical complexes and simplicial complexes are not affected (the labels are
either correct or consistently shifted, and no dimension-2 open-closed class
exists in the plane), which is why the existing 2-D cubical tests passed.

Ground truth here is independent of any zigzag code: at every zigzag layer the
number of bars alive in dimension k must equal the GF(2) Betti number of that
layer's complex (helpers shared with ``test_zigzag_deletion_death``).
"""

from collections import Counter

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from zztop.zigzag.engine import ZigzagEngine
from zztop.builders.cubical_builder import build_cubical_zigzag

from test_zigzag_deletion_death import barcode_betti, per_layer_betti

BACKENDS = ["python"]


def _smooth_field(shape, n_frames, seed):
    """Positive, spatio-temporally smooth random grid and its p30 threshold.

    Uses zztop's negated convention (a cell is active iff all its corners are
    >= the 30th percentile), as in the Sensorium turnover pipeline.  The seeds
    below were found to break the old engine on grids this small.
    """
    rng = np.random.default_rng(seed)
    g = gaussian_filter(rng.standard_normal(tuple(shape) + (n_frames,)), sigma=1.0)
    g = g - g.min() + 0.01
    return g, -float(np.percentile(g, 30))


#: (spatial shape, frames, seed) -- each triggered the old mispairing.
FIELDS = [((4, 4, 4), 6, 10), ((5, 5, 4), 6, 12)]


def _hollow_cube():
    """3x3x3 shell present in frames 1..3 (centre vertex off): one H2 cavity."""
    g = np.zeros((3, 3, 3, 5))
    g[..., 1:4] = 1.0
    g[1, 1, 1, :] = 0.0
    return g, -0.5


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape,n_frames,seed", FIELDS)
class TestCubical3DConeDimension:

    def test_per_layer_betti_matches_ground_truth(self, backend, shape, n_frames, seed):
        g, thr = _smooth_field(shape, n_frames, seed)
        filt = build_cubical_zigzag(g, threshold=thr)
        bars = ZigzagEngine(backend=backend).run(filt)
        for k in (0, 1, 2):
            truth = per_layer_betti(filt, lambda c: c.dim, lambda c: c.boundary(), k=k)
            got = barcode_betti(bars, k=k, n_layers=filt.n_layers)
            assert got == truth, f"H{k} per-layer Betti mismatch ({backend}): {got} vs {truth}"

    def test_no_bars_above_ambient_dimension(self, backend, shape, n_frames, seed):
        """A subcomplex of R^3 has no H3; the old readout reported H2 as H3."""
        g, thr = _smooth_field(shape, n_frames, seed)
        bars = ZigzagEngine(backend=backend).run(build_cubical_zigzag(g, threshold=thr))
        assert max(d for d, _, _ in bars) <= 2

    def test_prefix_restriction(self, backend, shape, n_frames, seed):
        """Bars of a prefix zigzag equal the full barcode restricted to it.

        The first L frames produce identical layer complexes in both runs, so
        every bar that dies strictly before the cut must coincide.
        """
        g, thr = _smooth_field(shape, n_frames, seed)
        L = n_frames - 2
        cut = 2 * L - 1
        full = ZigzagEngine(backend=backend).run(build_cubical_zigzag(g, threshold=thr))
        short = ZigzagEngine(backend=backend).run(build_cubical_zigzag(g[..., :L], threshold=thr))
        interior = lambda bars: Counter(b for b in bars if b[2] < cut)
        assert interior(short) == interior(full)


@pytest.mark.parametrize("backend", BACKENDS)
def test_hollow_cube_cavity(backend):
    g, thr = _hollow_cube()
    filt = build_cubical_zigzag(g, threshold=thr)
    bars = ZigzagEngine(backend=backend).run(filt)
    h2 = [(b, d) for dim, b, d in bars if dim == 2 and b < d]
    # the shell exists in frames 1..3 = layers 3..7; it dies at layer 8 (K3 ∩ K4)
    assert h2 == [(3, 8)], f"expected one cavity over layers [3, 8), got {h2}"
    truth = per_layer_betti(filt, lambda c: c.dim, lambda c: c.boundary(), k=2)
    assert barcode_betti(bars, k=2, n_layers=filt.n_layers) == truth

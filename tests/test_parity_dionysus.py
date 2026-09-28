"""Test parity against Dionysus zigzag_homology_persistence (if installed).

Skipped when Dionysus is not installed.  Compares **persistence diagrams**
(Betti curves at each layer) between zztop and the Dionysus reference.
"""

import pytest
import numpy as np

try:
    import dionysus as d
    HAS_DIONYSUS = True
except ImportError:
    HAS_DIONYSUS = False

from conftest import bars_to_diagrams, diagram_betti_at
from zztop.complex.simplicial import Simplex
from zztop.zigzag.filtration import ZigzagFiltration
from zztop.zigzag.engine import ZigzagEngine

pytestmark = pytest.mark.skipif(
    not HAS_DIONYSUS,
    reason="Dionysus not installed — parity test skipped",
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_triangle_zigzag():
    v0, v1, v2 = Simplex([0]), Simplex([1]), Simplex([2])
    e01, e02, e12 = Simplex([0, 1]), Simplex([0, 2]), Simplex([1, 2])
    t012 = Simplex([0, 1, 2])
    boundary = {v0, v1, v2, e01, e02, e12}
    filled = boundary | {t012}
    return [boundary, filled, boundary]


def _run_dionysus(frames):
    import itertools

    all_simplices = set()
    for f in frames:
        all_simplices |= f
    simp_list = sorted(all_simplices)

    n = len(frames)
    layers = []
    for i in range(n):
        layers.append(frames[i])
        if i < n - 1:
            layers.append(frames[i] & frames[i + 1])

    n_layers = len(layers)
    times = {}
    for s in simp_list:
        present = [k for k in range(n_layers) if s in layers[k]]
        flat = []
        for r in _ranges(present):
            flat.extend(r)
        times[s] = flat

    f_simplices = []
    f_times = []
    for s in simp_list:
        f_simplices.append(d.Simplex(list(s.vertices)))
        f_times.append(times[s])

    filt = d.Filtration(f_simplices)
    zz, _, _ = d.zigzag_homology_persistence(filt, f_times)

    # Convert to our (dim, birth, death) format
    bars = []
    for i, interval in enumerate(zz):
        dim = zz[i].data
        b = interval.birth
        death = interval.death
        if death == float("inf"):
            death = n_layers
        bars.append((dim, int(b), int(death)))
    return bars


def _ranges(indices):
    import itertools
    for _, grp in itertools.groupby(
        enumerate(indices), lambda pair: pair[1] - pair[0]
    ):
        grp = list(grp)
        yield grp[0][1] + 1, grp[-1][1] + 2


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestDionysusSimplicialParity:

    def test_triangle_zigzag_betti_curves_match(self):
        """Betti numbers at every layer must agree between zztop and Dionysus."""
        frames = _build_triangle_zigzag()

        filt = ZigzagFiltration(
            frames=frames,
            cell_dim_fn=lambda s: s.dimension,
            cell_boundary_fn=lambda s: s.faces(),
        )
        pd_zz = bars_to_diagrams(ZigzagEngine(backend="python").run(filt))
        pd_dio = bars_to_diagrams(_run_dionysus(frames))

        for dim in [0, 1]:
            for layer in range(1, filt.n_layers + 1):
                b_zz = diagram_betti_at(pd_zz, dim, layer)
                b_dio = diagram_betti_at(pd_dio, dim, layer)
                assert b_zz == b_dio, (
                    f"β_{dim} at layer {layer}: zztop={b_zz}, dionysus={b_dio}\n"
                    f"  zztop PD[{dim}]    = {pd_zz.get(dim, [])}\n"
                    f"  dionysus PD[{dim}] = {pd_dio.get(dim, [])}"
                )

    def test_single_vertex_diagram(self):
        """Single vertex → one H0 point persisting the whole zigzag."""
        v0 = Simplex([0])
        frames = [{v0}, {v0}, {v0}]

        filt = ZigzagFiltration(
            frames=frames,
            cell_dim_fn=lambda s: s.dimension,
            cell_boundary_fn=lambda s: s.faces(),
        )
        pd_zz = bars_to_diagrams(ZigzagEngine(backend="python").run(filt))
        pd_dio = bars_to_diagrams(_run_dionysus(frames))

        # Both should have the same number of H0 points
        assert len(pd_zz.get(0, [])) == len(pd_dio.get(0, [])), (
            f"H0 count: zztop={len(pd_zz.get(0, []))}, "
            f"dionysus={len(pd_dio.get(0, []))}"
        )

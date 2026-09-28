"""Tests for :mod:`zztop.zigzagdiff`, the differentiable zigzag layer.

The file name sorts after ``test_gpu_cubical.py`` on purpose: importing torch
into the pytest process before the GUDHI GPU-fork tests run breaks them with
"CUDA driver version is insufficient".  For the same reason torch and the
``zigzagdiff`` package are imported lazily, through fixtures, never at
collection time.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path
from typing import Dict, List, Sequence, Set

import numpy as np
import pytest

from zztop.complex.simplicial import SimplicialComplex

FD_EPS = 1e-6
FD_TOL = 1e-8

BACKENDS = ["python"]      # the only engine since 0.8.0; "auto" is an alias


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def torch():
    t = pytest.importorskip("torch")
    t.set_default_dtype(t.float64)
    return t


@pytest.fixture(scope="module")
def zzd(torch):
    import zztop.zigzagdiff as module

    return module


# ----------------------------------------------------------------------
# geometry helpers (torch-free)
# ----------------------------------------------------------------------

def square_complex() -> SimplicialComplex:
    """The boundary of a square: 4 vertices, 4 edges, no 2-cell.

    Canonical cell order is ``v0 v1 v2 v3 | e01 e03 e12 e23`` (indices 0-7).
    """
    return SimplicialComplex([[0, 1], [1, 2], [2, 3], [0, 3]])


def two_squares_complex() -> SimplicialComplex:
    """Two disjoint 4-cycles on vertices 0-3 and 4-7."""
    return SimplicialComplex(
        [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7]]
    )


def column(complex_, cell: Sequence[int]) -> int:
    """Canonical column index of *cell*."""
    from zztop.zigzagdiff.events import cell_order
    from zztop.complex.simplicial import Simplex

    return cell_order(complex_).index(Simplex(cell))


def staggered_square_values() -> np.ndarray:
    """Square values where the loop is born at t*=1.5 and dies at t*=2.5.

    Three edges are present throughout; ``e23`` enters in interval [1,2] and
    exits in interval [2,3], so the H1 class both appears and dies by a single
    interpolated crossing with a hand-computable time.
    """
    K = square_complex()
    V = np.full((5, 8), 10.0)          # vertices (and default) always present
    for cell in ([0, 1], [0, 3], [1, 2]):
        V[:, column(K, cell)] = 1.0
    V[:, column(K, [2, 3])] = [0.0, 0.25, 0.75, 0.25, 0.0]
    return V


# ----------------------------------------------------------------------
# independent GF(2) Betti oracle (no engine involved)
# ----------------------------------------------------------------------

def gf2_rank(columns: List[Set[int]]) -> int:
    """Rank over GF(2) of a matrix given as a list of row-index sets."""
    pivots: Dict[int, Set[int]] = {}
    rank = 0
    for col in columns:
        col = set(col)
        while col:
            p = max(col)
            if p not in pivots:
                pivots[p] = col
                rank += 1
                break
            col ^= pivots[p]
    return rank


def betti_k(present: Set[int], cell_dim: np.ndarray, face_idx: List[List[int]], k: int) -> int:
    """Betti number of the subcomplex spanned by *present*, over GF(2)."""
    rows = {d: {} for d in (k - 1, k, k + 1)}
    for d in rows:
        for i, c in enumerate(sorted(c for c in present if cell_dim[c] == d)):
            rows[d][c] = i

    def boundary_columns(d: int) -> List[Set[int]]:
        return [
            {rows[d - 1][f] for f in face_idx[c] if f in present}
            for c in sorted(c for c in present if cell_dim[c] == d)
        ]

    n_k = len(rows[k])
    rank_k = gf2_rank(boundary_columns(k)) if k > 0 else 0
    rank_k1 = gf2_rank(boundary_columns(k + 1))
    return n_k - rank_k - rank_k1


def replay_present(schedule, n_ops: int) -> Set[int]:
    """Set of cells present after applying the first *n_ops* events."""
    present: Set[int] = set()
    for ev in schedule.events[:n_ops]:
        if ev.kind == "i":
            present.add(ev.cell)
        else:
            present.discard(ev.cell)
    return present


def batch_ends(schedule) -> List[int]:
    """Event counts at which a simultaneity batch closes."""
    return [i + 1 for i in range(schedule.n_events) if schedule.s[i] == 0]


def central_difference(fn, x, index, eps: float = FD_EPS) -> float:
    """Central finite difference of scalar *fn* along one coordinate of *x*."""
    base = x.detach().clone()
    up, down = base.clone(), base.clone()
    up[index] += eps
    down[index] -= eps
    return float((fn(up) - fn(down)) / (2 * eps))


# ----------------------------------------------------------------------
# ordering and front-ends
# ----------------------------------------------------------------------

def test_cell_order_refines_faces():
    from zztop.zigzagdiff.events import cell_order

    K = SimplicialComplex([[0, 1, 2], [2, 3]])
    order = cell_order(K)
    rank = {cell: i for i, cell in enumerate(order)}
    for cell in order:
        for face in cell.faces():
            assert rank[face] < rank[cell], f"{face} must precede {cell}"


def test_lower_star_matches_min_and_face_condition(torch, zzd):
    K = SimplicialComplex([[0, 1, 2], [2, 3]])
    order = zzd.cell_order(K)
    theta = torch.rand(4, 4, generator=torch.Generator().manual_seed(0))
    V = zzd.lower_star(theta, K)

    assert V.shape == (4, len(order))
    for j, cell in enumerate(order):
        expected = theta[:, list(cell)].amin(dim=-1)
        assert torch.allclose(V[:, j], expected)

    # min over a subset is never smaller, so the face condition is automatic
    zzd.validate_face_condition(V, K)


def test_passthrough_rejects_violation(torch, zzd):
    K = square_complex()
    V = torch.as_tensor(staggered_square_values())
    V[2, column(K, [0, 1])] = 99.0          # edge above its own vertices
    with pytest.raises(ValueError, match="face condition violated at frame 2"):
        zzd.passthrough(V, K)


# ----------------------------------------------------------------------
# event construction
# ----------------------------------------------------------------------

def test_frame0_block_order_and_stamps(torch, zzd):
    K = square_complex()
    V = torch.as_tensor(staggered_square_values())
    schedule = zzd.build_events(K, V.numpy(), eps=0.5, validate=True)

    block = [ev for ev in schedule.events if ev.k == -1]
    assert len(block) == 7            # 4 vertices + 3 permanently present edges
    assert schedule.events[: len(block)] == block, "frame-0 block must come first"
    assert [ev.cell for ev in block] == sorted(ev.cell for ev in block)
    assert all(ev.kind == "i" and ev.t_star == 0.0 for ev in block)

    t = zzd.event_times(V, 0.5, schedule)
    assert torch.all(t[: len(block)] == 0.0)
    # the whole block is one simultaneity batch
    assert schedule.s[: len(block) - 1].all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_event_closure_random(torch, zzd, backend):
    """Random lower-star fields keep the complex closed after every arrow."""
    K = SimplicialComplex([[0, 1, 2], [1, 2, 3], [3, 4], [4, 0]])
    gen = torch.Generator().manual_seed(7)
    for _ in range(10):
        theta = torch.rand(6, 5, generator=gen)
        V = zzd.lower_star(theta, K)
        schedule = zzd.build_events(K, V.numpy(), eps=0.5)
        zzd.replay_validate(schedule)               # raises on any violation
        zzd.zigzag_pairs(schedule, backend=backend)  # engine accepts the stream


def test_epsilon_touch_is_a_phantom_pair(torch, zzd):
    """A value that touches eps exactly with both neighbours above emits NO event
    (paper Def. B.9: the exit and the re-entry at the same instant cancel).  For a
    p-cell this keeps the bar whole (Remark B.6); the old detour split it."""
    K = SimplicialComplex([[0, 1]])
    edge = column(K, [0, 1])
    # (p+1)-cell touch: the edge never leaves, the H0 diagram is that of the untouched sequence
    V = torch.tensor([[2.0, 2.0, 1.0], [2.0, 2.0, 0.5], [2.0, 2.0, 1.0]])
    schedule = zzd.build_events(K, V.numpy(), eps=0.5, validate=True)
    assert [ev for ev in schedule.events if ev.cell == edge and ev.k >= 0] == [], "phantom pair cancelled"
    res = zzd.zigzag_diagrams(K, V, eps=0.5, dims=(0,))
    ref = zzd.zigzag_diagrams(K, torch.tensor([[2.0, 2.0, 1.0]] * 3), eps=0.5, dims=(0,))
    assert torch.equal(res[0].stamped(), ref[0].stamped())
    # p-cell touch (a vertex) with sentinels: ONE H0 bar, not two split at the touch
    Vp = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0],
                       [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
    h0 = zzd.zigzag_diagrams(K, Vp, eps=0.0, dims=(0,))[0].stamped()
    assert h0.shape[0] == 1 and torch.allclose(h0[0], torch.tensor([0.5, 4.5]))


# ----------------------------------------------------------------------
# forward correctness
# ----------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_square_loop_p1_closed_form(torch, zzd, backend):
    """The square-boundary loop, births and deaths by hand.

    ``e23`` rises through eps=0.5 between frames 1 and 2 at
    ``1 + (0.5-0.25)/(0.75-0.25) = 1.5`` and falls back between frames 2 and 3
    at ``2 + (0.5-0.75)/(0.25-0.75) = 2.5``.
    """
    K = square_complex()
    V = torch.as_tensor(staggered_square_values())
    res = zzd.zigzag_diagrams(K, V, eps=0.5, dims=(0, 1), backend=backend, validate=True)

    h1 = res[1]
    assert len(h1) == 1 and h1.essential.numel() == 0
    assert torch.allclose(h1.finite[0], torch.tensor([1.5, 2.5]))

    h0 = res[0]
    assert h0.finite.numel() == 0, "the square is connected throughout"
    assert h0.essential.numel() == 1
    assert float(h0.essential[0]) == 0.0
    assert torch.allclose(h0.stamped()[0], torch.tensor([0.0, 4.0]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_deletion_death_regression(torch, zzd, backend):
    """A loop that dies by a deletion must survive the read-out.

    The engine reports closed operation intervals, so the killing operation is
    ``d`` in 0-based event space.  Reading ``d - 1`` collapses this bar onto a
    single event, where the zero-length rule then discards it -- the loop
    disappears silently.  Nothing else happens between the two events here, so
    the bar is exactly the canary for that bug.
    """
    K = square_complex()
    V = torch.as_tensor(staggered_square_values())
    res = zzd.zigzag_diagrams(K, V, eps=0.5, dims=(1,), backend=backend)

    assert len(res[1]) == 1, "the deletion-born death was dropped"
    birth_ev, death_ev = res[1].finite_events[0]
    assert death_ev == birth_ev + 1
    killing = res.schedule.events[death_ev]
    assert killing.kind == "d" and killing.cell == column(K, [2, 3])


def test_pairing_matches_native_integer(torch, zzd):
    """The abstract-stream pairing equals the classic engine's on the same stream,
    "auto" is an alias of "python", and the removed C++ backend is refused."""
    from zztop.zigzag._python_backend import compute_zigzag_python
    K = square_complex()
    V = torch.as_tensor(staggered_square_values())
    schedule = zzd.build_events(K, V.numpy(), eps=0.5)
    reference = compute_zigzag_python(zzd.build_operations(schedule))
    assert sorted(zzd.zigzag_pairs(schedule, backend="python")) == sorted(reference)
    assert sorted(zzd.zigzag_pairs(schedule, backend="auto")) == sorted(reference)
    with pytest.raises(ValueError):
        zzd.zigzag_pairs(schedule, backend="cpp")


def test_oracle_betti_replay(torch, zzd):
    """Bars alive at each batch end reproduce an independent GF(2) Betti count."""
    K = SimplicialComplex([[0, 1, 2], [1, 2, 3], [3, 4], [4, 0]])
    gen = torch.Generator().manual_seed(11)
    for _ in range(5):
        theta = torch.rand(6, 5, generator=gen)
        V = zzd.lower_star(theta, K)
        res = zzd.zigzag_diagrams(K, V, eps=0.5, dims=(0, 1))
        schedule = res.schedule
        for n_ops in batch_ends(schedule):
            present = replay_present(schedule, n_ops)
            for p in (0, 1):
                expected = betti_k(present, schedule.cell_dim, schedule.face_idx, p)
                diagram = res[p]
                alive = sum(
                    1 for b, d in diagram.finite_events if b < n_ops <= d
                ) + sum(1 for b in diagram.essential_events if b < n_ops)
                assert alive == expected, f"dim {p} after {n_ops} events"


def test_s_elimination_counts(torch, zzd):
    """Classes born and killed inside one batch never reach the diagram."""
    K = square_complex()
    # every cell appears at once, so all but one H0 class is zero-length
    V = torch.tensor(np.vstack([np.zeros(8), np.ones(8), np.ones(8)]))
    res = zzd.zigzag_diagrams(K, V, eps=0.5, dims=(0, 1))

    assert res[0].finite.numel() == 0
    assert res[0].essential.numel() == 1
    # every raw H0 pair except the essential one was eliminated
    raw_h0 = [pair for pair in res.raw_pairs if pair[2] == 0]
    assert len(raw_h0) > 1


# ----------------------------------------------------------------------
# gradients
# ----------------------------------------------------------------------

def test_grad_fd_wrt_V(torch, zzd):
    K = square_complex()
    base = torch.as_tensor(staggered_square_values())
    col = column(K, [2, 3])

    def loss_of(V):
        return zzd.zigzag_diagrams(K, V, eps=0.5, dims=(1,)).total_persistence()

    V = base.clone().requires_grad_(True)
    loss_of(V).backward()

    for frame in (1, 2, 3):
        fd = central_difference(loss_of, base, (frame, col))
        assert abs(float(V.grad[frame, col]) - fd) < FD_TOL, f"frame {frame}"

    # the analytic form of the birth stamp, for good measure
    v1, v2 = float(base[1, col]), float(base[2, col])
    assert abs(float(V.grad[1, col]) - -(0.5 - v2) / (v2 - v1) ** 2) < FD_TOL


def test_grad_fd_wrt_theta(torch, zzd):
    K = square_complex()
    gen = torch.Generator().manual_seed(3)
    base = torch.rand(6, 4, generator=gen)

    def loss_of(theta):
        V = zzd.lower_star(theta, K)
        return zzd.zigzag_diagrams(K, V, eps=0.5, dims=(0, 1)).total_persistence()

    theta = base.clone().requires_grad_(True)
    loss_of(theta).backward()
    assert torch.isfinite(theta.grad).all()
    assert theta.grad.abs().sum() > 0, "the loss must depend on theta"

    for index in itertools.product(range(6), range(4)):
        fd = central_difference(loss_of, base, index)
        assert abs(float(theta.grad[index]) - fd) < FD_TOL, f"at {index}"


def test_grad_fd_wrt_eps(torch, zzd):
    """The threshold is just another coordinate: dt*/deps = 1/(v_{k+1}-v_k)."""
    K = square_complex()
    V = torch.as_tensor(staggered_square_values())
    col = column(K, [2, 3])

    def loss_of(eps):
        return zzd.zigzag_diagrams(K, V, eps=eps, dims=(1,)).total_persistence()

    eps = torch.tensor(0.5, requires_grad=True)
    loss_of(eps).backward()

    fd = (loss_of(0.5 + FD_EPS) - loss_of(0.5 - FD_EPS)) / (2 * FD_EPS)
    assert abs(float(eps.grad) - float(fd)) < FD_TOL

    # bar = (death - birth); d/deps of each endpoint is 1/(v_{k+1} - v_k)
    v = V[:, col]
    d_birth = 1.0 / float(v[2] - v[1])
    d_death = 1.0 / float(v[3] - v[2])
    assert abs(float(eps.grad) - (d_death - d_birth)) < FD_TOL


def test_subgradient_ties_finite(torch, zzd):
    """Argmin ties and near-tangencies stay finite; no NaN escapes the clamp."""
    K = square_complex()

    tied = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]],
        requires_grad=True,
    )
    zzd.zigzag_diagrams(K, zzd.lower_star(tied, K), eps=0.5, dims=(0, 1)) \
        .total_persistence().backward()
    assert torch.isfinite(tied.grad).all()

    # a crossing whose bracketing values are 1e-13 apart: steep but finite
    V = torch.as_tensor(staggered_square_values())
    V[2, column(K, [2, 3])] = 0.5 + 1e-13
    V = V.requires_grad_(True)
    res = zzd.zigzag_diagrams(K, V, eps=0.5, dims=(1,))
    assert torch.isfinite(res.t).all()
    res.total_persistence().backward()
    assert torch.isfinite(V.grad).all()


# ----------------------------------------------------------------------
# the central claims: value ordering and non-zero gradient
# ----------------------------------------------------------------------

def test_collision_wall_attribution(torch, zzd):
    """Two deaths in one frame interval are separated, and follow the values.

    Frame ordering would stamp both with the same integer, losing which loop
    died when; value ordering gives them their own crossing times.  Swapping
    the two times across the wall must swap the deaths, with each loop keeping
    its own birth.
    """
    K = two_squares_complex()
    n_cells = len(zzd.cell_order(K))

    def values(fall_a: float, fall_b: float):
        V = np.full((4, n_cells), 10.0)
        for cell in ([0, 1], [0, 3], [1, 2], [4, 5], [4, 7], [5, 6]):
            V[:, column(K, cell)] = 1.0
        # loop A closes at t*=0 (present from the start), loop B at t*=0.5
        V[:, column(K, [2, 3])] = [1.0, 1.0, fall_a, fall_a]
        V[:, column(K, [6, 7])] = [0.0, 1.0, fall_b, fall_b]
        return torch.as_tensor(V)

    # falls chosen so the two exits land at distinct times inside [1, 2]
    for a, b in ((0.25, -0.5), (-0.5, 0.25)):
        res = zzd.zigzag_diagrams(K, values(a, b), eps=0.5, dims=(1,), validate=True)
        bars = {round(float(x), 10): round(float(y), 10) for x, y in res[1].finite}
        assert len(bars) == 2, "the two loops must stay distinct"

        death_a = 1 + (0.5 - 1.0) / (a - 1.0)
        death_b = 1 + (0.5 - 1.0) / (b - 1.0)
        assert bars[0.0] == pytest.approx(death_a)
        assert bars[0.5] == pytest.approx(death_b)
        assert death_a != death_b
        # frame ordering would have collapsed both onto the same integer stamp
        assert 1.0 < death_a < 2.0 and 1.0 < death_b < 2.0


def test_training_beats_index_baseline(torch, zzd):
    """Value stamping reaches the analytic optimum; integer stamps cannot move.

    Maximising the H1 lifetime on the square has a known optimum: the loop can
    at best live from the frame-0 boundary to the essential boundary death, so
    ``N - 1 = 4``.  The value-stamped pipeline climbs to it.  The integer-index
    baseline has no gradient at all, so it can never leave its starting point.
    """
    K = square_complex()
    base = staggered_square_values()
    col = column(K, [2, 3])

    def index_stamped(res):
        """The null control: endpoints read as integer frame indices."""
        stamps = torch.tensor(
            [float(ev.k + 1) for ev in res.schedule.events], dtype=torch.get_default_dtype()
        )
        idx = res[1].finite_events
        return (stamps[idx[:, 1]] - stamps[idx[:, 0]]).sum()

    V = torch.as_tensor(base).clone().requires_grad_(True)
    res = zzd.zigzag_diagrams(K, V, eps=0.5, dims=(1,))
    control = index_stamped(res)
    assert control.grad_fn is None, "integer stamps carry no gradient"
    assert float(control) == 1.0

    V = torch.as_tensor(base).clone().requires_grad_(True)
    opt = torch.optim.Adam([V], lr=0.02)
    start = None
    for step in range(60):
        opt.zero_grad()
        lifetime = zzd.zigzag_diagrams(K, V, eps=0.5, dims=(1,)).total_persistence()
        (-lifetime).backward()
        if step == 0:
            start = float(lifetime)
            assert V.grad[:, col].abs().sum() > 0, "value stamping must give gradient"
        opt.step()

    final = float(zzd.zigzag_diagrams(K, V, eps=0.5, dims=(1,)).total_persistence())
    assert start == pytest.approx(1.0)
    assert final == pytest.approx(4.0), f"did not reach the optimum: {start} -> {final}"

    # at the optimum the bar is essential and both endpoints are boundary
    # constants, so the gradient legitimately vanishes -- that is convergence,
    # not the flat-everywhere failure of the index baseline.
    optimum = zzd.zigzag_diagrams(K, V, eps=0.5, dims=(1,))
    assert optimum[1].finite.numel() == 0 and optimum[1].essential.numel() == 1


# ----------------------------------------------------------------------
# conventions and architecture
# ----------------------------------------------------------------------

def test_boundary_conventions_locked(torch, zzd):
    """Frame-0 births and essential deaths are constants, and stay so."""
    K = square_complex()
    V = torch.as_tensor(staggered_square_values()).requires_grad_(True)
    res = zzd.zigzag_diagrams(K, V, eps=0.5, dims=(0,))

    h0 = res[0]
    assert float(h0.essential[0]) == 0.0, "frame-0 births stamp at exactly 0.0"
    assert h0.boundary_death == 4.0, "essential deaths stamp at N-1"
    assert torch.allclose(h0.stamped(), torch.tensor([[0.0, 4.0]]))

    # no teardown: a cell alive at the last frame yields an essential bar
    assert res.schedule.events[-1].t_star < float(res.schedule.n_frames - 1)
    h0.total_persistence().backward()
    assert float(V.grad.abs().sum()) == 0.0, "a fully boundary bar is constant"


def test_empty_diagram_is_still_differentiable(torch, zzd):
    """A degree with no bars must not break backward.

    The gather runs even on empty index arrays, so the result stays attached to
    the event-time vector and the loss is a differentiable zero rather than a
    detached constant.
    """
    K = square_complex()
    V = torch.as_tensor(staggered_square_values()).requires_grad_(True)
    # the square has no 2- or 3-cells, so these degrees are always empty
    res = zzd.zigzag_diagrams(K, V, eps=0.5, dims=(2, 3))

    assert len(res[2]) == 0 and len(res[3]) == 0
    loss = res.total_persistence()
    assert loss.grad_fn is not None, "empty diagrams must stay in the graph"
    loss.backward()
    assert float(V.grad.abs().sum()) == 0.0


def test_layer_roundtrip_and_learnable_eps(torch, zzd):
    K = square_complex()
    theta = torch.rand(5, 4, generator=torch.Generator().manual_seed(5), requires_grad=True)

    layer = zzd.ZigzagDiffLayer(K, dims=(0, 1), eps=0.5, learnable_eps=True)
    loss = layer(theta).total_persistence()
    loss.backward()

    assert theta.grad is not None and torch.isfinite(theta.grad).all()
    assert layer.eps.grad is not None and torch.isfinite(layer.eps.grad)

    with pytest.raises(ValueError, match="unknown frontend"):
        zzd.ZigzagDiffLayer(K, frontend="nope")


def test_core_is_geometry_free(torch, zzd):
    """The core and the event builder must never reach a front-end."""
    forbidden = ("frontend", "lower_star", "cubical", "knn", "mesh", "gudhi",
                 "sklearn", "builders", "layer")
    root = Path(zzd.__file__).parent
    for name in ("core.py", "events.py"):
        tree = ast.parse((root / name).read_text())
        imported: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported += [node.module or ""] + [a.name for a in node.names]
        for mod in imported:
            assert not any(bad in mod.lower() for bad in forbidden), \
                f"{name} imports {mod!r}"


def test_ulp_tie_face_before_coface(torch, zzd):
    """Regression (exp10): crossing times computed from values that
    differ in the last ulp must not order a coface's insertion before
    its face's.  Observed in the wild: a vertex (max over incident
    edges) crossing 5e-15 after its argmax edge -> KeyError in
    build_operations.  TIE_QUANTUM batching restores the face-first
    order; stamps stay raw."""
    complex_ = SimplicialComplex([[0], [1], [0, 1]])
    # values reproduce the observed 1-ulp mismatch: vertex 1 carries
    # -0.761541143964483, the edge -0.7615411439644831 (1 ulp lower),
    # both rising to the same value -> t* differs by ~5e-15 with the
    # coface numerically FIRST.
    theta = torch.tensor([
        [1.0, -0.761541143964483, -0.7615411439644831],
        [1.0, 0.30796851341167075, 0.30796851341167075],
        [1.0, -1.0, -1.0],
    ], dtype=torch.float64)
    layer = zzd.ZigzagDiffLayer(complex_, dims=(0,), eps=0.0,
                                frontend="passthrough")
    result = layer(theta)                     # must not raise
    bars = result[0].finite.shape[0] + result[0].essential.shape[0]
    assert bars >= 1                          # at least vertex 0's bar
    # stamps are NOT quantised: the raw interpolated values survive
    stamps = torch.cat([result[0].finite.reshape(-1),
                        result[0].essential.reshape(-1)])
    frac = stamps - stamps.floor()
    assert bool(((frac > 0) & (frac < 1)).any())


def test_crossing_clamped_to_interval(torch, zzd):
    """Tangent grazes with guarded denominators must not stamp a
    crossing outside its own frame interval."""
    from zztop.zigzagdiff.events import build_events
    complex_ = SimplicialComplex([[0]])
    values = np.array([[-1e-13], [1e-16], [-1.0]])
    sched = build_events(complex_, values, 0.0)
    for ev in sched.events:
        assert ev.k <= ev.t_star <= ev.k + 1


def test_instant_presence_insert_before_delete(torch, zzd):
    """A cell present at one frame by ~1e-17 enters and exits at the same ``t*``.

    Its entry crossing on ``[0, 1]`` rounds to exactly ``1.0`` and its exit on
    ``[1, 2]`` to ``1.0`` as well; the deletions-first tie-break would delete a
    cell that was never inserted (observed as a ``KeyError`` in
    ``build_operations`` when Adam drove a pixel to the threshold).  The repair
    pass must hoist the cell's own insertion first, keep the face order, and the
    zero-length bar must vanish.
    """
    from zztop.complex.simplicial import Simplex

    K = SimplicialComplex([[0, 1]])
    theta = torch.tensor([[-1.0, 1.0], [1e-17, 1.0], [-1.0, 1.0]], dtype=torch.float64)
    V = zzd.lower_star(theta, K)
    schedule = zzd.build_events(K, V.numpy(), eps=0.0, validate=True)
    order = schedule.order
    v0, edge = order.index(Simplex((0,))), order.index(Simplex((0, 1)))
    at_one = [(ev.kind, ev.cell) for ev in schedule.events if ev.t_star == 1.0]
    assert at_one == [("i", v0), ("i", edge), ("d", edge), ("d", v0)]

    result = zzd.zigzag_diagrams(K, V, eps=0.0, dims=(0, 1))
    assert result[0].finite.shape[0] == 0, "the instant bar is zero-length and dropped"
    assert result[0].essential.numel() == 1, "vertex 1 lives throughout"
    assert result[1].finite.shape[0] == 0 and result[1].essential.numel() == 0

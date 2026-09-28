"""Differentiable read-out on top of the ``zz-top`` zigzag pairing.

The factorisation this module implements is::

    Pers = (read values) o (combinatorial pairing)

The pairing is computed by ``zz-top`` on a purely integer event stream and is
held **constant**: it crosses the bridge as detached integers and no gradient
ever enters the engine.  Differentiability comes from the event-time vector
``t``, which is built from torch ops, and from gathering ``t`` at the paired
event positions -- literally an index-select, not a hand-written Jacobian.

This module is geometry-free: it never learns where the values came from.
Front-ends live in :mod:`zztop.zigzagdiff.frontends`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from zztop.zigzag import _python_backend

from .events import EventSchedule, build_events

__all__ = [
    "DiffDiagram",
    "ZigzagDiffResult",
    "build_operations",
    "event_times",
    "zigzag_pairs",
    "zigzag_diagrams",
]


def build_operations(schedule: EventSchedule) -> List[Tuple[str, int, List[int]]]:
    """Convert the event schedule into the engine's operation list.

    The engine expects ``(op, dim, boundary)`` where an insertion carries the
    sorted **0-based operation indices** at which each codimension-1 face was
    inserted, and a deletion carries the operation index of its own matching
    insertion.
    """
    ops: List[Tuple[str, int, List[int]]] = []
    insert_op: Dict[int, int] = {}
    for i, ev in enumerate(schedule.events):
        dim = int(schedule.cell_dim[ev.cell])
        if ev.kind == "i":
            ops.append(("i", dim, sorted(insert_op[f] for f in schedule.face_idx[ev.cell])))
            insert_op[ev.cell] = i
        else:
            ops.append(("d", dim, [insert_op.pop(ev.cell)]))
    return ops


def zigzag_pairs(
    schedule: EventSchedule, backend: str = "auto"
) -> List[Tuple[int, int, int]]:
    """Run ``zz-top`` on the schedule and return its raw ``(b, d, p)`` triples.

    Endpoints are 1-based **closed** operation intervals, exactly as the
    engine reports them; the conversion to event indices happens in
    :func:`_read_bars`.
    """
    ops = build_operations(schedule)
    if backend == "cpp":
        raise ValueError("the C++ backend was removed in zztop 0.8.0 (licensing); use 'python' or 'auto'")
    if backend not in ("auto", "python"):
        raise ValueError(f"unknown backend {backend!r}; use 'auto' or 'python'")
    return _python_backend.compute_zigzag_python(ops)


def _same_batch(s: np.ndarray, i: int, j: int) -> bool:
    """True iff events ``i..j`` are all simultaneous (one exact-``t*`` batch)."""
    return bool(np.all(s[i:j] == 1))


def _read_bars(
    raw: Sequence[Tuple[int, int, int]],
    schedule: EventSchedule,
    dims: Sequence[int],
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Convert raw engine output to per-dimension event-index arrays.

    ``b`` is 1-based and the interval is closed, so the creating operation is
    ``b - 1`` in 0-based event space and the **killing** operation is ``d``.
    Reading ``d - 1`` here is the historical deletion-death bug: it collapses
    classes that die by a deletion.  ``d >= n_events`` marks an essential bar.

    Zero-length pairs -- born and killed inside one simultaneity batch, hence
    with identical crossing times -- are dropped.
    """
    n = schedule.n_events
    finite: Dict[int, List[Tuple[int, int]]] = {p: [] for p in dims}
    essential: Dict[int, List[int]] = {p: [] for p in dims}

    for b, d, p in raw:
        if p not in finite:
            continue
        birth_ev = b - 1
        if d >= n:
            essential[p].append(birth_ev)
            continue
        death_ev = d
        if _same_batch(schedule.s, birth_ev, death_ev):
            continue
        finite[p].append((birth_ev, death_ev))

    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for p in dims:
        fin = np.array(finite[p], dtype=np.int64).reshape(-1, 2)
        ess = np.array(essential[p], dtype=np.int64).reshape(-1)
        out[p] = (fin, ess)
    return out


def event_times(
    values: torch.Tensor,
    eps: Union[float, torch.Tensor],
    schedule: EventSchedule,
    guard: float = 1e-12,
) -> torch.Tensor:
    """Build the differentiable event-time vector ``t``.

    Interior events get ``k + (eps - v_k) / (v_{k+1} - v_k)`` with the
    sign-preserving denominator clamp; frame-0 events get the exact constant
    ``0.0``.  Autograd then supplies the closed-form derivatives with respect
    to ``v_k``, ``v_{k+1}`` and ``eps`` -- there is no hand-written backward.
    """
    n = schedule.n_events
    dtype, device = values.dtype, values.device
    if n == 0:
        return torch.zeros(0, dtype=dtype, device=device)

    k_np = np.array([ev.k for ev in schedule.events], dtype=np.int64)
    c_np = np.array([ev.cell for ev in schedule.events], dtype=np.int64)
    interior_np = k_np >= 0
    # Frame-0 events index a dummy lane; torch.where masks it out and blocks
    # any gradient from flowing through the unselected branch.
    k_safe = np.where(interior_np, k_np, 0)

    k_idx = torch.as_tensor(k_safe, device=device)
    c_idx = torch.as_tensor(c_np, device=device)
    interior = torch.as_tensor(interior_np, device=device)

    lo = values[k_idx, c_idx]
    hi = values[k_idx + 1, c_idx]
    delta = hi - lo
    sign = torch.where(
        delta >= 0,
        torch.ones((), dtype=dtype, device=device),
        -torch.ones((), dtype=dtype, device=device),
    )
    safe = sign * delta.abs().clamp(min=guard)

    eps_t = torch.as_tensor(eps, dtype=dtype, device=device) if not isinstance(
        eps, torch.Tensor
    ) else eps.to(dtype=dtype, device=device)

    interp = k_idx.to(dtype) + (eps_t - lo) / safe
    return torch.where(interior, interp, torch.zeros((), dtype=dtype, device=device))


@dataclass
class DiffDiagram:
    """The diagram of one homological degree.

    ``finite``/``essential`` carry gradient; ``finite_events``/
    ``essential_events`` are the detached engine pairing, exposed so the
    value-read can be audited (``finite`` is exactly ``t[finite_events]``).
    """

    dim: int
    finite: torch.Tensor
    essential: torch.Tensor
    finite_events: np.ndarray
    essential_events: np.ndarray
    boundary_death: float

    def stamped(self) -> torch.Tensor:
        """All bars as ``(birth, death)`` rows, essentials at the boundary time."""
        if self.essential.numel() == 0:
            return self.finite
        death = torch.full_like(self.essential, self.boundary_death)
        ess = torch.stack([self.essential, death], dim=1)
        if self.finite.numel() == 0:
            return ess
        return torch.cat([self.finite, ess], dim=0)

    def lifetimes(self, include_essential: bool = True) -> torch.Tensor:
        """Per-bar ``death - birth``."""
        bars = self.stamped() if include_essential else self.finite
        return bars[:, 1] - bars[:, 0]

    def total_persistence(
        self, power: float = 1.0, include_essential: bool = True
    ) -> torch.Tensor:
        """Sum of ``|death - birth| ** power`` over the bars.

        An empty diagram sums to zero *and stays attached to the graph*, so a
        training loop survives a step in which the requested degree happens to
        carry no bars.
        """
        life = self.lifetimes(include_essential=include_essential).abs()
        return (life**power).sum()

    def __len__(self) -> int:
        return int(self.finite.shape[0] + self.essential.shape[0])


@dataclass
class ZigzagDiffResult:
    """Forward output: differentiable diagrams plus the detached combinatorics."""

    diagrams: Dict[int, DiffDiagram]
    t: torch.Tensor
    schedule: EventSchedule
    raw_pairs: List[Tuple[int, int, int]]

    def __getitem__(self, dim: int) -> DiffDiagram:
        return self.diagrams[dim]

    def total_persistence(
        self, power: float = 1.0, include_essential: bool = True
    ) -> torch.Tensor:
        """Total persistence summed over every requested degree."""
        terms = [
            d.total_persistence(power=power, include_essential=include_essential)
            for d in self.diagrams.values()
        ]
        return torch.stack(terms).sum()


def zigzag_diagrams(
    complex_,
    values: torch.Tensor,
    eps: Union[float, torch.Tensor] = 0.0,
    dims: Sequence[int] = (1,),
    backend: str = "auto",
    guard: float = 1e-12,
    validate: bool = False,
) -> ZigzagDiffResult:
    """Differentiable zigzag diagrams of a per-frame filtering function.

    Parameters
    ----------
    complex_ : SimplicialComplex
        The fixed complex ``K``.
    values : Tensor, shape (n_frames, n_cells)
        Per-simplex values in the canonical cell order
        (:func:`~zztop.zigzagdiff.events.cell_order`).  May require grad.
    eps : float or Tensor
        Threshold; a 0-dim tensor with ``requires_grad`` makes it trainable.
    dims : sequence of int
        Homological degrees to return.
    backend : {'auto', 'python'}
        Kept for compatibility; both select the pure-Python engine.
    guard : float
        Magnitude clamp for near-degenerate interpolation denominators.
    validate : bool
        Replay-check that the complex is closed after every event.
    """
    eps_value = float(eps.detach()) if isinstance(eps, torch.Tensor) else float(eps)
    schedule = build_events(
        complex_,
        values.detach().cpu().numpy(),
        eps_value,
        guard=guard,
        validate=validate,
    )
    raw = zigzag_pairs(schedule, backend=backend)
    per_dim = _read_bars(raw, schedule, tuple(dims))
    t = event_times(values, eps, schedule, guard=guard)

    diagrams: Dict[int, DiffDiagram] = {}
    for p in dims:
        fin_ev, ess_ev = per_dim[p]
        # Gathering even when empty keeps the result attached to ``t``, so a
        # loss over an empty diagram is still differentiable (gradient zero).
        idx = torch.as_tensor(fin_ev, device=values.device)
        finite = torch.stack([t[idx[:, 0]], t[idx[:, 1]]], dim=1)
        essential = t[torch.as_tensor(ess_ev, device=values.device)]
        diagrams[p] = DiffDiagram(
            dim=p,
            finite=finite,
            essential=essential,
            finite_events=fin_ev,
            essential_events=ess_ev,
            boundary_death=schedule.boundary_death,
        )

    return ZigzagDiffResult(diagrams=diagrams, t=t, schedule=schedule, raw_pairs=list(raw))

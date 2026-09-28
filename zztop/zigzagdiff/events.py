"""Value-ordered event construction for the differentiable zigzag layer.

This module is deliberately **detached and geometry-free**: it works with a
canonically ordered simplicial complex and a plain ``numpy`` array of
per-frame simplex values, and produces the combinatorial event schedule that
is handed to the ``zz-top`` pairing engine.  No torch, no front-ends.

Conventions
-----------
* A simplex is *present* at frame ``i`` iff ``v(sigma)_i > eps`` (strict,
  upper-level set).
* Face condition: a proper face carries the **larger** value,
  ``v(face)_i >= v(coface)_i``.  Under the strict inequality the crossing
  times alone already order faces before cofaces; equality (which the
  lower-star front-end produces generically) is resolved by the tie-break in
  :func:`_sort_key`.
* Crossing time of an event in frame interval ``[k, k+1]``::

      t* = k + (eps - v_k) / (v_{k+1} - v_k)

  Entry crossings stamp ``t* in [k, k+1)``, exit crossings ``t* in (k, k+1]``.
* Events are ordered **globally by t***, never by frame index.  This is what
  makes same-interval events attribute to the right bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from zztop.complex.simplicial import Simplex

__all__ = [
    "Event",
    "EventSchedule",
    "cell_order",
    "build_events",
    "replay_validate",
    "safe_denominator",
]


def cell_order(complex_) -> List[Simplex]:
    """Return the canonical total order on the cells of *complex_*.

    :class:`~zztop.complex.simplicial.Simplex` orders by ``(dimension,
    vertices)``, so the result refines the face partial order: every proper
    face has a strictly smaller index than its cofaces.
    """
    return sorted(complex_.all_cells())


def safe_denominator(delta, guard: float):
    """Sign-preserving magnitude clamp of an interpolation denominator.

    Keeps the tangency singularity finite instead of producing ``inf``/``nan``.
    ``sign`` is never zero (a flat interval maps to ``+guard``); such an
    interval cannot carry a crossing anyway.
    """
    sign = np.where(delta >= 0.0, 1.0, -1.0)
    return sign * np.maximum(np.abs(delta), guard)


@dataclass(frozen=True)
class Event:
    """A single cell insertion or deletion stamped with its crossing time.

    Attributes
    ----------
    kind : {'i', 'd'}
        Insertion or deletion.
    cell : int
        Index into the canonical cell order.
    k : int
        Lower frame of the crossing interval, or ``-1`` for the frame-0 block
        (cells already present at the first frame, which have no interior
        crossing and therefore carry a constant, gradient-free stamp).
    t_star : float
        Crossing time used for the global ordering.
    """

    kind: str
    cell: int
    k: int
    t_star: float


@dataclass
class EventSchedule:
    """The detached combinatorial object handed to the pairing engine."""

    order: List[Simplex]
    cell_dim: np.ndarray
    face_idx: List[List[int]]
    events: List[Event]
    s: np.ndarray
    n_frames: int
    n_ties: int

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def n_cells(self) -> int:
        return len(self.order)

    @property
    def boundary_death(self) -> float:
        """Time stamped on essential deaths (see module docs of the package)."""
        return float(self.n_frames - 1)


# Width of the numerical-noise window for the closure repair pass.
# Crossing times computed from values that differ in the last ulp can
# land on the wrong side of each other by ~1e-15 (observed: a vertex
# whose value is the max over incident edges crossing 5e-15 AFTER the
# argmax edge), which would insert a coface before its face.  The
# repair pass in :func:`build_events` reorders only such
# provably-inverted pairs within this window; everything else keeps
# the raw time order, and genuine violations still raise.
TIE_QUANTUM = 1e-9


def _sort_key(ev: Event, rank: int) -> Tuple[float, int, int]:
    """Global event order: by crossing time, then face-respecting tie-break.

    Within an exact-``t*`` batch, deletions come before insertions and cells
    are ordered by rank ascending on insertion (face before coface) and
    descending on deletion (coface before face).  Deletions-first keeps the
    complex closed when a coface is deleted and an unrelated cell inserted at
    the same ``t*``; a threshold touch itself (a phantom pair) emits no event.
    """
    phase = 0 if ev.kind == "d" else 1
    return (ev.t_star, phase, rank if ev.kind == "i" else -rank)


def _repair_closure(events: List[Event], face_idx) -> None:
    """Reorder events inverted by floating-point noise, in place.

    Replays presence; when an insertion's face is missing (or a
    deletion's coface still present), the matching face-insertion /
    coface-deletion is searched within ``TIE_QUANTUM`` ahead and hoisted
    before the offending event.  Outside that window the stream is
    genuinely invalid and a ``ValueError`` is raised.

    The same pass handles the mirror of the threshold-touch case: a cell
    present at frame ``k`` by a hair (``v_k - eps ~ 1e-17``) has its entry
    crossing on ``[k-1, k]`` and its exit crossing on ``[k, k+1]`` both round
    to ``t* = k`` exactly, and the deletions-first tie-break would then delete
    a cell that was never inserted.  The cell is present for an instant, so
    its own insertion is hoisted before the deletion; any zero-length bar this
    creates is removed downstream by the simultaneity rule.
    """
    n_cells = len(face_idx)
    coface_idx: List[List[int]] = [[] for _ in range(n_cells)]
    for c, faces in enumerate(face_idx):
        for f in faces:
            coface_idx[f].append(c)

    present: Set[int] = set()
    i = 0
    while i < len(events):
        ev = events[i]
        if ev.kind == "i":
            blockers = [(f, "i") for f in face_idx[ev.cell] if f not in present]
        else:
            blockers = [(c, "d") for c in coface_idx[ev.cell] if c in present]
            if ev.cell not in present:
                blockers.append((ev.cell, "i"))   # present for an instant
        hoisted = False
        for b, want in blockers:
            for j in range(i + 1, len(events)):
                if events[j].t_star - ev.t_star > TIE_QUANTUM:
                    break
                if events[j].kind == want and events[j].cell == b:
                    events.insert(i, events.pop(j))
                    hoisted = True
                    break
            else:
                raise ValueError(
                    f"closure violation beyond the noise window: event "
                    f"{ev.kind!r} of cell {ev.cell} at t*={ev.t_star!r} "
                    f"blocked by cell {b}"
                )
            if hoisted:
                break
        if hoisted:
            continue                      # reprocess from the hoisted event
        if ev.kind == "i":
            present.add(ev.cell)
        else:
            present.discard(ev.cell)
        i += 1


def build_events(
    complex_,
    values: np.ndarray,
    eps: float,
    guard: float = 1e-12,
    validate: bool = False,
) -> EventSchedule:
    """Build the value-ordered event schedule.

    Parameters
    ----------
    complex_ : SimplicialComplex
        The fixed complex ``K``.
    values : ndarray, shape (n_frames, n_cells)
        Per-frame filtering values in the canonical cell order of
        :func:`cell_order`.  Detached; float64 is used internally.
    eps : float
        Threshold.  A cell is present at frame ``i`` iff ``values[i] > eps``.
    guard : float
        Magnitude clamp for near-degenerate interpolation denominators.
    validate : bool
        If true, replay the stream and assert the complex is closed after
        every single event.
    """
    order = cell_order(complex_)
    n_cells = len(order)
    index_of = {cell: i for i, cell in enumerate(order)}

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != n_cells:
        raise ValueError(
            f"values must have shape (n_frames, {n_cells}), got {values.shape}"
        )
    n_frames = values.shape[0]
    if n_frames < 2:
        raise ValueError("at least two frames are required to stamp crossings")

    cell_dim = np.array([c.dimension for c in order], dtype=np.int64)
    face_idx = [[index_of[f] for f in c.faces()] for c in order]

    active = values > eps

    events: List[Event] = []
    # Frame-0 block: no interior crossing, stamped at the constant 0.0.
    for c in np.flatnonzero(active[0]):
        events.append(Event("i", int(c), -1, 0.0))

    # Interior crossings, one per (cell, frame interval) sign change.
    lo_state, hi_state = active[:-1], active[1:]
    entry = np.argwhere(~lo_state & hi_state)
    exit_ = np.argwhere(lo_state & ~hi_state)
    # Phantom crossings (paper, Def. B.9): a value EXACTLY at eps at an interior frame k with
    # both neighbours above eps is an exit on [k-1, k] and a re-entry on [k, k+1] at the same
    # instant t* = k.  The construction cancels the pair: the cell never leaves and no event is
    # emitted.  (Emitting the pair as a detour is invisible for a (p+1)-cell, Lemma B.8, but
    # splits the bar carried by a p-cell, Remark B.6.)
    at_eps = values == eps
    phantom = np.zeros_like(active)
    phantom[1:-1] = at_eps[1:-1] & active[:-2] & active[2:]
    if phantom.any():
        keep_entry = np.array([not phantom[k, c] for k, c in entry], dtype=bool)
        keep_exit = np.array([not phantom[k + 1, c] for k, c in exit_], dtype=bool)
        entry, exit_ = entry[keep_entry], exit_[keep_exit]
    for kind, hits in (("i", entry), ("d", exit_)):
        for k, c in hits:
            k, c = int(k), int(c)
            lo = values[k, c]
            hi = values[k + 1, c]
            t_star = k + (eps - lo) / safe_denominator(hi - lo, guard)
            # the crossing lives inside its frame interval; the guarded
            # denominator can otherwise overshoot on tangent grazes
            t_star = min(max(float(t_star), float(k)), float(k + 1))
            events.append(Event(kind, c, k, t_star))

    events.sort(key=lambda ev: _sort_key(ev, ev.cell))
    _repair_closure(events, face_idx)

    n = len(events)
    s = np.zeros(n, dtype=np.int8)
    for i in range(n - 1):
        s[i] = 1 if events[i + 1].t_star == events[i].t_star else 0
    n_ties = int(s.sum())

    schedule = EventSchedule(
        order=order,
        cell_dim=cell_dim,
        face_idx=face_idx,
        events=events,
        s=s,
        n_frames=n_frames,
        n_ties=n_ties,
    )
    if validate:
        replay_validate(schedule)
    return schedule


def replay_validate(schedule: EventSchedule) -> None:
    """Replay the event stream, asserting the complex is closed throughout.

    ``zz-top`` requires a valid complex after *every* arrow, not merely at
    frame boundaries: an insertion needs all its codimension-1 faces present,
    and a deletion needs all its cofaces already gone.
    """
    cofaces: List[List[int]] = [[] for _ in range(schedule.n_cells)]
    for cell, faces in enumerate(schedule.face_idx):
        for f in faces:
            cofaces[f].append(cell)

    present = np.zeros(schedule.n_cells, dtype=bool)
    for i, ev in enumerate(schedule.events):
        name = schedule.order[ev.cell]
        if ev.kind == "i":
            if present[ev.cell]:
                raise ValueError(f"event {i}: {name} inserted while already present")
            missing = [
                schedule.order[f] for f in schedule.face_idx[ev.cell] if not present[f]
            ]
            if missing:
                raise ValueError(f"event {i}: inserting {name} with absent faces {missing}")
            present[ev.cell] = True
        else:
            if not present[ev.cell]:
                raise ValueError(f"event {i}: {name} deleted while absent")
            alive = [schedule.order[c] for c in cofaces[ev.cell] if present[c]]
            if alive:
                raise ValueError(f"event {i}: deleting {name} with live cofaces {alive}")
            present[ev.cell] = False

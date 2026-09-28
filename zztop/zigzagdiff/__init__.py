"""Differentiable zigzag persistence on top of the ``zz-top`` engine.

The pipeline this package makes differentiable is::

    parameters -> per-frame filtering values -> continuous-time zigzag
               -> barcode -> scalar loss

``zz-top`` returns a barcode as a combinatorial object: pairs of integer
operation positions.  Read that way, any loss built on the diagram is
piecewise constant and its gradient vanishes on an open set, so nothing can be
trained.  This package fixes that without touching the pairing algorithm.  It
factors persistence as::

    Pers = (read values) o (combinatorial pairing)

and makes only the left factor differentiable.  The pairing crosses the bridge
to the engine as detached integers; gradient never enters it.  What carries
gradient is the real-valued *time* stamped on each event.

Usage
-----
>>> import torch
>>> from zztop.complex.simplicial import SimplicialComplex
>>> from zztop.zigzagdiff import ZigzagDiffLayer
>>> K = SimplicialComplex([[0, 1], [1, 2], [2, 3], [0, 3]])
>>> theta = torch.rand(6, 4, dtype=torch.float64, requires_grad=True)
>>> layer = ZigzagDiffLayer(K, dims=(0, 1), eps=0.5)
>>> loss = layer(theta).total_persistence()
>>> loss.backward()

A degree that happens to carry no bars contributes zero *and stays attached to
the graph*, so a training loop never breaks on an empty diagram.

Conventions
-----------
Presence
    A simplex is present at frame ``i`` iff ``v(sigma)_i > eps`` (strict,
    **upper**-level set).
Face condition
    A proper face carries the larger value, ``v(face)_i >= v(coface)_i``, so
    every frame's present set is a subcomplex.  The specification states the
    strict inequality; the weak form is validated instead because
    :func:`~zztop.zigzagdiff.frontends.lower_star` produces equality
    generically (an edge's value *is* one endpoint's value).  Equality is
    resolved by simultaneity batching, not rejected.
Crossing time
    An event in frame interval ``[k, k+1]`` is stamped at
    ``t* = k + (eps - v_k) / (v_{k+1} - v_k)``.  Entry crossings land in
    ``[k, k+1)``, exits in ``(k, k+1]``.  Autograd supplies the closed forms::

        dt*/dv_k     = (eps - v_{k+1}) / (v_{k+1} - v_k)**2
        dt*/dv_{k+1} = (v_k - eps)     / (v_{k+1} - v_k)**2
        dt*/deps     = 1 / (v_{k+1} - v_k)

    No backward is written by hand.
Event order
    Events are ordered **globally by t***, never by frame index.  Ordering by
    frame misattributes bars whenever two events fall in the same frame
    interval, and the error is invisible to finite differences because it is
    constant on an open set.  Exact ``t*`` ties break as
    ``(t*, deletions before insertions, rank ascending on insertion /
    descending on deletion)``, where rank is the position in the canonical
    cell order.  That gives face-before-coface on insertion and
    coface-before-face on deletion, so the complex is closed after every
    single arrow.  Deletions-first is forced by the threshold-touch case,
    where one simplex exits and immediately re-enters at the same ``t*``.
Boundary frames
    The interpolation only stamps crossings *between* consecutive frames, so
    simplices already present at frame 0, or still present at frame ``N-1``,
    have no interior crossing.  Two conventions, both deliberate and both
    locked by tests:

    * Cells present at frame 0 are inserted in one batch stamped at the
      constant ``0.0``, detached -- those births are legitimately
      gradient-free.
    * There is no teardown at the end.  Classes still alive after the last
      event are **essential**, and
      :meth:`~zztop.zigzagdiff.core.DiffDiagram.stamped` gives them the
      constant death ``float(N - 1)``.  Their births still carry gradient.

    A bar with both endpoints on the boundary is entirely constant; that is
    correct, not a bug.
Zero-length bars
    Pairs born and killed inside a single simultaneity batch have identical
    crossing times and are dropped.

Stability contract
------------------
Local
    Within one cell of fixed combinatorial type -- same activity pattern, same
    event order, same pairing -- each diagram coordinate is a smooth rational
    function of the values and the threshold, and the barcode is 1-Lipschitz
    in the event-time vector.  On a top-dimensional cell it is Lipschitz in
    the field.  This is the metric shadow of differentiability.
Global (no-go)
    There is **no** global field-continuous stability.  A perturbation
    supported at a single frame can split one long bar into two long bars, so
    bottleneck distance decouples from the field's sup-norm distance.
    Training will therefore see genuine jumps when it crosses a cell wall.
    This is a property of non-monotone zigzag, not a defect of this
    implementation, and it cannot be fixed by a better read-out.  The correct
    global statement lives in the interleaving distance (algebraic
    stability), not in the field distance.

Subgradients, not derivatives
-----------------------------
Three lower strata are measure zero and are **not** smoothed over:

* threshold touches, ``v(sigma)_i == eps``;
* crossing-time collisions, two events with exactly equal ``t*``;
* argmin ties in :func:`~zztop.zigzagdiff.frontends.lower_star`.

The same closed-form expressions are differentiated there.  The result is a
valid one-sided directional subgradient, not a two-sided derivative; nothing
here claims smoothness at a wall.  Tie behaviour of ``torch.amin`` is not
pinned to a version -- only "a valid subgradient element" is promised.  Near
a tangency the denominator ``v_{k+1} - v_k`` is clamped in magnitude with its
sign preserved, so gradients grow large but stay finite instead of turning
into ``inf``/``nan``.

Scope
-----
Single-parameter only: time is an evolution parameter, not a second
filtration axis.  The pairing is exact -- there is no soft or relaxed
threshold on this path.  A soft threshold would trade exactness for
continuity across walls; it is not implemented and is mentioned only as a
hypothetical alternative.
"""

from __future__ import annotations

from .core import (
    DiffDiagram,
    ZigzagDiffResult,
    build_operations,
    event_times,
    zigzag_diagrams,
    zigzag_pairs,
)
from .events import Event, EventSchedule, build_events, cell_order, replay_validate
from .frontends import lower_star, passthrough, validate_face_condition, vertex_order
from .layer import ZigzagDiffLayer

__all__ = [
    # core
    "DiffDiagram",
    "ZigzagDiffResult",
    "build_operations",
    "event_times",
    "zigzag_diagrams",
    "zigzag_pairs",
    # events
    "Event",
    "EventSchedule",
    "build_events",
    "cell_order",
    "replay_validate",
    # front-ends
    "lower_star",
    "passthrough",
    "validate_face_condition",
    "vertex_order",
    # layer
    "ZigzagDiffLayer",
]

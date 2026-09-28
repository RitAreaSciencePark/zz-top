# `zztop.zigzagdiff`: differentiable zigzag persistence

`zz-top` returns a zigzag barcode as a combinatorial object: pairs of integer
operation positions.  Read that way, any loss built on the diagram is piecewise
constant, its gradient vanishes on an open set, and nothing can be trained.
`zigzagdiff` fixes this without touching the pairing algorithm.  It factors
persistence as

```
Pers = (read values) ∘ (combinatorial pairing)
```

and makes only the left factor differentiable.  The pairing crosses the bridge
to the engine as detached integers; gradient never enters it.  What carries
gradient is the real-valued *time* stamped on each event: the instant at which
a simplex's interpolated value crosses the threshold.

This document is the reference for the module.  Section 1 is the pipeline,
Sections 2 to 4 the conventions that make it exact, Section 5 the API, Sections
6 to 8 gradients, stability and numerics, Section 9 the tests, and Section 10
the dictionary to the paper's notation.

## 1. The pipeline

```
parameters θ ──G──▶ values V ∈ ℝ^{N×|K|} ──build_events──▶ EventSchedule (detached ints)
                        │                                        │
                        │                                   zigzag_pairs ──▶ (b, d, p) triples (detached)
                        │                                        │
                        └──event_times──▶ t ∈ ℝ^{n_events} ──gather──▶ DiffDiagram per degree ──▶ loss
```

1. A **front end** `G` maps parameters to a per-frame *filtering function*: a
   tensor `V` of shape `(n_frames, n_cells)` holding one value per simplex of a
   fixed complex `K` and per frame, in the canonical cell order.  Two front
   ends ship with the module (`lower_star`, `passthrough`); a Vietoris–Rips one
   is described in Section 4.
2. `build_events` (numpy, detached) turns `V` into the **event schedule**: the
   list of insertions and deletions with their crossing times, in global time
   order, plus the simultaneity flags.
3. `zigzag_pairs` hands the schedule to the engine as an abstract
   insertion/deletion stream and gets back the pairing as `(birth, death,
   degree)` triples of operation indices.
4. `event_times` (torch) recomputes every event's crossing time from `V` with
   autograd on.  The **diagram** of degree `p` is a gather of this vector at
   the paired event indices, so it is differentiable in `V` and, through `G`,
   in θ.  A degree with no bars gives an empty tensor that is still attached to
   the graph, so a training loop never breaks on an empty diagram.

Steps 2 and 3 are the "combinatorial pass", step 4 the "differentiable pass";
within one combinatorial type the diagram is a smooth function of `V`.

## 2. Modules

| module | role | depends on torch |
|---|---|---|
| `events.py` | canonical cell order, crossing detection, phantom-pair cancellation, global ordering with face-respecting tie-break, floating-point closure repair, simultaneity flags, replay validation | no |
| `core.py` | engine bridge (`build_operations`, `zigzag_pairs`), zero-length elimination and the engine-index reading (`_read_bars`), the differentiable event-time vector (`event_times`), the diagram gather (`zigzag_diagrams`), the containers `DiffDiagram` and `ZigzagDiffResult` | yes |
| `frontends.py` | `lower_star`, `passthrough`, `validate_face_condition`, `vertex_order` | yes |
| `layer.py` | `ZigzagDiffLayer`, a `torch.nn.Module` tying a front end to the core, with an optionally learnable threshold | yes |

`core.py` is geometry-free by construction: it never learns where the values
came from, and a test parses its AST to make sure it imports nothing from the
front ends.

## 3. Conventions

**Presence.** A simplex σ is present at frame `i` iff `v(σ)_i > eps`.  Strict,
and an *upper*-level set.

**Face condition.** A proper face carries the larger value,
`v(face)_i ≥ v(coface)_i`, so every frame's present set is a subcomplex.  The
paper states the strict inequality; the module validates the weak form because
the lower-star front end produces equality generically (an edge's value *is*
one endpoint's value).  Equality is resolved by the tie-break below, not
rejected.  A strict violation raises `ValueError`.

**Crossing time.** An event in the frame interval `[k, k+1]` is stamped at

```
t* = k + (eps − v_k) / (v_{k+1} − v_k)
```

Entry crossings land in `[k, k+1)`, exit crossings in `(k, k+1]`.  The
denominator is clamped in magnitude at `guard` (default `1e-12`) with its sign
preserved, and the result is clamped to `[k, k+1]`, so a near-tangency gives a
large but finite gradient instead of `inf`/`nan`.  Autograd supplies the closed
forms

```
∂t*/∂v_k = (eps − v_{k+1}) / (v_{k+1} − v_k)²
∂t*/∂v_{k+1} = (v_k − eps) / (v_{k+1} − v_k)²
∂t*/∂eps = 1 / (v_{k+1} − v_k)
```

No backward pass is written by hand.

**Global event order.** Events are sorted by `t*`, never by frame index.
Ordering by frame misattributes bars whenever two events fall in the same frame
interval, and the error is invisible to finite differences because it is
constant on an open set.  Exact ties are broken as

```
(t*, deletions before insertions, rank ascending on insertion / rank descending on deletion)
```

where the rank is the position in the canonical cell order `(dimension,
vertices)`, which refines the face partial order.  Insertions therefore go
face-before-coface and deletions coface-before-face, so the complex is closed
after every single arrow.  Deletions-first is forced by the case of a coface
deleted and an unrelated cell inserted at the same instant.

**Phantom pairs.** A value *exactly* at `eps` at an interior frame `k` whose two
neighbours are above `eps` would be an exit on `[k−1, k]` and a re-entry on
`[k, k+1]` at the same instant `t* = k`.  The module emits no event for it: the
cell never leaves.  Emitting the pair as a detour is invisible for a
`(p+1)`-cell but splits the bar carried by a `p`-cell, so cancelling it is a
correctness rule, not an optimisation.  Locked by
`test_epsilon_touch_is_a_phantom_pair`.

**Boundary frames.** Interpolation only stamps crossings *between* consecutive
frames, so cells present at frame 0, or still present at the last frame, have
no interior crossing.  Two deliberate conventions:

* cells present at frame 0 are inserted in one batch stamped at the constant
  `0.0`, detached: those births are legitimately gradient-free (`Event.k == -1`);
* there is no teardown at the end.  Classes still alive after the last event
  are **essential**; `DiffDiagram.stamped()` gives them the constant death
  `float(n_frames − 1)`, while their births keep their gradient.

A bar with both endpoints on the boundary is entirely constant, which is
correct.  Section 4 explains how the experiments avoid boundary bars
altogether.

**Zero-length bars.** A pair born and killed inside one simultaneity batch (all
events in between share the same `t*`) has identical endpoints and is dropped.
The simultaneity flags `EventSchedule.s` record, for each event, whether the
next event has the same stamp.

**Engine reading.** The engine reports 1-based *closed* operation intervals
`(b, d, p)`: `b` is the first operation at which the class is alive and `d` the
last one.  In 0-based event space the creating event is therefore `b − 1` and
the **killing** event is `d`; `d ≥ n_events` marks an essential class.  Reading
`d − 1` is the historical deletion-death bug: it collapses every class that
dies by a deletion, which then vanishes through the zero-length rule.
`test_deletion_death_regression` is the canary, and the classic engine
(`ZigzagEngine.run`) applies the same correction.

**Floating-point closure repair.** Crossing times computed from values that
differ in the last ulp can land on the wrong side of each other by about
`1e-15`, which would insert a coface before its face.  After sorting,
`build_events` replays presence and, when an insertion finds a face missing (or
a deletion finds a coface present), looks ahead within `TIE_QUANTUM = 1e-9` for
the blocking event and hoists it.  Outside that window the stream is genuinely
invalid and a `ValueError` is raised.  The same pass handles a cell present for
a single instant (entry and exit both rounding to `t* = k`).

## 4. Front ends, and the padding recipe

A front end is any differentiable map from parameters to a `(n_frames,
n_cells)` tensor in the canonical order `cell_order(K)` (all vertices sorted,
then all edges sorted, and so on) that satisfies the weak face condition.

* **`lower_star(theta, K)`**: `theta` has shape `(n_frames, n_vertices)`;
  `v(σ)_i = min over vertices p of σ of theta(p)_i`.  A simplex is present
  exactly while all of its vertices are.  The paper calls this map the
  **upper-star extension** (presence is an upper-level set); the function
  keeps the name under which it was written.  The minimum over a *subset* makes the
  face condition automatic; backward sends gradient to the argmin vertex, and
  at ties `torch.amin` returns a valid subgradient element.  This is the
  filtration of the temporal-graph experiment.
* **`passthrough(values, K, validate=True)`**: per-simplex values supplied
  directly, with the face condition checked unless `validate=False`.  Use it
  when the values are cheaper to compute once and reuse: computing the
  lower-star values once and evaluating the zigzag at several thresholds
  through `passthrough` gives the same diagrams and the same gradients as
  calling the layer with the `lower_star` front end at each threshold.
* **Vietoris–Rips** (the front end of the paper's sensor-coverage experiment, not
  part of the package): `v(σ)_i = −diam_i(σ)` with `eps = −R`, so that σ is present iff its diameter
  is below `R`; a face's vertex set is a subset, so `diam(face) ≤ diam(coface)`
  and the weak face condition holds.  Gradient reaches the two points that
  realise the diameter.

**Padding.** To be in the paper's class of filtering functions whose first and
last complexes are empty, so that *every* bar is born and dies at a crossing
and carries gradient at both ends, a front end adjoins one frame at each end
with a value strictly below `eps`: the coverage experiment uses `−2R` against
`eps = −R`, the temporal-graph experiment `−2` against `eps ∈ (−1, 1)`.  With
padding the module's frame index is the user's frame index plus one; bar
lengths are read as overlaps with the window `[1, N]`.  Padding frames must be
*constants*, not parameters: treating them as parameters breaks the structural
ties of the lower-star map asymmetrically and the loss acquires a kink.

A constant shift of all values and of `eps` changes neither presence nor any
crossing time, so `eps` can always be normalised to 0.

## 5. API

All names below are importable from `zztop.zigzagdiff`.

### `ZigzagDiffLayer(complex_, dims=(1,), eps=0.0, learnable_eps=False, frontend="lower_star", backend="auto", guard=1e-12, validate=False)`

`torch.nn.Module`.  `forward(x) -> ZigzagDiffResult`, where `x` is `theta`
(`(n_frames, n_vertices)`) for `frontend="lower_star"` and `V` (`(n_frames,
n_cells)`) for `frontend="passthrough"`.  With `learnable_eps=True` the
threshold is an `nn.Parameter` and receives gradient through `∂t*/∂eps`;
otherwise it is a buffer.  `backend` is `"auto"` or `"python"` (the same
pure-Python engine; kept for compatibility).  `validate=True` replays the event stream on every forward (a
Python loop, for debugging).

### `zigzag_diagrams(complex_, values, eps=0.0, dims=(1,), backend="auto", guard=1e-12, validate=False) -> ZigzagDiffResult`

The functional core behind the layer: `values` is a `(n_frames, n_cells)`
tensor (may require grad), `eps` a float or a 0-dim tensor.

### `ZigzagDiffResult`

| member | content |
|---|---|
| `diagrams: Dict[int, DiffDiagram]`, `result[p]` | one diagram per requested degree |
| `t: Tensor (n_events,)` | the differentiable event-time vector |
| `schedule: EventSchedule` | the detached combinatorics handed to the engine |
| `raw_pairs: List[(b, d, p)]` | the engine's output, 1-based closed operation intervals |
| `total_persistence(power=1.0, include_essential=True)` | `Σ_p Σ_bars |death − birth|^power` |

### `DiffDiagram`

| member | content |
|---|---|
| `dim` | homological degree |
| `finite: Tensor (n, 2)` | `(birth, death)` crossing times, differentiable |
| `essential: Tensor (m,)` | births of essential classes, differentiable |
| `finite_events: ndarray (n, 2)`, `essential_events: ndarray (m,)` | the detached pairing in event indices; `finite == t[finite_events]` |
| `boundary_death: float` | `n_frames − 1`, the constant death given to essentials |
| `stamped()` | all bars as rows, essentials closed at `boundary_death` |
| `lifetimes(include_essential=True)` | per-bar `death − birth` |
| `total_persistence(power=1.0, include_essential=True)` | sum of `|lifetime|^power`; zero *and attached to the graph* when empty |
| `len(diagram)` | number of bars, finite plus essential |

### `build_events(complex_, values, eps, guard=1e-12, validate=False) -> EventSchedule`

Detached.  `values` is a numpy array `(n_frames, n_cells)` (at least two
frames).  Raises `ValueError` on a shape mismatch or on a closure violation
beyond the noise window.

### `EventSchedule`

| member | content |
|---|---|
| `order: List[Simplex]` | the canonical cell order |
| `cell_dim: ndarray`, `face_idx: List[List[int]]` | dimension and codimension-1 faces of each cell, as indices into `order` |
| `events: List[Event]` | the ordered stream; `Event(kind ∈ {'i','d'}, cell, k, t_star)`, `k = -1` for the frame-0 block |
| `s: ndarray (n_events,) int8` | `s[i] = 1` iff event `i+1` has the same `t*` as event `i` |
| `n_frames`, `n_ties`, `n_events`, `n_cells`, `boundary_death` | sizes and the essential-death constant |

### Engine bridge and value read

| function | content |
|---|---|
| `build_operations(schedule) -> List[(op, dim, boundary)]` | the abstract operation list: an insertion carries the sorted 0-based operation indices of its faces' insertions, a deletion the index of its own insertion |
| `zigzag_pairs(schedule, backend="auto") -> List[(b, d, p)]` | runs the engine on the abstract operation stream |
| `event_times(values, eps, schedule, guard=1e-12) -> Tensor` | the differentiable event-time vector (Section 3, crossing time); frame-0 events get the constant `0.0` through a `torch.where` that blocks gradient |
| `replay_validate(schedule)` | asserts closure after every event |
| `cell_order(complex_) -> List[Simplex]` | `sorted(K.all_cells())` |
| `safe_denominator(delta, guard)` | numpy version of the sign-preserving clamp |

### Front ends

| function | content |
|---|---|
| `lower_star(theta, complex_) -> Tensor (n_frames, n_cells)` | the minimum rule; checks `theta.shape[1] == n_vertices` |
| `passthrough(values, complex_, validate=True) -> Tensor` | identity with an optional face-condition check |
| `validate_face_condition(values, complex_)` | raises `ValueError` naming the first violating `(frame, face, coface)` |
| `vertex_order(complex_) -> List[Simplex]` | the 0-cells in canonical order (the first `n_vertices` positions of `cell_order`) |

## 6. Gradients and subgradients

Within one cell of fixed combinatorial type (same activity pattern, same event
order, same pairing) each diagram coordinate is the rational function of
Section 3 in the two values that bracket its crossing and in `eps`, and the
chain rule through the front end is whatever autograd does for it: the argmin
vertex for `lower_star`, the diameter-realising pair for Vietoris–Rips.  The
module's gradient agrees with central finite differences to about `1e-9` on
random instances (`test_grad_fd_wrt_V`, `test_grad_fd_wrt_theta`,
`test_grad_fd_wrt_eps`).

Three lower strata are measure zero and are **not** smoothed over: threshold
touches (`v(σ)_i = eps`), crossing-time collisions (two events with exactly
equal `t*`) and argmin ties in `lower_star`.  The same closed forms are
differentiated there, which gives a valid one-sided directional subgradient
rather than a two-sided derivative; nothing claims smoothness at a wall.  Tie
behaviour of `torch.amin` is not pinned to a version: only "a valid subgradient
element" is promised.  `test_subgradient_ties_finite` checks that these
gradients are finite.

## 7. Stability

**Local.** Within one combinatorial cell the barcode is 1-Lipschitz in the
event-time vector, and on a top-dimensional cell it is Lipschitz in the field.
This is the metric shadow of differentiability.

**Global (no-go).** There is *no* global field-continuous stability.  A
perturbation supported at a single frame can split one long bar into two long
bars, so the bottleneck distance decouples from the sup-norm distance of the
field.  Training therefore sees genuine jumps when it crosses a cell wall
(`test_collision_wall_attribution` exhibits one).  This is a property of
non-monotone zigzag, not a defect of the implementation, and no read-out can
remove it; the correct global statement lives in the interleaving distance.
Read-outs whose weights vanish on the diagonal (persistence images with
ramp-type weights, as in the temporal-graph experiment) make a bar that appears
or disappears enter with weight zero, which is what buys a locally Lipschitz
loss.

## 8. Numerics and performance

* Run in `float64`.  Crossing times are differences of nearly equal numbers
  divided by small denominators; the experiments set
  `torch.set_default_dtype(torch.float64)`.
* `build_events` is `O(n_frames · n_cells)` numpy work plus a Python sort of
  the events and the replay pass; `event_times` is a gather; the pairing is
  the Dey–Hou algorithm on `n_events` operations, run by the package's
  pure-Python engine (a column reduction over GF(2) with the twist
  optimisation).  For the sizes of the paper's experiments (a few hundred
  simplices, tens of frames, a few thousand events) a forward pass is a few
  tens of milliseconds; measured timings against the reference C++
  implementation are in [parity_with_fzz.md](parity_with_fzz.md).
* The engine's pairing was verified against the reference implementation
  `fzz` on thousands of random instances before that code was removed from the
  package ([parity_with_fzz.md](parity_with_fzz.md)); the `backend` argument
  of `zigzag_diagrams` and of the layer accepts `"auto"` or `"python"` and is
  kept only for compatibility.
* The complex `K` is fixed.  If the candidate complex depends on the parameters
  (as the union-over-frames Rips complex does), rebuild it from *detached*
  coordinates between steps; within a step only the values carry gradient.

## 9. Tests

`tests/test_zigzagdiff.py` (torch imported lazily so that the GPU tests of the
classic library are unaffected):

| test | what it locks |
|---|---|
| `test_cell_order_refines_faces` | the canonical order puts every face before its cofaces |
| `test_lower_star_matches_min_and_face_condition` | the minimum rule and the weak face condition |
| `test_passthrough_rejects_violation` | a strict violation raises |
| `test_frame0_block_order_and_stamps` | frame-0 births are one batch at `t* = 0`, face-first, gradient-free |
| `test_event_closure_random` | closure after every event on random values |
| `test_epsilon_touch_is_a_phantom_pair` | a value exactly at `eps` emits no event |
| `test_square_loop_p1_closed_form` | the H₁ bar of the square matches its closed form |
| `test_deletion_death_regression` | classes killed by a deletion are read correctly |
| `test_pairing_matches_native_integer` | the pairing equals the classic engine's on the same stream; `"auto"` is an alias of `"python"` and `"cpp"` is refused |
| `test_oracle_betti_replay` | Betti numbers read from the diagram agree with an independent GF(2) computation |
| `test_s_elimination_counts` | zero-length pairs are exactly the same-batch ones |
| `test_grad_fd_wrt_V`, `..._theta`, `..._eps` | autograd against central differences |
| `test_subgradient_ties_finite` | finite gradients at ties |
| `test_collision_wall_attribution` | a wall crossing is a jump of the diagram, attributed to the right cell |
| `test_training_beats_index_baseline` | value stamping trains where the integer read-out has no gradient path |
| `test_boundary_conventions_locked` | frame-0 births at 0, essential deaths at `n_frames − 1` |
| `test_empty_diagram_is_still_differentiable` | an empty degree stays attached to the graph |
| `test_layer_roundtrip_and_learnable_eps` | the layer equals the functional core; `eps` receives gradient |
| `test_core_is_geometry_free` | AST guard on `core.py` |
| `test_ulp_tie_face_before_coface` | the closure repair pass on last-ulp inversions |
| `test_crossing_clamped_to_interval` | `t*` stays inside its frame interval under the guard |
| `test_instant_presence_insert_before_delete` | a cell present for one instant is inserted before it is deleted |

## 10. Dictionary to the paper

Numbering as in the accompanying manuscript on differentiable zigzag
persistence (under review).

| paper | module |
|---|---|
| filtering function `v ∈ FF_N(K)`, with `v(·)_0 = v(·)_{N+1}` below the threshold | `values`, `(N+2, n_cells)`, padding rows supplied by the front end (Section 4) |
| weak face condition, Eq. (2) | `validate_face_condition` |
| crossing time, Eq. (3) | `event_times` (torch) and the `t_star` of each `Event` (numpy) |
| the event sequence `S(v) ∈ Seq(K)` | `EventSchedule.events`, ordered as in Section 3 |
| `PH_p` of the sequence | `zigzag_pairs` followed by the engine-index reading in `core._read_bars` |
| the two-phase evaluation of the differential | detached `build_events` + `zigzag_pairs`, then the differentiable `event_times` gather |
| phantom pairs (Def. B.9) | the `phantom` mask in `build_events` |
| the exclusion sets: threshold touch, collision, argmin tie | Section 6; measured along trajectories by the experiments' diagnostics |
| `G` semialgebraic, differentiability off a null set (Prop. 3.8) | any front end built from `min`, distances and affine maps qualifies; `lower_star`, Vietoris–Rips |

## 11. Scope

Single-parameter only: time is an evolution parameter, not a second filtration
axis.  The pairing is exact; there is no soft or relaxed threshold on this
path.  A soft threshold would trade exactness for continuity across walls and
is not implemented.

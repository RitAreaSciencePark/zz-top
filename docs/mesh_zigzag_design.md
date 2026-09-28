# Mesh Zigzag Persistence — Design & Mathematical Justification

## 1. Summary

The **mesh** pipeline (`zztop/builders/mesh_builder.py`) computes zigzag
persistent homology of a scalar field that evolves in time on a **fixed**
triangulated surface mesh — for example a FreeSurfer cortical surface where
every vertex carries a neural-activity time series.  It tracks how the topology
of the *co-active* region reorganises frame to frame: components merging and
splitting (H0), loops of activity opening and closing (H1), and — for a whole
closed hemisphere — the global surface class (H2).

The geometry (vertices, edges, faces) is **fixed**; only the *sub-complex* that
is "switched on" at each frame changes.  This is exactly the setting the
Carlsson–de Silva zigzag construction was designed for, so the builder simply
materialises one sub-complex per frame and hands the list to
`ZigzagFiltration`, which already builds the intersection layers and the
insertion/deletion stream (see `docs/cubical_zigzag_design.md`, §4–5, for how
that machinery works — it is shared verbatim).

---

## 2. The Ambient Complex `M`

Let `M` be the fixed 2-complex of the mesh:

- **vertices** `0, …, V-1` (one per row of `activity`);
- **edges** — the unique undirected edge set derived from the triangle list
  `faces`; every mesh edge is an edge of at least one triangle;
- **faces** — the triangles themselves.

`M` is a valid CW-complex with the usual simplicial boundary
(`Simplex.faces()`), so `∂∂ = 0` holds and the abstract Dey–Hou backend applies
unchanged (the algorithm is purely algebraic — it only ever inspects a cell's
dimension and the filtration indices of its boundary; see
`docs/cubical_zigzag_design.md`, §2).

---

## 3. Per-Frame Construction (edge co-activity)

Given per-vertex values `x[:, t]` at frame `t` and a scalar `threshold` θ, the
frame sub-complex `K_t ⊆ M` is defined by an **instantaneous endpoint model**:

| cell | present at frame `t` iff |
|------|--------------------------|
| edge `(i, j)` | `reduce(x[i,t], x[j,t]) > θ` |
| triangle `(i, j, k)` | all three of its edges are present |
| vertex `v` | `x[v,t] > θ` **or** `v` is an endpoint of a present edge |

`reduce` is the **co-activity reducer**.  It defaults to `min`, and `"mean"`,
`"product"`, `"max"`, or any elementwise `callable(a, b) -> ndarray` are also
accepted.  With `strict=False` the strict `>` comparisons become `>=`
(applied consistently to both the edge reduction and the vertex "own value"
clause).

### 3.1 Reducer semantics

`reduce` decides what "an edge is co-active" *means*:

- **`min`** — the edge is on only when **both** endpoints exceed θ.  This is
  the natural "both neurons firing" rule and yields the excursion-set complex
  (§4).
- **`mean`** — the edge is on when the endpoints are *on average* active; a
  very active vertex can pull in a below-threshold neighbour.
- **`product`** — like a soft AND for non-negative signals; sensitive to the
  magnitude of both endpoints.
- **`max`** — the edge is on when **either** endpoint exceeds θ, i.e. the edge
  set is the sublevel/`OR` neighbourhood of the supra-threshold vertices.
- **callable** — any user-defined coupling, e.g. `|a − b|` for co-variation.

### 3.2 The vertex rule and the closure guarantee

The vertex rule has two clauses joined by **or**:

1. *own value*: `x[v,t] > θ` — included via `include_isolated_vertices=True`
   (the default);
2. *edge incidence*: `v` is an endpoint of a present edge — always included.

**Closure holds for _any_ reducer.**  A complex is closed iff every face of a
present cell is present.  There are only two non-trivial cases:

- A present **edge** `(i, j)` needs its two vertices.  Both are endpoints of a
  present edge (itself), so clause 2 includes them regardless of the reducer.
- A present **triangle** needs its three edges (present by definition) and its
  three vertices (endpoints of those present edges → clause 2).

So every `K_t` is a genuine closed sub-complex of `M`, and the dimension-sorted
insertion order in `ZigzagFiltration.to_operations()` (vertices → edges → faces
on insertion, reverse on deletion) is always satisfiable.  The test suite checks
this for random θ and every reducer / option combination
(`test_mesh_zigzag.py::TestClosureInvariant`).

Setting `include_isolated_vertices=False` drops clause 1: a vertex then appears
**only** when it has a co-active incident edge (a pure edge-cut complex).
Closure is unaffected because clause 2 alone already covers every face.

---

## 4. Excursion-Set Equivalence (`reduce=min`)

With `reduce=min` and `include_isolated_vertices=True`, `K_t` is **exactly** the
sub-complex of `M` induced on the supra-threshold vertex set
`A_t = { v : x[v,t] > θ }` — the discrete **excursion / superlevel set**:

- edge `(i, j)` present ⇔ `min(x[i], x[j]) > θ` ⇔ `x[i] > θ ∧ x[j] > θ`
  ⇔ both endpoints ∈ `A_t`;
- triangle present ⇔ all three edges present ⇔ all three vertices ∈ `A_t`;
- vertex present ⇔ `x[v] > θ` (clause 1) ⇔ `v ∈ A_t` — clause 2 adds nothing
  here, since an incident present edge already forces `x[v] > θ`.

Hence `K_t` is precisely the full induced sub-complex `M[A_t]`, isolated
supra-threshold vertices included.  This is verified against an independently
constructed induced sub-complex in
`test_mesh_zigzag.py::TestExcursionEquivalence`.

This is the mesh analogue of a superlevel-set filtration, but taken **frame by
frame in time** rather than by sweeping θ — the zigzag then measures how the
excursion set's topology *reorganises* as the activity field moves.

---

## 5. Frames → Zigzag

The builder materialises `[K_0, K_1, …, K_{T-1}]` as `Set[Simplex]` and passes
them straight to `ZigzagFiltration`, which interleaves intersection layers

$$K_0 \hookleftarrow (K_0 \cap K_1) \hookrightarrow K_1
  \hookleftarrow (K_1 \cap K_2) \hookrightarrow K_2 \hookleftarrow \cdots$$

and emits the operation stream (§5 of the cubical design note; the code path is
identical).  `run_mesh_zigzag` returns bars in the 1-based **zigzag-layer**
numbering of `ZigzagEngine.run` — `1 = K_0`, `2 = K_0∩K_1`, `3 = K_1`, …,
`2N−1 = K_{N-1}` — so the Betti number of original frame `t` is read at odd
layer `2t + 1` (after discarding zero-length cell-insertion bars).

### 5.1 The top-dimensional essential class

A closed surface (e.g. a whole hemisphere, or the octahedron test) carries an
essential **H2** class.  Because it is the *globally top-dimensional* essential
class, the Dey–Hou engine reports it at its birth layer as a single
zero-length bar `(1, 1)` rather than a full-length interval — the same happens
to an essential H1 class in a 1-complex with no 2-cells.  This is a property of
the engine, not of the mesh construction; lower-dimensional essential classes
(H0 everywhere, H1 whenever 2-cells are present) get full-length bars as usual.
`test_mesh_zigzag.py::TestClosedSurface` documents this: it is the
representation users will see for the global surface class of a hemisphere.

---

## 6. Complexity and the Memory Lever

Materialising every active vertex/edge/face as a `Simplex` for every frame costs
`O(mesh_size × n_frames)` in both time and memory.  A full FreeSurfer
hemisphere (~150k vertices) over hundreds of frames is large, so the intended
lever is `vertex_mask`: a boolean `(V,)` array that removes out-of-ROI vertices
(the medial wall, or everything outside a functional parcel) **and every
incident edge and face** from `M` entirely, before any frame is built.  The
per-frame boolean columns (`W = reduce(x[Ei], x[Ej])` computed once, then sliced
per frame) keep the vectorised core numpy-only — no scipy/nibabel/gudhi hard
dependency.  An optional `load_freesurfer_surface` helper lives behind a lazy
`nibabel` import and is never required.

---

## 7. Summary of Guarantees

| Property | Status |
|----------|--------|
| Each frame is a closed sub-complex of `M` (any reducer) | ✅ Proven (§3.2) & tested |
| `reduce=min` ⇒ excursion-set induced sub-complex | ✅ Proven (§4) & tested |
| Filled disk ⇒ β0=1, β1=0 | ✅ Tested |
| Annulus H1 born, dies when the ring is cut | ✅ Tested (birth/death layers) |
| Closed surface ⇒ single essential H2 class | ✅ Tested (engine `(1,1)` convention) |
| parity with the reference fzz implementation | ✅ campaign before its removal, see `parity_with_fzz.md` |
| `vertex_mask` removes a vertex + incident simplices | ✅ Tested |
| Coefficients: `F₂` | ✅ (inherited from the backend) |

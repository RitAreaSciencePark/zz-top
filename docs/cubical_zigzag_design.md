# Native Cubical Zigzag Persistence — Design & Mathematical Justification

## 1. Summary

The original pipeline (`zigzag.py`) computed zigzag persistence on grid data
by:

1. Using GUDHI to build a `CubicalComplex` from each frame.
2. **Triangulating** every cubical cell into simplices (converting each
   square into 2 triangles, each cube into tetrahedra, etc.).
3. Feeding the resulting simplicial complex into the Dey-Hou zigzag algorithm
   (via Dionysus / fzz).

The new `zztop` pipeline skips step 2 entirely.  It feeds cubical cells
**directly** into the Dey-Hou algorithm through an abstract cell interface.
This document explains how and why this is mathematically valid.

---

## 2. Background: What the Dey-Hou Algorithm Actually Needs

The Dey-Hou fast zigzag algorithm (ESA 2022) is a **reduction-based** method.
It receives:

- A stream of **cell insertions** and **cell deletions**, forming a zigzag
  filtration: $K_1 \hookrightarrow K_2 \hookleftarrow K_3 \hookrightarrow
  \cdots$
- For each insertion, the **boundary chain** of the inserted cell (as a list of
  already-inserted boundary cells).

The algorithm then:

1. **Cones off** every deletion to convert the zigzag into a standard (monotone)
   filtration of a larger complex $\Delta$.
2. **Reduces** the boundary matrix of $\Delta$ using column reduction with the
   twist optimisation (as in phat).
3. **Maps** the resulting persistence pairs back to zigzag intervals.

**Key observation:** The algorithm never inspects vertices, coordinates, or
geometric structure.  It only uses:

| Input per cell | What it is |
|---|---|
| `dim` | Integer dimension of the cell |
| `boundary` | List of already-inserted cells forming the mod-2 boundary |
| `op` | Whether this step is an insertion (`i`) or deletion (`d`) |

This means the algorithm is **purely algebraic** — it works on any
finite CW-complex, not just simplicial complexes.

---

## 3. Cubical Cells as a Valid CW-Complex

A **cubical complex** on a grid is a CW-complex whose cells are elementary
interval products.  In zztop, a cell in a $d$-dimensional grid is represented
as a tuple of intervals:

$$
c = I_1 \times I_2 \times \cdots \times I_d, \qquad I_k \in \{[a_k, a_k],\; [a_k, a_k+1]\}
$$

- **Degenerate** interval $[a, a]$: vertex-like along axis $k$.
- **Non-degenerate** interval $[a, a+1]$: edge-like along axis $k$.
- The **dimension** of $c$ is the number of non-degenerate intervals.

### 3.1. Boundary Operator

For a $k$-cell with non-degenerate axes $\{j_1, \ldots, j_k\}$, the mod-2
boundary is:

$$
\partial c = \sum_{m=1}^{k} \bigl( c|_{I_{j_m} \to [a_{j_m}, a_{j_m}]} + c|_{I_{j_m} \to [a_{j_m}+1, a_{j_m}+1]} \bigr)
$$

i.e., for each non-degenerate axis, we produce two $(k-1)$-faces by collapsing
that interval to its lower or upper endpoint.  A $k$-cell thus has exactly
$2k$ boundary faces.

### 3.2. $\partial\partial = 0$

This is the fundamental requirement for a valid chain complex.  For cubical
cells it holds because each $(k-2)$-face of a $k$-cell is shared by exactly
**two** of the $2k$ boundary $(k-1)$-faces, so over $\mathbb{F}_2$ it cancels.

**Proof sketch for a 2-cell (square):** Take $c = [0,1] \times [0,1]$.

$$
\partial c = \{[0,0]\times[0,1]\} + \{[1,1]\times[0,1]\} + \{[0,1]\times[0,0]\} + \{[0,1]\times[1,1]\}
$$

Now apply $\partial$ to each 1-face.  Each vertex appears in exactly two of
the four edge boundaries, so $\partial\partial c = 0 \pmod{2}$. This
generalises by induction to any dimension.

The test suite verifies $\partial\partial = 0$ computationally for all cells
in 2D and 3D grids (`test_cubical_zigzag.py::TestCubicalBoundaryOperator`).

---

## 4. The Abstract Interface

The connection between cubical cells and the Dey-Hou algorithm is the
`to_abstract_operations()` method in `ZigzagFiltration`.  For each cell
insertion, it:

1. Computes the boundary cells via `CubicalCell.boundary()`.
2. Looks up the **filtration-operation indices** of those boundary cells
   (which must have been inserted earlier — enforced by the dimension-sorted
   insertion order).
3. Passes `(op='i', dim=k, boundary_indices=[...])` to the backend.

For deletions, it passes `(op='d', dim=k, [insertion_index])`.

The engine (`_python_backend.compute_zigzag_python`
`compute_zigzag_abstract`) receives **only** these abstract triples.  It
never knows or cares that the cells are cubical rather than simplicial.

### 4.1. Insertion Order Guarantees

`ZigzagFiltration.to_operations()` sorts operations at each time step:

- **Insertions**: lowest dimension first (vertices → edges → faces → ...)
- **Deletions**: highest dimension first (faces → edges → vertices)

This ensures the closure property is maintained: when a $k$-cell is inserted,
all its $(k-1)$-boundary cells are already present.

---

## 5. How Frames Become a Zigzag

Given time-evolving grid data with $T$ frames, the builder
(`build_cubical_zigzag`) constructs:

1. **Frames** $\{K_0, K_1, \ldots, K_{T-1}\}$: for each time step, extract
   the set of active cubical cells (those with filtration value below a
   threshold) and close under boundary (all faces included).

2. **Intersection layers**: The `ZigzagFiltration` interleaves intersections:
   $$K_0 \hookleftarrow (K_0 \cap K_1) \hookrightarrow K_1
     \hookleftarrow (K_1 \cap K_2) \hookrightarrow K_2 \hookleftarrow \cdots$$
   
   This is the standard Carlsson-de Silva zigzag construction.  The
   intersections are computed set-theoretically on cell IDs.

3. **Appearance matrix → times**: Each cell's presence across layers is
   encoded as a binary vector.  Contiguous runs of "present" are converted to
   `(birth_time, death_time)` ranges (the `times` vector consumed by the
   original Dionysus/fzz interface).

4. **Operation stream**: The times are expanded into an insertion/deletion
   sequence, sorted by (time, dimension), and passed to the backend.

This is **exactly the same** zigzag construction used for simplicial
complexes — only the cell type and boundary operator are different.

---

## 6. What Changes vs. the Simplicial Pipeline

| Aspect | Old (simplicial) | New (cubical) |
|--------|-------------------|---------------|
| Cell type | `Simplex` (vertex tuple) | `CubicalCell` (interval product) |
| Boundary | Remove one vertex at a time | Collapse one axis at a time |
| # of boundary faces of a $k$-cell | $k + 1$ | $2k$ |
| # cells for a $d$-cube | $d! \cdot 2^d$ (triangulation) | $3^d$ (exact) |
| Zigzag construction | Carlsson-de Silva | Carlsson-de Silva (unchanged) |
| Algorithm | Dey-Hou (via fzz) | Dey-Hou (via abstract interface, same code) |

### 6.1. Are the Barcodes Identical?

**No, and they are not expected to be.**  The cubical and simplicial cellulations
of the same grid are different CW-complexes.  They are **homotopy-equivalent**
(they triangulate the same underlying space), so:

- The **Betti numbers** of each individual frame $K_t$ are the same.
- The **Betti numbers at each layer** of the zigzag should also match (since
  intersections $K_t \cap K_{t+1}$ are taken at the topological level).
- The **individual bars** in the barcode may differ in birth/death times
  because the cell-by-cell insertion/deletion order differs between a cubical
  complex and its triangulation.

**Analogy:** Standard (non-zigzag) persistence on a simplicial vs. cubical
filtration of the same data will generally have different barcodes — this is
well established in computational topology.

### 6.2. Is There a Correctness Concern?

The correctness guarantee is:

> The zigzag barcode computed by zztop on cubical cells is the correct
> barcode of the zigzag module:
> $$H_*(K_0) \to H_*(K_0 \cap K_1) \leftarrow H_*(K_1) \to \cdots$$
> where each $K_t$ is a cubical complex and the maps are induced by
> inclusion.

This follows from:

1. The cubical cells form a valid chain complex ($\partial\partial = 0$).
2. The boundary information passed to the Dey-Hou algorithm is correct.
3. The Dey-Hou algorithm is algebraically exact (it computes the correct
   decomposition of any zigzag module presented as an insertion/deletion
   stream over $\mathbb{F}_2$).

---

## 7. Potential Issues & Open Questions

### 7.1. Cell Activation / Threshold Logic

The `_grid_cells_from_gudhi()` and `_grid_cells_direct()` functions determine
which cells are "active" in each frame.  Currently:

- **Vertices** are always active.
- **Higher cells** are active if their GUDHI filtration value (or max-corner
  value in the direct path) is below the threshold.

**Question:** Should vertex activity also be thresholded?  In the original
`zigzag.py`, all cells from the GUDHI `CubicalComplex` were extracted, with
the threshold applied uniformly.  The current code always includes all vertices,
which means the underlying space is always the full grid — only the higher
cells (edges, faces) are toggled by the threshold.  This affects the
interpretation: the zigzag tracks the topology of the **graph/surface**
induced by the threshold on the grid, not of a subset of grid points.

**If you want vertex-thresholding**, the code would need to be modified to
exclude vertices whose value is above the threshold and then propagate that
exclusion upward (any higher cell touching an excluded vertex is also excluded).

### 7.2. GUDHI vs. Direct Cell Extraction

The two cell-extraction paths (`_grid_cells_from_gudhi` and
`_grid_cells_direct`) should produce identical results on the same input.
The test `test_gudhi_vs_direct_cells` verifies this, but only on small grids.
For production use, the GUDHI path is preferred (faster C++ implementation).

### 7.3. Cubical Complex Size

A $d$-dimensional grid with $N$ vertices per axis has $O(N^d \cdot 3^d)$ cells
in the full cubical complex.  This grows quickly.  For large grids, the
bottleneck is likely the set operations (union, intersection) on cells, not
the zigzag algorithm itself.  This is the same scaling as the simplicial
approach (which has $O(N^d \cdot 2^d \cdot d!)$ cells after triangulation — 
actually worse for $d \geq 3$).

### 7.4. Oriented vs. Unoriented Boundaries

We work over $\mathbb{F}_2$, so orientations do not matter — the boundary
is a mod-2 chain.  If we ever want $\mathbb{Z}$ or $\mathbb{Q}$ coefficients,
the boundary operator would need signs:

$$
\partial c = \sum_{m=1}^{k} (-1)^m \bigl( c|_{\text{lower face } j_m} - c|_{\text{upper face } j_m} \bigr)
$$

This is not implemented and would require changes to both the cell class and
the zigzag backend.

---

## 8. Summary of Guarantees

| Property | Status |
|----------|--------|
| $\partial\partial = 0$ | ✅ Proven and tested |
| Closure maintained during zigzag | ✅ Enforced by dimension-sorted ops |
| Dey-Hou algorithm works on abstract cells | ✅ By construction (algebraic) |
| Betti numbers match simplicial triangulation | ✅ Per-frame (same homotopy type) |
| Individual bars match simplicial triangulation | ❌ Not expected (different CW structure) |
| Coefficients: $\mathbb{F}_2$ | ✅ |
| Coefficients: $\mathbb{Z}$, $\mathbb{Q}$ | ❌ Not supported (boundary needs signs) |
| Arbitrary spatial dimension | ✅ |

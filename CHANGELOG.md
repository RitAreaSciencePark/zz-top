# Changelog

All notable changes to this project will be documented in this file.

## [0.8.0] — 2026-09-28

### Added
- **`zztop.zigzagdiff`** — differentiable zigzag persistence over a fixed
  simplicial complex.  Persistence is factored as *(read values) ∘
  (combinatorial pairing)*: the engine pairs a detached, value-ordered
  insertion/deletion stream exactly, and each bar endpoint is the interpolated
  crossing time `t* = k + (eps − v_k)/(v_{k+1} − v_k)` of a simplex's value
  through the threshold, which carries the gradient (PyTorch autograd; no
  hand-written backward).  Modules: `events` (canonical cell order, crossing
  detection, phantom-pair cancellation, global ordering with a face-respecting
  tie-break, floating-point closure repair, simultaneity flags), `core`
  (engine bridge, engine-index reading, zero-length elimination, differentiable
  event times, `DiffDiagram` / `ZigzagDiffResult`), `frontends` (`lower_star`,
  `passthrough`, `validate_face_condition`) and `layer` (`ZigzagDiffLayer`, an
  `nn.Module` with an optionally learnable threshold).  Conventions: strict
  upper-level presence, weak face condition, frame-0 births at a constant 0,
  essential deaths at `n_frames − 1`, events ordered globally by crossing time
  with deletions before insertions at exact ties, a value exactly at the
  threshold with both neighbours above emits no event.
- `tests/test_zigzagdiff.py` (28 tests): closure after every event on both
  backends, pairing equal to the integer engine, deletion-death regression,
  oracle Betti replay, finite differences with respect to values, parameters
  and threshold, tie and collision behaviour, boundary conventions, an AST
  guard keeping the core geometry-free.
- `docs/zigzagdiff.md` (design, conventions, API, gradients, stability,
  numerics, tests) and the corresponding section of `docs/api.md` and of the
  README.
- `docs/parity_with_fzz.md`: the campaign that verified the Python engine
  against the reference C++ implementation before its removal (see *Removed*),
  and `THIRD_PARTY_NOTICES.md` listing the dependencies and their licences.

### Removed
- **The C++ backend and all vendored third-party code.**  Earlier releases
  compiled `zztop._fzz_core` from a modified copy of the authors' reference
  implementation `fzz` (CGTDA group, Purdue University) and the PHAT headers.
  The `fzz` README restricts that code to academic research use and asks that
  it not be redistributed, so `zztop/_cpp/` (fzz, pybind11 bindings, PHAT) and
  `zztop/zigzag/_cpp_backend.py` are gone; the package is pure Python and
  vendors nothing.  The Python engine, written from the paper, is the only
  engine.  Before the removal both engines were compared on the same operation
  streams: 2280 random zigzag filtrations in three rounds (simplicial, kNN point clouds, 2-D/3-D/4-D cubical grids, meshes, multi-threshold, the zigzagdiff event stream, edge cases) gave identical barcodes, with 965 independent GF(2) ground-truth checks, 1052 prefix-restriction checks and no bar above the ambient dimension; the Python engine is 2-11x slower on large instances.  `fzz` remains the reference the engine is
  measured against and is cited in the README and in `THIRD_PARTY_NOTICES.md`.
- `tests/test_backends.py` (its only purpose was Python-vs-C++ parity); the
  backend parametrisations of the other tests now cover `"python"` and its
  alias `"auto"`.

### Changed
- `ZigzagEngine(backend=...)`, `zigzag_diagrams(..., backend=...)`,
  `ZigzagDiffLayer(..., backend=...)` and the builders accept `"auto"` or
  `"python"`, both selecting the same engine; `"cpp"` raises `ValueError`.
  `ZigzagEngine.backend_name` is always `"python"`.
- Packaging: no build-time dependency on `pybind11`, no compilation step;
  `setup.py` is a stub and `pyproject.toml` carries the metadata.
- The standard GUDHI release is the supported configuration.  The batched
  static-persistence functions (`run_cubical_persistence_gpu`,
  `run_cubical_persistence_sklearn`) keep their GPU dispatch, but the CUDA
  extension of GUDHI they can use is not publicly available; without it they
  run on the CPU, and the documentation now says so instead of linking to a
  private fork.  Verified under GUDHI 3.13: the full suite passes with the
  GPU-parity tests skipped.
- The cluster-specific scripts under `scripts/` (GPU-fork rebuild and test
  jobs, a data-processing batch job) were removed.

### Fixed
- **Wrong cone dimension for cubical (abstract) cells** — affects every
  barcode of a **3-D or higher** cubical zigzag computed so far. The Dey–Hou
  construction cones each deleted cell. `compute_abstract` (C++) and
  `compute_zigzag_python` set the cone's dimension to `|chain| - 1`. That is
  correct for simplices (a k-simplex has k+1 faces), but a k-cube has 2k
  faces. As a result, the cone of a square was labelled dimension 4 (true
  value 3) and the cone of a cube 6 (true value 4). The cone's dimension is
  now `dim(cell) + 1`. The two backends were affected differently:
  - **C++ backend** — PHAT's twist reduction reduces columns grouped by
    dimension. In 3-D the true 3-dimensional columns (cubes, and cones of
    squares) were split across two passes. H1/H2 bars were paired
    incorrectly, the barcode stopped being a zigzag invariant (a prefix run
    disagreed with the restricted full run), and it disagreed with the
    Python backend.
  - **Both backends** — the dimension of an open–closed interval is read
    from its birth column. Genuine H2 classes were therefore reported as
    "H3", which `max_dim=2` callers then discarded.
- **Not affected:** simplicial complexes and 2-D cubical complexes, where the
  labels are either correct or consistently shifted and no dimension-2
  open–closed class exists. This is why the existing 2-D cubical parity test
  passed.
- Verified against dionysus 2 on Freudenthal-triangulated 3-D cubical
  zigzags (random and real-data cases): the fixed engine gives identical
  barcodes.
- New `tests/test_cubical_cone_dimension.py`. It checks per-layer GF(2) Betti
  ground truth, absence of bars above the ambient dimension, prefix
  restriction, backend parity on small 3-D fields, and a hollow-cube cavity.
  These tests fail on the previous code.
- Removed the now-unused `_compute_dim_from_boundary` helper.

## [0.4.0] — 2026-07-01

### Added
- **Mesh zigzag persistence**: track the zigzag persistent homology of a scalar
  field (e.g. per-vertex neural activity) evolving in time on a **fixed**
  triangulated surface mesh, such as a FreeSurfer cortical surface.
  - `run_mesh_zigzag()` — one-call entry point: `(faces, activity, threshold)`
    → barcode.
  - `build_mesh_zigzag()` — lower-level builder returning a `ZigzagFiltration`.
  - Per-frame co-active sub-complex via an instantaneous endpoint model: an edge
    is present when `reduce(x[i], x[j]) > threshold`, a triangle when all three
    of its edges are present, and a vertex when it exceeds the threshold or is
    incident to a present edge — keeping every frame a **closed** complex for
    any reducer.
  - Co-activity reducers: `"min"` (default; reproduces the superlevel /
    excursion-set induced sub-complex), `"mean"`, `"product"`, `"max"`, or any
    elementwise `callable(a, b) -> ndarray`.
  - `include_isolated_vertices` toggle, `vertex_mask` for restricting the mesh
    to an ROI (e.g. medial-wall removal — the intended memory lever), and
    `strict` to select `>` vs `>=` comparisons.
  - Optional `load_freesurfer_surface()` helper behind a lazy `nibabel` import
    (never required by the numpy-only core).
- Test suite for the mesh pipeline (`tests/test_mesh_zigzag.py`, 28 tests):
  closure invariant across all reducers, excursion-set equivalence, disk /
  annulus / closed-surface homology, python/cpp backend parity, and
  `vertex_mask`.
- Documentation: `docs/mesh_zigzag_design.md`.

## [0.3.0] — 2026-04-04

### Added
- **Multi-threshold cubical zigzag persistence**: compute zigzag persistence
  across all threshold levels simultaneously for time-varying grid data.
  - `run_cubical_zigzag_multi_threshold()` — main entry point with parallel
    execution via joblib.
  - `build_cubical_zigzag_multi_threshold()` — lower-level builder.
  - `MultiThresholdResult` — container with Betti surface, per-threshold
    diagrams, total persistence, and frame-level accessors.
- **Cell filtration precomputation**: `precompute_cell_filtrations()` and
  `threshold_cells()` for amortised multi-threshold extraction (~16× faster
  than per-threshold cell enumeration).
- **Threshold selection strategies**: adaptive threshold selection via
  `select_thresholds()` with `"percentile"`, `"uniform"`, `"histogram"`, and
  `"custom"` modes.
- **Multi-threshold vectorizations**: `BettiSurface`,
  `ThresholdIntegratedPersistence`, `ThresholdBirthFrequency` — scikit-learn–
  compatible transformers operating on the (threshold × time) plane.
- Performance benchmarks (`tests/bench_multi_threshold.py`).
- Comprehensive test suite for multi-threshold functionality (29 tests).
- Documentation: `docs/multi_threshold_zigzag.md`.

### Changed
- `build_cubical_zigzag()` now uses named module-level functions for
  `cell_dim_fn` / `cell_boundary_fn` instead of lambdas (enables pickling
  for joblib parallelism).
- `_grid_cells_from_gudhi()` and `_grid_cells_direct()` refactored to
  delegate to precomputation + thresholding primitives (backward-compatible).

## [0.2.0] — 2026-04-02

### Changed
- **Breaking:** `run_cubical_zigzag()` and `run_simplicial_zigzag()` now return
  **0-based frame indices** for birth/death (previously returned raw operation
  indices from the fzz backend).
- `ZigzagEngine.run()` now returns **1-based zigzag-step indices** (matching
  the layer numbering K₀=1, K₀∩K₁=2, K₁=3, …) instead of operation indices.
- `filter_bars()` simplified: `layer_indices` parameter is deprecated (now
  optional, ignored).  Method simply removes zero-length bars (birth == death).

### Added
- `ZigzagFiltration.operation_to_layer_map()` — maps each operation index to
  its 1-based zigzag-step index.

## [0.1.0] — 2026-03-26

### Added
- **Simplicial zigzag**: point clouds → k-NN → GUDHI flag expansion → zigzag barcode.
- **Cubical zigzag**: grid data → native cubical cells → zigzag barcode (no triangulation).
- Fast C++ backend via vendored phat + Dey-Hou coning.
- Pure-Python fallback backend (identical algorithm over GF(2)).
- GPU-accelerated per-frame static persistence via GUDHI GPU fork integration.
- Comprehensive test suite: complexes, backends, parity, GPU.
- Documentation: README, API reference, migration guide, cubical design notes.

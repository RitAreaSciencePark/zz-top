# API Reference

## High-Level Builders

### `run_simplicial_zigzag(point_clouds, knn=10, dim=2, labels=None, backend="auto", output_file=None)`

One-call convenience for simplicial zigzag on point cloud sequences.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `point_clouds` | sequence of ndarray (n_points, n_features) | One point cloud per frame |
| `knn` | int | Number of nearest neighbours for the k-NN graph |
| `dim` | int | Maximum simplex dimension for GUDHI flag expansion |
| `labels` | sequence of ndarray, optional | Per-point labels; if given, edges connect only points with different labels |
| `backend` | str | `"auto"` or `"python"` (the same engine; kept for compatibility) |
| `output_file` | str, optional | CSV path to write the barcode |

**Returns:** `list[(dimension, birth_frame, death_frame)]` — 0-based frame indices.

---

### `run_cubical_zigzag(grid_data, threshold=0.0, use_gudhi=True, backend="auto", output_file=None)`

One-call convenience for cubical zigzag on grid/image data.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `grid_data` | ndarray of shape (N1, …, Nd, n_frames) | Grid values over time; last axis is the frame axis |
| `threshold` | float | Cells with (negated) filtration value below this are active |
| `use_gudhi` | bool | Use GUDHI `CubicalComplex` for cell extraction (True) or pure-Python fallback (False) |
| `backend` | str | `"auto"` or `"python"` (the same engine; kept for compatibility) |
| `output_file` | str, optional | CSV path to write the barcode |

**Returns:** `list[(dimension, birth_frame, death_frame)]` — 0-based frame indices.

---

### `build_simplicial_zigzag(point_clouds, knn=10, dim=2, labels=None, backend="auto")`

Build a `ZigzagFiltration` from point clouds without running the engine.

**Returns:** `ZigzagFiltration`

---

### `build_cubical_zigzag(grid_data, threshold=0.0, use_gudhi=True)`

Build a `ZigzagFiltration` from grid data without running the engine.

**Returns:** `ZigzagFiltration`

---

### `run_cubical_zigzag_multi_threshold(grid_data, thresholds=20, threshold_strategy="percentile", use_gudhi=True, backend="auto", n_jobs=-1, verbose=0, **threshold_kwargs)`

Compute zigzag persistence at **multiple threshold levels** in parallel.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `grid_data` | ndarray (N1, …, Nd, n_frames) | Grid values over time |
| `thresholds` | int or ndarray | Number of auto-selected thresholds, or explicit array |
| `threshold_strategy` | str | `"percentile"`, `"uniform"`, `"histogram"`, or `"custom"` |
| `use_gudhi` | bool | Use GUDHI for cell extraction |
| `backend` | str | `"auto"` or `"python"` (the same engine; kept for compatibility) |
| `n_jobs` | int | Parallel jobs for zigzag (-1 = all cores) |
| `verbose` | int | joblib verbosity |

**Returns:** `MultiThresholdResult` — see below.

---

### `MultiThresholdResult`

Container for multi-threshold zigzag results.

| Attribute / Method | Description |
|--------------------|-------------|
| `thresholds` | `ndarray(m,)` — sorted threshold values |
| `bars` | `dict[float → list[(dim, birth, death)]]` |
| `grid_shape` | Spatial grid dimensions |
| `n_frames` | Number of time frames |
| `threshold_slice(a)` | Barcode at threshold *a* |
| `diagrams(a)` | Per-dim `{int: ndarray(n,2)}` at threshold *a* |
| `all_diagrams()` | Dict of threshold → diagrams |
| `betti_surface(dim=0)` | `ndarray(m, T)` — β_k(a, t) surface |
| `frame_betti(frame, dim)` | `ndarray(m,)` — Betti at fixed frame vs threshold |
| `total_persistence(dim)` | `ndarray(m,)` — total persistence per threshold |

---

## Engine

### `ZigzagEngine(backend="auto")`

Compute zigzag persistent homology with the pure-Python Dey–Hou engine.
`backend` accepts `"auto"` or `"python"` and is kept for compatibility.

**Methods:**

| Method | Description |
|--------|-------------|
| `run(filtration, *, output_file=None)` | Compute barcode; returns `list[(dim, birth, death)]` with 1-based zigzag-step indices |
| `run_raw(filtration)` | Returns raw `list[(birth, death, dim)]` with operation indices (fzz convention) |
| `filter_bars(bars, layer_indices=None)` | Remove zero-length bars (`birth == death`). `layer_indices` is deprecated. |
| `backend_name` (property) | always `"python"` |

---

## Filtration

### `ZigzagFiltration(frames, cell_dim_fn, cell_boundary_fn, use_intersections=True)`

Encodes a zigzag sequence of complexes.  Complex-type agnostic — works with
any hashable cell objects.

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `frames` | list of sets of hashable cells | `frames[t]` is the set of cells at time t |
| `cell_dim_fn` | callable(cell) → int | Returns the dimension of a cell |
| `cell_boundary_fn` | callable(cell) → list[cell] | Returns the mod-2 boundary cells |
| `use_intersections` | bool | Interleave intersection layers between frames (default True) |

**Attributes:**

| Attribute | Description |
|-----------|-------------|
| `cell_to_id` | Dict mapping each unique cell to a canonical int ID |
| `n_cells` | Total number of unique cells across all frames |
| `n_layers` | Number of layers (frames + intersections) |
| `n_frames` | Number of original frames |

**Methods:**

| Method | Description |
|--------|-------------|
| `appearance_matrix()` | (n_layers × n_cells) binary ndarray |
| `compute_times()` | Dionysus-style times list per cell |
| `to_operations()` | Cell-wise insertion/deletion sequence |
| `to_simplicial_operations()` | `(op, vertex_list)` simplicial operation stream (the fzz interface convention) |
| `to_abstract_operations()` | `(op, dim, boundary_indices)` abstract-cell operation stream |
| `operation_to_layer_map()` | Map each operation index to its 1-based zigzag-step index |
| `cell(cell_id)` | Look up cell object by canonical ID |

---

## Complexes

### `Simplex(vertices)`

Immutable simplex defined by a sorted tuple of vertex integers.

| Attribute / Method | Description |
|--------------------|-------------|
| `vertices` | Sorted tuple of vertex ints |
| `dimension` | Number of vertices − 1 |
| `faces()` | List of codimension-1 faces |
| `closure()` | Full closure (all sub-simplices) |

### `SimplicialComplex(cells)`

Closed simplicial complex.  Auto-closes on construction.

### `CubicalCell(intervals)`

Cubical cell as a product of intervals `((lo, hi), ...)`.

| Attribute / Method | Description |
|--------------------|-------------|
| `intervals` | Tuple of (lo, hi) pairs |
| `dim` | Number of non-degenerate intervals |
| `boundary()` | List of boundary (dim−1)-cells |
| `closure()` | Full closure (all sub-cells) |

### `CubicalComplex(cells)`

Closed cubical complex.  Auto-closes on construction.

| Class Method | Description |
|-------------|-------------|
| `from_grid_shape(shape)` | Build complete cubical complex for an n-D grid |

---

## Layer numbering

The zigzag sequence for 3 frames is:

```
K₀ ↔ K₀∩K₁ ↔ K₁ ↔ K₁∩K₂ ↔ K₂
```

Layers are **1-based**: K₀ = layer 1, K₀∩K₁ = layer 2, K₁ = layer 3, etc.

Birth/death values in the barcode refer to these layer indices.

---

## Differentiable zigzag (`zztop.zigzagdiff`)

Full reference with conventions and gradients: [zigzagdiff.md](zigzagdiff.md).

| Name | Kind | Purpose |
|------|------|---------|
| `ZigzagDiffLayer(complex_, dims=(1,), eps=0.0, learnable_eps=False, frontend="lower_star", backend="auto", guard=1e-12, validate=False)` | `nn.Module` | parameters → differentiable diagrams; `forward(x) -> ZigzagDiffResult` |
| `zigzag_diagrams(complex_, values, eps=0.0, dims=(1,), backend="auto", guard=1e-12, validate=False)` | function | the functional core behind the layer |
| `ZigzagDiffResult` | dataclass | `diagrams`, `t`, `schedule`, `raw_pairs`, `total_persistence()` |
| `DiffDiagram` | dataclass | `finite (n,2)`, `essential (m,)`, `finite_events`, `essential_events`, `stamped()`, `lifetimes()`, `total_persistence()` |
| `build_events(complex_, values, eps, guard=1e-12, validate=False)` | function | detached, value-ordered `EventSchedule` |
| `EventSchedule`, `Event` | dataclasses | the combinatorics handed to the engine |
| `event_times(values, eps, schedule, guard=1e-12)` | function | differentiable crossing times, one per event |
| `zigzag_pairs(schedule, backend="auto")` | function | engine pairing, `(b, d, p)` 1-based closed operation intervals |
| `build_operations(schedule)` | function | abstract `(op, dim, boundary)` operation list |
| `replay_validate(schedule)` | function | closure check after every event |
| `cell_order(complex_)`, `vertex_order(complex_)` | functions | canonical cell / vertex order |
| `lower_star(theta, complex_)`, `passthrough(values, complex_, validate=True)` | front ends | vertex values → simplex values; identity with face check |
| `validate_face_condition(values, complex_)` | function | weak face condition `v(face) >= v(coface)` |

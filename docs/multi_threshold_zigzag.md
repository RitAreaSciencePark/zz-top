# Multi-Threshold Cubical Zigzag Persistence

## Overview

The **multi-threshold zigzag** feature extends zz-top to compute zigzag
persistence across *all* threshold levels simultaneously for time-varying
grid data.  Instead of choosing a single threshold *a* and running zigzag
across time frames, you sweep over a range of thresholds and obtain one
zigzag barcode per threshold.

This produces a rich two-parameter invariant over the *(threshold, time)*
plane, capturing how topology evolves both in space (via the threshold/
sublevel filtration) and in time (via zigzag persistence).

---

## Mathematical Background

### The bifiltered complex

Given a time-varying scalar field on a regular grid, we have data
`grid_data[x₁, …, xd, t]` for spatial indices `x` and time index `t`.
The full cubical complex at threshold *a* and time *t* is

```
K(a, t) = { cells c : filtration_value(c, t) < a }
```

where the filtration value of a cell is the maximum of its vertex values
(lower-star convention, matching GUDHI).

The pair *(a, t)* parameterises a **bifiltered** complex: increasing *a*
adds cells (sublevel inclusion), and varying *t* changes the field.

### Threshold-Parameterised Zigzag (TPZ)

For each fixed threshold *a*, we form the zigzag diagram across time:

```
K(a, t₀) ← K(a, t₀) ∩ K(a, t₁) → K(a, t₁) ← ⋯ → K(a, tₙ₋₁)
```

Computing zigzag persistence on this sequence yields a barcode
`B(a) = {(dim, birth, death)}` describing topological features that
appear and disappear as the field evolves.

The *multi-threshold zigzag* is the family `{B(a) : a ∈ Thresholds}`.
This is a 1-parameter family of zigzag barcodes, parameterised by
threshold.  The key derived quantity is the **Betti surface**:

```
β_k(a, t) = number of k-dimensional bars in B(a) alive at frame t
```

### Relationship to multi-parameter persistence

The bifiltered complex `K(a, t)` defines a 2-parameter persistence
module.  Full 2-parameter persistence does not decompose into barcodes
in general.  The TPZ approach is an alternative that *fibers* the
bi-filtration along one axis (threshold) and computes standard zigzag
along the other (time).  This produces a well-defined barcode at each
fiber, and the collection is tractable yet informative.

An analogous approach — fibering along time and computing standard
persistence along threshold — gives the "per-frame persistence diagram"
already available via `run_cubical_persistence_gpu`.  The TPZ approach
is complementary: it tracks temporal evolution at each scale.

---

## Algorithm

### Step 1: Cell filtration precomputation

The key efficiency insight: for a given frame, the filtration value of
every cubical cell depends only on the grid values, not on the threshold.
We compute these values **once per frame** and store them:

```python
cell_filtrations[t] = precompute_cell_filtrations(grid_data[..., t])
# → Dict[CubicalCell, float]
```

### Step 2: Threshold selection

Thresholds are chosen adaptively from the distribution of cell
filtration values.  Three built-in strategies:

| Strategy      | Description                                      |
|---------------|--------------------------------------------------|
| `percentile`  | Equal spacing in percentile space (default)      |
| `uniform`     | Equal spacing between min and max values         |
| `histogram`   | Bin edges of an equal-count histogram            |
| `custom`      | User-supplied array                              |

```python
from zztop.builders.threshold import select_thresholds
thresholds = select_thresholds(cell_filtrations, n_thresholds=20, strategy="percentile")
```

### Step 3: Active cell extraction

For each threshold, extract active cells by simple comparison:

```python
active_cells = {c for c, v in cell_filtrations[t].items() if v < threshold}
```

Under the lower-star convention, filtration values are monotone
(face ≤ coface), so the active set is automatically closed under
taking faces — no explicit closure computation needed.

### Step 4: Parallel zigzag computation

Each threshold produces an independent zigzag filtration.  These are
computed in parallel using `joblib`:

```python
from joblib import Parallel, delayed
barcodes = Parallel(n_jobs=-1)(delayed(run_zigzag)(filt) for filt in filtrations)
```

### Complexity

Let *C* = number of cells, *T* = number of frames, *m* = number of
thresholds.

| Step                  | Time                          | Space              |
|-----------------------|-------------------------------|--------------------|
| Precomputation        | O(C × T)                      | O(C × T)           |
| Threshold selection   | O(C × T)                      | O(C × T)           |
| Cell extraction/frame | O(C) per (threshold, frame)   | O(C)               |
| Zigzag per threshold  | O(C² × T) worst case          | O(C × T)           |
| **Total**             | O(m × C² × T)                 | O(C × T)           |

With *n* cores, the zigzag step scales as O(m × C² × T / n).

---

## API Reference

### Builder functions

#### `run_cubical_zigzag_multi_threshold`

```python
from zztop import run_cubical_zigzag_multi_threshold

result = run_cubical_zigzag_multi_threshold(
    grid_data,                    # ndarray (N1, …, Nd, n_frames)
    thresholds=20,                # int or ndarray
    threshold_strategy="percentile",  # "percentile", "uniform", "histogram", "custom"
    use_gudhi=True,               # use GUDHI for cell extraction
    backend="auto",               # zigzag backend: "auto", "cpp", "python"
    n_jobs=-1,                    # parallel jobs (-1 = all cores)
    verbose=0,                    # joblib verbosity
)
```

**Returns:** `MultiThresholdResult`

#### `build_cubical_zigzag_multi_threshold`

Lower-level function that returns the per-threshold frame lists and
resolved thresholds without running zigzag:

```python
from zztop.builders.multi_threshold import build_cubical_zigzag_multi_threshold

all_frames, thresholds = build_cubical_zigzag_multi_threshold(
    grid_data, thresholds=20, use_gudhi=True,
)
# all_frames[i][t] = set of active CubicalCells at thresholds[i], frame t
```

### `MultiThresholdResult`

Container returned by `run_cubical_zigzag_multi_threshold`.

| Method / Attribute          | Description                                              |
|-----------------------------|----------------------------------------------------------|
| `.thresholds`               | `ndarray(m,)` — sorted threshold values                  |
| `.bars`                     | `dict[float → list[(dim, birth, death)]]`                |
| `.grid_shape`               | Spatial grid dimensions                                  |
| `.n_frames`                 | Number of time frames                                    |
| `.threshold_slice(a)`       | Barcode at threshold *a*                                 |
| `.diagrams(a)`              | Per-dim `{int: ndarray(n,2)}` at threshold *a*           |
| `.all_diagrams()`           | Dict of threshold → diagrams                             |
| `.betti_surface(dim=0)`     | `ndarray(m, T)` — β_k(a, t) surface                     |
| `.frame_betti(frame, dim)`  | `ndarray(m,)` — Betti at fixed frame vs threshold        |
| `.total_persistence(dim)`   | `ndarray(m,)` — total persistence at each threshold      |

### Threshold selection

```python
from zztop.builders.threshold import select_thresholds

thresholds = select_thresholds(
    cell_filtrations_per_frame,   # list of Dict[cell → float]
    n_thresholds=20,
    strategy="percentile",
    percentile_lo=2.0,            # lower percentile bound
    percentile_hi=98.0,           # upper percentile bound
)
```

### Vectorizations

All are scikit-learn `BaseEstimator + TransformerMixin`.

```python
from zztop import BettiSurface, ThresholdIntegratedPersistence, ThresholdBirthFrequency
```

| Vectorizer                       | Input                     | Output shape             |
|----------------------------------|---------------------------|--------------------------|
| `BettiSurface(dim, flatten)`     | `MultiThresholdResult`    | `(n_samples, m × T)` or `(n_samples, m, T)` |
| `ThresholdIntegratedPersistence` | `MultiThresholdResult`    | `(n_samples, m)`         |
| `ThresholdBirthFrequency`        | `MultiThresholdResult`    | `(n_samples, m × T)` or `(n_samples, m, T)` |

---

## Tutorial

### Basic usage: 2-D time-varying field

```python
import numpy as np
from zztop import run_cubical_zigzag_multi_threshold

# Synthetic 2-D field: 20×20 grid, 10 time frames
rng = np.random.default_rng(42)
data = rng.standard_normal((20, 20, 10))

# Run multi-threshold zigzag (auto-selects 20 thresholds)
result = run_cubical_zigzag_multi_threshold(
    data, thresholds=20, backend="auto", n_jobs=-1,
)

print(result)
# MultiThresholdResult(n_thresholds=20, n_frames=10, grid_shape=(20, 20))

# Betti surface: β₀(threshold, time)
surface = result.betti_surface(dim=0)
print(surface.shape)  # (20, 10)
```

### 3-D volumetric data

```python
# 15×15×15 volume, 20 frames
data_3d = rng.standard_normal((15, 15, 15, 20))

result = run_cubical_zigzag_multi_threshold(
    data_3d, thresholds=10, backend="python", n_jobs=4,
)
```

### Custom thresholds

```python
result = run_cubical_zigzag_multi_threshold(
    data, thresholds=np.linspace(-2, 2, 30),
)
```

### ML pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from zztop import BettiSurface

pipe = Pipeline([
    ("betti", BettiSurface(dim=0, flatten=True)),
    ("scale", StandardScaler()),
    ("svm", SVC()),
])

# X_train = list of MultiThresholdResult
# y_train = labels
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

### Parity check: single threshold

```python
from zztop import run_cubical_zigzag

# These should produce identical barcodes:
bars_legacy = run_cubical_zigzag(data, threshold=0.0, backend="python")
result_mt = run_cubical_zigzag_multi_threshold(
    data, thresholds=np.array([0.0]), backend="python", n_jobs=1,
)
bars_mt = result_mt.threshold_slice(0.0)

assert sorted(bars_legacy) == sorted(bars_mt)
```

---

## Performance Guide

### Parallelism

- **`n_jobs=-1`** (default): use all CPU cores.  Ideal when many thresholds.
- **`n_jobs=1`**: sequential; avoids joblib overhead for <5 thresholds.
- The precomputation step (cell filtration values) is always sequential
  but runs **once per frame** regardless of the number of thresholds.

### Memory

- Cell filtration dicts are `O(C)` per frame, where `C` is the total
  number of cubical cells.  For a `15³` grid, `C ≈ 3375 + 6750 + 4500 + 1000 ≈ 15625`.
- All `T` dicts are held in memory during the build step.
- Per-threshold frame lists are `O(C × T)` each, and `m` are created.
  Total: `O(m × C × T)`.

### Scaling expectations

From benchmarks on an 8×8 2-D grid, 5 frames:

| Thresholds | Time (s)  | Time per threshold |
|------------|-----------|-------------------|
| 1          | 0.007     | 0.007             |
| 5          | 0.029     | 0.006             |
| 10         | 0.052     | 0.005             |
| 20         | 0.099     | 0.005             |
| 50         | 0.244     | 0.005             |

Near-linear scaling in number of thresholds, confirming that precomputation
amortises the cell-extraction cost effectively.

### Precompute speedup

For 20 thresholds on a 10×10 2-D grid:

| Method      | Time (s) |
|-------------|----------|
| Naive       | 0.032    |
| Precompute  | 0.002    |
| **Speedup** | **~16×** |

---

## Design Decisions

1. **TPZ over full 2-parameter persistence**: Full 2-parameter persistence
   modules don't decompose into barcodes.  TPZ gives well-understood
   zigzag barcodes at every threshold, tractable and interpretable.

2. **Precompute-then-filter**: The expensive GUDHI/cell-enumeration step
   is done once per frame.  Thresholding is a cheap dict comprehension.
   This is the key to scaling beyond ~5 thresholds.

3. **joblib for parallelism**: Already a transitive dependency via
   scikit-learn.  Familiar API, minimal code overhead.

4. **Named module-level functions instead of lambdas**: Required for
   `joblib` pickling of `ZigzagFiltration` objects.

5. **Separate module** (`multi_threshold.py`): Keeps `cubical_builder.py`
   focused on single-threshold logic and avoids growing it beyond
   manageable size.

6. **Automatic closure guarantee**: Under the lower-star convention,
   `filt(face) ≤ filt(coface)`, so thresholding gives a valid
   subcomplex without explicit closure computation.

---

## Future Work: Vineyard Tracking (Phase 2)

The next phase will add **feature tracking across thresholds**:
given barcodes `B(a₁)` and `B(a₂)` at neighbouring thresholds,
match bars via bottleneck/Wasserstein optimal matching to produce
*vines* — trajectories of topological features across thresholds.

This enables:
- **Vine length**: how stable a feature is across thresholds
- **Vine stability**: how much birth/death coordinates shift
- **Vineyard visualisation**: 3-D plots of bars evolving in the
  (threshold, birth, death) space

Detailed design deferred until TPZ benchmarks validate parameter choices.

# Migration Guide

This guide shows how to replace code using the original `zigzag.py`,
`zigzagllms.py`, and `dionysus` + `fastzigzag` with **zztop**.

---

## 1. Simplicial zigzag (replacing `zigzagllms.py` + Dionysus)

### Before (zigzagllms.py)

```python
from zigzagllms import ZIGZAG

zz = ZIGZAG(dataset, labels, knn=10, dim=2)
zz.compute_zigzag()
# Barcode in zz.bars as (birth, death, dim)
```

### After (zztop)

```python
from zztop import run_simplicial_zigzag

bars = run_simplicial_zigzag(dataset, knn=10, dim=2, labels=labels)
# bars is [(dimension, birth, death), ...]
```

**Note:** The output order is `(dim, birth, death)` not `(birth, death, dim)`.
Use `ZigzagEngine().run_raw(filt)` if you need the old `(birth, death, dim)`
convention.

---

## 2. Cubical zigzag (replacing `zigzag.py` + Dionysus)

### Before (zigzag.py)

```python
# The old pipeline triangulated cubical cells into simplices:
from zigzag import zigzag_from_cubical_complex

# This internally:
# 1. Builds GUDHI CubicalComplex
# 2. Converts to SimplicialComplex (triangulation!)
# 3. Feeds simplicial complex to Dionysus
```

### After (zztop)

```python
from zztop import run_cubical_zigzag
import numpy as np

# grid_data shape: (N1, N2, ..., n_frames)
bars = run_cubical_zigzag(grid_data, threshold=0.0)
```

No triangulation.  Cubical cells go directly into the zigzag algorithm.

---

## 3. More control (mid-level API)

### Before

```python
import dionysus as d
import pyfzz

# Manual construction of filtrations and times lists
filt = d.Filtration(simplices)
times = [...]
bars = d.zigzag_homology_persistence(filt, times)
```

### After

```python
from zztop import ZigzagFiltration, ZigzagEngine, Simplex

frames = [set_of_simplices_for_frame_0, set_of_simplices_for_frame_1, ...]

filt = ZigzagFiltration(
    frames=frames,
    cell_dim_fn=lambda s: s.dimension,
    cell_boundary_fn=lambda s: s.faces(),
)

engine = ZigzagEngine(backend="auto")
bars = engine.run(filt)
```

---

## 4. Key differences

| Aspect | Old pipeline | zztop |
|--------|-------------|-------|
| External dependencies | Dionysus, fastzigzag (separate installs) | Self-contained (pure Python) |
| Cubical complexes | Triangulated → simplicial | Native cubical cells |
| Output format | `(birth, death, dim)` | `(dim, birth, death)` via `run()` |
| Backend | Dionysus C++ only | pure Python (Dey–Hou algorithm) |
| Grid threshold | Hard-coded `-0.0` | Configurable `threshold` parameter |

---

## 5. Barcode interpretation

The zigzag filtration uses 1-based step indices internally:

```
K₀(step 1) ↔ K₀∩K₁(step 2) ↔ K₁(step 3) ↔ K₁∩K₂(step 4) ↔ K₂(step 5) ↔ ...
```

High-level builders (`run_cubical_zigzag`, `run_simplicial_zigzag`) return
**0-based frame indices** (0 … N−1), mapping intersection steps to the next
frame: `frame = step // 2` (1-based step numbering).  This follows the
convention in Gardinazzi et al. (2025).

The mid-level `ZigzagEngine.run()` returns 1-based **zigzag-step** indices
if you need intersection-level resolution.

**Caveat:** The cubical barcode will generally differ in detail from the
old triangulated-simplicial barcode.  However, Betti numbers at each original
frame are identical because a cubical complex and its triangulation are
homotopy equivalent.  See `docs/cubical_zigzag_design.md` for the full
mathematical justification.

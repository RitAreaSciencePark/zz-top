# zz-top — ZigZag TOpological Persistence

**zz-top** computes zigzag persistent homology for both **simplicial** and
**cubical** complexes through a single, self-contained Python package.

Its engine is an independent, pure-Python implementation of the Dey–Hou fast
zigzag algorithm (ESA 2022): the zigzag filtration is converted into a standard
filtration by coning off deletions and reduced over GF(2) with the twist
optimisation.  The authors' reference implementation, `fzz`, is not
redistributed here (its terms restrict it to academic use); the engine's
output was verified bar-for-bar against it before the vendored copy was
removed ([docs/parity_with_fzz.md](docs/parity_with_fzz.md)).

## Features

| | |
|---|---|
| **Simplicial zigzag** | Point clouds → k-NN graph → GUDHI flag expansion → zigzag barcode |
| **Cubical zigzag** | Grid data → native cubical cells → zigzag barcode (no triangulation) |
| **Mesh zigzag** | Scalar field on a fixed triangulated surface (e.g. FreeSurfer cortex) → co-active sub-complex → zigzag barcode |
| **Differentiable zigzag** | `zztop.zigzagdiff`: bar endpoints as interpolated crossing times of an evolving filtering function, exact pairing, gradients through PyTorch autograd |
| **Pure-Python engine** | Independent implementation of the Dey–Hou algorithm over GF(2); no compiler, no vendored third-party code |
| **Arbitrary dimension** | Works on 2-D grids, 3-D volumes, or higher |

## Installation

```bash
pip install -e ".[all]"
```

The package is pure Python; nothing is compiled.

### Dependencies

- **numpy** ≥ 1.20
- **scipy** ≥ 1.7
- **scikit-learn** ≥ 1.0
- **gudhi** ≥ 3.5

Optional: `pytest` (tests), `matplotlib` (plotting), `torch` (the
differentiable layer `zztop.zigzagdiff` and tensor output of the vectorizers):
`pip install -e ".[all,torch]"`.

### Installing with GPU support

The GPU-accelerated persistence path requires the
[GUDHI GPU fork](https://github.com/matteobiagetti/gudhi-devel/tree/gpu-cubical-personal-20260226)
compiled with CUDA for your GPU architecture.

**Follow the installation instructions in the
[gudhi-devel GPU branch README](https://github.com/matteobiagetti/gudhi-devel/tree/gpu-cubical-personal-20260226)**
to build and install the CUDA extension.  The GUDHI installation is kept
external to zztop to avoid complicating the install process.

Once the GPU-enabled GUDHI is installed, install zztop as usual:

```bash
pip install -e ".[all]"
```

Verify the GPU path is working:

```python
from zztop import gpu_available, gpu_supports_3d

print(gpu_available())      # True
print(gpu_supports_3d())    # True
```

## Quick start

### Simplicial zigzag (point clouds)

```python
from zztop import run_simplicial_zigzag

# point_clouds: list of ndarrays, each of shape (n_points, n_features)
bars = run_simplicial_zigzag(point_clouds, knn=10, dim=2)
# bars is a list of (dimension, birth, death) tuples
```

### Cubical zigzag (grid / image data)

```python
from zztop import run_cubical_zigzag
import numpy as np

# grid_data: ndarray of shape (N1, N2, ..., n_frames)
grid_data = np.random.randn(8, 8, 5)
bars = run_cubical_zigzag(grid_data, threshold=0.0)
```

### Mesh zigzag (scalar field on a fixed surface mesh)

Track how the topology of a *co-active* region reorganises over time when a
scalar field (e.g. per-vertex neural activity) evolves on a **fixed**
triangulated surface — for example a FreeSurfer cortical surface.

```python
from zztop import run_mesh_zigzag
import numpy as np

# faces:    (F, 3) triangle vertex indices of a fixed surface mesh
# activity: (V, T) per-vertex values over T frames (e.g. neural activity)
faces = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 0]])
activity = np.random.randn(6, 20)

bars = run_mesh_zigzag(faces, activity, threshold=0.5, coactivity="min")
```

Per frame, an **edge** is present when `reduce(x[i], x[j]) > threshold`, a
**triangle** when all three of its edges are, and a **vertex** when it exceeds
the threshold or is incident to a present edge — so every frame is a closed
sub-complex.  The reducer defaults to `"min"` (an edge is on when **both**
endpoints are active), which makes each frame the superlevel / **excursion-set**
induced sub-complex of the mesh; `"mean"`, `"product"`, `"max"`, or any
elementwise callable are also accepted.  Restrict large surfaces to a region of
interest (e.g. remove the medial wall) with `vertex_mask`.  See
[docs/mesh_zigzag_design.md](docs/mesh_zigzag_design.md) for the full
construction and its guarantees.

### GPU-accelerated per-frame persistence (learning pipelines)

For learning pipelines you often need standard (non-zigzag) persistence
diagrams computed independently for each frame, as fast as possible.
`run_cubical_persistence_gpu` auto-dispatches to the GPU when using the
[GUDHI GPU fork](https://github.com/matteobiagetti/gudhi-devel/tree/gpu-cubical-personal-20260226),
falling back seamlessly to CPU when running with standard GUDHI:

```python
from zztop import run_cubical_persistence_gpu, gpu_available, gpu_supports_3d

print(f"GPU available: {gpu_available()}")
print(f"3D GPU supported: {gpu_supports_3d()}")

# 2D grids — shape (H, W, n_frames)
diagrams_2d = run_cubical_persistence_gpu(grid_2d, backend="auto")

# 3D volumes — shape (H, W, D, n_frames)
diagrams_3d = run_cubical_persistence_gpu(grid_3d, backend="auto")

# Each entry: diagrams[t][dim] → ndarray(n_bars, 2)
```

#### Backend selection

| `backend=` | Behaviour |
|---|---|
| `"auto"` (default) | GPU for batches (n_frames > 1) when available, otherwise CPU. Single-frame inputs always use CPU because kernel-launch overhead dominates. |
| `"gpu"` | Force GPU. Raises `RuntimeError` if GPU extension is missing. |
| `"cpu"` | Force CPU via GUDHI `CubicalComplex`. |

```python
# Force GPU
diagrams = run_cubical_persistence_gpu(grid_data, backend="gpu")

# Force CPU
diagrams = run_cubical_persistence_gpu(grid_data, backend="cpu")
```

#### 3D GPU persistence

The GUDHI GPU fork (branch `gpu-cubical-personal-20260226`) includes CUDA
kernels for both **2D** and **3D** cubical persistence.  When 3D kernels are
available, `run_cubical_persistence_gpu` handles 3D volumes
(shape `(H, W, D, n_frames)`) on the GPU automatically.

For 3D grids with `input_type="vertices"` (the default), the GPU path
internally converts vertex filtration values to top-dimensional cells via
`_vertices_to_top_cells_3d` (max of 8 corner vertices per voxel).  The CPU
fallback applies the **same conversion** so that `backend="cpu"` and
`backend="gpu"` produce identical diagrams.

> **Note:** This top-cells conversion means 3D persistence is computed on a
> reduced `(H-1, W-1, D-1)` complex rather than the full `(2H-1, 2W-1, 2D-1)`
> bitmap.  The 2D path uses the full bitmap and does not have this reduction.

**Benchmark** (15×15×10 neural grid, 300 frames, A100 GPU):

| | 20 frames | 300 frames |
|---|---|---|
| CPU | 1.73 s | 1.71 s |
| GPU | 0.13 s | 0.24 s |
| **Speedup** | **13.6×** | **7.1×** |

There is also an sklearn-compatible wrapper that uses the fork's
`CubicalPersistence(backend="gpu")` transformer:

```python
from zztop import run_cubical_persistence_sklearn

# Returns list of arrays in sklearn CubicalPersistence format
result = run_cubical_persistence_sklearn(
    grid_data, homology_dimensions=(0, 1), backend="auto"
)
```

> **Hardware:** See [Installing with GPU support](#installing-with-gpu-support)
> for how to build the CUDA extension for your GPU.  The default build targets
> V100 (sm_70), A100 (sm_80), and H100 (sm_90).

### Differentiable zigzag (`zztop.zigzagdiff`)

A zigzag barcode is a combinatorial object, so any loss built on it is
piecewise constant in the data.  `zztop.zigzagdiff` factors persistence as
*(read values) ∘ (combinatorial pairing)* and makes only the first factor
differentiable: the pairing is computed exactly by the engine on a detached
integer event stream, and each bar endpoint is the real-valued instant at
which a simplex's linearly interpolated value crosses the threshold, through
which PyTorch autograd reaches the filtering function and its parameters.

```python
import torch
from zztop.complex.simplicial import SimplicialComplex
from zztop.zigzagdiff import ZigzagDiffLayer

torch.set_default_dtype(torch.float64)

K = SimplicialComplex([[0, 1], [1, 2], [2, 3], [0, 3]])   # boundary of a square
theta = torch.tensor([[0.0] * 4] + [[1.0] * 4] * 4 + [[0.0] * 4],   # 6 frames, one value per vertex:
                     requires_grad=True)                            # the square is present in frames 1-4

layer = ZigzagDiffLayer(K, dims=(0, 1), eps=0.5)   # lower-star front end, present iff value > eps
result = layer(theta)

result[1].finite                    # tensor([[0.5, 4.5]]): the loop's birth and death crossing times
loss = result.total_persistence()   # sum of bar lengths over the requested degrees
loss.backward()                     # gradient reaches theta through the crossing times only
```

`result.schedule` is the detached event sequence handed to the engine,
`result.t` the differentiable event-time vector and `result[p].finite_events`
the integer pairing, so `result[p].finite == result.t[result[p].finite_events]`.
Two front ends are provided, `lower_star` (values on vertices, minimum over
the vertices of a simplex) and `passthrough` (values on simplices, face
condition checked); a degree with no bars contributes zero and stays attached
to the graph.  See [docs/zigzagdiff.md](docs/zigzagdiff.md) for the
conventions (presence, face condition, tie-breaking, padding frames, essential
bars), the full API, the gradient and stability contract and the tests.

### Mid-level API

Build the filtration yourself and inspect it before running:

```python
from zztop import build_cubical_zigzag, ZigzagEngine

filt = build_cubical_zigzag(grid_data, threshold=0.0)
print(f"{filt.n_cells} unique cells, {filt.n_layers} layers")

engine = ZigzagEngine()
bars = engine.run(filt)
```

### Low-level API

Construct complexes and filtrations from scratch:

```python
from zztop import Simplex, ZigzagFiltration, ZigzagEngine

# Two frames of a simplicial complex
frame0 = {Simplex([0, 1]), Simplex([0]), Simplex([1]), Simplex([2])}
frame1 = {Simplex([1, 2]), Simplex([1]), Simplex([2]), Simplex([3])}

filt = ZigzagFiltration(
    frames=[frame0, frame1],
    cell_dim_fn=lambda s: s.dimension,
    cell_boundary_fn=lambda s: s.faces(),
)
bars = ZigzagEngine().run(filt)
```

The same pattern works with `CubicalCell` objects:

```python
from zztop import CubicalCell, ZigzagFiltration, ZigzagEngine

v00 = CubicalCell(((0, 0), (0, 0)))  # vertex at (0,0)
v10 = CubicalCell(((1, 1), (0, 0)))  # vertex at (1,0)
e   = CubicalCell(((0, 1), (0, 0)))  # horizontal edge

frame0 = {v00, v10, e}
frame1 = {v00}

filt = ZigzagFiltration(
    frames=[frame0, frame1],
    cell_dim_fn=lambda c: c.dim,
    cell_boundary_fn=lambda c: c.boundary(),
)
bars = ZigzagEngine().run(filt)
```

## Output format

All functions return a list of `(dimension, birth, death)` tuples:

```
[(0, 0, 2), (0, 1, 3), (1, 0, 1)]
```

- **dimension**: homological dimension (0 = connected component, 1 = loop, …)
- **birth** / **death**: indices whose semantics depend on the API level:

| Function | Birth / death semantics |
|----------|------------------------|
| `run_cubical_zigzag()` | **0-based frame indices** (0 … N−1) |
| `run_simplicial_zigzag()` | **0-based frame indices** (0 … N−1) |
| `run_mesh_zigzag()` | **1-based zigzag-layer indices** (1 … 2N−1) |
| `ZigzagEngine.run()` | **1-based zigzag-step indices** (1 … 2N−1) |

Zigzag steps are numbered: K₀ (1), K₀∩K₁ (2), K₁ (3), K₁∩K₂ (4), …

`run_mesh_zigzag()` returns these layer indices directly (like
`ZigzagEngine.run()`), so both frame layers (odd) and intersection layers (even)
are visible; read the Betti number of frame `t` at layer `2t + 1`.

The high-level builders convert intersection steps to the next frame
following the convention in Gardinazzi et al. (2025):
`frame = step // 2` (1-based step indexing).

You can also write the barcode to CSV:

```python
bars = run_cubical_zigzag(grid_data, output_file="barcode.csv")
```

## The engine

`ZigzagEngine` has a single, pure-Python implementation.  Its `backend`
argument is kept for compatibility with earlier releases and accepts
`"auto"` or `"python"` (both the same); `engine.backend_name` is always
`"python"`.  Earlier releases also shipped a C++ backend built on the
authors' `fzz` code; it was removed in 0.8.0 for licensing reasons after a
parity campaign showed identical output
([docs/parity_with_fzz.md](docs/parity_with_fzz.md)).

## Package structure

```
zztop/
├── __init__.py              # Public API
├── complex/
│   ├── boundary.py          # CellComplex abstract base class
│   ├── simplicial.py        # Simplex, SimplicialComplex
│   └── cubical.py           # CubicalCell, CubicalComplex
├── zigzag/
│   ├── filtration.py        # ZigzagFiltration (complex-type agnostic)
│   ├── engine.py            # ZigzagEngine
│   └── _python_backend.py   # Dey-Hou fast zigzag over GF(2)
├── zigzagdiff/
│   ├── events.py            # Detached, value-ordered event schedule
│   ├── core.py              # Engine bridge + differentiable event times
│   ├── frontends.py         # lower_star / passthrough
│   └── layer.py             # ZigzagDiffLayer (torch.nn.Module)
├── builders/
│   ├── simplicial_builder.py  # Point clouds → simplicial zigzag
│   ├── cubical_builder.py     # Grid data → cubical zigzag
│   ├── mesh_builder.py        # Scalar field on a fixed mesh → mesh zigzag
│   └── utils.py               # Shared helpers
└── docs/
    ├── cubical_zigzag_design.md  # Mathematical justification (cubical)
    ├── mesh_zigzag_design.md     # Mathematical justification (mesh)
    └── zigzagdiff.md             # The differentiable read-out
```

## How cubical zigzag works

The Dey-Hou algorithm is **purely algebraic** — it only needs each cell's
dimension and boundary chain (mod 2).  It never inspects vertices or
coordinates.  Cubical cells satisfy the same axioms (∂∂ = 0) as simplices, so
they can be fed directly into the algorithm through an abstract cell interface.

A cubical *k*-cell on a grid is a product of intervals:

$$c = I_1 \times I_2 \times \cdots \times I_d$$

Each interval is either degenerate `[a, a]` (vertex-like) or non-degenerate
`[a, a+1]` (edge-like).  The number of non-degenerate intervals is the cell's
dimension.  Its boundary consists of 2k faces obtained by collapsing each
non-degenerate axis to its lower or upper endpoint.

See [docs/cubical_zigzag_design.md](docs/cubical_zigzag_design.md) for the
full mathematical justification.

## How mesh zigzag works

The geometry (vertices, edges, faces) is **fixed**; only the sub-complex that is
"switched on" at each frame changes as the scalar field moves.  For frame `t`
with per-vertex values `x[:, t]` and threshold θ:

- edge `(i, j)` is present iff `reduce(x[i], x[j]) > θ`;
- triangle `(i, j, k)` is present iff all three of its edges are present;
- vertex `v` is present iff `x[v] > θ` **or** `v` is an endpoint of a present
  edge.

The vertex rule keeps every frame **closed** for any reducer, and for
`reduce=min` it reproduces exactly the excursion-set (superlevel) induced
sub-complex on `{v : x[v] > θ}`.  Frames are handed straight to
`ZigzagFiltration`, which builds the intersection layers and insertion/deletion
stream — the same engine used by the simplicial and cubical pipelines.

> A closed surface (a whole hemisphere) carries an essential **H2** class.
> Being the globally top-dimensional essential class, the engine reports it
> at its birth layer as a single `(1, 1)` bar rather than a full-length
> interval; lower-dimensional classes (H0, and H1 when 2-cells are present) get
> full-length bars as usual.

See [docs/mesh_zigzag_design.md](docs/mesh_zigzag_design.md) for the full
construction, reducer semantics, and closure/excursion guarantees.

## Vectorizations

The `zztop.vectorizations` sub-package converts persistence diagrams (from
zigzag or static GPU persistence) into **fixed-size feature vectors** suitable
for machine-learning pipelines.  All vectorizers are scikit-learn–compatible
(`fit` / `transform`) and work inside `sklearn.pipeline.Pipeline`.

### Quick example

```python
from zztop import run_cubical_zigzag
from zztop.vectorizations import PersistenceImage, BettiCurve

bars = run_cubical_zigzag(grid_data, threshold=0.0)

# Persistence image (Adams et al., 2017)
pi = PersistenceImage(resolution=(20, 20), sigma=1.0)
features = pi.fit_transform([bars])   # shape (1, n_dims × 400)

# Betti curve (1-D summary)
bc = BettiCurve(resolution=100)
curve = bc.fit_transform([bars])       # shape (1, n_dims × 100)
```

Any zztop output format is accepted: zigzag barcodes (`list` of `(dim, birth,
death)` tuples), GPU diagram dicts (`{dim: ndarray}`), or raw `ndarray`.

### Available vectorizers

| Category | Class | Description |
|----------|-------|-------------|
| **Standard** | `PersistenceImage` | Smoothed 2-D histogram in birth–persistence plane |
| | `PersistenceLandscape` | Piecewise-linear envelope functions (Bubenik, 2015) |
| | `Silhouette` | Weighted power-mean of tent functions |
| | `PersistenceEntropy` | Shannon entropy of normalised lifetimes |
| | `Amplitude` | Scalar summary (Wasserstein, bottleneck, landscape norms) |
| | `PersistenceStatistics` | Count, mean, std, max of births/deaths/lifetimes + entropy |
| **Curves** | `BettiCurve` | Number of bars alive at each filtration value |
| | `LifeCurve` | Total persistence of alive bars at each value |
| | `BirthCurve` / `DeathCurve` | Cumulative birth / death counts |
| | `MidlifeCurve` | KDE of midpoint values |
| | `MultiplicativeLifeCurve` | Log-product of alive-bar lifetimes |
| | `PersistenceCurve` | Generic: user-supplied summary function |
| **Zigzag** | `BettiProfile` | Betti number at each frame (layer) |
| | `BirthFrequency` | Relative birth frequency per frame |
| | `PersistenceProfile` | Mean persistence of alive bars per frame |
| | `TurnoverRate` | Topological event rate per frame |
| | `EffectivePersistenceImage` | PI weighted by birth frequency |
| | `CumulativePersistence` | Cumulative total persistence over frames |

Experimental zigzag descriptors (not in `__all__`): `LayerTransitionEntropy`,
`PersistenceFlux`, `CrossDimCorrelation`, `LifetimeSpectrum`.

### sklearn pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from zztop.vectorizations import PersistenceImage

pipe = Pipeline([
    ("pi", PersistenceImage(resolution=(20, 20), sigma=1.0, dimensions=[0, 1])),
    ("scaler", StandardScaler()),
    ("svm", SVC()),
])
pipe.fit(train_diagrams, train_labels)
```

### Optional torch support

If PyTorch is installed, any vectorizer can return tensors:

```python
pi = PersistenceImage(resolution=(20, 20), sigma=1.0)
pi.fit(train_diagrams)
tensor = pi.to_torch(test_diagrams)  # torch.Tensor

# Or wrap any vectorizer for torch output:
from zztop.vectorizations._torch import TorchVectorizer
tpi = TorchVectorizer(PersistenceImage(resolution=(20, 20)), device="cuda")
tensor = tpi.fit_transform(diagrams)
```

See [docs/vectorizations.md](docs/vectorizations.md) for the full
mathematical reference.

## Running tests

```bash
cd zztop
pip install -e ".[test]"
python3 -m pytest tests/ -v
```

### GPU tests on a Slurm cluster

The GPU parity tests (verifying GPU and CPU produce identical diagrams) are
skipped on machines without a GPU.  To run them on a GPU node:

```bash
sbatch scripts/run_gpu_tests.sbatch
```

The included sbatch script activates a conda environment with the GPU-enabled
GUDHI fork, installs zztop, and runs the full test suite.  **Edit the script**
to match your cluster's partition names, GPU types, conda paths, and module
loads before submitting.

The test suite covers:

- **Complexes** — Simplex/CubicalCell construction, boundary, closure
- **Simplicial zigzag** — Verified against the worked example of the fzz README and against GF(2) ground truth
- **Cubical zigzag** — ∂∂=0, Euler characteristic, GUDHI vs direct extraction
- **Cubical cone dimension** — 3-D regression tests for the abstract-cell path (per-layer GF(2) Betti ground truth, no bars above the ambient dimension, prefix restriction, backend parity, hollow cube)
- **Differentiable zigzag** — closure after every event, pairing equal to the integer engine, deletion-death regression, finite-difference checks of the gradient with respect to values, parameters and threshold, phantom-pair and tie rules, boundary conventions (requires `torch`)
- **Mesh zigzag** — closure invariant (all reducers), excursion-set equivalence, disk/annulus/sphere homology, `vertex_mask`
- **Parity** — Betti numbers match between native cubical and triangulated-simplicial pipelines
- **Parity with fzz** — the campaign that compared the engine with the reference C++ implementation before its removal is documented in [docs/parity_with_fzz.md](docs/parity_with_fzz.md)
- **GPU cubical** — Detection, CPU fallback, GPU ↔ CPU parity (on GPU nodes)
- **Dionysus** — Optional parity check (skipped if Dionysus not installed)

## Algorithm reference

> T. K. Dey and T. Hou.
> *Fast Computation of Zigzag Persistence.*
> ESA 2022.

The algorithm converts a zigzag filtration into a standard filtration by
coning off deletions, then reduces the boundary matrix over GF(2) with the
twist optimisation and maps the pairs back to zigzag intervals.  The authors'
reference implementation is [`fzz`](https://github.com/taohou01/fzz) (CGTDA
group, Purdue University), whose terms restrict it to academic research use
and forbid redistribution; `zztop` therefore re-implements the published
algorithm in Python and does not include any of that code.  The abstract-cell
interface (cubical and other non-simplicial cells) and the cone-dimension rule
`dim(cone) = dim(cell) + 1` are documented in
[docs/cubical_zigzag_design.md](docs/cubical_zigzag_design.md).

## License

BSD-3-Clause.  The package vendors no third-party code; its dependencies and
their licences are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

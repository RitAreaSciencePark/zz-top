"""Parity campaign: zztop's pure-Python engine against the vendored fzz (C++) backend.

Every case builds one zigzag filtration, runs both backends on the *same*
operation stream and compares the raw (birth, death, dim) triples as exact
multisets, plus the layer-mapped barcode.  A subset of every family is also
checked against GF(2) Betti numbers computed independently at every layer.
Writes parity_fzz.json and parity_fzz.md next to this file.
"""
import sys, time, json, random, itertools, platform, collections, importlib.util, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
W = sys.argv[1]                                   # repo root with the compiled extension
sys.path.insert(0, W); sys.path.insert(0, str(pathlib.Path(W) / "tests"))
import zztop
from zztop.zigzag import _cpp_backend
from zztop.zigzag.engine import ZigzagEngine
from zztop.zigzag.filtration import ZigzagFiltration
from zztop.complex.simplicial import Simplex, SimplicialComplex
from zztop.builders.simplicial_builder import build_simplicial_zigzag
from zztop.builders.cubical_builder import build_cubical_zigzag
from zztop.builders.mesh_builder import build_mesh_zigzag
from zztop.builders.multi_threshold import run_cubical_zigzag_multi_threshold
from zztop.zigzagdiff.events import build_events
from zztop.zigzagdiff.core import zigzag_pairs
from zztop.zigzagdiff.frontends import lower_star
import torch
spec = importlib.util.spec_from_file_location("tzdd", pathlib.Path(W) / "tests" / "test_zigzag_deletion_death.py")
tzdd = importlib.util.module_from_spec(spec); spec.loader.exec_module(tzdd)
assert _cpp_backend.AVAILABLE, "C++ backend must be compiled for the campaign"
import gudhi

rng = np.random.default_rng(20260928); random.seed(20260928)
PY, CPP = ZigzagEngine(backend="python"), ZigzagEngine(backend="cpp")
assert PY.backend_name == "python" and CPP.backend_name == "cpp"
results = collections.OrderedDict()

def compare(name, filt, gt_dims=(), dim_fn=None, bdry_fn=None):
    """Both backends on one filtration; optional GF(2) ground truth."""
    r = results.setdefault(name, dict(cases=0, mismatch=0, gt_checked=0, gt_fail=0, ops=0, max_ops=0,
                                      bars=0, essential=0, t_py=0.0, t_cpp=0.0, examples=[]))
    n_ops = len(filt.to_operations())
    t0 = time.perf_counter(); raw_py = PY.run_raw(filt); t1 = time.perf_counter()
    raw_cpp = CPP.run_raw(filt); t2 = time.perf_counter()
    same = sorted(raw_py) == sorted(raw_cpp)
    bars_py, bars_cpp = PY.run(filt), CPP.run(filt)
    same &= sorted(bars_py) == sorted(bars_cpp)
    r["cases"] += 1; r["ops"] += n_ops; r["max_ops"] = max(r["max_ops"], n_ops)
    r["bars"] += len(bars_py); r["essential"] += sum(1 for _, _, d in bars_py if d == filt.n_layers + 1)
    r["t_py"] += t1 - t0; r["t_cpp"] += t2 - t1
    if not same:
        r["mismatch"] += 1
        if len(r["examples"]) < 3:
            r["examples"].append(dict(n_ops=n_ops, only_py=sorted(set(raw_py) - set(raw_cpp))[:5],
                                      only_cpp=sorted(set(raw_cpp) - set(raw_py))[:5]))
    if gt_dims:
        r["gt_checked"] += 1
        for k in gt_dims:
            truth = tzdd.per_layer_betti(filt, dim_fn, bdry_fn, k)
            if tzdd.barcode_betti(bars_py, k, filt.n_layers) != truth or tzdd.barcode_betti(bars_cpp, k, filt.n_layers) != truth:
                r["gt_fail"] += 1; break
    return same

# --------------------------------------------------------------------------- families
def random_flag_complex(n, p, max_dim):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    K = SimplicialComplex([[v] for v in range(n)] + [list(e) for e in edges])
    adj = {v: set() for v in range(n)}
    for i, j in edges: adj[i].add(j); adj[j].add(i)
    if max_dim >= 2:
        for i, j in edges:
            for k in sorted(adj[i] & adj[j]):
                if k > j: K.add([i, j, k])
    if max_dim >= 3:
        tris = [c for c in K.all_cells() if c.dimension == 2]
        for t in tris:
            common = set.intersection(*(adj[v] for v in t))
            for w in common:
                if w > max(t): K.add(list(t) + [w])
    return K

def lower_star_frames(K, T, eps, quantise=None):
    verts = sorted(v for c in K.all_cells() if c.dimension == 0 for v in c)
    vals = rng.normal(size=(T, len(verts)))
    if quantise: vals = np.round(vals * quantise) / quantise
    frames = []
    for t in range(T):
        present = {c for c in K.all_cells() if min(vals[t, v] for v in c) > eps}
        frames.append(present)
    return frames

def fam_simplicial_lowerstar(n_cases, label, quantise=None, max_dim=2):
    for _ in range(n_cases):
        K = random_flag_complex(int(rng.integers(4, 22)), float(rng.uniform(0.2, 0.7)), max_dim)
        frames = lower_star_frames(K, int(rng.integers(2, 12)), float(rng.normal(0, 0.6)), quantise)
        filt = ZigzagFiltration(frames, lambda s: s.dimension, lambda s: s.faces())
        gt = (0, 1, 2) if results.get(label, {}).get("gt_checked", 0) < 60 and filt.n_cells < 120 else ()
        compare(label, filt, gt, lambda s: s.dimension, lambda s: s.faces())

def fam_knn(n_cases):
    for _ in range(n_cases):
        T, n, d = int(rng.integers(2, 9)), int(rng.integers(8, 36)), int(rng.integers(2, 4))
        clouds = [rng.normal(size=(n, d)) + 0.3 * t for t in range(T)]
        knn = int(rng.integers(2, min(8, n - 1)))
        filt = build_simplicial_zigzag(clouds, knn=knn, dim=int(rng.integers(1, 4)))
        gt = (0, 1) if results.get("knn point clouds", {}).get("gt_checked", 0) < 40 and filt.n_cells < 150 else ()
        compare("knn point clouds", filt, gt, lambda s: s.dimension, lambda s: s.faces())

def smooth_grid(shape, T, sigma=1.0):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(rng.normal(size=tuple(shape) + (T,)), sigma=sigma)

def fam_cubical(n_cases, ndim, label, use_gudhi=True):
    for _ in range(n_cases):
        shape = [int(rng.integers(3, 9 if ndim == 2 else 6)) for _ in range(ndim)]
        T = int(rng.integers(2, 8 if ndim == 2 else 6))
        g = smooth_grid(shape, T, float(rng.uniform(0.5, 1.5)))
        thr = float(np.percentile(g, rng.uniform(20, 80))) * float(rng.choice([-1.0, 1.0]))
        filt = build_cubical_zigzag(g, threshold=thr, use_gudhi=use_gudhi)
        if filt.n_cells == 0: continue
        gt = (0, 1, 2) if results.get(label, {}).get("gt_checked", 0) < 40 and filt.n_cells < 200 else ()
        compare(label, filt, gt, lambda c: c.dim, lambda c: c.boundary())

def grid_mesh(nx, ny, torus=False):
    idx = lambda i, j: (i % nx) * ny + (j % ny) if torus else i * ny + j
    faces = []
    for i in range(nx if torus else nx - 1):
        for j in range(ny if torus else ny - 1):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1)
            faces += [[a, b, d], [a, d, c]]
    return np.array(faces), nx * ny

def fam_mesh(n_cases):
    reducers = ["min", "mean", "max", "product"]
    for _ in range(n_cases):
        torus = bool(rng.random() < 0.4)
        faces, V = grid_mesh(int(rng.integers(3, 7)), int(rng.integers(3, 7)), torus)
        T = int(rng.integers(2, 9))
        act = np.cumsum(rng.normal(size=(V, T)) * 0.5, axis=1) + rng.normal(size=(V, 1))
        red = reducers[int(rng.integers(0, len(reducers)))]
        thr = float(np.percentile(act, rng.uniform(20, 80)))
        filt = build_mesh_zigzag(faces, act, threshold=thr, coactivity=red)
        if filt.n_cells == 0: continue
        gt = (0, 1, 2) if results.get("mesh zigzag", {}).get("gt_checked", 0) < 40 and filt.n_cells < 200 else ()
        compare("mesh zigzag", filt, gt, lambda s: s.dimension, lambda s: s.faces())

def fam_multi_threshold(n_cases):
    r = results.setdefault("multi-threshold cubical", dict(cases=0, mismatch=0, gt_checked=0, gt_fail=0, ops=0, max_ops=0, bars=0, essential=0, t_py=0.0, t_cpp=0.0, examples=[]))
    for _ in range(n_cases):
        g = smooth_grid([int(rng.integers(4, 8)), int(rng.integers(4, 8))], int(rng.integers(3, 7)))
        m = int(rng.integers(3, 8))
        t0 = time.perf_counter(); a = run_cubical_zigzag_multi_threshold(g, thresholds=m, backend="python", n_jobs=1); t1 = time.perf_counter()
        b = run_cubical_zigzag_multi_threshold(g, thresholds=m, backend="cpp", n_jobs=1); t2 = time.perf_counter()
        same = list(a.thresholds) == list(b.thresholds) and all(sorted(a.bars[t]) == sorted(b.bars[t]) for t in a.thresholds)
        r["cases"] += 1; r["mismatch"] += 0 if same else 1; r["t_py"] += t1 - t0; r["t_cpp"] += t2 - t1
        r["bars"] += sum(len(v) for v in a.bars.values()); r["ops"] += m

def fam_zigzagdiff(n_cases, label, quantise=None, eps_touch=False):
    r = results.setdefault(label, dict(cases=0, mismatch=0, gt_checked=0, gt_fail=0, ops=0, max_ops=0, bars=0, essential=0, t_py=0.0, t_cpp=0.0, examples=[]))
    for _ in range(n_cases):
        K = random_flag_complex(int(rng.integers(4, 20)), float(rng.uniform(0.25, 0.75)), 2)
        n_v = sum(1 for c in K.all_cells() if c.dimension == 0)
        T = int(rng.integers(2, 12))
        theta = torch.tensor(rng.normal(size=(T, n_v)))
        if quantise: theta = torch.round(theta * quantise) / quantise
        eps = 0.0 if (quantise or eps_touch) else float(rng.normal(0, 0.5))
        if eps_touch:                                   # plant exact threshold touches
            mask = torch.rand(T, n_v) < 0.15; theta[mask] = eps
        V = lower_star(theta, K).numpy()
        sched = build_events(K, V, eps)
        t0 = time.perf_counter(); p = zigzag_pairs(sched, backend="python"); t1 = time.perf_counter()
        c = zigzag_pairs(sched, backend="cpp"); t2 = time.perf_counter()
        same = sorted(p) == sorted(c)
        r["cases"] += 1; r["ops"] += sched.n_events; r["max_ops"] = max(r["max_ops"], sched.n_events)
        r["bars"] += len(p); r["essential"] += sum(1 for _, d, _ in p if d >= sched.n_events)
        r["t_py"] += t1 - t0; r["t_cpp"] += t2 - t1
        if not same:
            r["mismatch"] += 1
            if len(r["examples"]) < 3: r["examples"].append(dict(n_ops=sched.n_events, only_py=sorted(set(p) - set(c))[:5], only_cpp=sorted(set(c) - set(p))[:5]))

def fam_edge_cases():
    label = "edge cases"
    S = lambda *v: Simplex(list(v))
    cases = [
        [set()],                                                  # single empty frame
        [set(), set(), set()],                                    # all empty
        [{S(0)}, {S(0)}, {S(0)}],                                 # nothing ever changes
        [{S(0), S(1), S(0, 1)}, set(), {S(0), S(1), S(0, 1)}],    # full teardown and rebuild
        [{S(0)}, {S(1)}, {S(2)}, {S(3)}],                         # disjoint frames
        [{S(0), S(1), S(2), S(0, 1), S(1, 2), S(0, 2)}, {S(0), S(1), S(2), S(0, 1), S(1, 2), S(0, 2), S(0, 1, 2)}] * 4,  # loop filled and emptied repeatedly
    ]
    for frames in cases:
        filt = ZigzagFiltration(frames, lambda s: s.dimension, lambda s: s.faces())
        compare(label, filt, (0, 1), lambda s: s.dimension, lambda s: s.faces())

# --------------------------------------------------------------------------- run
plan = [
    ("simplicial, random lower-star complexes (dim ≤ 2)", lambda: fam_simplicial_lowerstar(300, "simplicial lower-star, dim<=2")),
    ("simplicial, random lower-star complexes (dim ≤ 3)", lambda: fam_simplicial_lowerstar(80, "simplicial lower-star, dim<=3", max_dim=3)),
    ("simplicial, quantised values (many simultaneous events)", lambda: fam_simplicial_lowerstar(120, "simplicial lower-star, tied values", quantise=2)),
    ("kNN point clouds (GUDHI flag expansion)", lambda: fam_knn(120)),
    ("cubical 2-D, GUDHI extraction", lambda: fam_cubical(150, 2, "cubical 2-D")),
    ("cubical 2-D, direct extraction", lambda: fam_cubical(50, 2, "cubical 2-D (direct)", use_gudhi=False)),
    ("cubical 3-D", lambda: fam_cubical(80, 3, "cubical 3-D")),
    ("mesh zigzag (disk and torus meshes, four reducers)", lambda: fam_mesh(60)),
    ("multi-threshold cubical", lambda: fam_multi_threshold(20)),
    ("zigzagdiff event stream, generic values", lambda: fam_zigzagdiff(300, "zigzagdiff stream, generic")),
    ("zigzagdiff event stream, tied values", lambda: fam_zigzagdiff(120, "zigzagdiff stream, ties", quantise=2)),
    ("zigzagdiff event stream, threshold touches", lambda: fam_zigzagdiff(80, "zigzagdiff stream, eps touches", eps_touch=True)),
    ("edge cases", fam_edge_cases),
]
T0 = time.perf_counter()
for desc, fn in plan:
    t = time.perf_counter(); fn(); print(f"[{time.perf_counter() - T0:7.1f}s] done: {desc} ({time.perf_counter() - t:.1f}s)", flush=True)

total_cases = sum(r["cases"] for r in results.values()); total_mm = sum(r["mismatch"] for r in results.values())
total_gt = sum(r["gt_checked"] for r in results.values()); total_gtf = sum(r["gt_fail"] for r in results.values())
meta = dict(python=platform.python_version(), numpy=np.__version__, gudhi=gudhi.__version__, torch=torch.__version__,
            zztop=zztop.__version__, total_cases=total_cases, total_mismatch=total_mm, gt_checked=total_gt, gt_fail=total_gtf,
            wall_seconds=time.perf_counter() - T0)
json.dump(dict(meta=meta, families=results), open(HERE / "parity_fzz.json", "w"), indent=1, default=float)
lines = ["| family | cases | operations (total / max) | bars (essential) | mismatches | GF(2) checks (failures) | Python s | C++ s |", "|---|---|---|---|---|---|---|---|"]
for k, r in results.items():
    lines.append(f"| {k} | {r['cases']} | {r['ops']} / {r['max_ops']} | {r['bars']} ({r['essential']}) | {r['mismatch']} | {r['gt_checked']} ({r['gt_fail']}) | {r['t_py']:.1f} | {r['t_cpp']:.2f} |")
lines.append(f"\n**{total_cases} cases, {total_mm} mismatches; {total_gt} ground-truth checks, {total_gtf} failures.**  "
             f"Python {platform.python_version()}, numpy {np.__version__}, gudhi {gudhi.__version__}, wall {meta['wall_seconds']:.0f} s.")
(HERE / "parity_fzz.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

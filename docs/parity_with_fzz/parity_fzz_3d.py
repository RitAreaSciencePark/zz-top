"""Round 3: 3-D (and 4-D) cubical zigzag, the path touched by the cone-dimension fix.

Every case is run through the pure-Python engine and the fixed C++/fzz
backend of the same checkout and compared as exact multisets of raw
(birth, death, dim) triples and of layer-indexed bars.  Independent oracles on
every case where affordable: GF(2) Betti numbers of every zigzag layer in
every degree; no bar above the ambient dimension; prefix-restriction
invariance (the barcode of the first L frames equals the full barcode
restricted to bars dying before the cut) -- the property the old C++ code
violated.  Also: GUDHI vs direct cell extraction give the same barcode.
"""
import sys, time, json, pathlib, collections, platform, importlib.util
import numpy as np
from scipy.ndimage import gaussian_filter
HERE = pathlib.Path(__file__).resolve().parent
W = sys.argv[1]; sys.path.insert(0, W)
import zztop, gudhi
from zztop.zigzag import _cpp_backend
from zztop.zigzag.engine import ZigzagEngine
from zztop.builders.cubical_builder import build_cubical_zigzag, run_cubical_zigzag
spec = importlib.util.spec_from_file_location("tzdd", pathlib.Path(W) / "tests" / "test_zigzag_deletion_death.py")
tzdd = importlib.util.module_from_spec(spec); spec.loader.exec_module(tzdd)
assert _cpp_backend.AVAILABLE
PY, CPP = ZigzagEngine(backend="python"), ZigzagEngine(backend="cpp")
assert (PY.backend_name, CPP.backend_name) == ("python", "cpp")
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
DIM_FN, BDRY_FN = (lambda c: c.dim), (lambda c: c.boundary())
R = collections.OrderedDict()
def rec(name):
    return R.setdefault(name, dict(cases=0, mismatch=0, gt_checked=0, gt_fail=0, above_ambient=0, prefix_checked=0, prefix_fail=0,
                                   extraction_checked=0, extraction_fail=0, ops=0, max_ops=0, bars=0, essential=0, t_py=0.0, t_cpp=0.0, examples=[]))
T0 = time.perf_counter()
def log(m): print(f"[{time.perf_counter() - T0:7.1f}s] {m}", flush=True)

def check(name, g, thr, ambient, gt_limit=900, prefix=True, extraction=False):
    r = rec(name)
    filt = build_cubical_zigzag(g, threshold=thr)
    if filt.n_cells == 0: return
    n_ops = len(filt.to_operations())
    t0 = time.perf_counter(); raw_py = PY.run_raw(filt); t1 = time.perf_counter(); raw_cpp = CPP.run_raw(filt); t2 = time.perf_counter()
    bars_py, bars_cpp = PY.run(filt), CPP.run(filt)
    same = sorted(raw_py) == sorted(raw_cpp) and sorted(bars_py) == sorted(bars_cpp)
    r["cases"] += 1; r["ops"] += n_ops; r["max_ops"] = max(r["max_ops"], n_ops); r["bars"] += len(bars_py)
    r["essential"] += sum(1 for _, _, d in bars_py if d == filt.n_layers + 1); r["t_py"] += t1 - t0; r["t_cpp"] += t2 - t1
    if not same:
        r["mismatch"] += 1
        if len(r["examples"]) < 3: r["examples"].append(dict(shape=list(g.shape), thr=thr, only_py=sorted(set(raw_py) - set(raw_cpp))[:5], only_cpp=sorted(set(raw_cpp) - set(raw_py))[:5]))
    if max((d for d, _, _ in bars_py), default=0) > ambient or max((d for d, _, _ in bars_cpp), default=0) > ambient:
        r["above_ambient"] += 1
    if filt.n_cells <= gt_limit:
        r["gt_checked"] += 1
        for k in range(ambient + 1):
            truth = tzdd.per_layer_betti(filt, DIM_FN, BDRY_FN, k)
            if tzdd.barcode_betti(bars_py, k, filt.n_layers) != truth or tzdd.barcode_betti(bars_cpp, k, filt.n_layers) != truth:
                r["gt_fail"] += 1; break
    T = g.shape[-1]
    if prefix and T >= 4:
        L = int(rng.integers(2, T - 1)); cut = 2 * L - 1
        for eng, full in ((PY, bars_py), (CPP, bars_cpp)):
            short = eng.run(build_cubical_zigzag(g[..., :L], threshold=thr))
            r["prefix_checked"] += 1
            if collections.Counter(b for b in short if b[2] < cut) != collections.Counter(b for b in full if b[2] < cut):
                r["prefix_fail"] += 1
    if extraction:
        r["extraction_checked"] += 1
        alt = build_cubical_zigzag(g, threshold=thr, use_gudhi=False)
        if sorted(PY.run(alt)) != sorted(bars_py): r["extraction_fail"] += 1

def smooth(shape, T, sigma):
    return gaussian_filter(rng.normal(size=tuple(shape) + (T,)), sigma=sigma)

# A. random smooth 3-D fields, many shapes / thresholds / signs
for i in range(400):
    shape = [int(rng.integers(3, 9)) for _ in range(3)]; T = int(rng.integers(2, 10))
    g = smooth(shape, T, float(rng.uniform(0.4, 1.6)))
    thr = float(np.percentile(g, rng.uniform(10, 90))) * float(rng.choice([-1.0, 1.0]))
    check("3-D smooth random fields", g, thr, 3, extraction=(i % 8 == 0))
    if i % 100 == 99: log(f"A {i + 1}")
# B. binary fields: plateaus and massive simultaneity
for i in range(150):
    shape = [int(rng.integers(3, 8)) for _ in range(3)]; T = int(rng.integers(2, 9))
    g = (rng.random(tuple(shape) + (T,)) < rng.uniform(0.3, 0.7)).astype(float)
    check("3-D binary fields", g, -0.5, 3, extraction=(i % 6 == 0))
log("B done")
# C. structured cavities and tunnels: shells / rings switched on in random frame windows, several at once
for i in range(80):
    n = int(rng.integers(4, 8)); T = int(rng.integers(4, 10)); g = np.zeros((n, n, n, T))
    for _ in range(int(rng.integers(1, 4))):
        a = rng.integers(0, n - 2, size=3); b = a + rng.integers(3, n + 1 - a)  # a box
        t0_, t1_ = sorted(rng.integers(0, T + 1, size=2)); t1_ = max(t1_, t0_ + 1)
        box = np.zeros((n, n, n), bool); box[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = True
        if rng.random() < 0.6:                                     # hollow: carve the interior -> cavity
            inner = np.zeros_like(box); inner[a[0]+1:b[0]-1, a[1]+1:b[1]-1, a[2]+1:b[2]-1] = True; box &= ~inner
        else:                                                      # ring: carve a through-hole -> tunnel
            axis = int(rng.integers(0, 3)); sl = [slice(a[k]+1, b[k]-1) for k in range(3)]; sl[axis] = slice(None); box[tuple(sl)] = False
        g[..., t0_:t1_][box] = 1.0
    check("3-D shells and tunnels (hand-built)", g, -0.5, 3)
log("C done")
# D. the two seeds that broke the old C++ code, and the hollow cube of the regression tests
for shape, T, seed in [((4, 4, 4), 6, 10), ((5, 5, 4), 6, 12)]:
    r0 = np.random.default_rng(seed); g = gaussian_filter(r0.standard_normal(tuple(shape) + (T,)), sigma=1.0); g = g - g.min() + 0.01
    check("3-D regression seeds", g, -float(np.percentile(g, 30)), 3, extraction=True)
g = np.zeros((3, 3, 3, 5)); g[..., 1:4] = 1.0; g[1, 1, 1, :] = 0.0; check("3-D regression seeds", g, -0.5, 3, extraction=True)
# E. larger 3-D grids (timing; ground truth only where affordable)
for i in range(12):
    shape = [int(rng.integers(7, 11)) for _ in range(3)]; T = int(rng.integers(4, 9))
    g = smooth(shape, T, 1.0); thr = float(np.percentile(g, rng.uniform(30, 70)))
    check("3-D large fields", g, thr, 3, gt_limit=1500, prefix=(i % 3 == 0)); log(f"E {i + 1}")
# F. 4-D grids: the abstract path in one more dimension (cones of cubes are 4-cells)
for i in range(60):
    shape = [int(rng.integers(2, 5)) for _ in range(4)]; T = int(rng.integers(2, 6))
    g = smooth(shape, T, float(rng.uniform(0.5, 1.2))); thr = float(np.percentile(g, rng.uniform(20, 80)))
    check("4-D smooth random fields", g, thr, 4, gt_limit=700)
log("F done")
# G. high-level API parity (frame-indexed output of run_cubical_zigzag)
r = rec("run_cubical_zigzag (frame indices)")
for i in range(60):
    shape = [int(rng.integers(3, 7)) for _ in range(3)]; g = smooth(shape, int(rng.integers(2, 7)), 1.0); thr = float(np.percentile(g, 50))
    a = run_cubical_zigzag(g, threshold=thr, backend="python"); b = run_cubical_zigzag(g, threshold=thr, backend="cpp")
    r["cases"] += 1; r["mismatch"] += 0 if sorted(a) == sorted(b) else 1; r["bars"] += len(a)
log("G done")

tot = {k: sum(r[k] for r in R.values()) for k in ("cases", "mismatch", "gt_checked", "gt_fail", "above_ambient", "prefix_checked", "prefix_fail", "extraction_checked", "extraction_fail")}
meta = dict(python=platform.python_version(), numpy=np.__version__, gudhi=gudhi.__version__, zztop=zztop.__version__, host=platform.node(), wall_s=time.perf_counter() - T0, **tot)
json.dump(dict(meta=meta, families=R), open(HERE / "parity_fzz_3d.json", "w"), indent=1, default=float)
lines = ["| family | cases | operations (total / max) | bars (essential) | mismatches | GF(2) checks (failures) | bars above ambient dim | prefix checks (failures) | extraction checks (failures) | Python s | C++ s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for k, r in R.items():
    lines.append(f"| {k} | {r['cases']} | {r['ops']} / {r['max_ops']} | {r['bars']} ({r['essential']}) | {r['mismatch']} | {r['gt_checked']} ({r['gt_fail']}) | {r['above_ambient']} | {r['prefix_checked']} ({r['prefix_fail']}) | {r['extraction_checked']} ({r['extraction_fail']}) | {r['t_py']:.1f} | {r['t_cpp']:.2f} |")
lines.append(f"\n**Round 3: {tot['cases']} cases, {tot['mismatch']} mismatches; GF(2) {tot['gt_checked']} checks / {tot['gt_fail']} failures; "
             f"bars above the ambient dimension {tot['above_ambient']}; prefix restriction {tot['prefix_checked']} / {tot['prefix_fail']}; extraction {tot['extraction_checked']} / {tot['extraction_fail']}.**  "
             f"{platform.node()}, wall {meta['wall_s']:.0f} s.")
(HERE / "parity_fzz_3d.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

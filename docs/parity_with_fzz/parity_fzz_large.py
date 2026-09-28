"""Round 2 of the parity campaign: few, large instances (timing and scaling)."""
import sys, time, json, pathlib, collections
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
src = (HERE / "parity_fzz.py").read_text().split("# --------------------------------------------------------------------------- run")[0]
ns = {"__file__": str(HERE / "parity_fzz.py"), "__name__": "parity_round2"}
exec(compile(src, "parity_fzz.py", "exec"), ns)
rng = ns["rng"] = np.random.default_rng(777); ns["results"].clear(); results = ns["results"]
compare, ZigzagFiltration = ns["compare"], ns["ZigzagFiltration"]
random_flag_complex, lower_star_frames, smooth_grid, grid_mesh = ns["random_flag_complex"], ns["lower_star_frames"], ns["smooth_grid"], ns["grid_mesh"]
build_simplicial_zigzag, build_cubical_zigzag, build_mesh_zigzag = ns["build_simplicial_zigzag"], ns["build_cubical_zigzag"], ns["build_mesh_zigzag"]
build_events, zigzag_pairs, lower_star, torch = ns["build_events"], ns["zigzag_pairs"], ns["lower_star"], ns["torch"]
T0 = time.perf_counter()
def log(msg): print(f"[{time.perf_counter() - T0:7.1f}s] {msg}", flush=True)

for i in range(6):                                                    # kNN, large
    T, n = 10 + 2 * (i % 3), 60 + 10 * (i // 3)
    clouds = [rng.normal(size=(n, 3)) + 0.25 * t for t in range(T)]
    filt = build_simplicial_zigzag(clouds, knn=7, dim=2)
    compare("LARGE knn point clouds", filt); log(f"knn n={n} T={T} ops={len(filt.to_operations())}")
for i in range(5):                                                    # cubical 3-D, large
    g = smooth_grid([7, 7, 7], 8, 1.0); thr = float(np.percentile(g, 50)) * (-1.0 if i % 2 else 1.0)
    filt = build_cubical_zigzag(g, threshold=thr); compare("LARGE cubical 3-D", filt); log(f"cubical3d ops={len(filt.to_operations())}")
for i in range(5):                                                    # cubical 2-D, large
    g = smooth_grid([30, 30], 10, 1.5); thr = float(np.percentile(g, 40 + 5 * i))
    filt = build_cubical_zigzag(g, threshold=thr); compare("LARGE cubical 2-D", filt); log(f"cubical2d ops={len(filt.to_operations())}")
for i in range(4):                                                    # torus mesh, large
    faces, V = grid_mesh(20, 20, torus=True); T = 12
    act = np.cumsum(rng.normal(size=(V, T)) * 0.5, axis=1) + rng.normal(size=(V, 1))
    filt = build_mesh_zigzag(faces, act, threshold=float(np.percentile(act, 50)), coactivity=["min", "mean", "max", "product"][i])
    compare("LARGE torus mesh", filt); log(f"mesh ops={len(filt.to_operations())}")
for i in range(4):                                                    # dense flag complexes, dim<=3
    K = random_flag_complex(28, 0.45, 3); frames = lower_star_frames(K, 12, 0.0)
    filt = ZigzagFiltration(frames, lambda s: s.dimension, lambda s: s.faces())
    compare("LARGE simplicial lower-star, dim<=3", filt); log(f"flag ops={len(filt.to_operations())} cells={filt.n_cells}")
r = results.setdefault("LARGE zigzagdiff stream", dict(cases=0, mismatch=0, gt_checked=0, gt_fail=0, ops=0, max_ops=0, bars=0, essential=0, t_py=0.0, t_cpp=0.0, examples=[]))
for i in range(5):
    K = random_flag_complex(40, 0.35, 2); n_v = 40; T = 16
    V = lower_star(torch.tensor(rng.normal(size=(T, n_v))), K).numpy(); sched = build_events(K, V, 0.0)
    t0 = time.perf_counter(); p = zigzag_pairs(sched, backend="python"); t1 = time.perf_counter(); c = zigzag_pairs(sched, backend="cpp"); t2 = time.perf_counter()
    r["cases"] += 1; r["mismatch"] += 0 if sorted(p) == sorted(c) else 1; r["ops"] += sched.n_events; r["max_ops"] = max(r["max_ops"], sched.n_events)
    r["bars"] += len(p); r["t_py"] += t1 - t0; r["t_cpp"] += t2 - t1; log(f"zigzagdiff events={sched.n_events}")
lines = ["| family | cases | operations (total / max) | bars (essential) | mismatches | GF(2) checks | Python s | C++ s | ratio |", "|---|---|---|---|---|---|---|---|---|"]
for k, r in results.items():
    lines.append(f"| {k} | {r['cases']} | {r['ops']} / {r['max_ops']} | {r['bars']} ({r['essential']}) | {r['mismatch']} | {r['gt_checked']} | {r['t_py']:.1f} | {r['t_cpp']:.2f} | {r['t_py'] / max(r['t_cpp'], 1e-9):.0f}x |")
tot = sum(r["cases"] for r in results.values()); mm = sum(r["mismatch"] for r in results.values())
lines.append(f"\n**Round 2: {tot} large cases, {mm} mismatches; wall {time.perf_counter() - T0:.0f} s.**")
(HERE / "parity_fzz_large.md").write_text("\n".join(lines) + "\n"); json.dump(results, open(HERE / "parity_fzz_large.json", "w"), indent=1, default=float)
print("\n".join(lines))

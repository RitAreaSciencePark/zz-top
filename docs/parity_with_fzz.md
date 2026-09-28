# Parity of the zztop engine with the reference implementation `fzz`

Releases of `zztop` up to 0.7.x shipped two zigzag engines: a pure-Python
implementation of the Dey–Hou fast zigzag algorithm (ESA 2022), written from
the paper, and a C++ extension built on the authors' reference implementation
[`fzz`](https://github.com/taohou01/fzz) (CGTDA group, Purdue University) with
an added abstract-cell interface.  The `fzz` README restricts that code to
academic research use and asks that it not be redistributed, so release 0.8.0
removed the C++ extension and the PHAT headers it depended on; the Python
engine is the only one.

This page records the campaign that was run, immediately before the removal,
to establish that the two engines produce the same barcodes.  It is kept as
evidence and as the specification of what "same" means; the numbers cannot be
regenerated from this repository, since the C++ code is gone, but the
ground-truth checks it used live on in the test suite.

## Protocol

Every case builds one zigzag filtration, runs **both engines on the same
operation stream** (the simplex-wise stream `(op, vertex_list)` for
simplicial cells, the abstract stream `(op, dim, boundary_indices)` for
everything else) and compares

1. the raw `(birth, death, dimension)` triples, 1-based closed operation
   intervals, as exact multisets;
2. the layer-indexed barcode returned by `ZigzagEngine.run`, as exact
   multisets.

Independent oracles, applied to a subset of every family where affordable:

* **GF(2) Betti numbers** of every zigzag layer in every degree, computed by
  Gaussian elimination on the boundary matrices of that layer's cells, must
  equal the number of bars alive at that layer (`tests/test_zigzag_deletion_death.py`
  helpers);
* **no bar above the ambient dimension** of a cubical grid;
* **prefix restriction**: the barcode of the first `L` frames must equal the
  full barcode restricted to bars dying before the cut (the property the C++
  code violated before the cone-dimension fix of 0.8.0);
* **extraction independence** (cubical): GUDHI-based and direct cell
  extraction give the same barcode.

Families, all with fresh random instances from fixed seeds:

| family | what varies |
|---|---|
| simplicial lower-star complexes | random flag complexes on 4–22 vertices up to dimension 2 or 3, per-vertex random values, 2–12 frames, random thresholds; a variant with values quantised to a few levels (many simultaneous events) |
| kNN point clouds | 8–36 points in 2–3 dimensions, 2–9 frames, `knn` 2–8, flag expansion to dimension 1–3 (`build_simplicial_zigzag`) |
| cubical 2-D | smooth random fields up to 9×9, 2–8 frames, thresholds at random percentiles of either sign; GUDHI and direct extraction |
| cubical 3-D | smooth random fields up to 6³ (round 1) and 10³ (round 2), 2–9 frames; plus the dedicated round 3 below |
| mesh zigzag | triangulated grids (disk) and periodic grids (torus), 2–9 frames, reducers `min`, `mean`, `max`, `product` (`build_mesh_zigzag`) |
| multi-threshold cubical | 3–8 thresholds per grid (`run_cubical_zigzag_multi_threshold`) |
| zigzagdiff event stream | random flag complexes with lower-star values through `build_events`; generic values, quantised values (ties), and planted exact threshold touches |
| edge cases | empty frames, constant frames, full teardown and rebuild, disjoint frames, a loop filled and emptied repeatedly |

The dedicated 3-D round (round 3) targets the path touched by the
cone-dimension fix: smooth random 3-D fields with random shapes, signs and
thresholds; binary fields (plateaus, massive simultaneity); hand-built shells
(cavities) and rings (tunnels) switched on in random frame windows, several at
once; the two random seeds and the hollow cube that broke the old C++ code;
larger grids up to 10³ × 8 frames; **4-D grids**, where the cone of a cube is
a 4-cell; and the frame-indexed output of `run_cubical_zigzag`.

All runs used Python 3.12, numpy 2.4, GUDHI 3.12, on one core of an AMD EPYC
node (Slurm, `--cpus-per-task=1`), with the C++ engine compiled from the last
commit that contained it (`fix/cubical-cone-dimension`, after the
cone-dimension fix).

## Round 1: many small instances

| family | cases | operations (total / max) | bars (essential) | mismatches | GF(2) checks (failures) | Python s | C++ s |
|---|---|---|---|---|---|---|---|
| simplicial lower-star, dim<=2 | 300 | 68458 / 2182 | 34229 (5492) | 0 | 60 (0) | 0.7 | 0.23 |
| simplicial lower-star, dim<=3 | 80 | 16598 / 1970 | 8299 (1112) | 0 | 60 (0) | 0.1 | 0.05 |
| simplicial lower-star, tied values | 120 | 22518 / 1502 | 11259 (1732) | 0 | 60 (0) | 0.3 | 0.08 |
| knn point clouds | 120 | 149078 / 5946 | 74539 (12743) | 0 | 35 (0) | 4.7 | 0.96 |
| cubical 2-D | 150 | 22504 / 472 | 11252 (5211) | 0 | 40 (0) | 0.2 | 0.14 |
| cubical 2-D (direct) | 50 | 6926 / 396 | 3463 (1736) | 0 | 40 (0) | 0.1 | 0.04 |
| cubical 3-D | 80 | 35094 / 1404 | 17547 (7841) | 0 | 40 (0) | 0.4 | 0.24 |
| mesh zigzag | 60 | 8428 / 418 | 4214 (1538) | 0 | 40 (0) | 0.1 | 0.04 |
| multi-threshold cubical | 20 | 98 / 0 | 7754 (0) | 0 | 0 (0) | 0.3 | 0.23 |
| zigzagdiff stream, generic | 300 | 52034 / 1762 | 26806 (1578) | 0 | 0 (0) | 0.3 | 0.07 |
| zigzagdiff stream, ties | 120 | 14497 / 1028 | 7379 (261) | 0 | 0 (0) | 0.1 | 0.02 |
| zigzagdiff stream, eps touches | 80 | 12092 / 1278 | 6171 (250) | 0 | 0 (0) | 0.0 | 0.01 |
| edge cases | 6 | 42 / 20 | 21 (8) | 0 | 6 (0) | 0.0 | 0.00 |

**1486 cases, 0 mismatches; 381 ground-truth checks, 0 failures.**  Python 3.12.12, numpy 2.4.2, gudhi 3.12.0a1, wall 25 s.

## Round 2: few large instances (scaling and timing)

| family | cases | operations (total / max) | bars (essential) | mismatches | GF(2) checks | Python s | C++ s | ratio |
|---|---|---|---|---|---|---|---|---|
| LARGE knn point clouds | 6 | 101714 / 21132 | 50857 (3039) | 0 | 0 | 8.4 | 0.88 | 10x |
| LARGE cubical 3-D | 5 | 22388 / 5222 | 11194 (2959) | 0 | 0 | 0.7 | 0.26 | 3x |
| LARGE cubical 2-D | 5 | 33070 / 7662 | 16535 (6089) | 0 | 0 | 0.6 | 0.36 | 2x |
| LARGE torus mesh | 4 | 20374 / 6582 | 10187 (2477) | 0 | 0 | 0.6 | 0.30 | 2x |
| LARGE simplicial lower-star, dim<=3 | 4 | 8046 / 2398 | 4023 (265) | 0 | 0 | 0.2 | 0.03 | 7x |
| LARGE zigzagdiff stream | 5 | 17128 / 4076 | 8615 (0) | 0 | 0 | 0.3 | 0.03 | 11x |

**Round 2: 29 large cases, 0 mismatches; wall 31 s.**

The Python engine is slower than the C++ one by the ratio in the last column;
the reduction is a plain left-to-right column reduction over Python sets.
For the sizes above it stays within seconds per barcode.

## Round 3: 3-D and 4-D cubical grids

| family | cases | operations (total / max) | bars (essential) | mismatches | GF(2) checks (failures) | bars above ambient dim | prefix checks (failures) | extraction checks (failures) | Python s | C++ s |
|---|---|---|---|---|---|---|---|---|---|---|
| 3-D smooth random fields | 400 | 625894 / 10300 | 312947 (108293) | 0 | 305 (0) | 0 | 616 (0) | 50 (0) | 11.7 | 5.68 |
| 3-D binary fields | 150 | 145446 / 3100 | 72723 (21908) | 0 | 140 (0) | 0 | 222 (0) | 25 (0) | 1.8 | 0.96 |
| 3-D shells and tunnels (hand-built) | 80 | 68508 / 1720 | 34254 (17429) | 0 | 80 (0) | 0 | 160 (0) | 0 (0) | 0.8 | 0.48 |
| 3-D regression seeds | 3 | 2072 / 1096 | 1036 (327) | 0 | 3 (0) | 0 | 6 (0) | 3 (0) | 0.0 | 0.01 |
| 3-D large fields | 12 | 89700 / 10158 | 44850 (13812) | 0 | 0 (0) | 0 | 8 (0) | 0 (0) | 2.5 | 0.89 |
| 4-D smooth random fields | 60 | 35604 / 2132 | 17802 (6884) | 0 | 56 (0) | 0 | 40 (0) | 0 (0) | 0.5 | 0.30 |
| run_cubical_zigzag (frame indices) | 60 | 0 / 0 | 19327 (0) | 0 | 0 (0) | 0 | 0 (0) | 0 (0) | 0.0 | 0.00 |

**Round 3: 765 cases, 0 mismatches; GF(2) 584 checks / 0 failures; bars above the ambient dimension 0; prefix restriction 1052 / 0; extraction 78 / 0.**  epyc001, wall 129 s.

## Conclusion


Across the three rounds, **2280 zigzag filtrations** (1486 small, 29 large, 765 three- and four-dimensional cubical) gave **identical barcodes** from the Python engine and from `fzz`, as exact multisets of raw operation-index triples and of layer-indexed bars.  Independently of `fzz`, the Python engine matched GF(2) Betti numbers at every layer in every degree on 965 of those filtrations, produced no bar above the ambient dimension in any 3-D or 4-D grid, satisfied prefix restriction in all 1052 checks, and was insensitive to the cubical cell-extraction route in all 78 checks.  The Python engine is between 2× and 11× slower than the C++ one on the large instances of round 2, which is the cost of the removal.

## Files

The scripts that ran the campaign, their raw JSON output and the Slurm
submission files are in `docs/parity_with_fzz/`.  They need a checkout that
still contains the C++ backend (any release up to 0.7.x with the
cone-dimension fix applied, i.e. commit `b3ee8d4`) and a copy of `fzz`
obtained from its authors under their terms; they will not run against 0.8.0.

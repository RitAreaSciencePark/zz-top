# Third-party components

`zztop` is distributed under the BSD-3-Clause licence (see `LICENSE`).  It
vendors no third-party source code.

## Algorithm and reference implementation

The zigzag engine (`zztop/zigzag/_python_backend.py`) is an independent
implementation, written from the paper, of

> T. K. Dey and T. Hou, *Fast Computation of Zigzag Persistence*, ESA 2022.

The authors' reference implementation is `fzz`
(https://github.com/taohou01/fzz), copyright of the CGTDA research group at
Purdue University, whose README restricts it to academic research use and asks
that it not be redistributed.  No code from `fzz`, and none of the PHAT headers
it depends on, is included in this package.  Releases before 0.8.0 shipped a
modified copy of `fzz` as an optional C++ backend; it was removed, and
`docs/parity_with_fzz.md` records the campaign that verified the Python
engine against it bar for bar.  `zztop` keeps the same operation-stream
conventions as `fzz` (1-based closed intervals of operation indices, `(birth,
death, dimension)` triples), so results are directly comparable with it.

## Dependencies (not vendored)

| component | use | licence |
|---|---|---|
| numpy, scipy, scikit-learn | runtime | BSD-3-Clause |
| GUDHI | `CubicalComplex`, `SimplexTree`, `gudhi.sklearn.cubical_persistence` (cell extraction, flag expansion, static persistence); an optional CUDA-enabled fork for the GPU path | MIT for the modules used (GUDHI's CGAL-dependent modules are GPL v3 and are not used) |
| PyTorch | optional: `zztop.zigzagdiff` and `vectorizations._torch` | BSD-3-Clause |
| matplotlib | optional: plotting | PSF-based licence |
| Dionysus 2 | optional: parity tests only | BSD-3-Clause |
| ZigZagLLMs (RitAreaSciencePark) | the read-out convention of the kNN pipeline was ported from its `run_fast_zigzag.py`; no file was copied | MIT |

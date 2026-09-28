"""
Zigzag persistence engine — the main entry point for running zigzag
persistence on a :class:`~zztop.zigzag.filtration.ZigzagFiltration`.

The engine is a pure-Python implementation of the Dey-Hou fast zigzag
algorithm (:mod:`zztop.zigzag._python_backend`).  Releases before 0.8.0 also
shipped a C++ backend built on the authors' ``fzz`` code; it was removed for
licensing reasons after a parity campaign showed identical output
(``docs/parity_with_fzz.md``).  The ``backend`` argument is kept so that
existing calls keep working.
"""

from __future__ import annotations

import csv
from typing import List, Optional, Tuple

from zztop.complex.simplicial import Simplex
from zztop.zigzag.filtration import ZigzagFiltration
from zztop.zigzag import _python_backend


class ZigzagEngine:
    """Compute zigzag persistent homology.

    Parameters
    ----------
    backend : str, optional
        ``'auto'`` (default) or ``'python'``; both select the pure-Python
        engine.  ``'cpp'`` is no longer available and raises ``ValueError``.

    Examples
    --------
    >>> engine = ZigzagEngine()
    >>> bars = engine.run(filtration)   # filtration is a ZigzagFiltration
    """

    def __init__(self, backend: str = "auto") -> None:
        if backend == "cpp":
            raise ValueError(
                "the C++ backend was removed in zztop 0.8.0 (licensing); "
                "use backend='python' or 'auto'"
            )
        if backend not in ("auto", "python"):
            raise ValueError(f"Unknown backend {backend!r}")
        self._backend = backend

    @property
    def backend_name(self) -> str:
        """Name of the engine in use; always ``"python"``."""
        return "python"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        filtration: ZigzagFiltration,
        *,
        output_file: Optional[str] = None,
    ) -> List[Tuple[int, int, int]]:
        """Compute the zigzag persistence barcode.

        Parameters
        ----------
        filtration : ZigzagFiltration
            The zigzag filtration to process.
        output_file : str, optional
            If given, write the barcode as CSV to this path.

        Returns
        -------
        bars : list of (dimension, birth, death)
            The persistence barcode.  Birth/death use the **zigzag-step**
            numbering from the filtration (1-based):
            1 = K₀, 2 = K₀∩K₁, 3 = K₁, …, 2N-1 = K_{N-1}.
            Intervals are **half-open** in layer space: ``birth`` is the layer
            at which the class first appears and ``death`` is the layer at which
            it is destroyed (the layer of the killing operation), so the class
            is alive in layers ``birth … death - 1``.  Essential classes report
            the sentinel ``death`` = ``n_layers + 1`` (one past the last layer),
            so they are never confused with a finite death on the last layer.
        """
        raw_bars = self._dispatch(filtration)

        # Convert (birth, death, dim) → (dim, birth, death) for output
        bars = [(dim, b, d) for b, d, dim in raw_bars]

        # Map operation indices → 1-based layer indices.
        # The raw backend returns indices into the flat operation list;
        # this converts them to the zigzag-layer numbering (1-based):
        #   1 = K₀, 2 = K₀∩K₁, 3 = K₁, 4 = K₁∩K₂, …, 2N-1 = K_{N-1}
        layer_map = filtration.operation_to_layer_map()
        n_ops = len(layer_map)
        n_layers = filtration.n_layers
        mapped: List[Tuple[int, int, int]] = []
        for dim, b, d in bars:
            # Birth: fzz returns ``b`` = the first operation at which the class
            # is alive (1-based), so the birth layer is layer(b) = layer_map[b-1].
            b_layer = layer_map[b - 1] if 1 <= b <= n_ops else b
            # Death: fzz returns closed op-intervals, so ``d`` is the *last*
            # operation at which the class is still alive.  The class dies at the
            # next (killing) operation d+1, hence the half-open death layer is
            # layer(d+1) = layer_map[d].  Using layer_map[d-1] here (the
            # last-alive op's layer) collapses deletion-deaths whose complex is
            # static up to the killing deletion, so filter_bars silently drops
            # real loops.  This matches the upstream reference readout
            # (ZigZagLLMs run_fast_zigzag.py: ``layers_res[i[1]]``).
            if 1 <= d < n_ops:
                d_layer = layer_map[d]          # layer of killing op (d+1)
            elif d >= n_ops:
                # Essential class: fzz clamps its death to the final op, so it
                # never dies.  Report the half-open sentinel n_layers + 1 ("one
                # past the last layer") so it is not confused with a finite
                # death that lands exactly on the last layer.
                d_layer = n_layers + 1
            else:
                d_layer = d
            mapped.append((dim, b_layer, d_layer))
        bars = mapped

        if output_file is not None:
            self._write_csv(bars, output_file)

        return bars

    def run_raw(
        self,
        filtration: ZigzagFiltration,
    ) -> List[Tuple[int, int, int]]:
        """Return raw ``(birth, death, dimension)`` tuples (fzz convention).

        Same as :meth:`run` but keeps the ``(b, d, dim)`` ordering used
        internally by the Dey-Hou algorithm and the original pyfzz.
        """
        return self._dispatch(filtration)

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self, filtration: ZigzagFiltration
    ) -> List[Tuple[int, int, int]]:
        """Run the engine on the filtration's operation stream.

        Simplicial filtrations go through the simplex-wise path (operations
        are ``(op, vertex_list)``); everything else through the abstract-cell
        path (``(op, dim, boundary_indices)``).
        """
        sample_cell = filtration.cell(0) if filtration.n_cells > 0 else None
        if isinstance(sample_cell, Simplex):
            ops = filtration.to_simplicial_operations()
            return _python_backend.compute_simplicial_python(ops)
        abs_ops, _ = filtration.to_abstract_operations()
        return _python_backend.compute_zigzag_python(abs_ops)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def filter_bars(
        bars: List[Tuple[int, int, int]],
        layer_indices: Optional[List[int]] = None,
    ) -> List[Tuple[int, int, int]]:
        """Filter out bars that are born and die in the same layer.

        Since :meth:`run` now returns layer-indexed bars directly, this
        simply removes zero-length bars (``birth == death``).  The
        *layer_indices* parameter is accepted for backward compatibility
        but ignored when *None*.

        Parameters
        ----------
        bars : list of (dim, birth, death)
            Barcode (output of :meth:`run`).
        layer_indices : list of int, optional
            **Deprecated.**  Previously needed to map operation indices
            to layers; now ignored because ``run()`` already returns
            layer indices.

        Returns
        -------
        filtered : list of (dim, birth, death)
        """
        return [
            (dim, b, d) for dim, b, d in bars if b != d
        ]

    @staticmethod
    def _write_csv(
        bars: List[Tuple[int, int, int]], path: str
    ) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["dimension", "birth", "death"])
            for row in bars:
                writer.writerow(row)

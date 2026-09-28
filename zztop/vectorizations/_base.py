"""
Base classes for persistence diagram vectorizers.

All vectorizers in :mod:`zztop.vectorizations` inherit from
:class:`BaseVectorizer`, which enforces a scikit-learn-compatible
``fit / transform`` interface and provides common plumbing (dimension
selection, normalisation, optional torch conversion).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from zztop.vectorizations._diagram import (
    DiagramDict,
    DiagramInput,
    normalize_diagram,
    normalize_diagrams,
)

__all__ = ["BaseVectorizer"]


class BaseVectorizer(BaseEstimator, TransformerMixin):
    """Abstract base class for persistence diagram vectorizers.

    Sub-classes must implement:

    * ``_fit_single(diagram)`` — learn any parameters from one diagram.
    * ``_transform_single(diagram)`` — vectorize one diagram → 1-D ndarray.

    Parameters
    ----------
    dimensions : int, list of int, or None
        Homology dimensions to include.  ``None`` → use every dimension
        present in the first training diagram.
    drop_inf : bool
        Remove bars with infinite death before vectorising (default True).
    """

    def __init__(
        self,
        dimensions: Optional[Union[int, List[int]]] = None,
        drop_inf: bool = True,
    ):
        self.dimensions = dimensions
        self.drop_inf = drop_inf

    # ------------------------------------------------------------------
    #  Public sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: Union[List[DiagramInput], DiagramInput], y=None):
        """Fit the vectorizer on a collection of diagrams.

        Parameters
        ----------
        X : list of diagram inputs (any zztop output format)
            Training diagrams.
        y : ignored
        """
        diagrams = self._normalise_batch(X)
        self._resolve_dimensions(diagrams)
        for dgm in diagrams:
            self._fit_single(dgm)
        self._post_fit(diagrams)
        return self

    def transform(self, X: Union[List[DiagramInput], DiagramInput]) -> np.ndarray:
        """Transform a collection of diagrams into feature vectors.

        Parameters
        ----------
        X : list of diagram inputs

        Returns
        -------
        features : ndarray of shape ``(n_diagrams, n_features)``
        """
        diagrams = self._normalise_batch(X)
        rows = [self._transform_single(dgm) for dgm in diagrams]
        return np.vstack(rows)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y=y).transform(X)

    # ------------------------------------------------------------------
    #  Optional torch support
    # ------------------------------------------------------------------

    def to_torch(self, X: Union[List[DiagramInput], DiagramInput]):
        """Transform *X* and return a :class:`torch.Tensor`.

        Requires ``torch`` to be installed.  The tensor is placed on CPU
        with ``requires_grad=False`` by default.
        """
        arr = self.transform(X)
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Optional dependency 'torch' is required for to_torch(). "
                "Install with: pip install torch"
            ) from exc
        return torch.from_numpy(arr)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _normalise_batch(self, X) -> List[DiagramDict]:
        """Accept any zztop format and return list of canonical dicts."""
        if isinstance(X, (list, tuple)) and len(X) > 0:
            first = X[0]
            # Already a list of diagram dicts → normalise each
            if isinstance(first, dict):
                return [normalize_diagram(d, drop_inf=self.drop_inf) for d in X]
            # List of zigzag-style tuples: check if inner element is a tuple of 3
            if isinstance(first, (list, tuple)):
                # Could be a single zigzag barcode or a list of barcodes.
                # A zigzag barcode has inner tuples of length 3 with numeric types.
                inner = first
                if isinstance(inner, (list, tuple)) and len(inner) >= 2:
                    if isinstance(inner[0], (int, float, np.integer, np.floating)):
                        # Single zigzag barcode as list of tuples
                        return [normalize_diagram(X, drop_inf=self.drop_inf)]
                    else:
                        # List of zigzag barcodes
                        return [normalize_diagram(d, drop_inf=self.drop_inf) for d in X]
            # list of ndarrays
            if isinstance(first, np.ndarray):
                return [normalize_diagram(d, drop_inf=self.drop_inf) for d in X]
        # Single diagram of any kind
        return [normalize_diagram(X, drop_inf=self.drop_inf)]

    def _resolve_dimensions(self, diagrams: List[DiagramDict]):
        """Set ``self.dimensions_`` from user spec or data."""
        if self.dimensions is not None:
            if isinstance(self.dimensions, int):
                self.dimensions_ = [self.dimensions]
            else:
                self.dimensions_ = list(self.dimensions)
        else:
            # infer from the union of all dimensions in all training diagrams
            all_dims: set = set()
            for dgm in diagrams:
                all_dims.update(dgm.keys())
            self.dimensions_ = sorted(all_dims)

    def _get_bars(self, diagram: DiagramDict, dim: int) -> np.ndarray:
        """Get ``(n, 2)`` birth-death array for *dim*, or empty array."""
        if dim in diagram:
            arr = diagram[dim]
            if arr.size == 0:
                return np.empty((0, 2), dtype=np.float64)
            return arr
        return np.empty((0, 2), dtype=np.float64)

    # ------------------------------------------------------------------
    #  Template methods for sub-classes
    # ------------------------------------------------------------------

    @abstractmethod
    def _fit_single(self, diagram: DiagramDict) -> None:
        """Learn any per-diagram statistics (called once per training sample)."""
        ...

    @abstractmethod
    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        """Vectorize one diagram → 1-D ndarray."""
        ...

    def _post_fit(self, diagrams: List[DiagramDict]) -> None:
        """Hook called after all ``_fit_single`` calls.  Override as needed."""
        pass

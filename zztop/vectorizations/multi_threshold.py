"""
Multi-threshold zigzag vectorizations.
======================================

Scikit-learn–compatible transformers that operate on
:class:`~zztop.builders.multi_threshold.MultiThresholdResult` objects,
converting the two-parameter (threshold × time) barcode family into
fixed-size feature vectors.

All transformers follow the ``fit / transform`` API and can be chained
in :class:`sklearn.pipeline.Pipeline`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


__all__ = [
    "BettiSurface",
    "ThresholdIntegratedPersistence",
    "ThresholdBirthFrequency",
]


# ======================================================================
# Helpers
# ======================================================================

def _ensure_list(X):
    """Wrap a single result in a list for uniform handling."""
    # Avoid circular import — check by class name rather than isinstance
    if hasattr(X, "betti_surface") and not isinstance(X, (list, tuple)):
        return [X]
    return list(X)


# ======================================================================
# BettiSurface
# ======================================================================

class BettiSurface(BaseEstimator, TransformerMixin):
    r"""Betti surface :math:`\beta_k(a, t)` on the (threshold × time) plane.

    For each sample (a :class:`MultiThresholdResult`), produces a 2-D
    array of shape ``(n_thresholds, n_frames)`` counting live bars at
    each (threshold, frame) pair.  The array is flattened for ML use.

    Parameters
    ----------
    dim : int
        Homological dimension (default 0).
    flatten : bool
        If True (default), flatten the ``(m, T)`` surface to a 1-D
        feature vector of length ``m × T``.  Set False to keep the
        2-D surface (useful for visualisation).
    """

    def __init__(self, dim: int = 0, *, flatten: bool = True):
        self.dim = dim
        self.flatten = flatten

    def fit(self, X, y=None):
        """Fit is a no-op (stateless transformer)."""
        return self

    def transform(self, X) -> np.ndarray:
        """Transform a list of :class:`MultiThresholdResult` into features.

        Parameters
        ----------
        X : MultiThresholdResult or list thereof

        Returns
        -------
        features : ndarray
            Shape ``(n_samples, m * T)`` if *flatten* is True, else
            ``(n_samples, m, T)``.
        """
        results = _ensure_list(X)
        surfaces = [r.betti_surface(self.dim) for r in results]
        if self.flatten:
            return np.array([s.ravel() for s in surfaces])
        return np.array(surfaces)


# ======================================================================
# ThresholdIntegratedPersistence
# ======================================================================

class ThresholdIntegratedPersistence(BaseEstimator, TransformerMixin):
    """Total persistence integrated over thresholds.

    For each sample, computes the total persistence (sum of bar
    lifetimes) at each threshold level, returning a 1-D vector of
    length ``n_thresholds``.

    Parameters
    ----------
    dim : int
        Homological dimension (default 0).
    """

    def __init__(self, dim: int = 0):
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        results = _ensure_list(X)
        rows = [r.total_persistence(self.dim) for r in results]
        return np.array(rows)


# ======================================================================
# ThresholdBirthFrequency
# ======================================================================

class ThresholdBirthFrequency(BaseEstimator, TransformerMixin):
    r"""Birth frequency on the (threshold × time) plane.

    For each (threshold, frame), computes the fraction of bars born at
    that frame, producing a surface of shape ``(n_thresholds, n_frames)``.

    Parameters
    ----------
    dim : int
        Homological dimension (default 0).
    flatten : bool
        Flatten to 1-D for ML (default True).
    """

    def __init__(self, dim: int = 0, *, flatten: bool = True):
        self.dim = dim
        self.flatten = flatten

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        results = _ensure_list(X)
        surfaces = []
        for r in results:
            m = len(r.thresholds)
            T = r.n_frames
            freq = np.zeros((m, T), dtype=np.float64)
            for i, a in enumerate(r.thresholds):
                bars_at_a = r.bars[float(a)]
                n_dim = sum(1 for d, b, de in bars_at_a if d == self.dim)
                if n_dim == 0:
                    continue
                for d, b, de in bars_at_a:
                    if d == self.dim and 0 <= b < T:
                        freq[i, b] += 1.0 / n_dim
            surfaces.append(freq)
        if self.flatten:
            return np.array([s.ravel() for s in surfaces])
        return np.array(surfaces)

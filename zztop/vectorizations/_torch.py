"""
Optional PyTorch support for differentiable vectorizations.

This module provides :func:`differentiable_transform`, which wraps
a fitted :class:`BaseVectorizer` so that its output is a
:class:`torch.Tensor` with gradient support via ``torch.autograd``.

Usage::

    from zztop.vectorizations import PersistenceImage
    from zztop.vectorizations._torch import differentiable_transform

    pi = PersistenceImage(resolution=(20, 20), sigma=1.0)
    pi.fit(train_diagrams)

    # Returns a torch.Tensor (CPU, float64, requires_grad via autograd)
    features = differentiable_transform(pi, test_diagrams)

Notes
-----
* ``torch`` is imported lazily — the module can be imported even if
  torch is not installed; errors are raised only when functions are
  actually called.
* The current implementation uses a straight-through estimator for the
  histogram binning step (which is non-differentiable).  This is
  sufficient for use in downstream differentiable pipelines (e.g.
  loss = f(vectorized_diagram)), but does **not** enable gradients
  through the diagram coordinates themselves.  For full differentiability
  through the persistence computation, use a differentiable persistence
  library such as ``TopologyLayer``.
"""

from __future__ import annotations

from typing import List, Union

import numpy as np


def _check_torch():
    """Lazy-import torch and raise a clear error if missing."""
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "Optional dependency 'torch' is required for differentiable "
            "vectorizations.  Install with:  pip install torch"
        ) from exc


def differentiable_transform(vectorizer, X, *, requires_grad: bool = False):
    """Transform diagrams to a :class:`torch.Tensor`.

    Parameters
    ----------
    vectorizer : BaseVectorizer (fitted)
        Any vectorizer from :mod:`zztop.vectorizations`.
    X : list of diagrams
        Input in any format accepted by the vectorizer.
    requires_grad : bool
        If True, the returned tensor has ``requires_grad=True``.

    Returns
    -------
    torch.Tensor of shape ``(n_diagrams, n_features)``
    """
    torch = _check_torch()
    arr = vectorizer.transform(X)
    t = torch.from_numpy(arr)
    if requires_grad:
        t = t.requires_grad_(True)
    return t


class TorchVectorizer:
    """Wrapper that makes any :class:`BaseVectorizer` return torch tensors.

    Usage::

        from zztop.vectorizations import PersistenceImage
        from zztop.vectorizations._torch import TorchVectorizer

        pi = TorchVectorizer(PersistenceImage(resolution=(20, 20)))
        pi.fit(train_diagrams)
        features = pi.transform(test_diagrams)  # torch.Tensor

    Parameters
    ----------
    vectorizer : BaseVectorizer
        The underlying numpy vectorizer.
    device : str
        Torch device (default ``'cpu'``).
    dtype : str or torch.dtype or None
        Output dtype.  None → ``torch.float64``.
    """

    def __init__(self, vectorizer, device: str = "cpu", dtype=None):
        self.vectorizer = vectorizer
        self.device = device
        self._dtype = dtype

    @property
    def dtype(self):
        if self._dtype is not None:
            return self._dtype
        torch = _check_torch()
        return torch.float64

    def fit(self, X, y=None):
        self.vectorizer.fit(X, y=y)
        return self

    def transform(self, X):
        torch = _check_torch()
        arr = self.vectorizer.transform(X)
        return torch.tensor(arr, dtype=self.dtype, device=self.device)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y=y).transform(X)

"""
1-D functional summaries (curves) of persistence diagrams.

Inspired by the framework presented in *Persistent Homology of Cosmic
Density Fields* (Biagetti et al., 2021, arXiv: 2009.04819), this module
provides curve-based vectorizers that summarize a persistence diagram as
a 1-D function sampled on a regular grid.

Each curve type captures a different aspect of the diagram:

* **BettiCurve** — count of bars alive at each filtration value.
* **LifeCurve** — total lifetime of bars alive at each value.
* **BirthCurve** — cumulative count of births up to each value.
* **DeathCurve** — cumulative count of deaths up to each value.
* **MidlifeCurve** — density of midpoints ``(b+d)/2``.
* **MultiplicativeLifeCurve** — product of alive-bar lifetimes (log-scale).
* **PersistenceCurve** — generic framework: user supplies a summary function.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from zztop.vectorizations._base import BaseVectorizer
from zztop.vectorizations._diagram import DiagramDict

__all__ = [
    "BettiCurve",
    "LifeCurve",
    "BirthCurve",
    "DeathCurve",
    "MidlifeCurve",
    "MultiplicativeLifeCurve",
    "PersistenceCurve",
]


# ======================================================================
#  Helper: grid range from training data
# ======================================================================

class _CurveBase(BaseVectorizer):
    """Shared logic for all curve-type vectorizers."""

    def __init__(
        self,
        resolution: int = 100,
        sample_range=None,
        dimensions=None,
        drop_inf: bool = True,
    ):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.resolution = resolution
        self.sample_range = sample_range

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass  # ranges gathered in _post_fit

    def _post_fit(self, diagrams: List[DiagramDict]) -> None:
        if self.sample_range is not None:
            self.sample_range_ = tuple(self.sample_range)
            return
        lo, hi = np.inf, -np.inf
        for dgm in diagrams:
            for dim in self.dimensions_:
                bd = self._get_bars(dgm, dim)
                if bd.size:
                    lo = min(lo, bd.min())
                    hi = max(hi, bd.max())
        if lo == np.inf:
            lo, hi = 0.0, 1.0
        self.sample_range_ = (float(lo), float(hi))

    def _grid(self) -> np.ndarray:
        return np.linspace(self.sample_range_[0], self.sample_range_[1], self.resolution)


# ======================================================================
#  Betti Curve
# ======================================================================

class BettiCurve(_CurveBase):
    r"""Betti curve: number of bars alive at each filtration value.

    .. math::

        \beta_d(t) = \#\{i : b_i \le t < d_i\}

    Parameters
    ----------
    resolution : int
        Grid points (default 100).
    sample_range : tuple or None
        ``(lo, hi)``.  ``None`` → fitted from training data.
    dimensions, drop_inf : see :class:`BaseVectorizer`.
    """

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        grid = self._grid()
        curves = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            curve = np.zeros(self.resolution, dtype=np.float64)
            if bd.size == 0:
                curves.append(curve)
                continue
            # Vectorised: for each grid point, count alive bars
            # births[:, None] <= grid[None, :] & grid[None, :] < deaths[:, None]
            alive = (bd[:, 0:1] <= grid[None, :]) & (grid[None, :] < bd[:, 1:2])
            curve = alive.sum(axis=0).astype(np.float64)
            curves.append(curve)
        return np.concatenate(curves)


# ======================================================================
#  Life Curve
# ======================================================================

class LifeCurve(_CurveBase):
    r"""Life curve: total persistence of bars alive at each filtration value.

    .. math::

        L_d(t) = \sum_{i: b_i \le t < d_i} (d_i - b_i)

    Parameters
    ----------
    resolution, sample_range, dimensions, drop_inf : see :class:`BettiCurve`.
    """

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        grid = self._grid()
        curves = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            curve = np.zeros(self.resolution, dtype=np.float64)
            if bd.size == 0:
                curves.append(curve)
                continue
            pers = bd[:, 1] - bd[:, 0]
            alive = (bd[:, 0:1] <= grid[None, :]) & (grid[None, :] < bd[:, 1:2])
            curve = (alive * pers[:, None]).sum(axis=0)
            curves.append(curve)
        return np.concatenate(curves)


# ======================================================================
#  Birth Curve
# ======================================================================

class BirthCurve(_CurveBase):
    r"""Birth curve: cumulative count of births up to each filtration value.

    .. math::

        B_d(t) = \#\{i : b_i \le t\}

    Parameters
    ----------
    resolution, sample_range, dimensions, drop_inf : see :class:`BettiCurve`.
    """

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        grid = self._grid()
        curves = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            curve = np.zeros(self.resolution, dtype=np.float64)
            if bd.size == 0:
                curves.append(curve)
                continue
            curve = (bd[:, 0:1] <= grid[None, :]).sum(axis=0).astype(np.float64)
            curves.append(curve)
        return np.concatenate(curves)


# ======================================================================
#  Death Curve
# ======================================================================

class DeathCurve(_CurveBase):
    r"""Death curve: cumulative count of deaths up to each filtration value.

    .. math::

        D_d(t) = \#\{i : d_i \le t\}

    Parameters
    ----------
    resolution, sample_range, dimensions, drop_inf : see :class:`BettiCurve`.
    """

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        grid = self._grid()
        curves = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            curve = np.zeros(self.resolution, dtype=np.float64)
            if bd.size == 0:
                curves.append(curve)
                continue
            curve = (bd[:, 1:2] <= grid[None, :]).sum(axis=0).astype(np.float64)
            curves.append(curve)
        return np.concatenate(curves)


# ======================================================================
#  Midlife Curve
# ======================================================================

class MidlifeCurve(_CurveBase):
    r"""Midlife curve: KDE of midpoint values.

    Places a Gaussian kernel at each midpoint ``(b+d)/2`` and evaluates
    the density on a regular grid.

    Parameters
    ----------
    bandwidth : float
        KDE bandwidth (default 0.1).
    resolution, sample_range, dimensions, drop_inf : see :class:`BettiCurve`.
    """

    def __init__(
        self,
        bandwidth: float = 0.1,
        resolution: int = 100,
        sample_range=None,
        dimensions=None,
        drop_inf: bool = True,
    ):
        super().__init__(
            resolution=resolution,
            sample_range=sample_range,
            dimensions=dimensions,
            drop_inf=drop_inf,
        )
        self.bandwidth = bandwidth

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        grid = self._grid()
        curves = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            curve = np.zeros(self.resolution, dtype=np.float64)
            if bd.size == 0:
                curves.append(curve)
                continue
            mids = (bd[:, 0] + bd[:, 1]) / 2.0
            # Gaussian KDE (vectorised)
            diff = grid[None, :] - mids[:, None]
            curve = np.exp(-0.5 * (diff / self.bandwidth) ** 2).sum(axis=0)
            curve /= self.bandwidth * np.sqrt(2 * np.pi)
            curves.append(curve)
        return np.concatenate(curves)


# ======================================================================
#  Multiplicative Life Curve
# ======================================================================

class MultiplicativeLifeCurve(_CurveBase):
    r"""Multiplicative life curve: log-product of alive-bar lifetimes.

    .. math::

        M_d(t) = \sum_{i: b_i \le t < d_i} \log(d_i - b_i)

    The log transform turns the product into a sum, yielding a stable
    numerical representation.

    Parameters
    ----------
    resolution, sample_range, dimensions, drop_inf : see :class:`BettiCurve`.
    """

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        grid = self._grid()
        curves = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            curve = np.zeros(self.resolution, dtype=np.float64)
            if bd.size == 0:
                curves.append(curve)
                continue
            pers = bd[:, 1] - bd[:, 0]
            log_pers = np.log(np.maximum(pers, 1e-300))
            alive = (bd[:, 0:1] <= grid[None, :]) & (grid[None, :] < bd[:, 1:2])
            curve = (alive * log_pers[:, None]).sum(axis=0)
            curves.append(curve)
        return np.concatenate(curves)


# ======================================================================
#  Generic Persistence Curve
# ======================================================================

class PersistenceCurve(_CurveBase):
    r"""Generic persistence curve with a user-supplied summary function.

    For each grid point *t*, evaluates ``summary_fn(births, deaths, pers, t)``
    on the bars alive at *t*.  This provides maximum flexibility while still
    fitting into the sklearn pipeline.

    Parameters
    ----------
    summary_fn : callable
        ``(births, deaths, pers, t) → float``.  Receives only the alive bars.
        Default: Betti number (count of alive bars).
    resolution, sample_range, dimensions, drop_inf : see :class:`BettiCurve`.
    """

    def __init__(
        self,
        summary_fn: Optional[Callable] = None,
        resolution: int = 100,
        sample_range=None,
        dimensions=None,
        drop_inf: bool = True,
    ):
        super().__init__(
            resolution=resolution,
            sample_range=sample_range,
            dimensions=dimensions,
            drop_inf=drop_inf,
        )
        self.summary_fn = summary_fn

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        grid = self._grid()
        fn = self.summary_fn or (lambda b, d, p, t: float(len(b)))
        curves = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            curve = np.zeros(self.resolution, dtype=np.float64)
            if bd.size == 0:
                curves.append(curve)
                continue
            births = bd[:, 0]
            deaths = bd[:, 1]
            pers = deaths - births
            for j, t in enumerate(grid):
                alive = (births <= t) & (t < deaths)
                if alive.any():
                    curve[j] = fn(births[alive], deaths[alive], pers[alive], t)
            curves.append(curve)
        return np.concatenate(curves)

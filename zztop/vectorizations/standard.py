"""
Standard persistence diagram vectorizers.

These are well-established representations from the TDA literature,
each implemented as a scikit-learn–compatible transformer inheriting
from :class:`~zztop.vectorizations._base.BaseVectorizer`.

Classes
-------
PersistenceImage
    2-D weighted histogram on the birth-persistence plane,
    smoothed with a Gaussian kernel (Adams et al., 2017).
PersistenceLandscape
    Piecewise-linear envelope functions (Bubenik, 2015).
Silhouette
    Weighted power-mean of "tent" functions over bars.
PersistenceEntropy
    Shannon entropy of normalised bar lengths.
Amplitude
    Scalar summary of a diagram (Wasserstein norm, bottleneck, …).
PersistenceStatistics
    Basic statistics: count, mean/std/max of births, deaths, lifetimes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
from scipy.ndimage import gaussian_filter

from zztop.vectorizations._base import BaseVectorizer
from zztop.vectorizations._diagram import DiagramDict

__all__ = [
    "PersistenceImage",
    "PersistenceLandscape",
    "Silhouette",
    "PersistenceEntropy",
    "Amplitude",
    "PersistenceStatistics",
]


# ======================================================================
#  Persistence Image
# ======================================================================

class PersistenceImage(BaseVectorizer):
    r"""Persistence Image vectorizer (Adams et al., 2017).

    Maps a persistence diagram to a weighted 2-D histogram in the
    *(birth, persistence)* plane, smoothed by a Gaussian kernel.

    Parameters
    ----------
    resolution : tuple of int
        ``(ny, nx)`` — pixel resolution of the image.
    sigma : float
        Standard deviation of the Gaussian smoothing.
    weight : callable or None
        Weighting function ``w(birth, persistence) → float``.
        Default: ``w = persistence`` (linear ramp).
    birth_range, pers_range : tuple or None
        ``(lo, hi)`` ranges.  ``None`` → fitted from training data.
    dimensions, drop_inf : see :class:`BaseVectorizer`.
    """

    def __init__(
        self,
        resolution: tuple = (20, 20),
        sigma: float = 1.0,
        weight=None,
        birth_range=None,
        pers_range=None,
        dimensions=None,
        drop_inf: bool = True,
    ):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.resolution = resolution
        self.sigma = sigma
        self.weight = weight
        self.birth_range = birth_range
        self.pers_range = pers_range

    # -- fit -----------------------------------------------------------

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass  # statistics gathered in _post_fit

    def _post_fit(self, diagrams: List[DiagramDict]) -> None:
        # Determine birth/pers ranges from training data if not set
        all_births, all_pers = [], []
        for dgm in diagrams:
            for dim in self.dimensions_:
                bd = self._get_bars(dgm, dim)
                if bd.size == 0:
                    continue
                all_births.append(bd[:, 0])
                all_pers.append(bd[:, 1] - bd[:, 0])
        if all_births:
            all_b = np.concatenate(all_births)
            all_p = np.concatenate(all_pers)
        else:
            all_b = np.array([0.0, 1.0])
            all_p = np.array([0.0, 1.0])

        self.birth_range_ = self.birth_range or (float(all_b.min()), float(all_b.max()))
        self.pers_range_ = self.pers_range or (float(max(all_p.min(), 0)), float(all_p.max()))

        # Ensure non-degenerate ranges
        for attr in ("birth_range_", "pers_range_"):
            lo, hi = getattr(self, attr)
            if hi - lo < 1e-12:
                setattr(self, attr, (lo - 0.5, hi + 0.5))

    # -- transform -----------------------------------------------------

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        ny, nx = self.resolution
        images = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            img = np.zeros((ny, nx), dtype=np.float64)
            if bd.size == 0:
                images.append(img.ravel())
                continue
            births = bd[:, 0]
            pers = bd[:, 1] - bd[:, 0]

            b_lo, b_hi = self.birth_range_
            p_lo, p_hi = self.pers_range_

            # Weight
            if self.weight is not None:
                w = np.array([self.weight(b, p) for b, p in zip(births, pers)])
            else:
                w = pers  # default linear ramp

            # bin indices
            bx = np.clip(
                ((births - b_lo) / (b_hi - b_lo) * (nx - 1)).astype(int),
                0, nx - 1,
            )
            py = np.clip(
                ((pers - p_lo) / (p_hi - p_lo) * (ny - 1)).astype(int),
                0, ny - 1,
            )
            np.add.at(img, (py, bx), w)

            # Gaussian smoothing
            if self.sigma > 0:
                img = gaussian_filter(img, sigma=self.sigma)
            images.append(img.ravel())
        return np.concatenate(images)


# ======================================================================
#  Persistence Landscape
# ======================================================================

class PersistenceLandscape(BaseVectorizer):
    r"""Persistence Landscape (Bubenik, 2015).

    For each homology dimension, computes the first *k* landscape
    functions evaluated on a regular grid.

    Parameters
    ----------
    n_landscapes : int
        Number of landscape functions (default 5).
    resolution : int
        Number of sampling points on the grid.
    sample_range : tuple or None
        ``(lo, hi)`` grid range.  ``None`` → fitted from data.
    dimensions, drop_inf : see :class:`BaseVectorizer`.
    """

    def __init__(
        self,
        n_landscapes: int = 5,
        resolution: int = 100,
        sample_range=None,
        dimensions=None,
        drop_inf: bool = True,
    ):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.n_landscapes = n_landscapes
        self.resolution = resolution
        self.sample_range = sample_range

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass

    def _post_fit(self, diagrams: List[DiagramDict]) -> None:
        lo, hi = np.inf, -np.inf
        for dgm in diagrams:
            for dim in self.dimensions_:
                bd = self._get_bars(dgm, dim)
                if bd.size:
                    lo = min(lo, bd.min())
                    hi = max(hi, bd.max())
        if lo == np.inf:
            lo, hi = 0.0, 1.0
        self.sample_range_ = self.sample_range or (float(lo), float(hi))

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        lo, hi = self.sample_range_
        grid = np.linspace(lo, hi, self.resolution)
        all_ls = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            # tent function for each bar
            n_bars = bd.shape[0]
            tents = np.zeros((n_bars, self.resolution), dtype=np.float64)
            for i in range(n_bars):
                b, d = bd[i, 0], bd[i, 1]
                mid = (b + d) / 2.0
                tents[i] = np.maximum(
                    0,
                    np.minimum(grid - b, d - grid),
                )
            # sort descending at each grid point
            if n_bars > 0:
                tents_sorted = -np.sort(-tents, axis=0)
            else:
                tents_sorted = np.zeros((1, self.resolution), dtype=np.float64)
            # take first k
            k = min(self.n_landscapes, tents_sorted.shape[0])
            ls = np.zeros((self.n_landscapes, self.resolution), dtype=np.float64)
            ls[:k] = tents_sorted[:k]
            all_ls.append(ls.ravel())
        return np.concatenate(all_ls)


# ======================================================================
#  Silhouette
# ======================================================================

class Silhouette(BaseVectorizer):
    r"""Silhouette (Chazal et al., 2014).

    Weighted power-mean of tent functions arising from bars, sampled on
    a regular grid.

    Parameters
    ----------
    power : float
        Weight exponent ``p`` (default 1.0 = equal weight).
    resolution : int
        Grid resolution.
    sample_range : tuple or None
        ``None`` → fitted from training data.
    dimensions, drop_inf : see :class:`BaseVectorizer`.
    """

    def __init__(
        self,
        power: float = 1.0,
        resolution: int = 100,
        sample_range=None,
        dimensions=None,
        drop_inf: bool = True,
    ):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.power = power
        self.resolution = resolution
        self.sample_range = sample_range

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass

    def _post_fit(self, diagrams: List[DiagramDict]) -> None:
        lo, hi = np.inf, -np.inf
        for dgm in diagrams:
            for dim in self.dimensions_:
                bd = self._get_bars(dgm, dim)
                if bd.size:
                    lo = min(lo, bd.min())
                    hi = max(hi, bd.max())
        if lo == np.inf:
            lo, hi = 0.0, 1.0
        self.sample_range_ = self.sample_range or (float(lo), float(hi))

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        lo, hi = self.sample_range_
        grid = np.linspace(lo, hi, self.resolution)
        vectors = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            sil = np.zeros(self.resolution, dtype=np.float64)
            if bd.size == 0:
                vectors.append(sil)
                continue
            pers = bd[:, 1] - bd[:, 0]
            weights = pers ** self.power
            w_sum = weights.sum()
            for i in range(bd.shape[0]):
                b, d = bd[i, 0], bd[i, 1]
                tent = np.maximum(0, np.minimum(grid - b, d - grid))
                sil += weights[i] * tent
            if w_sum > 0:
                sil /= w_sum
            vectors.append(sil)
        return np.concatenate(vectors)


# ======================================================================
#  Persistence Entropy
# ======================================================================

class PersistenceEntropy(BaseVectorizer):
    r"""Shannon entropy of normalised bar lengths.

    For each dimension, returns a single scalar:

    .. math::

        H_d = -\sum_i p_i \log p_i, \qquad
        p_i = \frac{\ell_i}{\sum_j \ell_j}

    where :math:`\ell_i = d_i - b_i` is the lifetime of bar *i*.

    Parameters
    ----------
    normalize : bool
        If True, divide by ``log(n_bars)`` to get a value in ``[0, 1]``.
    dimensions, drop_inf : see :class:`BaseVectorizer`.
    """

    def __init__(
        self,
        normalize: bool = False,
        dimensions=None,
        drop_inf: bool = True,
    ):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.normalize = normalize

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass  # stateless

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        entropies = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            if bd.size == 0:
                entropies.append(0.0)
                continue
            pers = bd[:, 1] - bd[:, 0]
            pers = pers[pers > 0]
            if pers.size == 0:
                entropies.append(0.0)
                continue
            total = pers.sum()
            p = pers / total
            ent = -np.sum(p * np.log(p))
            if self.normalize and pers.size > 1:
                ent /= np.log(pers.size)
            entropies.append(float(ent))
        return np.array(entropies, dtype=np.float64)


# ======================================================================
#  Amplitude
# ======================================================================

class Amplitude(BaseVectorizer):
    r"""Scalar amplitude summary of an entire diagram.

    Computes a single number per dimension, measuring the "size" of the
    diagram according to the chosen metric.

    Parameters
    ----------
    metric : str
        ``'wasserstein'`` — p-Wasserstein distance to the empty diagram
        (= sum of ``pers^p``).
        ``'bottleneck'`` — maximum persistence.
        ``'landscape'`` — L-inf of the first landscape function.
    order : float
        Exponent *p* for the Wasserstein metric (default 2.0).
    dimensions, drop_inf : see :class:`BaseVectorizer`.
    """

    def __init__(
        self,
        metric: str = "wasserstein",
        order: float = 2.0,
        dimensions=None,
        drop_inf: bool = True,
    ):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.metric = metric
        self.order = order

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        amps = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            if bd.size == 0:
                amps.append(0.0)
                continue
            pers = bd[:, 1] - bd[:, 0]
            if self.metric == "wasserstein":
                amps.append(float(np.sum(np.abs(pers) ** self.order) ** (1.0 / self.order)))
            elif self.metric == "bottleneck":
                amps.append(float(np.max(np.abs(pers))))
            elif self.metric == "landscape":
                amps.append(float(np.max(np.abs(pers)) / 2.0))
            else:
                raise ValueError(f"Unknown metric {self.metric!r}")
        return np.array(amps, dtype=np.float64)


# ======================================================================
#  Persistence Statistics
# ======================================================================

class PersistenceStatistics(BaseVectorizer):
    r"""Basic statistics of birth, death, and persistence values.

    Output vector (per dimension): ``[n_bars, mean_b, std_b, mean_d,
    std_d, mean_p, std_p, max_p, entropy]`` — 9 features per dimension.

    Parameters
    ----------
    dimensions, drop_inf : see :class:`BaseVectorizer`.
    """

    N_FEATURES_PER_DIM = 9

    def __init__(self, dimensions=None, drop_inf: bool = True):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        stats_list = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            if bd.size == 0:
                stats_list.append(np.zeros(self.N_FEATURES_PER_DIM, dtype=np.float64))
                continue
            births = bd[:, 0]
            deaths = bd[:, 1]
            pers = deaths - births
            n = float(bd.shape[0])

            # entropy
            pers_pos = pers[pers > 0]
            if pers_pos.size > 0:
                total = pers_pos.sum()
                p = pers_pos / total
                ent = float(-np.sum(p * np.log(p)))
            else:
                ent = 0.0

            stats_list.append(np.array([
                n,
                births.mean(), births.std(),
                deaths.mean(), deaths.std(),
                pers.mean(), pers.std(),
                pers.max(),
                ent,
            ], dtype=np.float64))
        return np.concatenate(stats_list)

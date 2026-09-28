"""
Zigzag-specific persistence descriptors.

These vectorizers exploit the **multi-layer / temporal** structure inherent
in zigzag persistence, where bars track features across consecutive frames.
They are inspired by the descriptors in *Topological Analysis for Detecting
Anomalies in LLMs* (Gardinazzi et al., 2024, arXiv: 2410.11042), simplified
for practical use as 1-D profiles rather than 2-D inter-layer matrices.

Core zigzag descriptors
-----------------------
BettiProfile
    Betti number at each frame (layer).
BirthFrequency
    Fraction of features born at each frame.
PersistenceProfile
    Mean persistence of bars alive at each frame.
TurnoverRate
    Fraction of features that change (births + deaths) at each frame.
EffectivePersistenceImage
    Persistence image weighted by birth-relative frequency.
CumulativePersistence
    Cumulative sum of persistences at each frame.

Experimental descriptors
------------------------
LayerTransitionEntropy
    Shannon entropy of birth/death events per frame.
PersistenceFlux
    Net change in total persistence between consecutive frames.
CrossDimCorrelation
    Pearson correlation of Betti numbers across dimensions.
LifetimeSpectrum
    Histogram of bar lifetimes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from zztop.vectorizations._base import BaseVectorizer
from zztop.vectorizations._diagram import DiagramDict

__all__ = [
    "BettiProfile",
    "BirthFrequency",
    "PersistenceProfile",
    "TurnoverRate",
    "EffectivePersistenceImage",
    "CumulativePersistence",
]


# ======================================================================
#  Helpers for frame-indexed zigzag bars
# ======================================================================

def _max_frame(diagram: DiagramDict) -> int:
    """Return the maximum death/birth value (interpreted as frame index)."""
    mx = 0
    for arr in diagram.values():
        if arr.size > 0:
            mx = max(mx, int(np.ceil(arr.max())))
    return mx


# ======================================================================
#  Betti Profile
# ======================================================================

class BettiProfile(BaseVectorizer):
    r"""Betti number at each frame/layer.

    For a zigzag barcode with integer birth/death (frame indices), the
    Betti number at frame *t* is the count of bars alive at *t*:

    .. math::

        \beta_d(t) = \#\{i : b_i \le t < d_i\}

    Unlike :class:`BettiCurve`, this operates on *frame indices* and the
    output length equals ``n_frames`` (fitted or user-specified).

    Parameters
    ----------
    n_frames : int or None
        Number of frames.  ``None`` → inferred from ``max(death)`` in training data.
    dimensions, drop_inf : see :class:`BaseVectorizer`.
    """

    def __init__(self, n_frames=None, dimensions=None, drop_inf: bool = True):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.n_frames = n_frames

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass

    def _post_fit(self, diagrams: List[DiagramDict]) -> None:
        if self.n_frames is not None:
            self.n_frames_ = self.n_frames
        else:
            self.n_frames_ = max(_max_frame(d) for d in diagrams) if diagrams else 1

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        profiles = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            profile = np.zeros(self.n_frames_, dtype=np.float64)
            if bd.size == 0:
                profiles.append(profile)
                continue
            for t in range(self.n_frames_):
                profile[t] = np.sum((bd[:, 0] <= t) & (t < bd[:, 1]))
            profiles.append(profile)
        return np.concatenate(profiles)


# ======================================================================
#  Birth Frequency
# ======================================================================

class BirthFrequency(BaseVectorizer):
    r"""Relative frequency of births at each frame.

    Inspired by Eq. 5-6 of Gardinazzi et al. (2024):

    .. math::

        f_d(t) = \frac{\#\{i : b_i = t\}}{N_d}

    where :math:`N_d` is the total number of bars in dimension *d*.

    Parameters
    ----------
    n_frames : int or None
        ``None`` → inferred from training data.
    dimensions, drop_inf : see :class:`BaseVectorizer`.
    """

    def __init__(self, n_frames=None, dimensions=None, drop_inf: bool = True):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.n_frames = n_frames

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass

    def _post_fit(self, diagrams: List[DiagramDict]) -> None:
        if self.n_frames is not None:
            self.n_frames_ = self.n_frames
        else:
            self.n_frames_ = max(_max_frame(d) for d in diagrams) if diagrams else 1

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        profiles = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            profile = np.zeros(self.n_frames_, dtype=np.float64)
            if bd.size == 0:
                profiles.append(profile)
                continue
            n_bars = bd.shape[0]
            births = bd[:, 0].astype(int)
            for b in births:
                if 0 <= b < self.n_frames_:
                    profile[b] += 1.0
            if n_bars > 0:
                profile /= n_bars
            profiles.append(profile)
        return np.concatenate(profiles)


# ======================================================================
#  Persistence Profile
# ======================================================================

class PersistenceProfile(BaseVectorizer):
    r"""Mean persistence of bars alive at each frame.

    Inspired by the inter-layer persistence concept (Eq. 7-8 of Gardinazzi
    et al. 2024), simplified to a 1-D profile:

    .. math::

        P_d(t) = \frac{1}{\beta_d(t)} \sum_{i: b_i \le t < d_i} (d_i - b_i)

    Returns 0 where no bars are alive.

    Parameters
    ----------
    n_frames, dimensions, drop_inf : see :class:`BettiProfile`.
    """

    def __init__(self, n_frames=None, dimensions=None, drop_inf: bool = True):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.n_frames = n_frames

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass

    def _post_fit(self, diagrams: List[DiagramDict]) -> None:
        if self.n_frames is not None:
            self.n_frames_ = self.n_frames
        else:
            self.n_frames_ = max(_max_frame(d) for d in diagrams) if diagrams else 1

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        profiles = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            profile = np.zeros(self.n_frames_, dtype=np.float64)
            if bd.size == 0:
                profiles.append(profile)
                continue
            pers = bd[:, 1] - bd[:, 0]
            for t in range(self.n_frames_):
                alive = (bd[:, 0] <= t) & (t < bd[:, 1])
                n_alive = alive.sum()
                if n_alive > 0:
                    profile[t] = pers[alive].mean()
            profiles.append(profile)
        return np.concatenate(profiles)


# ======================================================================
#  Turnover Rate
# ======================================================================

class TurnoverRate(BaseVectorizer):
    r"""Topological turnover rate at each frame.

    Measures how much the topology changes between frames:

    .. math::

        T_d(t) = \frac{\text{births}_d(t) + \text{deaths}_d(t)}
                      {\beta_d(t-1) + \beta_d(t) + 1}

    The denominator is smoothed by +1 to avoid division by zero.
    Higher values indicate more topological activity (features appearing
    and disappearing).

    Parameters
    ----------
    n_frames, dimensions, drop_inf : see :class:`BettiProfile`.
    """

    def __init__(self, n_frames=None, dimensions=None, drop_inf: bool = True):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.n_frames = n_frames

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass

    def _post_fit(self, diagrams: List[DiagramDict]) -> None:
        if self.n_frames is not None:
            self.n_frames_ = self.n_frames
        else:
            self.n_frames_ = max(_max_frame(d) for d in diagrams) if diagrams else 1

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        profiles = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            profile = np.zeros(self.n_frames_, dtype=np.float64)
            if bd.size == 0:
                profiles.append(profile)
                continue
            births_at = np.zeros(self.n_frames_, dtype=np.float64)
            deaths_at = np.zeros(self.n_frames_, dtype=np.float64)
            betti = np.zeros(self.n_frames_, dtype=np.float64)

            for b_val in bd[:, 0].astype(int):
                if 0 <= b_val < self.n_frames_:
                    births_at[b_val] += 1
            for d_val in bd[:, 1].astype(int):
                if 0 <= d_val < self.n_frames_:
                    deaths_at[d_val] += 1
            for t in range(self.n_frames_):
                betti[t] = np.sum((bd[:, 0] <= t) & (t < bd[:, 1]))

            for t in range(self.n_frames_):
                prev = betti[t - 1] if t > 0 else 0
                denom = prev + betti[t] + 1
                profile[t] = (births_at[t] + deaths_at[t]) / denom
            profiles.append(profile)
        return np.concatenate(profiles)


# ======================================================================
#  Effective Persistence Image
# ======================================================================

class EffectivePersistenceImage(BaseVectorizer):
    r"""Persistence image weighted by birth-relative frequency.

    Inspired by Eq. 4 of Gardinazzi et al. (2024): the standard
    persistence image weight is modulated by the relative birth
    frequency, emphasising bars born at "busy" frames.

    .. math::

        w_i = (d_i - b_i) \cdot f_d(b_i)

    where :math:`f_d(b_i)` is the birth-frequency at frame :math:`b_i`.

    Parameters
    ----------
    resolution : tuple of int
        ``(ny, nx)`` — pixel resolution.
    sigma : float
        Gaussian smoothing bandwidth.
    n_frames : int or None
        For computing birth frequencies.  ``None`` → inferred.
    birth_range, pers_range : tuple or None
    dimensions, drop_inf : see :class:`BaseVectorizer`.
    """

    def __init__(
        self,
        resolution: tuple = (20, 20),
        sigma: float = 1.0,
        n_frames=None,
        birth_range=None,
        pers_range=None,
        dimensions=None,
        drop_inf: bool = True,
    ):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.resolution = resolution
        self.sigma = sigma
        self.n_frames = n_frames
        self.birth_range = birth_range
        self.pers_range = pers_range

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass

    def _post_fit(self, diagrams: List[DiagramDict]) -> None:
        if self.n_frames is not None:
            self.n_frames_ = self.n_frames
        else:
            self.n_frames_ = max((_max_frame(d) for d in diagrams), default=1)

        all_births, all_pers = [], []
        for dgm in diagrams:
            for dim in (self.dimensions_ if hasattr(self, 'dimensions_') else [0]):
                bd = self._get_bars(dgm, dim)
                if bd.size:
                    all_births.append(bd[:, 0])
                    all_pers.append(bd[:, 1] - bd[:, 0])
        if all_births:
            ab = np.concatenate(all_births)
            ap = np.concatenate(all_pers)
        else:
            ab = np.array([0.0, 1.0])
            ap = np.array([0.0, 1.0])
        self.birth_range_ = self.birth_range or (float(ab.min()), float(ab.max()))
        self.pers_range_ = self.pers_range or (float(max(ap.min(), 0)), float(ap.max()))
        for attr in ("birth_range_", "pers_range_"):
            lo, hi = getattr(self, attr)
            if hi - lo < 1e-12:
                setattr(self, attr, (lo - 0.5, hi + 0.5))

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        from scipy.ndimage import gaussian_filter

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

            # birth frequency at each bar's birth frame
            birth_counts = np.zeros(self.n_frames_, dtype=np.float64)
            for b in births.astype(int):
                if 0 <= b < self.n_frames_:
                    birth_counts[b] += 1
            n_bars = bd.shape[0]
            birth_freq = np.zeros(len(births), dtype=np.float64)
            for i, b in enumerate(births.astype(int)):
                if 0 <= b < self.n_frames_ and n_bars > 0:
                    birth_freq[i] = birth_counts[b] / n_bars

            # Weight = persistence * birth frequency
            w = pers * birth_freq

            b_lo, b_hi = self.birth_range_
            p_lo, p_hi = self.pers_range_
            bx = np.clip(
                ((births - b_lo) / (b_hi - b_lo) * (nx - 1)).astype(int), 0, nx - 1
            )
            py = np.clip(
                ((pers - p_lo) / (p_hi - p_lo) * (ny - 1)).astype(int), 0, ny - 1
            )
            np.add.at(img, (py, bx), w)
            if self.sigma > 0:
                img = gaussian_filter(img, sigma=self.sigma)
            images.append(img.ravel())
        return np.concatenate(images)


# ======================================================================
#  Cumulative Persistence
# ======================================================================

class CumulativePersistence(BaseVectorizer):
    r"""Cumulative total persistence up to each frame.

    .. math::

        C_d(t) = \sum_{i: d_i \le t} (d_i - b_i)

    Features that have died by frame *t* contribute their full lifetime.

    Parameters
    ----------
    n_frames, dimensions, drop_inf : see :class:`BettiProfile`.
    """

    def __init__(self, n_frames=None, dimensions=None, drop_inf: bool = True):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.n_frames = n_frames

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass

    def _post_fit(self, diagrams: List[DiagramDict]) -> None:
        if self.n_frames is not None:
            self.n_frames_ = self.n_frames
        else:
            self.n_frames_ = max(_max_frame(d) for d in diagrams) if diagrams else 1

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        profiles = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            profile = np.zeros(self.n_frames_, dtype=np.float64)
            if bd.size == 0:
                profiles.append(profile)
                continue
            pers = bd[:, 1] - bd[:, 0]
            deaths = bd[:, 1].astype(int)
            # Accumulate at death frames
            increments = np.zeros(self.n_frames_, dtype=np.float64)
            for i in range(bd.shape[0]):
                d_frame = deaths[i]
                if 0 <= d_frame < self.n_frames_:
                    increments[d_frame] += pers[i]
            profile = np.cumsum(increments)
            profiles.append(profile)
        return np.concatenate(profiles)


# ======================================================================
#  Experimental descriptors (not in __all__)
# ======================================================================

class LayerTransitionEntropy(BaseVectorizer):
    r"""Shannon entropy of topological events at each frame.

    At frame *t*, consider the event vector ``[n_births, n_deaths,
    n_surviving]`` and compute its entropy.  High entropy means events
    are spread evenly; low entropy means one type dominates.

    .. note:: Experimental — not included in ``__all__``.

    Parameters
    ----------
    n_frames, dimensions, drop_inf : see :class:`BettiProfile`.
    """

    def __init__(self, n_frames=None, dimensions=None, drop_inf: bool = True):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.n_frames = n_frames

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass

    def _post_fit(self, diagrams: List[DiagramDict]) -> None:
        if self.n_frames is not None:
            self.n_frames_ = self.n_frames
        else:
            self.n_frames_ = max(_max_frame(d) for d in diagrams) if diagrams else 1

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        profiles = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            profile = np.zeros(self.n_frames_, dtype=np.float64)
            if bd.size == 0:
                profiles.append(profile)
                continue

            for t in range(self.n_frames_):
                n_births = np.sum(bd[:, 0].astype(int) == t)
                n_deaths = np.sum(bd[:, 1].astype(int) == t)
                n_alive = np.sum((bd[:, 0] <= t) & (t < bd[:, 1]))
                counts = np.array([n_births, n_deaths, n_alive], dtype=np.float64)
                total = counts.sum()
                if total > 0:
                    p = counts / total
                    p = p[p > 0]
                    profile[t] = -np.sum(p * np.log(p))
            profiles.append(profile)
        return np.concatenate(profiles)


class PersistenceFlux(BaseVectorizer):
    r"""Net change in total persistence between consecutive frames.

    .. math::

        \Phi_d(t) = P_{\text{total},d}(t) - P_{\text{total},d}(t-1)

    where :math:`P_{\text{total},d}(t) = \sum_{i: b_i \le t < d_i} (d_i - b_i)`.

    Positive flux = topology is growing; negative = shrinking.

    .. note:: Experimental — not included in ``__all__``.
    """

    def __init__(self, n_frames=None, dimensions=None, drop_inf: bool = True):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.n_frames = n_frames

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass

    def _post_fit(self, diagrams: List[DiagramDict]) -> None:
        if self.n_frames is not None:
            self.n_frames_ = self.n_frames
        else:
            self.n_frames_ = max(_max_frame(d) for d in diagrams) if diagrams else 1

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        profiles = []
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            total_pers = np.zeros(self.n_frames_, dtype=np.float64)
            if bd.size > 0:
                pers = bd[:, 1] - bd[:, 0]
                for t in range(self.n_frames_):
                    alive = (bd[:, 0] <= t) & (t < bd[:, 1])
                    total_pers[t] = pers[alive].sum()
            flux = np.diff(total_pers, prepend=0.0)
            profiles.append(flux)
        return np.concatenate(profiles)


class CrossDimCorrelation(BaseVectorizer):
    r"""Pearson correlation of Betti profiles across dimension pairs.

    For each pair of dimensions ``(d1, d2)``, returns the Pearson-*r*
    between their Betti profiles.  Captures cross-dimensional topological
    coupling.

    .. note:: Experimental — not included in ``__all__``.
    """

    def __init__(self, n_frames=None, dimensions=None, drop_inf: bool = True):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.n_frames = n_frames

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass

    def _post_fit(self, diagrams: List[DiagramDict]) -> None:
        if self.n_frames is not None:
            self.n_frames_ = self.n_frames
        else:
            self.n_frames_ = max(_max_frame(d) for d in diagrams) if diagrams else 1

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        # Compute Betti profile for each dimension
        betti = {}
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            prof = np.zeros(self.n_frames_, dtype=np.float64)
            if bd.size > 0:
                for t in range(self.n_frames_):
                    prof[t] = np.sum((bd[:, 0] <= t) & (t < bd[:, 1]))
            betti[dim] = prof

        # Pairwise correlations
        corrs = []
        dims = sorted(self.dimensions_)
        for i in range(len(dims)):
            for j in range(i + 1, len(dims)):
                x, y = betti[dims[i]], betti[dims[j]]
                if x.std() < 1e-12 or y.std() < 1e-12:
                    corrs.append(0.0)
                else:
                    corrs.append(float(np.corrcoef(x, y)[0, 1]))
        if not corrs:
            corrs = [0.0]
        return np.array(corrs, dtype=np.float64)


class LifetimeSpectrum(BaseVectorizer):
    r"""Histogram of bar lifetimes.

    A simple, robust summary — just bin the ``death - birth`` values.

    .. note:: Experimental — not included in ``__all__``.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins (default 50).
    lifetime_range : tuple or None
        ``None`` → fitted from training data.
    dimensions, drop_inf : see :class:`BaseVectorizer`.
    """

    def __init__(
        self,
        n_bins: int = 50,
        lifetime_range=None,
        dimensions=None,
        drop_inf: bool = True,
    ):
        super().__init__(dimensions=dimensions, drop_inf=drop_inf)
        self.n_bins = n_bins
        self.lifetime_range = lifetime_range

    def _fit_single(self, diagram: DiagramDict) -> None:
        pass

    def _post_fit(self, diagrams: List[DiagramDict]) -> None:
        if self.lifetime_range is not None:
            self.lifetime_range_ = tuple(self.lifetime_range)
            return
        all_lt = []
        for dgm in diagrams:
            for dim in self.dimensions_:
                bd = self._get_bars(dgm, dim)
                if bd.size:
                    all_lt.append(bd[:, 1] - bd[:, 0])
        if all_lt:
            lt = np.concatenate(all_lt)
            self.lifetime_range_ = (float(lt.min()), float(lt.max()))
        else:
            self.lifetime_range_ = (0.0, 1.0)
        lo, hi = self.lifetime_range_
        if hi - lo < 1e-12:
            self.lifetime_range_ = (lo - 0.5, hi + 0.5)

    def _transform_single(self, diagram: DiagramDict) -> np.ndarray:
        vectors = []
        lo, hi = self.lifetime_range_
        for dim in self.dimensions_:
            bd = self._get_bars(diagram, dim)
            if bd.size == 0:
                vectors.append(np.zeros(self.n_bins, dtype=np.float64))
                continue
            lt = bd[:, 1] - bd[:, 0]
            hist, _ = np.histogram(lt, bins=self.n_bins, range=(lo, hi))
            vectors.append(hist.astype(np.float64))
        return np.concatenate(vectors)

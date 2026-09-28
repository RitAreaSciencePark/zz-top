"""
Persistence diagram vectorizations for zztop.
==============================================

This sub-package provides scikit-learn–compatible transformers that
convert persistence diagrams into fixed-size feature vectors suitable
for machine-learning pipelines.

Three families of vectorizers are included:

**Standard**  (``vectorizations.standard``)
    Classical methods from the TDA literature: persistence images,
    landscapes, silhouettes, entropy, amplitudes, and statistics.

**Curves**  (``vectorizations.curves``)
    1-D functional summaries sampled on a regular grid: Betti curve,
    life curve, birth/death curves, midlife KDE, and a generic
    persistence-curve framework.

**Zigzag**  (``vectorizations.zigzag``)
    Descriptors that exploit the multi-layer / temporal structure of
    zigzag persistence: Betti profile, birth frequency, persistence
    profile, turnover rate, effective persistence image, and several
    experimental descriptors.

Quick start
-----------
::

    from zztop import run_cubical_zigzag
    from zztop.vectorizations import BettiCurve, PersistenceImage

    bars = run_cubical_zigzag(grid_data, threshold=0.0)

    pi = PersistenceImage(resolution=(20, 20), sigma=1.0)
    pi.fit([bars])
    feature_vector = pi.transform([bars])  # shape (1, 800)

    bc = BettiCurve(resolution=50)
    bc.fit([bars])
    curve = bc.transform([bars])  # shape (1, n_dims * 50)

All vectorizers follow the ``fit / transform`` API and can be used
directly in :class:`sklearn.pipeline.Pipeline`.
"""

# -- Diagram normalisation (internal, but useful for advanced users) ------
from zztop.vectorizations._diagram import (
    normalize_diagram,
    normalize_diagrams,
)

# -- Standard vectorizers --------------------------------------------------
from zztop.vectorizations.standard import (
    PersistenceImage,
    PersistenceLandscape,
    Silhouette,
    PersistenceEntropy,
    Amplitude,
    PersistenceStatistics,
)

# -- Curve-based vectorizers -----------------------------------------------
from zztop.vectorizations.curves import (
    BettiCurve,
    LifeCurve,
    BirthCurve,
    DeathCurve,
    MidlifeCurve,
    MultiplicativeLifeCurve,
    PersistenceCurve,
)

# -- Zigzag-specific vectorizers -------------------------------------------
from zztop.vectorizations.zigzag import (
    BettiProfile,
    BirthFrequency,
    PersistenceProfile,
    TurnoverRate,
    EffectivePersistenceImage,
    CumulativePersistence,
)

# -- Multi-threshold vectorizers -------------------------------------------
from zztop.vectorizations.multi_threshold import (
    BettiSurface,
    ThresholdIntegratedPersistence,
    ThresholdBirthFrequency,
)

__all__ = [
    # Normalisation
    "normalize_diagram",
    "normalize_diagrams",
    # Standard
    "PersistenceImage",
    "PersistenceLandscape",
    "Silhouette",
    "PersistenceEntropy",
    "Amplitude",
    "PersistenceStatistics",
    # Curves
    "BettiCurve",
    "LifeCurve",
    "BirthCurve",
    "DeathCurve",
    "MidlifeCurve",
    "MultiplicativeLifeCurve",
    "PersistenceCurve",
    # Zigzag
    "BettiProfile",
    "BirthFrequency",
    "PersistenceProfile",
    "TurnoverRate",
    "EffectivePersistenceImage",
    "CumulativePersistence",
    # Multi-threshold
    "BettiSurface",
    "ThresholdIntegratedPersistence",
    "ThresholdBirthFrequency",
]

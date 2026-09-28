"""
zztop — Unified Zigzag Persistence Pipeline
=============================================

Compute zigzag persistent homology for both **simplicial** and **cubical**
complexes through a single, self-contained package.

Quick start (simplicial)::

    from zztop import run_simplicial_zigzag
    bars = run_simplicial_zigzag(point_clouds, knn=10, dim=2)

Quick start (cubical)::

    from zztop import run_cubical_zigzag
    bars = run_cubical_zigzag(grid_data, threshold=0.0)

The engine is a pure-Python implementation of the Dey-Hou fast zigzag
algorithm (ESA 2022) over GF(2); nothing is compiled.
"""

__version__ = "0.8.0"

# Public API — builders (high-level)
from zztop.builders.simplicial_builder import (
    build_simplicial_zigzag,
    run_simplicial_zigzag,
)
from zztop.builders.cubical_builder import (
    build_cubical_zigzag,
    run_cubical_zigzag,
)
from zztop.builders.mesh_builder import (
    build_mesh_zigzag,
    run_mesh_zigzag,
)
from zztop.builders.multi_threshold import (
    MultiThresholdResult,
    build_cubical_zigzag_multi_threshold,
    run_cubical_zigzag_multi_threshold,
)
from zztop.builders.gpu_cubical import (
    run_cubical_persistence_gpu,
    run_cubical_persistence_sklearn,
    gpu_available,
    gpu_info,
    _gpu_supports_3d as gpu_supports_3d,
)

# Public API — engine (mid-level)
from zztop.zigzag.engine import ZigzagEngine
from zztop.zigzag.filtration import ZigzagFiltration

# Public API — complexes (low-level)
from zztop.complex.simplicial import Simplex, SimplicialComplex
from zztop.complex.cubical import CubicalCell, CubicalComplex

# Public API — vectorizations
from zztop.vectorizations import (                     # noqa: F401
    normalize_diagram,
    normalize_diagrams,
    PersistenceImage,
    PersistenceLandscape,
    Silhouette,
    PersistenceEntropy,
    Amplitude,
    PersistenceStatistics,
    BettiCurve,
    LifeCurve,
    BirthCurve,
    DeathCurve,
    MidlifeCurve,
    MultiplicativeLifeCurve,
    PersistenceCurve,
    BettiProfile,
    BirthFrequency,
    PersistenceProfile,
    TurnoverRate,
    EffectivePersistenceImage,
    CumulativePersistence,
    BettiSurface,
    ThresholdIntegratedPersistence,
    ThresholdBirthFrequency,
)

__all__ = [
    # Builders
    "build_simplicial_zigzag",
    "run_simplicial_zigzag",
    "build_cubical_zigzag",
    "run_cubical_zigzag",
    # Mesh zigzag
    "build_mesh_zigzag",
    "run_mesh_zigzag",
    # Multi-threshold zigzag
    "MultiThresholdResult",
    "build_cubical_zigzag_multi_threshold",
    "run_cubical_zigzag_multi_threshold",
    # GPU-accelerated static persistence
    "run_cubical_persistence_gpu",
    "run_cubical_persistence_sklearn",
    "gpu_available",
    "gpu_info",
    # Engine
    "ZigzagEngine",
    "ZigzagFiltration",
    # Complexes
    "Simplex",
    "SimplicialComplex",
    "CubicalCell",
    "CubicalComplex",
    # Vectorizations — standard
    "normalize_diagram",
    "normalize_diagrams",
    "PersistenceImage",
    "PersistenceLandscape",
    "Silhouette",
    "PersistenceEntropy",
    "Amplitude",
    "PersistenceStatistics",
    # Vectorizations — curves
    "BettiCurve",
    "LifeCurve",
    "BirthCurve",
    "DeathCurve",
    "MidlifeCurve",
    "MultiplicativeLifeCurve",
    "PersistenceCurve",
    # Vectorizations — zigzag
    "BettiProfile",
    "BirthFrequency",
    "PersistenceProfile",
    "TurnoverRate",
    "EffectivePersistenceImage",
    "CumulativePersistence",
    # Vectorizations — multi-threshold
    "BettiSurface",
    "ThresholdIntegratedPersistence",
    "ThresholdBirthFrequency",
]

"""
Tests for zztop.vectorizations.

Covers:
  - Diagram normalisation (_diagram.py)
  - Standard vectorizers (standard.py)
  - Curve-based vectorizers (curves.py)
  - Zigzag-specific vectorizers (zigzag.py)
  - sklearn pipeline compatibility
  - Optional torch support (_torch.py)
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

# ======================================================================
#  Fixtures — synthetic diagrams
# ======================================================================

# --- Zigzag barcodes (list of (dim, birth, death)) ---

ZIGZAG_BARS_A = [
    (0, 0, 5),
    (0, 1, 3),
    (0, 2, 7),
    (1, 0, 4),
    (1, 3, 6),
]

ZIGZAG_BARS_B = [
    (0, 0, 10),
    (0, 2, 8),
    (1, 1, 5),
    (1, 4, 9),
    (2, 0, 3),
]

# --- GPU-style diagram dicts {dim: ndarray(n_bars, 2)} ---

GPU_DIAGRAM_A = {
    0: np.array([[0.0, 2.5], [1.0, 3.0], [0.5, 1.5]]),
    1: np.array([[0.0, 1.0], [0.5, 2.0]]),
}

GPU_DIAGRAM_B = {
    0: np.array([[0.0, 4.0], [1.5, 2.5]]),
    1: np.array([[0.0, 3.0]]),
}

# --- Raw ndarray (n, 3) ---

RAW_ARRAY_3COL = np.array([
    [0, 0.0, 2.0],
    [0, 1.0, 3.0],
    [1, 0.5, 1.5],
])

RAW_ARRAY_2COL = np.array([
    [0.0, 2.0],
    [1.0, 3.0],
    [0.5, 1.5],
])


# ======================================================================
#  Tests — Diagram normalisation
# ======================================================================

class TestNormalizeDiagram:

    def test_zigzag_barcode(self):
        from zztop.vectorizations._diagram import normalize_diagram
        result = normalize_diagram(ZIGZAG_BARS_A)
        assert isinstance(result, dict)
        assert 0 in result and 1 in result
        assert result[0].shape == (3, 2)
        assert result[1].shape == (2, 2)
        # Check values
        assert_allclose(result[0][0], [0, 5])
        assert_allclose(result[1][0], [0, 4])

    def test_gpu_dict(self):
        from zztop.vectorizations._diagram import normalize_diagram
        result = normalize_diagram(GPU_DIAGRAM_A)
        assert 0 in result and 1 in result
        assert result[0].shape == (3, 2)

    def test_ndarray_3col(self):
        from zztop.vectorizations._diagram import normalize_diagram
        result = normalize_diagram(RAW_ARRAY_3COL)
        assert 0 in result and 1 in result
        assert result[0].shape == (2, 2)
        assert result[1].shape == (1, 2)

    def test_ndarray_2col(self):
        from zztop.vectorizations._diagram import normalize_diagram
        result = normalize_diagram(RAW_ARRAY_2COL)
        assert result[0].shape == (3, 2)

    def test_drop_inf(self):
        from zztop.vectorizations._diagram import normalize_diagram
        bars_with_inf = [
            (0, 0, float("inf")),
            (0, 1, 5),
        ]
        result = normalize_diagram(bars_with_inf, drop_inf=True)
        assert result[0].shape == (1, 2)
        result_keep = normalize_diagram(bars_with_inf, drop_inf=False)
        assert result_keep[0].shape == (2, 2)

    def test_empty_input(self):
        from zztop.vectorizations._diagram import normalize_diagram
        result = normalize_diagram([])
        assert result == {}

    def test_gpu_batch(self):
        from zztop.vectorizations._diagram import normalize_diagrams
        batch = [GPU_DIAGRAM_A, GPU_DIAGRAM_B]
        results = normalize_diagrams(batch)
        assert len(results) == 2
        assert results[0][0].shape == (3, 2)

    def test_persistence_values(self):
        from zztop.vectorizations._diagram import (
            normalize_diagram, persistence_values,
            birth_values, death_values, midlife_values,
        )
        dgm = normalize_diagram(GPU_DIAGRAM_A)
        p = persistence_values(dgm, 0)
        assert_allclose(p, [2.5, 2.0, 1.0])
        b = birth_values(dgm, 0)
        assert_allclose(b, [0.0, 1.0, 0.5])
        d = death_values(dgm, 0)
        assert_allclose(d, [2.5, 3.0, 1.5])
        m = midlife_values(dgm, 0)
        assert_allclose(m, [1.25, 2.0, 1.0])


# ======================================================================
#  Tests — Standard vectorizers
# ======================================================================

class TestPersistenceImage:

    def test_shape(self):
        from zztop.vectorizations import PersistenceImage
        pi = PersistenceImage(resolution=(10, 10), sigma=0.5, dimensions=[0, 1])
        pi.fit([GPU_DIAGRAM_A, GPU_DIAGRAM_B])
        out = pi.transform([GPU_DIAGRAM_A])
        # 2 dims × 10×10 = 200
        assert out.shape == (1, 200)

    def test_nonzero(self):
        from zztop.vectorizations import PersistenceImage
        pi = PersistenceImage(resolution=(10, 10), sigma=0.5, dimensions=[0])
        pi.fit([GPU_DIAGRAM_A])
        out = pi.transform([GPU_DIAGRAM_A])
        assert np.any(out > 0)

    def test_batch(self):
        from zztop.vectorizations import PersistenceImage
        pi = PersistenceImage(resolution=(5, 5), sigma=0.3, dimensions=[0])
        pi.fit([GPU_DIAGRAM_A, GPU_DIAGRAM_B])
        out = pi.transform([GPU_DIAGRAM_A, GPU_DIAGRAM_B])
        assert out.shape == (2, 25)

    def test_zigzag_input(self):
        from zztop.vectorizations import PersistenceImage
        pi = PersistenceImage(resolution=(5, 5), sigma=0.5)
        pi.fit([ZIGZAG_BARS_A])
        out = pi.transform([ZIGZAG_BARS_A])
        assert out.shape[0] == 1
        assert out.shape[1] > 0


class TestPersistenceLandscape:

    def test_shape(self):
        from zztop.vectorizations import PersistenceLandscape
        pl = PersistenceLandscape(n_landscapes=3, resolution=50, dimensions=[0])
        pl.fit([GPU_DIAGRAM_A])
        out = pl.transform([GPU_DIAGRAM_A])
        assert out.shape == (1, 3 * 50)

    def test_known_single_bar(self):
        """Single bar (0,2) → landscape_1 has peak=1 at midpoint."""
        from zztop.vectorizations import PersistenceLandscape
        dgm = {0: np.array([[0.0, 2.0]])}
        pl = PersistenceLandscape(n_landscapes=1, resolution=101,
                                   sample_range=(0, 2), dimensions=[0])
        pl.fit([dgm])
        out = pl.transform([dgm]).ravel()
        # Peak at midpoint t=1.0 → tent value = (2-0)/2 = 1.0
        mid_idx = 50
        assert_allclose(out[mid_idx], 1.0, atol=0.02)


class TestSilhouette:

    def test_shape(self):
        from zztop.vectorizations import Silhouette
        sil = Silhouette(resolution=80, dimensions=[0, 1])
        sil.fit([GPU_DIAGRAM_A])
        out = sil.transform([GPU_DIAGRAM_A])
        assert out.shape == (1, 2 * 80)


class TestPersistenceEntropy:

    def test_shape(self):
        from zztop.vectorizations import PersistenceEntropy
        pe = PersistenceEntropy(dimensions=[0, 1])
        pe.fit([GPU_DIAGRAM_A])
        out = pe.transform([GPU_DIAGRAM_A])
        assert out.shape == (1, 2)

    def test_single_bar_entropy_zero(self):
        """One bar → entropy = 0."""
        from zztop.vectorizations import PersistenceEntropy
        dgm = {0: np.array([[0.0, 1.0]])}
        pe = PersistenceEntropy(dimensions=[0])
        pe.fit([dgm])
        out = pe.transform([dgm])
        assert_allclose(out[0, 0], 0.0, atol=1e-12)

    def test_normalized(self):
        from zztop.vectorizations import PersistenceEntropy
        pe = PersistenceEntropy(normalize=True, dimensions=[0])
        pe.fit([GPU_DIAGRAM_A])
        out = pe.transform([GPU_DIAGRAM_A])
        assert 0.0 <= out[0, 0] <= 1.0


class TestAmplitude:

    def test_wasserstein(self):
        from zztop.vectorizations import Amplitude
        amp = Amplitude(metric="wasserstein", order=2, dimensions=[0])
        amp.fit([GPU_DIAGRAM_A])
        out = amp.transform([GPU_DIAGRAM_A])
        assert out.shape == (1, 1)
        assert out[0, 0] > 0

    def test_bottleneck(self):
        from zztop.vectorizations import Amplitude
        amp = Amplitude(metric="bottleneck", dimensions=[0])
        amp.fit([GPU_DIAGRAM_A])
        out = amp.transform([GPU_DIAGRAM_A])
        # max persistence in dim 0 = 2.5
        assert_allclose(out[0, 0], 2.5)


class TestPersistenceStatistics:

    def test_shape(self):
        from zztop.vectorizations import PersistenceStatistics
        ps = PersistenceStatistics(dimensions=[0, 1])
        ps.fit([GPU_DIAGRAM_A])
        out = ps.transform([GPU_DIAGRAM_A])
        # 9 features per dim × 2 dims = 18
        assert out.shape == (1, 18)

    def test_count(self):
        from zztop.vectorizations import PersistenceStatistics
        ps = PersistenceStatistics(dimensions=[0])
        ps.fit([GPU_DIAGRAM_A])
        out = ps.transform([GPU_DIAGRAM_A])
        # First feature is n_bars for dim 0 = 3
        assert_allclose(out[0, 0], 3.0)


# ======================================================================
#  Tests — Curve vectorizers
# ======================================================================

class TestBettiCurve:

    def test_shape(self):
        from zztop.vectorizations import BettiCurve
        bc = BettiCurve(resolution=50, dimensions=[0])
        bc.fit([GPU_DIAGRAM_A])
        out = bc.transform([GPU_DIAGRAM_A])
        assert out.shape == (1, 50)

    def test_known_values(self):
        """Two bars: [0,2) and [1,3) → β(1.5)=2."""
        from zztop.vectorizations import BettiCurve
        dgm = {0: np.array([[0.0, 2.0], [1.0, 3.0]])}
        bc = BettiCurve(resolution=101, sample_range=(0, 3), dimensions=[0])
        bc.fit([dgm])
        out = bc.transform([dgm]).ravel()
        # t=1.5 → index = int(1.5/3*100) ≈ 50
        idx = 50
        assert out[idx] == 2.0


class TestLifeCurve:

    def test_shape(self):
        from zztop.vectorizations import LifeCurve
        lc = LifeCurve(resolution=40, dimensions=[0])
        lc.fit([GPU_DIAGRAM_A])
        out = lc.transform([GPU_DIAGRAM_A])
        assert out.shape == (1, 40)


class TestBirthCurve:

    def test_monotone(self):
        """Birth curve is monotone non-decreasing."""
        from zztop.vectorizations import BirthCurve
        bc = BirthCurve(resolution=100, dimensions=[0])
        bc.fit([GPU_DIAGRAM_A])
        out = bc.transform([GPU_DIAGRAM_A]).ravel()
        assert np.all(np.diff(out) >= 0)


class TestDeathCurve:

    def test_monotone(self):
        """Death curve is monotone non-decreasing."""
        from zztop.vectorizations import DeathCurve
        dc = DeathCurve(resolution=100, dimensions=[0])
        dc.fit([GPU_DIAGRAM_A])
        out = dc.transform([GPU_DIAGRAM_A]).ravel()
        assert np.all(np.diff(out) >= 0)


class TestMidlifeCurve:

    def test_shape(self):
        from zztop.vectorizations import MidlifeCurve
        mc = MidlifeCurve(bandwidth=0.2, resolution=50, dimensions=[0])
        mc.fit([GPU_DIAGRAM_A])
        out = mc.transform([GPU_DIAGRAM_A])
        assert out.shape == (1, 50)


class TestMultiplicativeLifeCurve:

    def test_shape(self):
        from zztop.vectorizations import MultiplicativeLifeCurve
        mlc = MultiplicativeLifeCurve(resolution=30, dimensions=[0])
        mlc.fit([GPU_DIAGRAM_A])
        out = mlc.transform([GPU_DIAGRAM_A])
        assert out.shape == (1, 30)


class TestPersistenceCurve:

    def test_custom_fn(self):
        """Custom summary: sum of persistences of alive bars."""
        from zztop.vectorizations import PersistenceCurve
        fn = lambda b, d, p, t: p.sum()
        pc = PersistenceCurve(summary_fn=fn, resolution=50, dimensions=[0])
        pc.fit([GPU_DIAGRAM_A])
        out = pc.transform([GPU_DIAGRAM_A])
        assert out.shape == (1, 50)
        assert np.any(out > 0)


# ======================================================================
#  Tests — Zigzag vectorizers
# ======================================================================

class TestBettiProfile:

    def test_shape(self):
        from zztop.vectorizations import BettiProfile
        bp = BettiProfile(n_frames=10, dimensions=[0, 1])
        bp.fit([ZIGZAG_BARS_A])
        out = bp.transform([ZIGZAG_BARS_A])
        assert out.shape == (1, 20)  # 2 dims × 10 frames

    def test_known_values(self):
        """Bars (0,0,5), (0,1,3), (0,2,7): at t=2, all 3 alive."""
        from zztop.vectorizations import BettiProfile
        bp = BettiProfile(n_frames=8, dimensions=[0])
        bp.fit([ZIGZAG_BARS_A])
        out = bp.transform([ZIGZAG_BARS_A]).ravel()
        # t=2: bars alive are [0,5), [1,3), [2,7) → 3
        assert out[2] == 3.0
        # t=0: [0,5) only → 1
        assert out[0] == 1.0
        # t=6: [2,7) only → 1
        assert out[6] == 1.0


class TestBirthFrequency:

    def test_sums_to_one(self):
        """Birth frequencies for each dim sum to 1."""
        from zztop.vectorizations import BirthFrequency
        bf = BirthFrequency(n_frames=8, dimensions=[0])
        bf.fit([ZIGZAG_BARS_A])
        out = bf.transform([ZIGZAG_BARS_A]).ravel()
        assert_allclose(out.sum(), 1.0, atol=1e-12)


class TestPersistenceProfile:

    def test_shape(self):
        from zztop.vectorizations import PersistenceProfile
        pp = PersistenceProfile(n_frames=8, dimensions=[0])
        pp.fit([ZIGZAG_BARS_A])
        out = pp.transform([ZIGZAG_BARS_A])
        assert out.shape == (1, 8)


class TestTurnoverRate:

    def test_shape(self):
        from zztop.vectorizations import TurnoverRate
        tr = TurnoverRate(n_frames=8, dimensions=[0])
        tr.fit([ZIGZAG_BARS_A])
        out = tr.transform([ZIGZAG_BARS_A])
        assert out.shape == (1, 8)


class TestEffectivePersistenceImage:

    def test_shape(self):
        from zztop.vectorizations import EffectivePersistenceImage
        epi = EffectivePersistenceImage(
            resolution=(5, 5), sigma=0.5, n_frames=8, dimensions=[0]
        )
        epi.fit([ZIGZAG_BARS_A])
        out = epi.transform([ZIGZAG_BARS_A])
        assert out.shape == (1, 25)


class TestCumulativePersistence:

    def test_monotone(self):
        from zztop.vectorizations import CumulativePersistence
        cp = CumulativePersistence(n_frames=10, dimensions=[0])
        cp.fit([ZIGZAG_BARS_A])
        out = cp.transform([ZIGZAG_BARS_A]).ravel()
        assert np.all(np.diff(out) >= 0)


# ======================================================================
#  Tests — Experimental zigzag descriptors
# ======================================================================

class TestExperimentalZigzag:

    def test_layer_transition_entropy(self):
        from zztop.vectorizations.zigzag import LayerTransitionEntropy
        lte = LayerTransitionEntropy(n_frames=8, dimensions=[0])
        lte.fit([ZIGZAG_BARS_A])
        out = lte.transform([ZIGZAG_BARS_A])
        assert out.shape == (1, 8)
        assert np.all(out >= 0)

    def test_persistence_flux(self):
        from zztop.vectorizations.zigzag import PersistenceFlux
        pf = PersistenceFlux(n_frames=8, dimensions=[0])
        pf.fit([ZIGZAG_BARS_A])
        out = pf.transform([ZIGZAG_BARS_A])
        assert out.shape == (1, 8)

    def test_cross_dim_correlation(self):
        from zztop.vectorizations.zigzag import CrossDimCorrelation
        cdc = CrossDimCorrelation(n_frames=8, dimensions=[0, 1])
        cdc.fit([ZIGZAG_BARS_A])
        out = cdc.transform([ZIGZAG_BARS_A])
        # 1 pair (0,1) → 1 value
        assert out.shape == (1, 1)
        assert -1.0 <= out[0, 0] <= 1.0

    def test_lifetime_spectrum(self):
        from zztop.vectorizations.zigzag import LifetimeSpectrum
        ls = LifetimeSpectrum(n_bins=10, dimensions=[0])
        ls.fit([ZIGZAG_BARS_A])
        out = ls.transform([ZIGZAG_BARS_A])
        assert out.shape == (1, 10)
        assert out.sum() == 3  # 3 bars in dim 0


# ======================================================================
#  Tests — sklearn pipeline compatibility
# ======================================================================

class TestSklearnPipeline:

    def test_pipeline_persistence_image(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from zztop.vectorizations import PersistenceImage

        pipe = Pipeline([
            ("pi", PersistenceImage(resolution=(5, 5), sigma=0.3, dimensions=[0])),
            ("scaler", StandardScaler()),
        ])
        data = [GPU_DIAGRAM_A, GPU_DIAGRAM_B]
        out = pipe.fit_transform(data)
        assert out.shape == (2, 25)

    def test_pipeline_betti_curve(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from zztop.vectorizations import BettiCurve

        pipe = Pipeline([
            ("bc", BettiCurve(resolution=20, dimensions=[0])),
            ("scaler", StandardScaler()),
        ])
        data = [GPU_DIAGRAM_A, GPU_DIAGRAM_B]
        out = pipe.fit_transform(data)
        assert out.shape == (2, 20)

    def test_fit_transform_equivalence(self):
        """fit().transform() == fit_transform()."""
        from zztop.vectorizations import PersistenceEntropy
        data = [GPU_DIAGRAM_A, GPU_DIAGRAM_B]
        pe1 = PersistenceEntropy(dimensions=[0, 1])
        out1 = pe1.fit(data).transform(data)
        pe2 = PersistenceEntropy(dimensions=[0, 1])
        out2 = pe2.fit_transform(data)
        assert_allclose(out1, out2)


# ======================================================================
#  Tests — Torch support
# ======================================================================

class TestTorchSupport:

    @pytest.fixture(autouse=True)
    def _skip_if_no_torch(self):
        pytest.importorskip("torch")

    def test_to_torch(self):
        from zztop.vectorizations import PersistenceImage
        pi = PersistenceImage(resolution=(5, 5), sigma=0.3, dimensions=[0])
        pi.fit([GPU_DIAGRAM_A])
        t = pi.to_torch([GPU_DIAGRAM_A])
        import torch
        assert isinstance(t, torch.Tensor)
        assert t.shape == (1, 25)

    def test_differentiable_transform(self):
        from zztop.vectorizations._torch import differentiable_transform
        from zztop.vectorizations import PersistenceEntropy
        pe = PersistenceEntropy(dimensions=[0])
        pe.fit([GPU_DIAGRAM_A])
        t = differentiable_transform(pe, [GPU_DIAGRAM_A], requires_grad=True)
        import torch
        assert t.requires_grad

    def test_torch_vectorizer_wrapper(self):
        from zztop.vectorizations._torch import TorchVectorizer
        from zztop.vectorizations import BettiCurve
        tv = TorchVectorizer(BettiCurve(resolution=20, dimensions=[0]))
        tv.fit([GPU_DIAGRAM_A, GPU_DIAGRAM_B])
        t = tv.transform([GPU_DIAGRAM_A])
        import torch
        assert isinstance(t, torch.Tensor)
        assert t.shape == (1, 20)


# ======================================================================
#  Tests — Edge cases
# ======================================================================

class TestEdgeCases:

    def test_empty_diagram(self):
        from zztop.vectorizations import PersistenceImage, BettiCurve
        empty = {0: np.empty((0, 2))}
        pi = PersistenceImage(resolution=(5, 5), sigma=0.3, dimensions=[0])
        pi.fit([GPU_DIAGRAM_A])
        out = pi.transform([empty])
        assert out.shape == (1, 25)
        assert_allclose(out, 0.0)

        bc = BettiCurve(resolution=20, dimensions=[0])
        bc.fit([GPU_DIAGRAM_A])
        out = bc.transform([empty])
        assert_allclose(out, 0.0)

    def test_single_bar(self):
        from zztop.vectorizations import PersistenceStatistics
        dgm = {0: np.array([[1.0, 3.0]])}
        ps = PersistenceStatistics(dimensions=[0])
        ps.fit([dgm])
        out = ps.transform([dgm])
        # n_bars=1, mean_b=1, std_b=0, mean_d=3, std_d=0, mean_p=2, std_p=0, max_p=2, ent=0
        assert_allclose(out[0, 0], 1.0)  # n_bars
        assert_allclose(out[0, 5], 2.0)  # mean_pers

    def test_dimension_mismatch(self):
        """Transform with dims not present in data → zeros."""
        from zztop.vectorizations import PersistenceEntropy
        pe = PersistenceEntropy(dimensions=[0, 1, 2])
        pe.fit([GPU_DIAGRAM_A])
        dgm_no_dim2 = {0: np.array([[0.0, 1.0]])}
        out = pe.transform([dgm_no_dim2])
        # dim 1 and 2 not present → entropy = 0
        assert_allclose(out[0, 1], 0.0)
        assert_allclose(out[0, 2], 0.0)

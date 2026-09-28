"""
Tests for multi-threshold cubical zigzag persistence.
=====================================================

Covers:
- Precomputation parity with legacy single-threshold extraction
- Threshold selection strategies
- Single-threshold parity: multi-threshold with 1 threshold == ``run_cubical_zigzag``
- Extreme thresholds (lowest → only vertices, highest → full complex)
- Betti surface shape and non-negativity
- Monotonicity of per-frame sublevel Betti numbers
- MultiThresholdResult accessors
- Vectorizer shapes and sklearn pipeline integration
"""

from __future__ import annotations

import numpy as np
import pytest

# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def small_grid_2d():
    """Random 3×3 grid, 4 frames."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((3, 3, 4))


@pytest.fixture
def small_grid_3d():
    """Random 3×3×3 grid, 3 frames."""
    rng = np.random.default_rng(123)
    return rng.standard_normal((3, 3, 3, 3))


@pytest.fixture
def constant_grid_2d():
    """All-ones 3×3 grid, 3 frames."""
    return np.ones((3, 3, 3))


def _gudhi_available():
    try:
        import gudhi  # noqa: F401
        return True
    except ImportError:
        return False


# ======================================================================
# Step 0: Precomputation parity
# ======================================================================

class TestPrecomputation:
    """Verify that the new precompute→threshold path produces the same
    cells as the legacy _grid_cells_direct function."""

    def test_precompute_matches_direct_random(self, small_grid_2d):
        """Precompute + threshold == _grid_cells_direct for random data."""
        from zztop.builders.cubical_builder import (
            _grid_cells_direct,
            _precompute_cell_filtrations_direct,
            threshold_cells,
        )
        frame = small_grid_2d[:, :, 0]
        for threshold in [-1.0, 0.0, 0.5, 2.0]:
            expected = _grid_cells_direct(frame, threshold)
            filt = _precompute_cell_filtrations_direct(frame)
            got = threshold_cells(filt, threshold)
            assert got == expected, (
                f"Mismatch at threshold={threshold}: "
                f"|expected|={len(expected)}, |got|={len(got)}"
            )

    def test_precompute_matches_direct_3d(self, small_grid_3d):
        """Same check on 3-D data."""
        from zztop.builders.cubical_builder import (
            _grid_cells_direct,
            _precompute_cell_filtrations_direct,
            threshold_cells,
        )
        frame = small_grid_3d[:, :, :, 0]
        for threshold in [-0.5, 0.0, 1.0]:
            expected = _grid_cells_direct(frame, threshold)
            filt = _precompute_cell_filtrations_direct(frame)
            got = threshold_cells(filt, threshold)
            assert got == expected

    @pytest.mark.skipif(
        not _gudhi_available(), reason="GUDHI not installed"
    )
    def test_precompute_gudhi_matches_direct(self, small_grid_2d):
        """GUDHI and direct precomputation give the same active cells."""
        from zztop.builders.cubical_builder import (
            _precompute_cell_filtrations_gudhi,
            _precompute_cell_filtrations_direct,
            threshold_cells,
        )
        frame = small_grid_2d[:, :, 0]
        filt_g = _precompute_cell_filtrations_gudhi(frame)
        filt_d = _precompute_cell_filtrations_direct(frame)
        for threshold in [-1.0, 0.0, 0.5, 2.0]:
            cells_g = threshold_cells(filt_g, threshold)
            cells_d = threshold_cells(filt_d, threshold)
            assert cells_g == cells_d, (
                f"GUDHI vs direct mismatch at threshold={threshold}"
            )


# ======================================================================
# Step 1: Threshold selection
# ======================================================================

class TestThresholdSelection:
    """Verify threshold selection strategies."""

    def _make_filts(self, grid):
        """Precompute filtrations for all frames."""
        from zztop.builders.cubical_builder import precompute_cell_filtrations
        ndim = grid.ndim - 1
        n_frames = grid.shape[-1]
        filts = []
        for t in range(n_frames):
            slicing = tuple([slice(None)] * ndim + [t])
            filts.append(precompute_cell_filtrations(grid[slicing], use_gudhi=False))
        return filts

    def test_percentile_covers_range(self, small_grid_2d):
        from zztop.builders.threshold import select_thresholds
        filts = self._make_filts(small_grid_2d)
        thresholds = select_thresholds(filts, n_thresholds=10, strategy="percentile")
        assert len(thresholds) >= 2
        assert thresholds[0] < thresholds[-1]

    def test_uniform_equispaced(self, small_grid_2d):
        from zztop.builders.threshold import select_thresholds
        filts = self._make_filts(small_grid_2d)
        thresholds = select_thresholds(filts, n_thresholds=5, strategy="uniform")
        assert len(thresholds) == 5
        diffs = np.diff(thresholds)
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-10)

    def test_custom_passthrough(self, small_grid_2d):
        from zztop.builders.threshold import select_thresholds
        filts = self._make_filts(small_grid_2d)
        custom = np.array([0.1, 0.5, 1.0])
        result = select_thresholds(filts, strategy="custom", thresholds=custom)
        np.testing.assert_array_equal(result, custom)

    def test_histogram_returns_edges(self, small_grid_2d):
        from zztop.builders.threshold import select_thresholds
        filts = self._make_filts(small_grid_2d)
        thresholds = select_thresholds(filts, n_thresholds=8, strategy="histogram")
        assert len(thresholds) >= 2
        assert thresholds[0] <= thresholds[-1]

    def test_unknown_strategy_raises(self, small_grid_2d):
        from zztop.builders.threshold import select_thresholds
        filts = self._make_filts(small_grid_2d)
        with pytest.raises(ValueError, match="Unknown threshold strategy"):
            select_thresholds(filts, strategy="bogus")


# ======================================================================
# Step 4.1: Single-threshold parity
# ======================================================================

class TestSingleThresholdParity:
    """run_cubical_zigzag_multi_threshold with one threshold must
    reproduce run_cubical_zigzag exactly."""

    def _run_both(self, grid, threshold, use_gudhi=False):
        from zztop.builders.cubical_builder import run_cubical_zigzag
        from zztop.builders.multi_threshold import (
            run_cubical_zigzag_multi_threshold,
        )
        legacy = run_cubical_zigzag(
            grid, threshold=threshold, use_gudhi=use_gudhi, backend="python",
        )
        result = run_cubical_zigzag_multi_threshold(
            grid,
            thresholds=np.array([threshold]),
            use_gudhi=use_gudhi,
            backend="python",
            n_jobs=1,
        )
        multi = result.threshold_slice(threshold)
        return sorted(legacy), sorted(multi)

    def test_parity_zero_threshold(self, small_grid_2d):
        legacy, multi = self._run_both(small_grid_2d, 0.0)
        assert legacy == multi

    def test_parity_positive_threshold(self, small_grid_2d):
        legacy, multi = self._run_both(small_grid_2d, 1.0)
        assert legacy == multi

    def test_parity_negative_threshold(self, small_grid_2d):
        legacy, multi = self._run_both(small_grid_2d, -0.5)
        assert legacy == multi

    def test_parity_3d(self, small_grid_3d):
        legacy, multi = self._run_both(small_grid_3d, 0.0)
        assert legacy == multi


# ======================================================================
# Step 4.2-4.3: Extreme thresholds and monotonicity
# ======================================================================

class TestExtremeThresholds:
    """Verify behaviour at extreme threshold values."""

    def test_very_low_threshold_only_vertices(self, small_grid_2d):
        """Below all cell values → only vertices active → only H0 bars."""
        from zztop.builders.multi_threshold import (
            run_cubical_zigzag_multi_threshold,
        )
        result = run_cubical_zigzag_multi_threshold(
            small_grid_2d,
            thresholds=np.array([-1e6]),
            backend="python",
            n_jobs=1,
        )
        bars = result.threshold_slice(-1e6)
        dims = {d for d, b, de in bars}
        assert dims <= {0}, f"Expected only H0, got dims={dims}"

    def test_very_high_threshold_full_complex(self, constant_grid_2d):
        """Constant field + high threshold → full complex at every frame,
        single long H0 bar."""
        from zztop.builders.multi_threshold import (
            run_cubical_zigzag_multi_threshold,
        )
        result = run_cubical_zigzag_multi_threshold(
            constant_grid_2d,
            thresholds=np.array([1e6]),
            backend="python",
            n_jobs=1,
        )
        bars = result.threshold_slice(1e6)
        h0_bars = [(b, d) for dim, b, d in bars if dim == 0]
        assert len(h0_bars) >= 1
        # The longest H0 bar should span most/all frames
        longest = max(d - b for b, d in h0_bars)
        assert longest > 0


class TestMonotonicity:
    """Check that per-frame sublevel Betti numbers are non-decreasing
    in threshold."""

    def test_sublevel_betti_monotone(self, small_grid_2d):
        """β_0 at each frame should not decrease as threshold rises."""
        from zztop.builders.cubical_builder import (
            precompute_cell_filtrations,
            threshold_cells,
        )
        from zztop.complex.cubical import CubicalCell

        frame = small_grid_2d[:, :, 0]
        filt = precompute_cell_filtrations(frame, use_gudhi=False)
        thresholds = np.linspace(-2, 3, 20)
        prev_n_cells = 0
        for a in thresholds:
            cells = threshold_cells(filt, a)
            # More cells at higher threshold → monotone inclusion
            assert len(cells) >= prev_n_cells
            prev_n_cells = len(cells)


# ======================================================================
# Step 4.4: MultiThresholdResult accessors
# ======================================================================

class TestMultiThresholdResult:
    """Test the result container's methods."""

    def _make_result(self, grid, n_thresholds=5):
        from zztop.builders.multi_threshold import (
            run_cubical_zigzag_multi_threshold,
        )
        return run_cubical_zigzag_multi_threshold(
            grid,
            thresholds=n_thresholds,
            threshold_strategy="uniform",
            backend="python",
            n_jobs=1,
            use_gudhi=False,
        )

    def test_len(self, small_grid_2d):
        result = self._make_result(small_grid_2d)
        assert len(result) == 5

    def test_iter(self, small_grid_2d):
        result = self._make_result(small_grid_2d)
        items = list(result)
        assert len(items) == 5
        for a, bars in items:
            assert isinstance(a, float)
            assert isinstance(bars, list)

    def test_repr(self, small_grid_2d):
        result = self._make_result(small_grid_2d)
        r = repr(result)
        assert "MultiThresholdResult" in r
        assert "n_thresholds=5" in r

    def test_betti_surface_shape(self, small_grid_2d):
        result = self._make_result(small_grid_2d)
        surface = result.betti_surface(dim=0)
        assert surface.shape == (5, small_grid_2d.shape[-1])

    def test_betti_surface_nonneg(self, small_grid_2d):
        result = self._make_result(small_grid_2d)
        surface = result.betti_surface(dim=0)
        assert np.all(surface >= 0)

    def test_diagrams_dict(self, small_grid_2d):
        result = self._make_result(small_grid_2d)
        dgm = result.diagrams(result.thresholds[0])
        # Should be a dict of int → ndarray
        for dim, arr in dgm.items():
            assert isinstance(dim, int)
            assert arr.ndim == 2
            assert arr.shape[1] == 2

    def test_all_diagrams(self, small_grid_2d):
        result = self._make_result(small_grid_2d)
        all_dgm = result.all_diagrams()
        assert len(all_dgm) == 5

    def test_total_persistence_nonneg(self, small_grid_2d):
        result = self._make_result(small_grid_2d)
        tp = result.total_persistence(dim=0)
        assert np.all(tp >= 0)

    def test_frame_betti_shape(self, small_grid_2d):
        result = self._make_result(small_grid_2d)
        fb = result.frame_betti(frame=0, dim=0)
        assert fb.shape == (5,)


# ======================================================================
# Step 4.7: Vectorizer tests
# ======================================================================

class TestMultiThresholdVectorizers:
    """Test the multi-threshold vectorization transformers."""

    def _make_result(self, grid, n_thresholds=3):
        from zztop.builders.multi_threshold import (
            run_cubical_zigzag_multi_threshold,
        )
        return run_cubical_zigzag_multi_threshold(
            grid,
            thresholds=n_thresholds,
            threshold_strategy="uniform",
            backend="python",
            n_jobs=1,
            use_gudhi=False,
        )

    def test_betti_surface_vectorizer_shape(self, small_grid_2d):
        from zztop.vectorizations.multi_threshold import BettiSurface
        result = self._make_result(small_grid_2d)
        bs = BettiSurface(dim=0, flatten=True)
        out = bs.fit_transform([result])
        assert out.shape == (1, 3 * small_grid_2d.shape[-1])

    def test_betti_surface_vectorizer_2d(self, small_grid_2d):
        from zztop.vectorizations.multi_threshold import BettiSurface
        result = self._make_result(small_grid_2d)
        bs = BettiSurface(dim=0, flatten=False)
        out = bs.fit_transform([result])
        assert out.shape == (1, 3, small_grid_2d.shape[-1])

    def test_threshold_integrated_persistence(self, small_grid_2d):
        from zztop.vectorizations.multi_threshold import (
            ThresholdIntegratedPersistence,
        )
        result = self._make_result(small_grid_2d)
        tip = ThresholdIntegratedPersistence(dim=0)
        out = tip.fit_transform([result])
        assert out.shape == (1, 3)
        assert np.all(out >= 0)

    def test_threshold_birth_frequency(self, small_grid_2d):
        from zztop.vectorizations.multi_threshold import (
            ThresholdBirthFrequency,
        )
        result = self._make_result(small_grid_2d)
        tbf = ThresholdBirthFrequency(dim=0, flatten=True)
        out = tbf.fit_transform([result])
        assert out.shape == (1, 3 * small_grid_2d.shape[-1])

    def test_sklearn_pipeline(self, small_grid_2d):
        """Vectorizers work in an sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from zztop.vectorizations.multi_threshold import BettiSurface

        result = self._make_result(small_grid_2d)
        pipe = Pipeline([
            ("betti", BettiSurface(dim=0)),
            ("scale", StandardScaler()),
        ])
        # Need at least 2 samples for StandardScaler
        X = [result, result]
        out = pipe.fit_transform(X)
        assert out.shape[0] == 2

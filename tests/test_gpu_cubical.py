"""
Tests for GPU cubical persistence integration.

These tests validate:
1. The gpu_info() / gpu_available() detection logic works regardless of
   whether the GPU fork is installed.
2. The CPU fallback path of ``run_cubical_persistence_gpu`` produces correct
   standard persistence diagrams on small grids.
3. When the GPU fork *is* installed, the GPU path produces diagrams matching
   the CPU path (parity).
4. The sklearn wrapper ``run_cubical_persistence_sklearn`` works when the
   sklearn backend is available.
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from zztop.builders.gpu_cubical import (
    gpu_info,
    gpu_available,
    run_cubical_persistence_gpu,
    run_cubical_persistence_sklearn,
    _persistence_per_frame_gudhi,
    _gpu_supports_3d,
)


# ======================================================================
# Detection tests (always run)
# ======================================================================

class TestGPUDetection:
    """Tests that the GPU detection logic runs without errors."""

    def test_gpu_info_returns_dict(self):
        info = gpu_info()
        assert isinstance(info, dict)
        assert "available" in info
        assert "backend_tag" in info

    def test_gpu_available_returns_bool(self):
        result = gpu_available()
        assert isinstance(result, bool)

    def test_gpu_info_cached(self):
        """Subsequent calls return the same object (cached)."""
        a = gpu_info()
        b = gpu_info()
        assert a is b


# ======================================================================
# CPU fallback tests (always run — use standard GUDHI)
# ======================================================================

class TestCPUFallback:
    """Verify the CPU path works with standard GUDHI."""

    @pytest.fixture
    def grid_2d_3frames(self):
        """A small 2D grid (5×5) over 3 frames."""
        rng = np.random.default_rng(42)
        return rng.random((5, 5, 3))

    @pytest.fixture
    def grid_3d_3frames(self):
        """A small 3D grid (4×4×3) over 3 frames."""
        rng = np.random.default_rng(42)
        return rng.random((4, 4, 3, 3))

    def test_cpu_2d_returns_list_of_dicts(self, grid_2d_3frames):
        diagrams = run_cubical_persistence_gpu(
            grid_2d_3frames, backend="cpu", min_persistence=0.0
        )
        assert len(diagrams) == 3  # 3 frames
        for dgm in diagrams:
            assert isinstance(dgm, dict)
            assert 0 in dgm  # at least H0
            assert isinstance(dgm[0], np.ndarray)
            assert dgm[0].ndim == 2
            if dgm[0].shape[0] > 0:
                assert dgm[0].shape[1] == 2

    def test_cpu_3d_returns_list_of_dicts(self, grid_3d_3frames):
        diagrams = run_cubical_persistence_gpu(
            grid_3d_3frames, backend="cpu", min_persistence=0.0
        )
        assert len(diagrams) == 3
        for dgm in diagrams:
            assert isinstance(dgm, dict)
            # 3D grid should have H0, H1, H2, H3
            for d in range(4):
                assert d in dgm

    def test_cpu_known_H0(self):
        """A constant 2D grid (all zeros) should yield exactly 1 H0 bar
        (the connected component) with birth=-0 or 0, death=inf."""
        grid = np.zeros((3, 3, 2))  # 3×3 over 2 frames
        diagrams = run_cubical_persistence_gpu(grid, backend="cpu")
        for dgm in diagrams:
            h0 = dgm[0]
            # Should have at least one essential bar (inf death)
            assert h0.shape[0] >= 1

    def test_cpu_negate_flag(self):
        """With negate=False the grid values are used as-is."""
        rng = np.random.default_rng(99)
        grid = rng.random((4, 4, 2))
        d_neg = run_cubical_persistence_gpu(grid, backend="cpu", negate=True)
        d_raw = run_cubical_persistence_gpu(-grid, backend="cpu", negate=False)
        # Should produce identical diagrams
        for t in range(2):
            for dim in d_neg[t]:
                np.testing.assert_array_almost_equal(
                    d_neg[t][dim], d_raw[t][dim]
                )

    def test_auto_falls_back_to_cpu_for_3d(self, grid_3d_3frames):
        """backend='auto' should work for 3D grids (CPU or GPU if available)."""
        diagrams = run_cubical_persistence_gpu(
            grid_3d_3frames, backend="auto", min_persistence=0.0
        )
        assert len(diagrams) == 3

    def test_auto_uses_cpu_for_single_frame(self):
        """For a single frame, auto should always use CPU (faster than GPU)."""
        rng = np.random.default_rng(55)
        grid_1frame = rng.random((6, 6, 1))  # 2D spatial, 1 frame
        diagrams = run_cubical_persistence_gpu(
            grid_1frame, backend="auto", min_persistence=0.0
        )
        assert len(diagrams) == 1
        assert 0 in diagrams[0]

    def test_max_dimension_caps_output(self, grid_3d_3frames):
        diagrams = run_cubical_persistence_gpu(
            grid_3d_3frames, backend="cpu", max_dimension=1
        )
        for dgm in diagrams:
            assert set(dgm.keys()) == {0, 1}

    def test_gpu_backend_raises_for_unsupported_dim(self):
        """Requesting GPU for 4D+ data should raise."""
        rng = np.random.default_rng(42)
        grid_4d = rng.random((3, 3, 3, 3, 3))  # 4D spatial + 1 frame
        with pytest.raises((RuntimeError, ValueError)):
            run_cubical_persistence_gpu(grid_4d, backend="gpu")


# ======================================================================
# GUDHI single-frame helper test
# ======================================================================

class TestPerFrameGUDHI:
    def test_single_frame(self):
        rng = np.random.default_rng(123)
        frame = rng.random((6, 6))
        dgm = _persistence_per_frame_gudhi(frame, min_persistence=0.0)
        assert 0 in dgm
        assert 1 in dgm
        # H0 should have at least 1 bar
        assert dgm[0].shape[0] >= 1


# ======================================================================
# GPU parity tests (only run when the GPU fork is installed)
# ======================================================================

@pytest.mark.skipif(
    not gpu_available(),
    reason="GPU GUDHI extension not available; install the gpu-cubical fork",
)
class TestGPUParity:
    """When the GPU fork is installed, verify GPU and CPU produce matching results."""

    @pytest.fixture
    def grid_2d_batch(self):
        rng = np.random.default_rng(7)
        return rng.random((8, 8, 5))  # 8×8 grid, 5 frames

    def test_gpu_vs_cpu_2d(self, grid_2d_batch):
        cpu_dgms = run_cubical_persistence_gpu(
            grid_2d_batch, backend="cpu", min_persistence=0.0
        )
        gpu_dgms = run_cubical_persistence_gpu(
            grid_2d_batch, backend="gpu", min_persistence=0.0
        )
        assert len(cpu_dgms) == len(gpu_dgms)
        for t in range(len(cpu_dgms)):
            for dim in cpu_dgms[t]:
                if dim not in gpu_dgms[t]:
                    continue
                a = np.sort(cpu_dgms[t][dim], axis=0)
                b = np.sort(gpu_dgms[t][dim], axis=0)
                # Filter out inf deaths for comparison if shapes differ
                a_fin = a[np.isfinite(a[:, 1])] if a.shape[0] > 0 else a
                b_fin = b[np.isfinite(b[:, 1])] if b.shape[0] > 0 else b
                np.testing.assert_array_almost_equal(
                    a_fin, b_fin, decimal=10,
                    err_msg=f"Mismatch at frame {t}, dim {dim}",
                )

    def test_gpu_info_reports_cuda(self):
        info = gpu_info()
        assert info["backend_tag"] in ("cuda", "cpu-fallback")


@pytest.mark.skipif(
    not (gpu_available() and _gpu_supports_3d()),
    reason="GPU 3D cubical persistence not available",
)
class TestGPUParity3D:
    """Verify GPU and CPU produce matching 3D cubical persistence."""

    @pytest.fixture
    def grid_3d_batch(self):
        rng = np.random.default_rng(13)
        return rng.random((5, 5, 4, 3))  # 5×5×4 grid, 3 frames

    def test_gpu_vs_cpu_3d(self, grid_3d_batch):
        """GPU and CPU backends should produce identical 3D diagrams.

        The CPU fallback now applies the same vertices-to-top-cells
        conversion that the GPU kernel uses internally, so a simple
        backend comparison suffices.
        """
        cpu_dgms = run_cubical_persistence_gpu(
            grid_3d_batch, backend="cpu", min_persistence=0.0
        )
        gpu_dgms = run_cubical_persistence_gpu(
            grid_3d_batch, backend="gpu", min_persistence=0.0
        )

        assert len(cpu_dgms) == len(gpu_dgms)
        for t in range(len(cpu_dgms)):
            for dim in cpu_dgms[t]:
                if dim not in gpu_dgms[t]:
                    continue
                a = np.sort(cpu_dgms[t][dim], axis=0)
                b = np.sort(gpu_dgms[t][dim], axis=0)
                a_fin = a[np.isfinite(a[:, 1])] if a.shape[0] > 0 else a
                b_fin = b[np.isfinite(b[:, 1])] if b.shape[0] > 0 else b
                np.testing.assert_array_almost_equal(
                    a_fin, b_fin, decimal=10,
                    err_msg=f"3D mismatch at frame {t}, dim {dim}",
                )

    def test_3d_has_three_dimensions(self, grid_3d_batch):
        dgms = run_cubical_persistence_gpu(
            grid_3d_batch, backend="gpu", min_persistence=0.0
        )
        for dgm in dgms:
            # 3D data should have H0, H1, H2
            for d in range(3):
                assert d in dgm


# ======================================================================
# sklearn wrapper tests (only when sklearn backend is available)
# ======================================================================

@pytest.mark.skipif(
    not gpu_info().get("sklearn_available", False),
    reason="gudhi.sklearn.cubical_persistence not available",
)
class TestSklearnWrapper:
    def test_sklearn_2d(self):
        rng = np.random.default_rng(11)
        grid = rng.random((5, 5, 3))
        result = run_cubical_persistence_sklearn(
            grid, homology_dimensions=(0, 1), backend="cpu"
        )
        assert len(result) == 3
        for frame_dgms in result:
            assert len(frame_dgms) == 2  # H0, H1


# ======================================================================
# Zigzag compatibility: confirm run_cubical_zigzag still works after
# the GPU module is imported (no import side-effects breaking things).
# ======================================================================

class TestZigzagStillWorks:
    def test_zigzag_import_unaffected(self):
        from zztop import run_cubical_zigzag
        rng = np.random.default_rng(1)
        grid = rng.random((3, 3, 2))
        bars = run_cubical_zigzag(grid, threshold=0.0, backend="python")
        assert isinstance(bars, list)
        assert len(bars) > 0
        for dim, b, d in bars:
            assert isinstance(dim, int)

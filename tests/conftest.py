"""Shared fixtures and helpers for the zztop test suite."""

import pytest
import sys
import os
from collections import Counter

# Ensure the zztop package is importable when running tests from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ======================================================================
# Persistence-diagram helpers
# ======================================================================

def bars_to_diagrams(bars):
    """Convert engine output to per-dimension persistence diagrams.

    Parameters
    ----------
    bars : list of (dim, birth, death)
        Output of ``ZigzagEngine.run()`` or equivalent.

    Returns
    -------
    diagrams : dict[int, list[tuple[int, int]]]
        ``diagrams[k]`` is a sorted list of ``(birth, death)`` points
        for homological dimension *k*.
    """
    from collections import defaultdict
    dgms = defaultdict(list)
    for dim, b, d in bars:
        dgms[dim].append((b, d))
    return {k: sorted(v) for k, v in dgms.items()}


def raw_bars_to_diagrams(raw_bars):
    """Convert raw ``(birth, death, dim)`` tuples (fzz/pyfzz convention)
    to per-dimension persistence diagrams.

    Parameters
    ----------
    raw_bars : list/set of (birth, death, dim)

    Returns
    -------
    diagrams : dict[int, list[tuple[int, int]]]
    """
    from collections import defaultdict
    dgms = defaultdict(list)
    for b, d, dim in raw_bars:
        dgms[dim].append((b, d))
    return {k: sorted(v) for k, v in dgms.items()}


def diagrams_equal(dgm_a, dgm_b):
    """Check if two persistence diagram dicts are equal as multisets.

    Returns *True* iff for every dimension the sorted lists of
    ``(birth, death)`` points match exactly.
    """
    all_dims = set(dgm_a) | set(dgm_b)
    for d in all_dims:
        if sorted(dgm_a.get(d, [])) != sorted(dgm_b.get(d, [])):
            return False
    return True


def diagram_betti_at(diagrams, dim, layer):
    """Betti number of dimension *dim* at zigzag *layer* (1-based).

    Counts the number of bars ``(b, d)`` in ``diagrams[dim]`` with
    ``b <= layer <= d``.
    """
    return sum(
        1 for b, d in diagrams.get(dim, []) if b <= layer <= d
    )


def diagram_total_persistence(diagrams, dim):
    """Sum of ``(death - birth)`` for all bars in ``diagrams[dim]``
    with ``death > birth`` (skip zero-length bars).
    """
    return sum(d - b for b, d in diagrams.get(dim, []) if d > b)


def assert_diagrams_equal(dgm_a, dgm_b, *, msg=""):
    """Assert two persistence diagrams are equal, with informative diff."""
    all_dims = sorted(set(dgm_a) | set(dgm_b))
    for d in all_dims:
        pts_a = sorted(dgm_a.get(d, []))
        pts_b = sorted(dgm_b.get(d, []))
        assert pts_a == pts_b, (
            f"{msg}Dimension {d} mismatch:\n"
            f"  got      {pts_a}\n"
            f"  expected {pts_b}"
        )

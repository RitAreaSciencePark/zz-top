"""
Pure-Python implementation of the Dey-Hou fast zigzag algorithm.

This module implements the Dey-Hou ESA 2022 algorithm entirely in Python
over GF(2), written from the paper.  It is the package's only engine: the
authors' reference implementation ``fzz`` is not redistributed (its terms
restrict it to academic use), and the output of this module was verified
against it bar for bar before the vendored copy was removed
(``docs/parity_with_fzz.md``).  Operation-stream and interval conventions
follow ``fzz`` so that results stay directly comparable.

Algorithm overview
------------------
1.  Convert the simplex-wise zigzag filtration into an equivalent
    *non-zigzag* (standard) filtration of a Delta-complex by building a
    **cone** on the deletions.
2.  Reduce the resulting boundary matrix using column reduction with the
    **twist** optimisation (process columns by anti-transpose order,
    skip columns whose anti-diagonal entry is already a pivot of an
    earlier column).
3.  Map the persistence pairs of the cone back to zigzag intervals.

Reference:
    Dey, T.K. & Hou, T. "Fast Computation of Zigzag Persistence."
    European Symposium on Algorithms (ESA), 2022.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple


# ======================================================================
# Low-level column algebra over GF(2)
# ======================================================================

def _pivot(col: Set[int]) -> int:
    """Return the largest index in *col* (the pivot), or -1 if empty."""
    return max(col) if col else -1


def _add_cols(a: Set[int], b: Set[int]) -> Set[int]:
    """Symmetric difference = addition over GF(2)."""
    return a ^ b


# ======================================================================
# Boundary matrix reduction (twist algorithm)
# ======================================================================

def _reduce_boundary_matrix(
    columns: List[Set[int]],
    dims: List[int],
) -> List[Tuple[int, int]]:
    """Reduce *columns* in-place and return ``(birth_col, death_col)`` pairs.

    Parameters
    ----------
    columns : list of set[int]
        ``columns[j]`` is the set of row indices of column *j*.
    dims : list of int
        ``dims[j]`` is the dimension of column *j*.

    Returns
    -------
    pairs : list of (int, int)
        Each ``(b, d)`` means column *b* is paired with column *d*.
    """
    n = len(columns)
    # pivot_to_col[p] = j  means column j has pivot row p
    pivot_to_col: Dict[int, int] = {}
    pairs: List[Tuple[int, int]] = []

    # Twist optimisation: we process columns in order 0..n-1.
    # Before reducing, mark columns that are already known to be zero
    # (they are the birth-columns of pairs identified so far).
    paired_as_birth: Set[int] = set()

    for j in range(n):
        if j in paired_as_birth:
            # This column's pivot was already claimed → skip
            continue

        col = columns[j]
        while True:
            p = _pivot(col)
            if p < 0:
                break
            if p in pivot_to_col:
                col = _add_cols(col, columns[pivot_to_col[p]])
            else:
                break

        columns[j] = col
        p = _pivot(col)

        if p >= 0:
            pivot_to_col[p] = j
            pairs.append((p, j))
            paired_as_birth.add(p)

    return pairs


# ======================================================================
# Dey-Hou cone construction + mapping
# ======================================================================

def compute_zigzag_python(
    operations: List[Tuple[str, int, List[int]]],
) -> List[Tuple[int, int, int]]:
    """Compute zigzag persistence entirely in Python.

    Parameters
    ----------
    operations : list of (op, dim, boundary_filt_indices)
        Identical format to the *abstract* C++ interface.

        * For insertions (``op='i'``): ``boundary_filt_indices`` is the
          sorted list of **operation indices** (0-based) of boundary
          cells that have already been inserted.
        * For deletions (``op='d'``): ``boundary_filt_indices`` is
          ``[insertion_op_index]`` pointing to the operation where the
          cell was originally inserted.

    Returns
    -------
    bars : list of (birth, death, dimension)
        Persistence barcode (1-based indexing, closed intervals, matching
        the convention of the C++ backend).
    """
    n = len(operations)
    if n == 0:
        return []

    # ----- Phase 1: build the upper half (insertions) ------------------
    # Column 0 = Omega vertex (empty boundary, dim 0)
    # Columns 1.. = one per insertion, in filtration order.
    # We also record deletion ids for later.

    simp_num = sum(1 for op, _, _ in operations if op == "i")

    # Map: filt_index → column id (for insertions)
    filt_to_col: Dict[int, int] = {}

    # columns and dims for the full cone boundary matrix
    total_cols = 2 * simp_num + 1
    columns: List[Set[int]] = [set() for _ in range(total_cols)]
    dims: List[int] = [0] * total_cols

    # Column 0 = Omega vertex
    # (already empty set, dim 0)

    orig_f_add_id: List[int] = []
    orig_f_del_id: List[int] = []
    del_ids: List[int] = []

    s_id = 1
    orig_f_id = 0

    for i, (op, dim, bdry) in enumerate(operations):
        if op == "i":
            # Build boundary column from previously-inserted boundary cells
            col: Set[int] = set()
            for b_filt_idx in bdry:
                col_id = filt_to_col.get(b_filt_idx)
                if col_id is not None:
                    col.add(col_id)
            columns[s_id] = col
            dims[s_id] = dim
            filt_to_col[i] = s_id
            orig_f_add_id.append(orig_f_id)
            s_id += 1
        else:
            # Deletion
            ins_idx = bdry[0] if bdry else -1
            col_id = filt_to_col.pop(ins_idx, -1) if ins_idx >= 0 else -1
            if col_id > 0:
                del_ids.append(col_id)
            orig_f_del_id.append(orig_f_id)
        orig_f_id += 1

    # ----- Clean up remaining (un-deleted) cells -----------------------
    # Collect remaining cells sorted by dimension descending
    remaining = []
    del_id_set = set(del_ids)
    for filt_idx, col_id in sorted(filt_to_col.items()):
        if col_id not in del_id_set:
            dim_val = operations[filt_idx][1]
            remaining.append((dim_val, col_id))
    remaining.sort(key=lambda x: -x[0])

    for dim_val, col_id in remaining:
        del_ids.append(col_id)
        orig_f_del_id.append(orig_f_id)
        orig_f_id += 1

    if len(del_ids) != simp_num:
        raise RuntimeError(
            f"Cone construction mismatch: del_ids={len(del_ids)} != simp_num={simp_num}"
        )

    # ----- Phase 2: build the cone (lower half) ------------------------
    cone_sid: Dict[int, int] = {}

    for del_id in reversed(del_ids):
        col: Set[int] = {del_id}

        orig_col = columns[del_id]
        if not orig_col:
            col.add(0)  # cone with Omega for 0-cells
        else:
            for b in orig_col:
                if b in cone_sid:
                    col.add(cone_sid[b])

        columns[s_id] = col
        # The cone over a k-cell is a (k+1)-cell; |col| - 1 is only right for simplices
        # (a k-cube has 2k faces).
        dims[s_id] = dims[del_id] + 1
        cone_sid[del_id] = s_id
        s_id += 1

    # ----- Phase 3: reduce & extract pairs -----------------------------
    pairs = _reduce_boundary_matrix(columns, dims)

    # ----- Phase 4: map pairs back to zigzag intervals -----------------
    bars: List[Tuple[int, int, int]] = []
    filt_size = n

    for b, d in pairs:
        d_mapped = d - 1
        p = dims[b]

        if d_mapped < simp_num:
            # Ordinary interval
            b_new = orig_f_add_id[b - 1] + 1
            d_new = orig_f_add_id[d_mapped]
        else:
            # Relative / extended interval
            b_new, d_new, p = _map_rel_ext(
                p, b, d_mapped, simp_num, orig_f_add_id, orig_f_del_id
            )

        if b_new > filt_size:
            continue
        if d_new > filt_size:
            d_new = filt_size

        bars.append((b_new, d_new, p))

    return bars


def _map_rel_ext(
    p: int,
    b: int,
    d: int,
    simp_num: int,
    orig_f_add_id: List[int],
    orig_f_del_id: List[int],
) -> Tuple[int, int, int]:
    """Map a relative/extended interval back to the zigzag.

    The interval map of Dey-Hou (the relative-to-extended-to-zigzag step of
    the paper; ``mapRelExtIntv`` in the reference implementation's interface).
    """
    if b > simp_num:
        # Open-closed: swap and map
        b, d = d, b
        b = 3 * simp_num - b
        d = 3 * simp_num - d
        p -= 1
        b_new = orig_f_del_id[b - 1 - simp_num] + 1
        d_new = orig_f_del_id[d - simp_num]
    else:
        # Closed-closed
        d = 3 * simp_num - d - 1
        b_new = orig_f_add_id[b - 1]
        d_new = orig_f_del_id[d - simp_num]
        if b_new < d_new:
            b_new = b_new + 1
        else:
            b_new, d_new = d_new, b_new
            b_new = b_new + 1
            p = p - 1
    return b_new, d_new, p


# ======================================================================
# Convenience: simplicial input (vertex lists)
# ======================================================================

def compute_simplicial_python(
    operations: List[Tuple[str, List[int]]],
) -> List[Tuple[int, int, int]]:
    """Compute zigzag persistence from a simplicial ``(op, vertex_list)``
    stream, entirely in Python.

    This converts the simplicial format to the abstract format (computing
    boundaries from the vertex lists) and then calls
    :func:`compute_zigzag_python`.
    """
    # First pass: assign operation indices, track insertions
    abstract_ops: List[Tuple[str, int, List[int]]] = []
    # cell_key (tuple of vertices) → operation index of current insertion
    cell_insert: Dict[tuple, int] = {}

    for i, (op, verts) in enumerate(operations):
        key = tuple(sorted(verts))
        dim = len(key) - 1

        if op == "i":
            # Compute boundary: remove each vertex in turn
            bdry_indices: List[int] = []
            if dim > 0:
                for j in range(len(key)):
                    face = key[:j] + key[j + 1:]
                    if face in cell_insert:
                        bdry_indices.append(cell_insert[face])
                bdry_indices.sort()
            abstract_ops.append(("i", dim, bdry_indices))
            cell_insert[key] = i
        else:
            ins_idx = cell_insert.pop(key, -1)
            abstract_ops.append(("d", dim, [ins_idx] if ins_idx >= 0 else []))

    return compute_zigzag_python(abstract_ops)

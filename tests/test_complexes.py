"""Tests for the fundamental complex data structures."""

import pytest
from zztop.complex.simplicial import Simplex, SimplicialComplex
from zztop.complex.cubical import CubicalCell, CubicalComplex


# =====================================================================
# Simplex tests
# =====================================================================

class TestSimplex:
    def test_sorted_vertices(self):
        s = Simplex([3, 1, 2])
        assert s.vertices == (1, 2, 3)

    def test_dimension(self):
        assert Simplex([0]).dimension == 0
        assert Simplex([0, 1]).dimension == 1
        assert Simplex([0, 1, 2]).dimension == 2

    def test_faces(self):
        s = Simplex([0, 1, 2])
        faces = s.faces()
        assert len(faces) == 3
        expected = {Simplex([1, 2]), Simplex([0, 2]), Simplex([0, 1])}
        assert set(faces) == expected

    def test_vertex_no_faces(self):
        assert Simplex([5]).faces() == []

    def test_closure(self):
        s = Simplex([0, 1, 2])
        closure = s.closure()
        # 3 vertices + 3 edges + 1 triangle = 7
        assert len(closure) == 7

    def test_hash_equality(self):
        a = Simplex([2, 0])
        b = Simplex([0, 2])
        assert a == b
        assert hash(a) == hash(b)
        assert {a, b} == {a}

    def test_ordering(self):
        s1 = Simplex([0])
        s2 = Simplex([0, 1])
        assert s1 < s2  # dim 0 < dim 1


class TestSimplicialComplex:
    def test_closure_property(self):
        sc = SimplicialComplex([[0, 1, 2]])
        assert len(sc) == 7  # 3 + 3 + 1

    def test_contains(self):
        sc = SimplicialComplex([[0, 1]])
        assert Simplex([0]) in sc
        assert Simplex([1]) in sc
        assert Simplex([0, 1]) in sc
        assert Simplex([2]) not in sc

    def test_cells_by_dimension(self):
        sc = SimplicialComplex([[0, 1, 2]])
        cbd = sc.cells_by_dimension()
        assert len(cbd[0]) == 3
        assert len(cbd[1]) == 3
        assert len(cbd[2]) == 1

    def test_boundary_interface(self):
        sc = SimplicialComplex([[0, 1]])
        bdry = sc.boundary(Simplex([0, 1]))
        assert len(bdry) == 2


# =====================================================================
# CubicalCell tests
# =====================================================================

class TestCubicalCell:
    def test_dimension_vertex(self):
        c = CubicalCell(((0, 0), (1, 1)))
        assert c.dim == 0

    def test_dimension_edge(self):
        c = CubicalCell(((0, 1), (2, 2)))
        assert c.dim == 1

    def test_dimension_square(self):
        c = CubicalCell(((0, 1), (0, 1)))
        assert c.dim == 2

    def test_dimension_cube3d(self):
        c = CubicalCell(((0, 1), (0, 1), (0, 1)))
        assert c.dim == 3

    def test_boundary_edge(self):
        c = CubicalCell(((0, 1), (2, 2)))
        bdry = c.boundary()
        assert len(bdry) == 2
        expected = {
            CubicalCell(((0, 0), (2, 2))),
            CubicalCell(((1, 1), (2, 2))),
        }
        assert set(bdry) == expected

    def test_boundary_square(self):
        c = CubicalCell(((0, 1), (0, 1)))
        bdry = c.boundary()
        assert len(bdry) == 4
        # 4 edges: 2 horizontal + 2 vertical
        expected = {
            CubicalCell(((0, 0), (0, 1))),  # left
            CubicalCell(((1, 1), (0, 1))),  # right
            CubicalCell(((0, 1), (0, 0))),  # bottom
            CubicalCell(((0, 1), (1, 1))),  # top
        }
        assert set(bdry) == expected

    def test_boundary_vertex(self):
        c = CubicalCell(((3, 3), (4, 4)))
        assert c.boundary() == []

    def test_closure_square(self):
        c = CubicalCell(((0, 1), (0, 1)))
        closure = c.closure()
        # 4 vertices + 4 edges + 1 square = 9
        assert len(set(closure)) == 9

    def test_hash_equality(self):
        a = CubicalCell(((0, 1), (2, 2)))
        b = CubicalCell(((0, 1), (2, 2)))
        assert a == b
        assert hash(a) == hash(b)


class TestCubicalComplex:
    def test_closure_property(self):
        cc = CubicalComplex([CubicalCell(((0, 1), (0, 1)))])
        # 4 vertices + 4 edges + 1 square = 9
        assert len(cc) == 9

    def test_from_grid_shape_2d(self):
        cc = CubicalComplex.from_grid_shape((3, 3))
        # 2x2 grid of squares: 9 verts + 12 edges + 4 squares = 25
        assert len(cc) == 25

    def test_from_grid_shape_3d(self):
        cc = CubicalComplex.from_grid_shape((2, 2, 2))
        # 1 cube: 8 verts + 12 edges + 6 faces + 1 cube = 27
        assert len(cc) == 27

    def test_boundary_interface(self):
        cc = CubicalComplex([CubicalCell(((0, 1), (0, 1)))])
        edge = CubicalCell(((0, 1), (0, 0)))
        bdry = cc.boundary(edge)
        assert len(bdry) == 2

    def test_cells_by_dimension(self):
        cc = CubicalComplex.from_grid_shape((3, 3))
        cbd = cc.cells_by_dimension()
        assert len(cbd[0]) == 9
        assert len(cbd[1]) == 12
        assert len(cbd[2]) == 4

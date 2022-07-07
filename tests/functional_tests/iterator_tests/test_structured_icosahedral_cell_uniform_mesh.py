import re
from dataclasses import dataclass

import pytest


# vertices have 1 color
# cells have 2 colors (0 - downward, 1 - upward)
# edges have 3 colors (0 - horizontal, 1 - vertical, 2 - diagonal)
#
#  ___0e__
#  |  0c /|
#  |    / |
# 1e| 2e/  |
#  |  /   |
#  | / 1c |
#  |/_____|
@dataclass
class StructuredIcosahedralCellUniformMesh:
    """Cell because each cell has the full set of neighbors.
    Uniform because we make each color dimension the same size:
    #c0-edges == #c1-edges == #c2-edges"""

    i_size: int  # number of edges in i direction
    j_size: int  # number of edges in j direction
    # => i_size * j_size = number of cells / 2
    # => (i_size + 1) * (j_size + 1) = number of vertices

    @staticmethod
    def _i_j_index(index, i_size):
        return (index % i_size, index // i_size)

    def vertex_index(self, i, j):
        assert i <= self.i_size and j <= self.j_size
        return i + j * (self.i_size + 1)

    def from_vertex_index(self, v):
        return self._i_j_index(v, self.i_size + 1)

    def edge_index(self, i, j, c):
        assert i <= self.i_size and j <= self.j_size
        return (self.i_size + 1) * (self.j_size + 1) * c + (self.i_size + 1) * j + i

    def edge_index_safe(self, i, j, c):
        if i < 0 or j < 0:
            return -1
        if c == 0 and (i >= self.i_size or j > self.j_size):
            return -1
        if c == 1 and (i > self.i_size or j >= self.j_size):
            return -1
        if c == 2 and (i >= self.i_size or j >= self.j_size):
            return -1
        return self.edge_index(i, j, c)

    def from_edge_index(self, e):
        ...

    def cell_index(self, i, j, c):
        return c * self.i_size * self.j_size + j * self.i_size + i

    def v2e(self, v):
        i, j = self.from_vertex_index(v)
        e = self.edge_index_safe
        return (
            e(i, j, 0),
            e(i, j, 1),
            e(i - 1, j, 2),
            e(i - 1, j, 0),
            e(i, j - 1, 1),
            e(i, j - 1, 2),
        )


@pytest.fixture
def mesh():
    return StructuredIcosahedralCellUniformMesh(3, 2)


def test_vertex_index(mesh):
    assert mesh.vertex_index(0, 0) == 0
    assert mesh.vertex_index(1, 0) == 1
    assert mesh.vertex_index(0, 1) == 4
    with pytest.raises(AssertionError):
        mesh.vertex_index(4, 2)


def test_edge_index(mesh):
    assert mesh.edge_index(0, 0, 0) == 0
    # assert mesh.from_edge_index(0) == (0, 0, 0)
    assert mesh.edge_index(1, 0, 0) == 1
    # assert mesh.from_edge_index(1) == (1, 0, 0)
    assert mesh.edge_index(0, 1, 0) == 4
    # assert mesh.from_edge_index(3) == (0, 1, 0)
    assert mesh.edge_index(0, 0, 1) == 12
    # assert mesh.from_edge_index(9) == (0, 0, 1)
    assert mesh.edge_index(0, 1, 1) == 16
    # assert mesh.from_edge_index(13) == (0, 1, 1)
    assert mesh.edge_index(2, 1, 2) == 30
    # assert mesh.from_edge_index(22) == (2, 1, 2)
    assert mesh.edge_index(3, 0, 0) == 3
    assert mesh.edge_index_safe(3, 0, 0) == -1


def test_cell_index(mesh):
    assert mesh.cell_index(0, 0, 0) == 0
    assert mesh.cell_index(1, 0, 0) == 1
    assert mesh.cell_index(0, 1, 0) == 3
    assert mesh.cell_index(0, 0, 1) == 6


def test_v2e(mesh: StructuredIcosahedralCellUniformMesh):
    assert mesh.v2e(mesh.vertex_index(0, 0)) == (0, 12, -1, -1, -1, -1)
    assert mesh.v2e(mesh.vertex_index(1, 1)) == (5, 17, 28, 4, 13, 25)
    assert mesh.v2e(mesh.vertex_index(3, 2)) == (-1, -1, -1, 10, 19, -1)

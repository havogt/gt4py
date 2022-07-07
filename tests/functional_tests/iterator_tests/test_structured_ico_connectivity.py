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
class StructuredIcosahedralMesh:
    i_size: int  # number of edges in i direction
    j_size: int  # number of edges in j direction
    # => i_size * j_size = number of cells / 2
    # => (i_size + 1) * (j_size + 1) = number of vertices

    def vertex_index(self, i, j):
        assert i <= self.i_size and j <= self.j_size
        return i + j * (self.i_size + 1)

    @property
    def _edge_c0_size(self):
        return self.i_size * (self.j_size + 1)

    @property
    def _edge_c1_size(self):
        return (self.i_size + 1) * self.j_size

    @staticmethod
    def _i_j_index(index, i_size):
        return (index % i_size, index // i_size)

    def edge_index(self, i, j, c):
        if c == 0:
            assert i < self.i_size and j < self.j_size + 1
            return i + j * self.i_size
        if c == 1:
            assert i < self.i_size + 1 and j < self.j_size
            return self._edge_c0_size + i + j * (self.i_size + 1)
        if c == 2:
            assert i < self.i_size and j < self.j_size
            return self._edge_c0_size + self._edge_c1_size + i + j * self.i_size

    def from_edge_index(self, e):
        c = 0
        if e - self._edge_c0_size >= 0:
            e -= self._edge_c0_size
            c += 1
        else:
            return (*self._i_j_index(e, self.i_size), c)
        if e - self._edge_c1_size >= 0:
            e -= self._edge_c1_size
            c += 1
        else:
            return (*self._i_j_index(e, self.i_size + 1), c)
        return (*self._i_j_index(e, self.i_size), c)

    def cell_index(self, i, j, c):
        return c * self.i_size * self.j_size + j * self.i_size + i

    def v2e(self):
        ...


@pytest.fixture
def mesh():
    return StructuredIcosahedralMesh(3, 2)


def test_vertex_index(mesh):
    assert mesh.vertex_index(0, 0) == 0
    assert mesh.vertex_index(1, 0) == 1
    assert mesh.vertex_index(0, 1) == 4
    with pytest.raises(AssertionError):
        mesh.vertex_index(4, 2)


def test_edge_index(mesh):
    assert mesh.edge_index(0, 0, 0) == 0
    assert mesh.from_edge_index(0) == (0, 0, 0)
    assert mesh.edge_index(1, 0, 0) == 1
    assert mesh.from_edge_index(1) == (1, 0, 0)
    assert mesh.edge_index(0, 1, 0) == 3
    assert mesh.from_edge_index(3) == (0, 1, 0)
    assert mesh.edge_index(0, 0, 1) == 9
    assert mesh.from_edge_index(9) == (0, 0, 1)
    assert mesh.edge_index(0, 1, 1) == 13
    assert mesh.from_edge_index(13) == (0, 1, 1)
    assert mesh.edge_index(2, 1, 2) == 22
    assert mesh.from_edge_index(22) == (2, 1, 2)


def test_cell_index(mesh):
    assert mesh.cell_index(0, 0, 0) == 0
    assert mesh.cell_index(1, 0, 0) == 1
    assert mesh.cell_index(0, 1, 0) == 3
    assert mesh.cell_index(0, 0, 1) == 6

from functional.language_server.hover import hover_info
from functional.language_server.parser import parse_ffront


source = """
import numpy as np

from functional.common import Field
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import FieldOffset
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import CartesianAxis, offset


IDim = CartesianAxis("IDim")
JDim = CartesianAxis("JDim")

Ioff = FieldOffset("Ioff", source=IDim, target=(IDim,))
Joff = FieldOffset("Joff", source=JDim, target=(JDim,))


@field_operator
def lap(
    in_field: Field[[IDim, JDim], "float"], a4: Field[[IDim, JDim], "float"]
) -> Field[[IDim, JDim], "float"]:
    return (
        a4 * in_field
        + in_field(Ioff[1])
        + in_field(Joff[1])
        + in_field(Ioff[-1]) 
        + in_field(Joff[-1])  
    )
 
@program
def lap_program(
    in_field: Field[[IDim, JDim], "float"],
    a4: Field[[IDim, JDim], "float"],
    out_field: Field[[IDim, JDim], "float"],
):
    lap(in_field, a4, out=out_field[1:-1, 1:-1])


def expected_solution(inp):
    return -4.0 * inp[1:-1, 1:-1] + inp[:-2, 1:-1] + inp[2:, 1:-1] + inp[1:-1, :-2] + inp[1:-1, 2:]


def test_ffront_lap():
    shape = (20, 20)
    as_ij = np_as_located_field(IDim, JDim)
    input = as_ij(np.fromfunction(lambda x, y: x ** 2 + y ** 2, shape))
    a4 = as_ij(np.ones(shape) * -4.0)  # TODO support scalar field

    result = as_ij(np.zeros_like(input))
    lap_program(input, a4, result, offset_provider={"Ioff": IDim, "Joff": JDim})

    assert np.allclose(np.asarray(result)[1:-1, 1:-1], expected_solution(np.asarray(input)))
"""


def test_parsing():
    nodes = parse_ffront("<foo>", source)
    assert len(nodes) == 2


def test_hover():
    nodes = parse_ffront("<foo>", source)
    hover = hover_info(nodes, 19, 6)

    assert hover
    assert hover.contents == "Field[[IDim, JDim], float64]"
    assert hover.range.start.line == 19
    assert hover.range.start.character == 4
    assert hover.range.end.line == 19
    assert hover.range.end.character == 42

    hover = hover_info(nodes, 26, 9)
    assert hover
    assert hover.contents == "Field[[IDim, JDim], float64]"
    assert hover.range.start.line == 22
    assert hover.range.start.character == 8
    assert hover.range.end.line == 26
    assert hover.range.end.character == 28
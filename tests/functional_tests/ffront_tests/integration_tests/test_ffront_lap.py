import numpy as np

from functional.common import Field
from functional.ffront.decorator import field_operator, program
from functional.ffront.fbuiltins import Dimension, FieldOffset
from functional.iterator.embedded import np_as_located_field


def field_operator(fun):
    def impl(*args, **kwargs):
        if "out" in kwargs:
            kwargs["out"] = fun(*args)
        else:
            return fun(*args)

    return impl


offset_provider = None


def program(fun):
    def impl(*args, **kwargs):
        global offset_provider
        offset_provider = kwargs["offset_provider"]
        fun(*args)

    return impl


IDim = Dimension("IDim")
JDim = Dimension("JDim")

Ioff = FieldOffset("Ioff", source=IDim, target=(IDim,))
Joff = FieldOffset("Joff", source=JDim, target=(JDim,))


class NpField:
    def __init__(self, buffer: np.ndarray, axes, *, origin: dict[str, int] = None):
        self.buffer = buffer
        self.axes = axes
        if origin is not None:
            assert axes == tuple(origin.keys())
        self.origin = origin or {a: 0 for a in axes}

    # def __getitem__(self, indices):
    # self.buffer.re
    # return self.buffer[indices]
    # assert len(indices) == len(self.axes)
    # for i, a in enumerate(self.axes):
    #     assert isinstance(indices[i], slice)
    #     self.origin[a] += indices[i].start
    # print(self.origin)

    def __setitem__(self, indices, value):
        # self.buffer[indices] = value
        assert isinstance(value, NpField)
        self.buffer[indices] = value.buffer

    def __rmul__(self, other):
        return NpField(self.buffer * other, self.axes, origin=self.origin.copy())

    def __add__(self, other):
        if isinstance(other, NpField):
            new_origin, shape = compute_origin_and_shape(self, other)
            return NpField(
                reshape_buffer(self.buffer, self.origin, new_origin, shape)
                + reshape_buffer(other.buffer, self.origin, new_origin, shape),
                self.axes,
                origin={a: new_origin[i] for i, a in enumerate(self.axes)},
            )

        return self.buffer + other

    def __array__(self):
        return self.buffer.__array__()

    @property
    def shape(self):
        return self.buffer.shape

    def __call__(self, offset):
        assert isinstance(offset, tuple)
        tag, index = offset
        assert tag.value in offset_provider
        origin = self.origin.copy()
        # if (cutoff := origin[offset_provider[tag.value]] + index) < 0:
        #     # cutoff = self.origin[offset_provider[tag.value]] + index
        #     origin[offset_provider[tag.value]] = 0
        #     slices = tuple(
        #         slice(0, cutoff) if a == offset_provider[tag.value] else slice(None)
        #         for a in self.axes
        #     )
        #     return NpField(self.buffer[slices], self.axes, origin=origin)
        # else:
        origin[offset_provider[tag.value]] += index
        return NpField(self.buffer, self.axes, origin=origin)


def compute_dimorigin_and_size(a: tuple[int, int], b: tuple[int, int]):
    pos = min(a[0], b[0])
    size = pos + min(a[1] - a[0], b[1] - b[0])
    return pos, size


def compute_origin_and_shape(a: NpField, b: NpField):
    assert a.axes == b.axes
    pos = []
    shape = []
    for i, axis in enumerate(a.axes):
        p, s = compute_dimorigin_and_size(
            (a.origin[axis], a.shape[i]), (b.origin[axis], b.shape[i])
        )
        pos.append(p)
        shape.append(s)
    return pos, shape


def compute_slice(old_origin, new_origin, old_size, new_size):
    first = old_origin - new_origin
    last = new_size + first
    return slice(first, last)


def reshape_buffer(buffer, old_origin, new_origin, shape):
    old_origin = list(old_origin.values())
    return buffer[
        tuple(
            compute_slice(old_origin[i], new_origin[i], buffer.shape[i], shape[i])
            for i in range(len(old_origin))
        )
    ]


@field_operator
def lap(in_field: Field[[IDim, JDim], "float"]) -> Field[[IDim, JDim], "float"]:
    # tmp = -4.0 * in_field
    # tmp = tmp + in_field(Ioff[1])
    # tmp = tmp + in_field(Joff[1])
    # tmp = tmp + in_field(Ioff[-1])
    # tmp = tmp + in_field(Joff[-1])
    tmp = in_field(Ioff[1])
    return tmp
    # return (
    #     -4.0 * in_field
    #     + in_field(Ioff[1])
    #     + in_field(Joff[1])
    #     + in_field(Ioff[-1])
    #     + in_field(Joff[-1])
    # )


# @field_operator
# def laplap(in_field: Field[[IDim, JDim], "float"]) -> Field[[IDim, JDim], "float"]:
#     return lap(lap(in_field))


@program
def lap_program(
    in_field: Field[[IDim, JDim], "float"],
    out_field: Field[[IDim, JDim], "float"],
):
    # lap(in_field, out=out_field[1:-1, 1:-1])
    out_field[1:-1, 1:-1] = lap(in_field)


# @program
# def laplap_program(
#     in_field: Field[[IDim, JDim], "float"],
#     out_field: Field[[IDim, JDim], "float"],
# ):
#     laplap(in_field, out=out_field[2:-2, 2:-2])


def lap_ref(inp):
    """Compute the laplacian using numpy"""
    return -4.0 * inp[1:-1, 1:-1] + inp[:-2, 1:-1] + inp[2:, 1:-1] + inp[1:-1, :-2] + inp[1:-1, 2:]


def np_as_located_field(*axes):
    def impl(buffer):
        return NpField(buffer, axes)

    return impl


def test_ffront_lap():
    shape = (20, 20)
    as_ij = np_as_located_field(IDim, JDim)
    input = as_ij(np.fromfunction(lambda x, y: x**2 + y**2, shape))
    input.origin = {IDim: 1, JDim: 1}

    result_lap = as_ij(np.zeros_like(input))
    lap_program(input, result_lap, offset_provider={"Ioff": IDim, "Joff": JDim})
    ref = lap_ref(np.asarray(input))
    assert np.allclose(np.asarray(result_lap)[1:-1, 1:-1], lap_ref(np.asarray(input)))

    # result_laplap = as_ij(np.zeros_like(input))
    # laplap_program(input, result_laplap, offset_provider={"Ioff": IDim, "Joff": JDim})
    # assert np.allclose(np.asarray(result_laplap)[2:-2, 2:-2], lap_ref(lap_ref(np.asarray(input))))


test_ffront_lap()

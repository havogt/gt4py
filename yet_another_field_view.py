import pytest


def stencil(sten):
    def impl(*args, **kwargs):
        return Stencil(sten, *args, **kwargs)

    return impl


@stencil
def my_stencil(a, b):
    return a + b


class Stencil:
    def __init__(self, fun, *args, **kwargs):
        self.fun = fun
        self.args = args
        self.kwargs = kwargs


class Proxy:
    def __init__(self, reference) -> None:
        self.reference = reference


def compute_domain(a, b):
    if a is not None:
        if b is not None:
            if a == b:
                return a
            raise RuntimeError("domains don't match")
    return None


class NpFake:
    def __init__(self, value, *, domain=None, bounds=None) -> None:
        self.value = value
        self.domain = domain
        self.bounds = bounds

    def __add__(self, other):
        return NpFake(self.value + other.value, domain=compute_domain(self.domain, other.domain))

    def __setitem__(self, index, value):
        if isinstance(index, slice) and index.stop is None:
            if isinstance(value, Stencil):
                if "backend" in value.kwargs:
                    print(f"executing for {value.kwargs['backend']}")
                self.value = value.fun(*value.args).value
                return
        raise NotImplementedError()

    # def __getitem__(self, index):
    #     if isinstance(index, slice) and index.stop is None:
    #         return Proxy(self)


a = NpFake(1)
b = NpFake(2)
c = NpFake(-1)


def program(a, b, c, *, backend):
    c[:] = my_stencil(a, b, backend=backend)


program(a, b, c, backend="whatever")


print(c.value)


class NpField:
    @classmethod
    def from_lst(cls, arr, *, domain=None, bounds=None):

        bounds = range(len(arr)) if bounds is None else bounds
        return cls(
            arr,
            domain=range(len(arr)) if domain is None else domain,
            bounds=bounds,
            offset=-bounds.start,
        )

    def __init__(self, data, *, domain=None, bounds=None, offset=0) -> None:
        self.data = data
        self.domain = domain
        self.bounds = bounds
        self.offset = offset
        if self.domain is not None and self.bounds is not None:
            assert self.domain[0] in self.bounds and self.domain[-1] in self.bounds

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.data[self.domain.start + self.offset + index]
        if isinstance(index, range) or isinstance(index, slice):
            return self.data[index.start + self.offset : index.stop + self.offset]
        raise NotImplementedError()

    def __setitem__(self, index, value):
        if isinstance(index, slice) and index.stop is None:
            if isinstance(value, NpField):
                if not self.domain == value.domain:
                    raise RuntimeError("output domain does not match")
                self.data[
                    self.domain.start + self.offset : self.domain.stop + self.offset
                ] = value.domain_data
                return
        raise NotImplementedError()

    def shift(self, offset):
        return NpField(
            self.data,
            domain=self.domain,
            bounds=range(self.bounds.start - offset, self.bounds.stop - offset),
            offset=self.offset + offset,
        )

    def op(self, other, op):
        domain = compute_domain(self.domain, other.domain)
        bounds = range(
            max(self.bounds.start, other.bounds.start), min(self.bounds.stop, other.bounds.stop)
        )
        data = [op(a, b) for a, b in zip(self[bounds], other[bounds])]
        return NpField(data=data, domain=domain, bounds=bounds, offset=-bounds.start)

    def __add__(self, other):
        # domain = compute_domain(self.domain, other.domain)
        # bounds = range(
        #     max(self.bounds.start, other.bounds.start), min(self.bounds.stop, other.bounds.stop)
        # )
        # data = [a + b for a, b in zip(self[bounds], other[bounds])]
        # return NpField(data=data, domain=domain, bounds=bounds, offset=-bounds.start)
        return self.op(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self.op(other, lambda a, b: a - b)

    def with_boundary(self, boundary: "NpField"):
        # TODO maybe don't care about the domain of boundary (just use as much as we have)
        data = boundary.domain_data
        data[
            self.domain.start - boundary.domain.start : self.domain.stop - boundary.domain.stop
        ] = self.domain_data

        return NpField(data, domain=self.domain, bounds=boundary.domain)

    @property
    def domain_data(self):
        return self[self.domain.start : self.domain.stop]


def test_index_access():
    testee = NpField.from_lst(list(range(5)))
    assert testee[3] == 3

    testee = NpField.from_lst(list(range(5)), domain=range(1, 5))
    assert testee[3] == 4
    assert testee[-1] == 0


test_index_access()


def test_bounds():
    testee = NpField.from_lst(list(range(5)), domain=range(1, 5), bounds=range(1, 6))
    assert testee[0] == 0, testee[0]
    assert testee[4] == 4
    testee.shift(1)


test_bounds()


def test_shift():
    inp = NpField.from_lst(list(range(5)), domain=range(1, 4))
    assert inp[0] == 1
    testee = inp.shift(-1)
    assert testee[0] == 0, testee.offset
    testee = inp.shift(1)
    assert testee[0] == 2

    # with pytest.raises:
    #     testee = inp.shift(2)


test_shift()


def test_add():
    a = NpField.from_lst(list(range(5)))
    b = NpField.from_lst(list(range(5)))

    c = a + b
    assert c.data == [0, 2, 4, 6, 8]

    a = NpField.from_lst(list(range(5)), domain=range(1, 4))
    b = NpField.from_lst(list(range(5)), domain=range(1, 4), bounds=range(1, 6))

    c = a + b
    assert c.domain_data == [1, 3, 5], c.domain_data


test_add()


def test_domain_data():
    inp = NpField.from_lst(list(range(5)), domain=range(1, 4))
    assert inp.domain_data == list(range(1, 4)), inp.domain_data


test_domain_data()


def lap(inp):
    return inp.shift(1) + inp.shift(-1) - inp - inp


def test_lap():
    inp = NpField.from_lst([9, 4, 1, 0, 1, 4, 9], domain=range(1, 4))
    res = lap(inp)


test_lap()


def inp_with_boundary(inp, boundary):
    return inp.with_boundary(boundary)


def test_inp_with_boundary():
    inp = NpField.from_lst(list(range(5)), domain=range(1, 4))
    boundary = NpField.from_lst(
        [None, -1, None, None, None, -2, None], domain=range(0, 5), bounds=range(-1, 6)
    )

    testee = inp.with_boundary(boundary)
    assert testee.domain_data == [1, 2, 3]
    assert testee.data == [-1, 1, 2, 3, -2]
    assert testee.bounds == range(5)


test_inp_with_boundary()


def test_assign():
    inp = NpField.from_lst(list(range(5)), domain=range(0, 3), bounds=range(-2, 3))
    out = NpField.from_lst([-1, -1, -1])

    out[:] = inp
    assert out.domain_data == [2, 3, 4]


test_assign()


def where(cond_field, true_field: NpField, false_field: NpField):
    domain = compute_domain(cond_field.domain, true_field.domain)
    domain = compute_domain(domain, false_field.domain)

    bounds = range(
        max(cond_field.bounds.start, true_field.bounds.start, false_field.bounds.start),
        min(cond_field.bounds.stop, true_field.bounds.stop, false_field.bounds.stop),
    )
    data = [t if c else f for c, t, f in zip(cond_field, true_field, false_field)]
    return NpField(data=data, domain=domain, bounds=bounds, offset=-bounds.start)


def test_where():
    cond = NpField.from_lst([True, False, True, False, True], domain=range(1, 4))
    tr = NpField.from_lst(list(range(10)), domain=range(1, 4), bounds=range(-3, 7))  # 3 [4,5,6] 7
    fa = NpField.from_lst(list(range(4)), domain=range(1, 4), bounds=range(1, 5))  # [0,1,2] 3

    testee = where(cond, tr, fa)
    assert testee.data == [0, 5, 2, 7], testee.data
    assert testee.domain_data == [0, 5, 2]


test_where()


def zero_gradient_boundary_left(inp, index, lower_bound):
    return where(index < lower_bound, inp.shift(1), inp)
    # TODO test and implement this


def zero_gradient_boundary_left_local(pos, inp, lower_bound):
    return inp[pos + 1] if pos < lower_bound else inp


# - Field view makes sense to me
# - we can add local operators but we are unsure about the syntax
# - we certainly can explain this to the user, it doesn't have to be the final version

# TODO zero gradient boundary

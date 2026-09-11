# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import ctypes
import importlib
import mmap
import os
import subprocess
import sys

import numpy as np
import pytest
from next_tests.integration_tests.cases import IDim, JDim, KDim, cartesian_case
from gt4py import next as gtx
from gt4py.next import broadcast, common
from gt4py.next.ffront.experimental import concat_where
from next_tests import definitions as test_definitions
from next_tests.integration_tests import cases
from next_tests.integration_tests.cases_utils import (
    exec_alloc_descriptor,
)

pytestmark = pytest.mark.uses_concat_where


@pytest.fixture(params=[False, True], ids=["dynamic_domains", "static_domains"])
def static_domains(request) -> bool:
    """Fixture to select compilation with dynamic or statically known domain bounds."""
    return request.param


def test_concat_where_simple(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(ground: cases.IJKField, air: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim > 0, air, ground)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case,
        testee,
        lambda ground, air: np.where(k[np.newaxis, np.newaxis, :] == 0, ground, air),
    )


def test_concat_where(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(ground: cases.IJKField, air: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, ground, air)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case,
        testee,
        lambda ground, air: np.where(k[np.newaxis, np.newaxis, :] == 0, ground, air),
    )


def test_concat_where_non_overlapping(cartesian_case, static_domains: bool):
    """Fields only defined in their respective region in concat_where."""

    @gtx.field_operator(static_domains=static_domains)
    def testee(ground: cases.IJKField, air: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, ground, air)

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    ground = cases.allocate(
        cartesian_case, testee, "ground", domain=out.domain.slice_at[:, :, 0:1]
    )()
    air = cases.allocate(cartesian_case, testee, "air", domain=out.domain.slice_at[:, :, 1:])()

    ref = np.concatenate((ground.asnumpy(), air.asnumpy()), axis=2)
    cases.verify(cartesian_case, testee, ground, air, out=out, ref=ref)


def test_concat_where_empty_branch(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(a: cases.IJKField, b: cases.IJKField, N: np.int32) -> cases.IJKField:
        return concat_where(IDim < N, a, b * 2)

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    a = cases.allocate(cartesian_case, testee, "a")()
    b = cases.allocate(cartesian_case, testee, "b")()

    N = out.shape[2] + 1
    cases.verify(cartesian_case, testee, a, b, N, out=out, ref=a.asnumpy())


@pytest.mark.embedded_concat_where_infinite_domain
def test_concat_where_scalar_broadcast(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(a: np.int32, b: cases.IJKField, N: np.int32) -> cases.IJKField:
        return concat_where(KDim < N - 1, a, b)

    a = 3
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    ref = np.concatenate(
        (
            np.full((*out.domain.shape[0:2], out.domain.shape[2] - 1), a),
            b.asnumpy()[:, :, -1:],
        ),
        axis=2,
    )
    cases.verify(cartesian_case, testee, a, b, cartesian_case.default_sizes[KDim], out=out, ref=ref)


@pytest.mark.embedded_concat_where_infinite_domain
def test_concat_where_scalar_broadcast_on_empty_branch(cartesian_case, static_domains: bool):
    """Output domain such that the scalar branch is never active."""

    @gtx.field_operator(static_domains=static_domains)
    def testee(a: np.int32, b: cases.KField, N: np.int32) -> cases.KField:
        return concat_where(KDim < N, a, b)

    a = 3
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN, domain=b.domain.slice_at[1:])()

    ref = b.asnumpy()[1:]
    cases.verify(cartesian_case, testee, a, b, 1, out=out, ref=ref)


def test_concat_where_single_level_broadcast(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(a: cases.KField, b: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, a, b)

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    a = cases.allocate(
        cartesian_case, testee, "a", domain=gtx.domain({KDim: out.domain.shape[2]})
    )()
    b = cases.allocate(cartesian_case, testee, "b", domain=out.domain.slice_at[:, :, 1:])()

    ref = np.concatenate(
        (
            np.tile(a.asnumpy()[0], (*b.domain.shape[0:2], 1)),
            b.asnumpy(),
        ),
        axis=2,
    )
    cases.verify(cartesian_case, testee, a, b, out=out, ref=ref)


def test_concat_where_single_level_restricted_domain_broadcast(
    cartesian_case, static_domains: bool
):
    @gtx.field_operator(static_domains=static_domains)
    def testee(a: cases.KField, b: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, a, b)

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    # note: this field is only defined on K: 0, 1, i.e., contains only a single value
    a = cases.allocate(cartesian_case, testee, "a", domain=gtx.domain({KDim: (0, 1)}))()
    b = cases.allocate(cartesian_case, testee, "b", domain=out.domain.slice_at[:, :, 1:])()

    ref = np.concatenate(
        (
            np.tile(a.asnumpy()[0], (*b.domain.shape[0:2], 1)),
            b.asnumpy(),
        ),
        axis=2,
    )
    cases.verify(cartesian_case, testee, a, b, out=out, ref=ref)


def test_boundary_single_layer_3d_bc(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(interior: cases.IJKField, boundary: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim == 0, boundary, interior)

    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary", sizes={KDim: 1})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    ref = np.where(
        k[np.newaxis, np.newaxis, :] == 0,
        np.broadcast_to(boundary.asnumpy(), interior.shape),
        interior.asnumpy(),
    )

    cases.verify(cartesian_case, testee, interior, boundary, out=out, ref=ref)


def test_boundary_single_layer_2d_bc(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(interior: cases.IJKField, boundary: cases.IJField) -> cases.IJKField:
        return concat_where(KDim == 0, boundary, interior)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case,
        testee,
        lambda interior, boundary: np.where(
            k[np.newaxis, np.newaxis, :] == 0, boundary[:, :, np.newaxis], interior
        ),
    )


def test_boundary_single_layer_2d_bc_on_empty_branch(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(interior: cases.IJKField, boundary: cases.IJField) -> cases.IJKField:
        return concat_where(KDim == 0, boundary, interior)

    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(
        cartesian_case, testee, cases.RETURN, domain=interior.domain.slice_at[:, :, 1:]
    )()

    ref = interior.asnumpy()[:, :, 1:]
    cases.verify(cartesian_case, testee, interior, boundary, out=out, ref=ref)


def test_dimension_two_nested_conditions(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(interior: cases.IJKField, boundary: cases.IJKField) -> cases.IJKField:
        return concat_where((KDim < 2), boundary, concat_where((KDim >= 5), boundary, interior))

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case,
        testee,
        lambda interior, boundary: np.where(
            (k[np.newaxis, np.newaxis, :] < 2) | (k[np.newaxis, np.newaxis, :] >= 5),
            boundary,
            interior,
        ),
    )


def test_dimension_two_conditions_and(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(interior: cases.KField, boundary: cases.KField, nlev: np.int32) -> cases.KField:
        return concat_where((0 < KDim) & (KDim < (nlev - 1)), interior, boundary)

    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    nlev = cartesian_case.default_sizes[KDim]
    k = np.arange(0, nlev)
    ref = np.where((0 < k) & (k < (nlev - 1)), interior.asnumpy(), boundary.asnumpy())
    cases.verify(cartesian_case, testee, interior, boundary, nlev, out=out, ref=ref)


def test_dimension_eq_in_middle_of_domain(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where((KDim == 2), interior, boundary)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case, testee, lambda interior, boundary: np.where(k == 2, interior, boundary)
    )


def _guarded_array(values: np.ndarray, at_end: bool) -> np.ndarray:
    """C-ordered copy of `values` flush against an inaccessible page, before it or after it."""
    page = mmap.PAGESIZE
    body = -(-values.nbytes // page) * page
    buffer = mmap.mmap(-1, page + body + page, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    libc = ctypes.CDLL(None, use_errno=True)
    for guard in (address, address + page + body):
        assert libc.mprotect(ctypes.c_void_p(guard), ctypes.c_size_t(page), 0) == 0
    offset = page + body - values.nbytes if at_end else page
    result = np.frombuffer(buffer, dtype=values.dtype, count=values.size, offset=offset)
    result = result.reshape(values.shape)
    result[...] = values
    return result


@gtx.field_operator
def _eq_at_bottom(a: cases.IJKField, b: cases.IJKField, nlev: np.int32) -> cases.IJKField:
    return concat_where(KDim == 0, b, a(KDim - 1))


@gtx.field_operator
def _eq_at_top(a: cases.IJKField, b: cases.IJKField, nlev: np.int32) -> cases.IJKField:
    return concat_where(KDim == nlev - 1, b, a(KDim + 1))


@gtx.field_operator
def _eq_at_top_shifted_temporary(
    a: cases.IJKField, b: cases.IJKField, nlev: np.int32
) -> cases.IJKField:
    tmp = a + b
    return concat_where(KDim == nlev - 1, tmp, tmp - tmp(KDim + 1))


def _run_eq_at_boundary(backend_id: str, form: str, shape: tuple[int, int, int]) -> None:
    module, _, name = backend_id.rpartition(".")
    backend = getattr(importlib.import_module(module), name)
    if isinstance(backend, test_definitions.EmbeddedDummyBackend):
        backend = None
    testee = {
        "bottom": _eq_at_bottom,
        "top": _eq_at_top,
        "temporary": _eq_at_top_shifted_temporary,
    }[form]

    a_np = np.arange(1, np.prod(shape) + 1, dtype=np.int32).reshape(shape)
    b_np = -a_np
    if form == "bottom":
        ref = np.concatenate((b_np[..., :1], a_np[..., :-1]), axis=2)
    elif form == "top":
        ref = np.concatenate((a_np[..., 1:], b_np[..., -1:]), axis=2)
    else:
        tmp = a_np + b_np
        ref = np.concatenate((tmp[..., :-1] - tmp[..., 1:], tmp[..., -1:]), axis=2)

    dims = [IDim, JDim, KDim]
    guarded_a_np = _guarded_array(a_np, at_end=form != "bottom")
    a = common._field(guarded_a_np, domain=gtx.domain(dict(zip(dims, shape))))
    assert np.shares_memory(a.ndarray, guarded_a_np)
    out = gtx.as_field(dims, np.full_like(ref, np.iinfo(ref.dtype).min))
    testee.with_backend(backend)(
        a, gtx.as_field(dims, b_np), np.int32(shape[2]), out=out, offset_provider={}
    )
    np.testing.assert_array_equal(out.asnumpy(), ref)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Guard pages need 'mprotect'.")
@pytest.mark.parametrize("form", ["bottom", "top", "temporary"])
def test_dimension_eq_at_boundary_reads_only_selected_levels(cartesian_case, request, form):
    # The false branch is evaluated only where it is selected, so it never reads the level beyond
    # the domain of `a`. Such a read faults on the page next to `a`, so the program runs in a
    # separate interpreter, where the fault fails this test instead of the test session. Serial
    # compilation keeps a crashing child from leaving compile workers behind.
    if not isinstance(cases.allocate(cartesian_case, _eq_at_bottom, "a")().ndarray, np.ndarray):
        pytest.skip("Guard pages need host memory.")
    backend_id = str(request.node.callspec.params["exec_alloc_descriptor"])
    shape = tuple(cartesian_case.default_sizes[dim] for dim in (IDim, JDim, KDim))

    child = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import {__name__} as m; m._run_eq_at_boundary({backend_id!r}, {form!r}, {shape})",
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(path for path in sys.path if path),
            "GT4PY_BUILD_JOBS_MODE": "serial",
        },
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr


@pytest.mark.embedded_concat_where_non_contiguous_domain
def test_dimension_not_eq_in_middle_of_domain(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where((KDim != 2), boundary, interior)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case, testee, lambda interior, boundary: np.where(k != 2, boundary, interior)
    )


def test_dimension_less_equal(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where((KDim <= 2), boundary, interior)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case, testee, lambda interior, boundary: np.where(k <= 2, boundary, interior)
    )


def test_dimension_reverse_greater(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where((2 > KDim), boundary, interior)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case, testee, lambda interior, boundary: np.where(2 > k, boundary, interior)
    )


def test_dimension_reverse_greater_equal(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where((2 >= KDim), boundary, interior)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case, testee, lambda interior, boundary: np.where(2 >= k, boundary, interior)
    )


def test_dimension_reverse_eq(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where((2 == KDim), interior, boundary)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case, testee, lambda interior, boundary: np.where(2 == k, interior, boundary)
    )


@pytest.mark.embedded_concat_where_non_contiguous_domain
def test_dimension_reverse_not_eq(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where((2 != KDim), boundary, interior)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case, testee, lambda interior, boundary: np.where(2 != k, boundary, interior)
    )


@pytest.mark.embedded_concat_where_non_contiguous_domain
def test_dimension_two_conditions_or(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(interior: cases.KField, boundary: cases.KField) -> cases.KField:
        return concat_where(((KDim < 2) | (KDim >= 5)), boundary, interior)

    k = np.arange(0, cartesian_case.default_sizes[KDim])
    cases.verify_with_default_data(
        cartesian_case,
        testee,
        lambda interior, boundary: np.where((k < 2) | (k >= 5), boundary, interior),
    )


def test_lap_like(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(
        inp: cases.IJField, boundary: np.int32, shape: tuple[np.int32, np.int32]
    ) -> cases.IJField:
        # TODO(havogt) add support for multi-dimensional concat_where and non-contiguous unions
        return concat_where(
            (IDim == 0),
            boundary,
            concat_where(
                IDim == shape[0] - 1,
                boundary,
                concat_where(
                    JDim == 0,
                    boundary,
                    concat_where(JDim == shape[1] - 1, boundary, inp),
                ),
            ),
        )

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    inp = cases.allocate(cartesian_case, testee, "inp", domain=out.domain.slice_at[1:-1, 1:-1])()
    boundary = 2

    ref = np.full(out.domain.shape, np.nan)
    ref[0, :] = boundary
    ref[:, 0] = boundary
    ref[-1, :] = boundary
    ref[:, -1] = boundary
    ref[1:-1, 1:-1] = inp.asnumpy()
    cases.verify(cartesian_case, testee, inp, boundary, out.domain.shape, out=out, ref=ref)


@pytest.mark.uses_tuple_returns
def test_with_tuples(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(
        interior0: cases.IJKField,
        boundary0: cases.IJField,
        interior1: cases.IJKField,
        boundary1: cases.IJField,
    ) -> tuple[cases.IJKField, cases.IJKField]:
        return concat_where(KDim == 0, (boundary0, boundary1), (interior0, interior1))

    k = np.arange(0, cartesian_case.default_sizes[KDim])

    def ref(interior0, boundary0, interior1, boundary1):
        return (
            np.where(k[np.newaxis, np.newaxis, :] == 0, boundary0[:, :, np.newaxis], interior0),
            np.where(k[np.newaxis, np.newaxis, :] == 0, boundary1[:, :, np.newaxis], interior1),
        )

    cases.verify_with_default_data(cartesian_case, testee, ref)


def test_nested_conditions_with_empty_branches(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(interior: cases.IField, boundary: cases.IField, N: gtx.int32) -> cases.IField:
        interior = concat_where(IDim == 0, boundary, interior)
        interior = concat_where((1 <= IDim) & (IDim < N - 1), interior * 2, interior)
        interior = concat_where(IDim == N - 1, boundary, interior)
        return interior

    interior = cases.allocate(cartesian_case, testee, "interior")()
    boundary = cases.allocate(cartesian_case, testee, "boundary")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    N = cartesian_case.default_sizes[IDim]

    i = np.arange(0, cartesian_case.default_sizes[IDim])
    ref = np.where(
        (i[:] == 0) | (i[:] == N - 1),
        boundary.asnumpy(),
        interior.asnumpy() * 2,
    )
    cases.verify(cartesian_case, testee, interior, boundary, N, out=out, ref=ref)


@pytest.mark.uses_tuple_returns
def test_with_tuples_different_domain(cartesian_case, static_domains: bool):
    @gtx.field_operator(static_domains=static_domains)
    def testee(
        interior0: cases.IJKField,
        boundary0: cases.IJKField,
        interior1: cases.KField,
        boundary1: cases.KField,
    ) -> tuple[cases.IJKField, cases.IJKField]:
        a, b = concat_where(KDim == 0, (boundary0, boundary1), (interior0, interior1))
        # the broadcast is only needed since we can not return fields on different domains yet
        return a, broadcast(b, (IDim, JDim, KDim))

    k = np.arange(0, cartesian_case.default_sizes[KDim])

    def ref(interior0, boundary0, interior1, boundary1):
        return (
            np.where(k[np.newaxis, np.newaxis, :] == 0, boundary0, interior0),
            np.where(k == 0, boundary1, interior1),
        )

    cases.verify_with_default_data(cartesian_case, testee, ref)


def test_concat_where_field_broadcast_on_empty_branch(cartesian_case, static_domains: bool):
    """
    A field branch with fewer dimensions than the expression is implicitly broadcast.

    `b` only has the `K` dimension, but is selected everywhere, so it is broadcast to the
    three-dimensional result. With `static_domains` this also tests pruning: the domain bounds
    are statically known, so `prune_empty_concat_where` decides that `a` is never selected and
    must reintroduce the implicit broadcast of the `concat_where` instead of replacing the
    three-dimensional expression by a one-dimensional one.
    """

    @gtx.field_operator(static_domains=static_domains)
    def testee(a: cases.IJKField, b: cases.KField) -> cases.IJKField:
        return concat_where(KDim < 0, a, b)

    cases.verify_with_default_data(
        cartesian_case,
        testee,
        lambda a, b: np.broadcast_to(b[np.newaxis, np.newaxis, :], a.shape),
    )

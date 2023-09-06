# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Sequence

import pytest

from gt4py.next import common
from gt4py.next.common import UnitRange
from gt4py.next.embedded import exceptions as embedded_exceptions
from gt4py.next.embedded.common import _slice_range, sub_domain


@pytest.mark.parametrize(
    "rng, slce, expected",
    [
        (UnitRange(2, 10), slice(2, -2), UnitRange(4, 8)),
        (UnitRange(2, 10), slice(2, None), UnitRange(4, 10)),
        (UnitRange(2, 10), slice(None, -2), UnitRange(2, 8)),
        (UnitRange(2, 10), slice(None), UnitRange(2, 10)),
    ],
)
def test_slice_range(rng, slce, expected):
    result = _slice_range(rng, slce)
    assert result == expected


I = common.Dimension("I")
J = common.Dimension("J")
K = common.Dimension("K")


@pytest.mark.parametrize(
    "domain, index, expected",
    [
        ([(I, (2, 5))], 1, []),
        ([(I, (2, 5))], slice(1, 2), [(I, (3, 4))]),
        ([(I, (2, 5))], (I, 2), []),
        ([(I, (2, 5))], (I, UnitRange(2, 3)), [(I, (2, 3))]),
        ([(I, (-2, 3))], 1, []),
        ([(I, (-2, 3))], slice(1, 2), [(I, (-1, 0))]),
        ([(I, (-2, 3))], (I, 1), []),
        ([(I, (-2, 3))], (I, UnitRange(2, 3)), [(I, (2, 3))]),
        ([(I, (-2, 3))], -5, []),
        ([(I, (-2, 3))], -6, IndexError),
        ([(I, (-2, 3))], slice(-7, -6), IndexError),
        ([(I, (-2, 3))], slice(-6, -7), IndexError),
        ([(I, (-2, 3))], 4, []),
        ([(I, (-2, 3))], 5, IndexError),
        ([(I, (-2, 3))], slice(4, 5), [(I, (2, 3))]),
        ([(I, (-2, 3))], slice(5, 6), IndexError),
        ([(I, (-2, 3))], (I, -3), IndexError),
        ([(I, (-2, 3))], (I, UnitRange(-3, -2)), IndexError),
        ([(I, (-2, 3))], (I, 3), IndexError),
        ([(I, (-2, 3))], (I, UnitRange(3, 4)), IndexError),
        (
                [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
                2,
                [(J, (3, 6)), (K, (4, 7))],
        ),
        (
                [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
                slice(2, 3),
                [(I, (4, 5)), (J, (3, 6)), (K, (4, 7))],
        ),
        (
                [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
                (I, 2),
                [(J, (3, 6)), (K, (4, 7))],
        ),
        (
                [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
                (I, UnitRange(2, 3)),
                [(I, (2, 3)), (J, (3, 6)), (K, (4, 7))],
        ),
        (
                [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
                (J, 3),
                [(I, (2, 5)), (K, (4, 7))],
        ),
        (
                [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
                (J, UnitRange(4, 5)),
                [(I, (2, 5)), (J, (4, 5)), (K, (4, 7))],
        ),
        (
                [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
                ((J, 3), (I, 2)),
                [(K, (4, 7))],
        ),
        (
                [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
                ((J, UnitRange(4, 5)), (I, 2)),
                [(J, (4, 5)), (K, (4, 7))],
        ),
        (
                [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
                (slice(1, 2), slice(2, 3)),
                [(I, (3, 4)), (J, (5, 6)), (K, (4, 7))],
        ),
        (
                [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
                (Ellipsis, slice(2, 3)),
                [(I, (2, 5)), (J, (3, 6)), (K, (6, 7))],
        ),
        (
                [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
                (slice(1, 2), Ellipsis, slice(2, 3)),
                [(I, (3, 4)), (J, (3, 6)), (K, (6, 7))],
        ),
        (
                [(I, (2, 5)), (J, (3, 6)), (K, (4, 7))],
                (slice(1, 2), slice(1, 2), Ellipsis),
                [(I, (3, 4)), (J, (4, 5)), (K, (4, 7))],
        ),
        ([], Ellipsis, []),
        ([], slice(None), IndexError),
        ([], 0, IndexError),
        ([], (I, 0), IndexError),
        # ([], (), []), # once we implement the array API standard
    ],
)
def test_sub_domain(domain, index, expected):
    domain = common.domain(domain)
    if expected is IndexError:
        with pytest.raises(embedded_exceptions.IndexOutOfBounds):
            sub_domain(domain, index)
    else:
        expected = common.domain(expected)
        result = sub_domain(domain, index)
        assert result == expected


@pytest.fixture
def finite_domain():
    I = common.Dimension("I")
    J = common.Dimension("J")
    return common.Domain((I, UnitRange(-1, 3)), (J, UnitRange(2, 4)))


@pytest.fixture
def infinite_domain():
    I = common.Dimension("I")
    return common.Domain((I, UnitRange.infinity()))


@pytest.fixture
def mixed_domain():
    I = common.Dimension("I")
    J = common.Dimension("J")
    return common.Domain((I, UnitRange(-1, 3)), (J, UnitRange.infinity()))


def test_finite_domain_is_finite(finite_domain):
    assert finite_domain.is_finite() == True


def test_infinite_domain_is_finite(infinite_domain):
    assert infinite_domain.is_finite() == False


def test_mixed_domain_is_finite(mixed_domain):
    assert mixed_domain.is_finite() == False

# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

from gt4py.next import containers
from gt4py.next.type_system import type_specifications as ts, type_translation


def _type_of_dataclass(cls: type) -> ts.NamedTupleType:
    assert dataclasses.is_dataclass(cls)
    fields = [f for f in dataclasses.fields(cls) if not f.name.startswith("_")]

    types = [type_translation.from_type_hint(field.type) for field in fields]
    keys = [f.name for f in fields]
    is_vector = getattr(cls, "__gt_is_vector__", False)
    # return_type = ts.NamedTupleType(types=types, keys=keys)
    return_type = ts.NamedTupleType(types=types, keys=keys, is_vector=is_vector)
    return return_type

    # TODO: the following is the type of the constructor
    # res = ts.FunctionType(
    #     pos_only_args=[],
    #     pos_or_kw_args=dict(zip(keys, types, strict=True)),
    #     kw_only_args={},
    #     returns=return_type,
    # )
    # return res


containers.register(lambda cls: dataclasses.is_dataclass(cls), _type_of_dataclass)

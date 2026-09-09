# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""``shard_map`` across jax versions, and GT4Py field construction from jax tracers."""

from __future__ import annotations

import inspect

import jax

# jax >= 0.11: ``jax.shard_map(..., check_vma=)``; ``jax.experimental.shard_map`` is an
# empty shim. jax 0.6.2: ``jax.experimental.shard_map(..., check_rep=)``.
_NEW_SHARD_MAP = hasattr(jax, "shard_map") and (
    "check_vma" in inspect.signature(jax.shard_map).parameters
)


def shard_map(f, mesh, in_specs, out_specs, check_rep: bool = True):
    if _NEW_SHARD_MAP:
        return jax.shard_map(
            f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=check_rep
        )
    from jax.experimental.shard_map import shard_map as _legacy

    return _legacy(f, mesh, in_specs, out_specs, check_rep=check_rep)


def shard_map_api() -> str:
    return (
        "jax.shard_map(check_vma=)" if _NEW_SHARD_MAP else "jax.experimental.shard_map(check_rep=)"
    )


def patch_gt4py_tracer_dispatch() -> str:
    """Register ``jax.core.Tracer`` with GT4Py's ``singledispatch`` field constructor.

    On jax >= 0.11 a tracer is still an ``isinstance`` of ``jax.Array`` but no longer
    dispatches through it, so ``gtx.as_field`` raises ``NotImplementedError`` under
    ``grad``/``jit``/``scan``. Returns ``"patched"``, ``"not needed"`` or ``"no gt4py"``.
    """
    try:
        from gt4py.next import common
        from gt4py.next.embedded import nd_array_field as _ndf
    except ImportError:
        return "no gt4py"
    tracer = jax.core.Tracer
    if common._field.dispatch(tracer) is not common._field.dispatch(object):
        return "not needed"
    common._field.register(tracer, _ndf.JaxArrayField.from_array)
    common._connectivity.register(tracer, _ndf.JaxArrayConnectivityField.from_array)
    return "patched"

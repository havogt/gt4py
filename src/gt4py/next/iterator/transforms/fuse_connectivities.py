# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses

import numpy as np

from gt4py import eve
from gt4py.next import NeighborTableOffsetProvider, common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm


class HashableArray:
    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.hash = hash(arr.data.tobytes())

    def __hash__(self):
        return self.hash

    def __eq__(self, other: HashableArray):
        return np.array_equal(self.arr, other.arr)


def analyze(first, other):  # TODO extend to more
    stack = []
    for i in range(first.shape[1]):
        for j in range(other.shape[1]):
            stack.append(other[:, j][first[:, i]])
            # print(f"{i}/{j}: {other[:, j][first[:, i]]}")
    s = set(HashableArray(s) for s in stack)
    print(f"Compressed from {len(stack)} to {len(s)} elements.")


def apply(ir: itir.Program, offset_provider: common.OffsetProvider):
    ir = FuseConnectivities().visit(ir)
    print(ir)
    ir = ReplaceShifts(offset_provider=offset_provider).visit(ir)
    print(ir)
    return ir


def _new_tag(*tags):
    return "_".join([tag.value for tag in tags])


def compress(first, *other):
    composed = first
    for o in other:
        composed = o[composed]
    _, ind, inv = np.unique(composed[0], return_index=True, return_inverse=True)
    compressor = np.unravel_index(ind, composed[0].shape)
    decompressor = inv.reshape(composed[0].shape)
    print(f"Compressed to {len(ind)} elements with {compressor}.")
    return composed[(slice(None), *compressor)], decompressor


@dataclasses.dataclass
class ReplaceShifts(eve.NodeTranslator):
    offset_provider: common.OffsetProvider
    _combined_connectivity_cache: dict[str, np.ndarray] = dataclasses.field(
        init=False, default_factory=dict
    )

    def visit_FunCall(self, node: itir.FunCall, **kwargs):
        if cpm.is_applied_shift(node):
            offsets = node.fun.args
            if len(offsets) == 2:
                return node

            tags = offsets[::2]
            connectivities = [self.offset_provider[tag.value] for tag in tags]

            new_tag = _new_tag(*tags)
            if new_tag in self._combined_connectivity_cache:
                compressed = self.offset_provider[new_tag]
                decompressor = self._combined_connectivity_cache[new_tag]
            else:
                analyze(*[c.table for c in connectivities])
                compressed, decompressor = compress(*[c.table for c in connectivities])
                self.offset_provider[new_tag] = NeighborTableOffsetProvider(
                    compressed,
                    connectivities[0].origin_axis,
                    connectivities[-1].neighbor_axis,
                    has_skip_values=False,  # TODO
                    max_neighbors=compressed.shape[1],
                )
                self._combined_connectivity_cache[new_tag] = decompressor

            index = tuple(o.value for o in offsets[1::2])
            new_offset = decompressor[index]

            return itir.FunCall(
                fun=itir.FunCall(
                    fun=itir.SymRef(id="shift"),
                    args=[
                        itir.OffsetLiteral(value=new_tag),
                        itir.OffsetLiteral(value=int(new_offset)),
                    ],
                ),
                args=node.args,
            )
        node = self.generic_visit(node)
        return node


class FuseConnectivities(eve.NodeTranslator):
    def visit_FunCall(self, node: itir.FunCall, **kwargs):
        if cpm.is_let(node):
            if node.fun.params[0].id.startswith("_step"):
                # TODO generalize the shift inlining
                from gt4py.next.iterator.transforms import inline_lambdas

                res = inline_lambdas.InlineLambdas(
                    opcount_preserving=False,
                    force_inline_lambda_args=True,
                    force_inline_lift_args=True,
                    force_inline_trivial_lift_args=True,
                ).visit(node)
                print(res)
                res = inline_lambdas.InlineLambdas(
                    opcount_preserving=False,
                    force_inline_lambda_args=True,
                    force_inline_lift_args=True,
                    force_inline_trivial_lift_args=True,
                ).visit(res)
                print(res)
                return res

        node = self.generic_visit(node)
        return node

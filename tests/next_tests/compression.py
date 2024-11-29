# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import toy_connectivity


# import toy_connectivity_non_periodic


v2v = toy_connectivity.v2v_arr

v2v2v = v2v[v2v]

v2v2v_0 = v2v2v[0]


def compress(first, *other):
    composed = first
    for o in other:
        composed = o[composed]
    _, ind, inv = np.unique(composed[0], return_index=True, return_inverse=True)
    compressor = np.unravel_index(ind, composed[0].shape)
    decompressor = inv.reshape(composed[0].shape)
    print(f"Compressed to {len(ind)} elements with {compressor}.")
    return composed[(slice(None), *compressor)], decompressor


def full_analysis(first, *other):
    composed = first
    for o in other:
        composed = o[composed]
    for i in range(composed.shape[0]):
        _, ind, inv = np.unique(composed[i], return_index=True, return_inverse=True)
        # print(f"Could compress to {len(ind)} elements with {np.sort(ind)}.")
        print(composed[i])


# c_v2v_v2v, dec_v2v_v2v = compress(v2v, v2v)

# assert np.array_equal(c_v2v_v2v[slice(None), dec_v2v_v2v], v2v2v)


class HashableArray:
    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.hash = hash(arr.data.tobytes())

    def __hash__(self):
        return self.hash

    def __eq__(self, other: HashableArray):
        return np.array_equal(self.arr, other.arr)


class StranglyHashableArray:
    def __init__(self, arr: np.ndarray):
        self.arr = arr

    def __hash__(self):
        return 0

    def __eq__(self, other: StranglyHashableArray):
        # ignore -1 (skip) values
        mask_self = np.equal(self.arr, -1)
        mask_other = np.equal(other.arr, -1)
        mask = np.logical_or(mask_self, mask_other)
        clean_self = np.where(mask, 0, self.arr)
        clean_other = np.where(mask, 0, other.arr)
        return np.array_equal(clean_self, clean_other)


def analyze(first, other):  # TODO extend to more
    stack = []
    for i in range(first.shape[1]):
        for j in range(other.shape[1]):
            stack.append(other[:, j][first[:, i]])
            # print(f"{i}/{j}: {other[:, j][first[:, i]]}")
    s = set(HashableArray(s) for s in stack)
    print(f"Compressed from {len(stack)} to {len(s)} elements.")


def analyze_skip_value(first, other):  # TODO extend to more
    stack = []
    for i in range(first.shape[1]):
        for j in range(other.shape[1]):
            tmp = other[:, j][first[:, i]]
            tmp[first[:, i] == -1] = -1  # fixes places where we did wrap-around indexing with -1
            # print(f"{i}/{j}: {tmp}")
            stack.append(tmp)
    s = set(StranglyHashableArray(s) for s in stack)
    print(f"Compressed from {len(stack)} to {len(s)} elements.")


def invert(arr):
    _, inv = np.unique(arr, return_inverse=True)
    return inv  # np.unravel_index(inv, toy_connectivity.v2e_arr.shape)


print(toy_connectivity.e2v_arr)
print(invert(toy_connectivity.e2v_arr))
print(toy_connectivity.v2e_arr)


# analyze(v2v, v2v)
# analyze(toy_connectivity.e2v_arr, toy_connectivity.v2v_arr)
# print("v2e-e2v")
# analyze(toy_connectivity.v2e_arr, toy_connectivity.e2v_arr)
# analyze(toy_connectivity.e2v_arr, toy_connectivity.v2e_arr)
# analyze(toy_connectivity.c2e_arr, toy_connectivity.e2v_arr)
# compressed, decompressor = compress(toy_connectivity.c2e_arr, toy_connectivity.e2v_arr)
# assert np.array_equal(
#     compressed[(slice(None), decompressor)], toy_connectivity.e2v_arr[toy_connectivity.c2e_arr]
# )

# analyze_skip_value(toy_connectivity_non_periodic.v2v_arr, toy_connectivity_non_periodic.v2v_arr)

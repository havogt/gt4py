# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any
from gt4py._core import ndarray_utils, definitions
import pytest


def cuda_device(id_: int) -> Any:
    if ndarray_utils.cupy is not None:
        return ndarray_utils.cupy.cuda.Device(id_)
    return NotImplemented  # something that's not `None`


@pytest.mark.parametrize(
    "array_ns, gt4py_device, expected_device",
    [
        (ndarray_utils.np, None, None),
        (ndarray_utils.np, definitions.Device(definitions.DeviceType.CPU, 0), None),
        pytest.param(ndarray_utils.cupy, None, None, marks=pytest.mark.requires_gpu),
        pytest.param(
            ndarray_utils.cupy,
            definitions.Device(definitions.CUPY_DEVICE_TYPE, 42),
            cuda_device(42),
            marks=pytest.mark.requires_gpu,
        ),
    ],
)
def test_get_device_translator(array_ns, gt4py_device, expected_device):
    translator = ndarray_utils.get_device_translator(array_ns)
    assert translator(gt4py_device) == expected_device


@pytest.mark.parametrize(
    "array_ns",
    [
        ndarray_utils.np,
        pytest.param(ndarray_utils.cupy, marks=pytest.mark.requires_gpu),
    ],
)
def test_is_array_namespace(array_ns):
    assert ndarray_utils.is_array_namespace(array_ns)


@pytest.mark.requires_torch
@pytest.mark.parametrize(
    "gt4py_device, expected_device",
    [
        (None, None),
        (definitions.Device(definitions.DeviceType.CPU, 0), "cpu"),
        (definitions.Device(definitions.DeviceType.CUDA, 1), "cuda:1"),
        (definitions.Device(definitions.DeviceType.ROCM, 0), "cuda:0"),
    ],
)
def test_get_device_translator_torch(gt4py_device, expected_device):
    import torch

    translator = ndarray_utils.get_device_translator(torch)
    expected = None if expected_device is None else torch.device(expected_device)
    assert translator(gt4py_device) == expected


@pytest.mark.requires_torch
def test_is_array_namespace_torch():
    import torch

    assert ndarray_utils.is_array_namespace(torch)


@pytest.mark.requires_torch
@pytest.mark.parametrize(
    "name",
    [
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_dtype_from_torch(name):
    import numpy as np
    import torch

    assert definitions.dtype(getattr(torch, name)) == definitions.dtype(getattr(np, name))

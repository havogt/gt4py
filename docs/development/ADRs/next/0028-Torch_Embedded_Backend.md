---
tags: [embedded]
---

# PyTorch as Embedded Array Backend

- **Status**: draft
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-09-29
- **Updated**: 2026-09-29

In the context of embedded execution of `gt4py.next` field operators and programs, facing
users who hold their data in `torch` tensors (on CPU or CUDA), we decided to support
`torch.Tensor` as a backing array of `NdArrayField`, next to NumPy, CuPy and JAX, as an
optional dependency (`gt4py[torch]`), to let such users run embedded programs on their
tensors without copying them to NumPy or CuPy.

## Context

`NdArrayField` implements all embedded operations against a NumPy-like array namespace
(`array_ns`). NumPy, CuPy and `jax.numpy` provide it directly. `torch` does not:

- It is not Array API compliant, and several NumPy names that `NdArrayField` calls are
  missing (`power`, `mod`, `invert`, `cbrt`, `gamma`) or have other signatures
  (`transpose`).
- `torch.dtype` objects are not NumPy dtypes. They have no `.type`, `torch.dtype(...)` is not
  a constructor, and tensors have no `.astype`.
- Array creation functions (`arange`, `empty`, `asarray`) allocate on the CPU unless a
  device is passed. CuPy uses the current GPU device instead. Mixing CPU and CUDA tensors
  in an operation is an error.
- `uint16`, `uint32` and `uint64` exist only as limited dtypes: arithmetic is not implemented
  for them.

Unlike JAX, tensors are mutable, so in-place `__setitem__` works as for NumPy.

## Decision

- `array_ns` of the torch fields is a namespace based on `array_api_compat.torch`
  (`array-api-compat` is already a core dependency), extended with the missing NumPy names.
  Its array creation functions accept NumPy dtypes and default to the device of the namespace.
- There are two field classes per kind: `TorchArrayField` (CPU) and `TorchCUDAArrayField`
  (the current CUDA device, which also covers ROCm builds of `torch`), plus the matching
  connectivity classes. Registration on `torch.Tensor` dispatches on `tensor.device.type`.
  With a namespace per device, arrays created internally (index arrays, masks, 0-d scalars)
  land on the device of the field without changes to the generic `NdArrayField` code.
- `NdArrayField` gets three overridable hooks: `_scalar_type_of(array)` and
  `_astype(array, type_)` for the NumPy dtype assumptions, and `_as_assignable(value)` for
  `__setitem__`, because torch does not accept NumPy scalars or arrays as assigned values.
- `core_defs.dtype()` accepts `torch.dtype` by name, without importing `torch`.
- `constructors` accept `allocator=torch`. Without `device`, tensors are allocated on the torch
  default device (`torch.set_default_device`, or a `with torch.device(...)` block), as
  `allocator=jax.numpy` allocates on the JAX default device. An explicit `device` goes through a
  device translator in `_core.ndarray_utils`. The translator matches `torch` by module name, so importing
  `gt4py._core` does not import `torch`.
- Testing: two embedded entries in the test matrix, `EmbeddedTorch` and `EmbeddedTorchCUDA`,
  with a `requires_torch` marker. For now `torch` is not installed in the nox sessions, and
  `requires_torch` tests are deselected there, so CI does not run them.

## Consequences

- If `torch` is installed, importing `gt4py.next` imports `torch`, as it already does for `jax`.
- Features that need unsigned integer arithmetic do not work with torch fields. No embedded
  feature test currently uses unsigned integers, so the torch entries need no exclusions beyond
  the common embedded skip list.
- `__gt_buffer_info__` is not implemented, as for JAX, so torch fields cannot be passed to
  compiled backends yet.
- Out of scope, possible follow-ups: `torch.compile` of embedded programs, autograd through
  fields, and zero-copy passing of torch fields to gtfn/dace via DLPack.

## Alternatives considered

- **Raw `torch` module as `array_ns`**: rejected. Too many names and signatures differ.
- **One torch field class with a per-instance namespace**: rejected. `array_ns` is used at
  class level (`from_array`, `_make_builtin`, `_concat`), and dynamically created classes
  per device cannot be pickled.
- **`torch.set_default_device` in the tests**: rejected. It would hide device bugs that
  users hit without a default device.

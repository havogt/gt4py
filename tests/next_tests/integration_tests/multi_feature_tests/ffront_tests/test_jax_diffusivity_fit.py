# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The notebook `examples/next/jax_diffusivity_fit.ipynb`: the `jax_jit` backend inside `jax.jit` and `jax.grad`."""

import functools
import importlib.util
import json
import pathlib
import types

import numpy as np
import pytest

import gt4py.next as gtx


pytestmark = pytest.mark.requires_jax

N, N_STEPS = 12, 16
KAPPA_TRUE, KAPPA = 5e-3, 2e-3
OBS_CELLS = np.arange(0, N * N, 7)


@pytest.fixture(scope="module")
def example(tmp_path_factory) -> types.ModuleType:
    """The notebook's definitions: its code cells without the twin experiment."""
    notebook = pathlib.Path(__file__).parents[5] / "examples" / "next" / "jax_diffusivity_fit.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    # gt4py reads the source of field operators, so the cells go to a file
    path = tmp_path_factory.mktemp("notebook") / "jax_diffusivity_fit.py"
    path.write_text(
        "\n\n".join(
            "".join(cell["source"])
            for cell in cells
            if cell["cell_type"] == "code" and "experiment" not in cell["metadata"].get("tags", [])
        )
    )
    spec = importlib.util.spec_from_file_location("jax_diffusivity_fit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numpy_forecast(example, kappa, q0=None):
    """The forecast on NumPy fields, embedded in a Python loop: the reference, without JAX."""
    mesh, vn = example.make_mesh(N, np), example.cellular_flow(N, np)
    q = example.gaussian_blob(N, np) if q0 is None else gtx.as_field([example.Cell], q0)
    dt = example.CFL * mesh.h / np.max(np.abs(vn.asnumpy()))
    embedded = example.advect_diffuse.with_backend(None)
    for _ in range(N_STEPS):
        q = example.advance(q, vn, kappa, dt, mesh, stencil=embedded)
    return q.asnumpy()


def numpy_misfit(example, kappa, q0=None):
    observed = numpy_forecast(example, KAPPA_TRUE)[OBS_CELLS]
    simulated = numpy_forecast(example, kappa, q0)[OBS_CELLS]
    return 0.5 * np.mean((simulated - observed) ** 2) / np.var(observed)


@pytest.fixture
def inputs(example):
    import jax.numpy as jnp

    observed = jnp.asarray(numpy_forecast(example, KAPPA_TRUE)[OBS_CELLS])
    return types.SimpleNamespace(
        q0=example.gaussian_blob(N),
        vn=example.cellular_flow(N),
        mesh=example.make_mesh(N),
        obs_cells=jnp.asarray(OBS_CELLS),
        observed=observed,
    )


def test_forecast_jitted_equals_eager(example, inputs):
    import jax

    assert example.advect_diffuse.backend.name == "jax_jit"
    forecast = functools.partial(example.forecast, n_steps=N_STEPS)
    args = (inputs.q0, inputs.vn, KAPPA, inputs.mesh)

    with jax.checking_leaks():
        jitted = jax.jit(forecast)(*args)
    with jax.disable_jit():
        eager = forecast(*args)

    assert jitted.domain == eager.domain == inputs.q0.domain
    np.testing.assert_allclose(jitted.asnumpy(), eager.asnumpy(), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        jitted.asnumpy(), numpy_forecast(example, KAPPA), rtol=1e-10, atol=1e-12
    )
    assert np.abs(jitted.asnumpy() - inputs.q0.asnumpy()).max() > 0.1


def test_forecast_is_traced_once_for_new_values(example, inputs):
    import jax

    traces = []

    def forecast(q0, vn, kappa, mesh):
        traces.append(None)
        return example.forecast(q0, vn, kappa, mesh, N_STEPS)

    jitted = jax.jit(forecast)
    first = jitted(inputs.q0, inputs.vn, KAPPA, inputs.mesh)
    second = jitted(2.0 * inputs.q0, inputs.vn, 2.0 * KAPPA, inputs.mesh)

    assert len(traces) == 1
    assert not np.allclose(first.asnumpy(), second.asnumpy())


def _misfit_gradients(example, inputs, wrt):
    import jax

    def misfit(kappa, q0):
        return example.misfit(
            kappa, q0, inputs.vn, inputs.mesh, inputs.obs_cells, inputs.observed, N_STEPS
        )

    grad = jax.grad(misfit, argnums=wrt)
    with jax.disable_jit():
        eager = grad(KAPPA, inputs.q0)
    return jax.jit(grad)(KAPPA, inputs.q0), eager


def test_misfit_gradient_wrt_diffusivity(example, inputs):
    grad, eager = _misfit_gradients(example, inputs, wrt=0)

    delta = 1e-6
    finite_difference = (
        numpy_misfit(example, KAPPA + delta) - numpy_misfit(example, KAPPA - delta)
    ) / (2 * delta)
    np.testing.assert_allclose(grad, eager, rtol=1e-10)
    np.testing.assert_allclose(grad, finite_difference, rtol=1e-6)
    # the misfit decreases towards the true diffusivity
    assert finite_difference < 0


def test_misfit_gradient_wrt_initial_state(example, inputs):
    grad, eager = _misfit_gradients(example, inputs, wrt=1)

    assert isinstance(grad, gtx.Field)
    assert grad.domain == inputs.q0.domain
    np.testing.assert_allclose(grad.asnumpy(), eager.asnumpy(), rtol=1e-10, atol=1e-14)
    direction = np.random.default_rng(0).standard_normal(N * N)
    q0, epsilon = inputs.q0.asnumpy(), 1e-3
    finite_difference = (
        numpy_misfit(example, KAPPA, q0 + epsilon * direction)
        - numpy_misfit(example, KAPPA, q0 - epsilon * direction)
    ) / (2 * epsilon)
    assert abs(finite_difference) > 1e-6
    np.testing.assert_allclose(np.dot(grad.asnumpy(), direction), finite_difference, rtol=1e-7)


def test_fit_recovers_diffusivity(example, inputs):
    import jax

    with jax.checking_leaks():
        kappa, result = example.fit_diffusivity(
            5e-4, inputs.q0, inputs.vn, inputs.mesh, inputs.obs_cells, inputs.observed, N_STEPS
        )

    assert result.success
    np.testing.assert_allclose(kappa, KAPPA_TRUE, rtol=1e-4)

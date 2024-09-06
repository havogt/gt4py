# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools

import factory

import gt4py._core.definitions as core_defs
from gt4py.next import allocators as next_allocators, backend as next_backend, config
from gt4py.next.ffront import foast_to_gtir, past_to_itir, stages as ffront_stages
from gt4py.next.otf import recipes, stages, workflow
from gt4py.next.program_processors import modular_executor
from gt4py.next.program_processors.runners.dace_fieldview import workflow as dace_workflow


def _no_bindings(inp: stages.ProgramSource) -> stages.CompilableSource:
    return stages.CompilableSource(program_source=inp, binding_source=None)


class DaCeWorkflowFactory(factory.Factory):
    class Meta:
        model = recipes.OTFCompileWorkflow

    class Params:
        device_type: core_defs.DeviceType = core_defs.DeviceType.CPU
        cmake_build_type: config.CMakeBuildType = factory.LazyFunction(
            lambda: config.CMAKE_BUILD_TYPE
        )

    translation = factory.SubFactory(
        dace_workflow.DaCeTranslationStepFactory,
        device_type=factory.SelfAttribute("..device_type"),
    )
    bindings = _no_bindings
    compilation = factory.SubFactory(
        dace_workflow.DaCeCompilationStepFactory,
        cache_lifetime=factory.LazyFunction(lambda: config.BUILD_CACHE_LIFETIME),
        cmake_build_type=factory.SelfAttribute("..cmake_build_type"),
    )
    decoration = factory.LazyAttribute(
        lambda o: functools.partial(
            dace_workflow.convert_args,
            device=o.device_type,
        )
    )


# class DaCeBackendFactory(GTFNBackendFactory):
#     class Params:
#         otf_workflow = factory.SubFactory(
#             DaCeWorkflowFactory,
#             device_type=factory.SelfAttribute("..device_type"),
#             use_field_canonical_representation=factory.SelfAttribute(
#                 "..use_field_canonical_representation"
#             ),
#         )
#         name = factory.LazyAttribute(
#             lambda o: f"run_dace_{o.name_device}{o.name_temps}{o.name_cached}{o.name_postfix}"
#         )
#         auto_optimize = factory.Trait(
#             otf_workflow__translation__auto_optimize=True, name_temps="_opt"
#         )
#         use_field_canonical_representation: bool = False


# run_dace_cpu = DaCeBackendFactory(cached=True, auto_optimize=True)
# run_dace_cpu_noopt = DaCeBackendFactory(cached=True, auto_optimize=False)

# run_dace_gpu = DaCeBackendFactory(gpu=True, cached=True, auto_optimize=True)
# run_dace_gpu_noopt = DaCeBackendFactory(gpu=True, cached=True, auto_optimize=False)

run_dace_cpu = next_backend.Backend(
    executor=modular_executor.ModularExecutor(otf_workflow=DaCeWorkflowFactory(), name="dace_cpu"),
    allocator=next_allocators.StandardCPUFieldBufferAllocator(),
    transforms_fop=next_backend.FieldopTransformWorkflow(
        past_to_itir=past_to_itir.PastToItirFactory(to_gtir=True),
        foast_to_itir=workflow.CachedStep(
            step=foast_to_gtir.foast_to_gtir, hash_function=ffront_stages.fingerprint_stage
        ),
    ),
    transforms_prog=next_backend.ProgramTransformWorkflow(
        past_to_itir=past_to_itir.PastToItirFactory(to_gtir=True)
    ),
)

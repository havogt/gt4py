# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import factory

from gt4py.next import allocators as next_allocators, backend
from gt4py.next.ffront import foast_to_gtir, past_to_itir, foast_to_past
from gt4py.next.program_processors import modular_executor
from gt4py.next.program_processors.runners.dace_fieldview import workflow as dace_fieldview_workflow
from gt4py.next.program_processors.runners.dace_iterator import workflow as dace_iterator_workflow
from gt4py.next.program_processors.runners.gtfn import GTFNBackendFactory


class DaCeIteratorBackendFactory(GTFNBackendFactory):
    class Params:
        otf_workflow = factory.SubFactory(
            dace_iterator_workflow.DaCeWorkflowFactory,
            device_type=factory.SelfAttribute("..device_type"),
            use_field_canonical_representation=factory.SelfAttribute(
                "..use_field_canonical_representation"
            ),
        )
        name = factory.LazyAttribute(
            lambda o: f"run_dace_{o.name_device}{o.name_temps}{o.name_cached}{o.name_postfix}"
        )
        auto_optimize = factory.Trait(
            otf_workflow__translation__auto_optimize=True, name_temps="_opt"
        )
        use_field_canonical_representation: bool = False

    transforms = backend.DEFAULT_TRANSFORMS


run_dace_cpu = DaCeIteratorBackendFactory(cached=True, auto_optimize=True)
run_dace_cpu_noopt = DaCeIteratorBackendFactory(cached=True, auto_optimize=False)

run_dace_gpu = DaCeIteratorBackendFactory(gpu=True, cached=True, auto_optimize=True)
run_dace_gpu_noopt = DaCeIteratorBackendFactory(gpu=True, cached=True, auto_optimize=False)

itir_cpu = run_dace_cpu
itir_gpu = run_dace_gpu

gtir_cpu = backend.Backend(
    executor=modular_executor.ModularExecutor(
        otf_workflow=dace_fieldview_workflow.DaCeWorkflowFactory(), name="dace.gtir.cpu"
    ),
    allocator=next_allocators.StandardCPUFieldBufferAllocator(),
    transforms=backend.Transforms(
        past_to_itir=past_to_itir.past_to_itir_factory(to_gtir=True),
        foast_to_itir=foast_to_gtir.adapted_foast_to_gtir_factory(cached=True),
        field_view_op_to_prog=foast_to_past.operator_to_program_factory(foast_to_itir_step=foast_to_gtir.adapted_foast_to_gtir_factory(cached=True)),
    ),
)

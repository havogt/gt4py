from gt4py.gtc import gtir


def prune_unused_parameters(node: gtir.Stencil):
    assert isinstance(node, gtir.Stencil)
    used_variables = (
        node.iter_tree()
        .if_isinstance(gtir.FieldAccess, gtir.ScalarAccess)
        .getattr("name")
        .to_list()
    )
    used_params = list(filter(lambda param: param.name in used_variables, node.params))
    return gtir.Stencil(name=node.name, params=used_params, vertical_loops=node.vertical_loops)

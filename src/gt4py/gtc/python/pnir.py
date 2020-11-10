from typing import List, Union

from eve import Str

from gt4py.gtc.gtir import AxisBound, Expr, FieldAccess, FieldDecl, FieldsMetadata, LocNode, Stmt


# TODO here we inject a new node to gtirs stmt system -> need generic stmt to decouple
class AssignStmt(Stmt):
    left: FieldAccess
    right: Expr


class IJLoop(LocNode):
    body: List[Union[Expr, Stmt]]


class IndexFromStart(LocNode):
    idx: int


class IndexFromEnd(LocNode):
    idx: int


class KLoop(LocNode):
    lower: AxisBound
    upper: AxisBound
    ij_loops: List[IJLoop]


class RunFunction(LocNode):
    field_params: List[Str]
    scalar_params: List[Str]
    k_loops: List[KLoop]


class Module(LocNode):
    run: RunFunction


class StencilObject(LocNode):
    name: Str
    params: List[FieldDecl]
    fields_metadata: FieldsMetadata


class Stencil(LocNode):
    computation: Module
    stencil_obj: StencilObject

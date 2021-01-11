import enum
from typing import List, Optional, Tuple, Union

import eve
from eve import Str, StrEnum, SymbolName, SymbolTableTrait
from eve.type_definitions import SymbolRef
from pydantic.class_validators import validator

from gt4py.gtc import common
from gt4py.gtc.common import LocNode


class Expr(common.Expr):
    dtype: Optional[common.DataType]

    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args, **kwargs):
        if type(self) is Expr:
            raise TypeError("Trying to instantiate `Expr` abstract class.")
        super().__init__(*args, **kwargs)


class Stmt(common.Stmt):
    # TODO Eve could provide support for making a node abstract
    def __init__(self, *args, **kwargs):
        if type(self) is Stmt:
            raise TypeError("Trying to instantiate `Stmt` abstract class.")
        super().__init__(*args, **kwargs)


class Offset(common.CartesianOffset):
    pass


class VarDecl(Stmt):
    name: SymbolName
    init: Expr
    dtype: common.DataType


class Literal(common.Literal, Expr):  # type: ignore
    pass


class ScalarAccess(common.ScalarAccess, Expr):  # type: ignore
    pass


class AccessorRef(common.FieldAccess, Expr):  # type: ignore
    pass


class BlockStmt(common.BlockStmt[Stmt], Stmt):
    pass


class AssignStmt(common.AssignStmt[Union[ScalarAccess, AccessorRef], Expr], Stmt):
    # TODO remove duplication of this check
    @validator("left")
    def no_horizontal_offset_in_assignment(cls, v):
        if isinstance(v, AccessorRef) and (v.offset.i != 0 or v.offset.j != 0):
            raise ValueError("Lhs of assignment must not have a horizontal offset.")
        return v


class IfStmt(common.IfStmt[Stmt, Expr], Stmt):
    pass


class UnaryOp(common.UnaryOp[Expr], Expr):
    pass


class BinaryOp(common.BinaryOp[Expr], Expr):
    _dtype_propagation = common.binary_op_dtype_propagation(strict=True)


class TernaryOp(common.TernaryOp[Expr], Expr):
    _dtype_propagation = common.ternary_op_dtype_propagation(strict=True)


class NativeFuncCall(common.NativeFuncCall[Expr], Expr):
    _dtype_propagation = common.native_func_call_dtype_propagation(strict=True)


class Cast(common.Cast[Expr], Expr):  # type: ignore
    pass


class VerticalDimension(LocNode):
    pass


class Temporary(LocNode):
    name: SymbolName
    dtype: common.DataType


class GTGrid(LocNode):
    pass


class GTLevel(LocNode):
    splitter: int
    offset: int
    # TODO validator offset != 0


class GTInterval(LocNode):
    from_level: GTLevel
    to_level: GTLevel


class GTApplyMethod(LocNode):
    interval: GTInterval
    body: List[Stmt]


@enum.unique
class Intent(StrEnum):
    IN = "in"
    INOUT = "inout"


class GTExtent(LocNode):
    i: Tuple[int, int]
    j: Tuple[int, int]
    k: Tuple[int, int]

    @classmethod
    def zero(cls):
        return cls(i=(0, 0), j=(0, 0), k=(0, 0))

    def __add__(self, other):
        if isinstance(other, common.CartesianOffset):
            return GTExtent(
                i=(min(self.i[0], other.i), max(self.i[1], other.i)),
                j=(min(self.j[0], other.j), max(self.j[1], other.j)),
                k=(min(self.k[0], other.k), max(self.k[1], other.k)),
            )
        else:
            assert "Can only add CartesianOffsets"


class GTAccessor(LocNode):
    name: SymbolName
    id: int
    intent: Intent
    extent: GTExtent


class GTParamList(LocNode):
    accessors: List[GTAccessor]


class GTFunctor(LocNode, SymbolTableTrait):
    name: SymbolName
    applies: List[GTApplyMethod]
    param_list: GTParamList


class Param(LocNode):
    name: SymbolName

    class Config(eve.concepts.FrozenModelConfig):
        pass

    # TODO see https://github.com/eth-cscs/eve_toolchain/issues/40
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class Arg(LocNode):
    name: SymbolRef

    class Config(eve.concepts.FrozenModelConfig):
        pass

    # TODO see https://github.com/eth-cscs/eve_toolchain/issues/40
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class ApiParamDecl(LocNode):
    name: SymbolName
    dtype: common.DataType

    def __init__(self, *args, **kwargs):
        if type(self) is ApiParamDecl:
            raise TypeError("Trying to instantiate `ApiParamDecl` abstract class.")
        super().__init__(*args, **kwargs)


class FieldDecl(ApiParamDecl):
    # TODO dimensions (or mask?)
    pass


# class ScalarDecl(Decl):
#     pass


class GlobalParamDecl(ApiParamDecl):
    pass


class GTStage(LocNode):
    functor: SymbolRef
    # `args` are SymbolRefs to GTComputation `arguments` (interpreted as parameters)
    # or `temporaries`
    args: List[Arg]

    @validator("args")
    def at_least_one_argument(cls, v):
        if len(v) == 0:
            raise ValueError("A GTStage needs at least one argument.")
        return v


class IJCache(LocNode):
    name: SymbolRef  # symbol ref to GTComputation params or temporaries


class GTMultiStage(LocNode):
    loop_order: common.LoopOrder
    stages: List[GTStage]  # TODO at least one
    caches: List[Union[IJCache]]


class GTComputationCall(LocNode, SymbolTableTrait):
    # In the generated C++ code `arguments` represent both the arguments in the call to `run`
    # and the parameters of the function object.
    # We could represent this closer to the C++ code by splitting call and definition of the
    # function object.
    arguments: List[Arg]
    temporaries: List[Temporary]
    multi_stages: List[GTMultiStage]  # TODO at least one


class Program(LocNode, SymbolTableTrait):
    name: Str
    parameters: List[
        ApiParamDecl
    ]  # in the current implementation these symbols can be accessed by the functor body
    functors: List[GTFunctor]
    gt_computation: GTComputationCall  # here could be the CtrlFlow region

    _validate_dtype_is_set = common.validate_dtype_is_set()
    _validate_symbol_refs = common.validate_symbol_refs()

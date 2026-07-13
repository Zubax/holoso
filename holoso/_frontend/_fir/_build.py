"""
AST -> FIR builder: syntax-directed, NO analysis. It encodes exact Python evaluation order into ANF (every
subexpression lands in a write-once temp, left to right), ordered stores (``x, x = 1, 2`` leaves x == 2; chained
targets assign left to right), eager boolean semantics (and/or/chained comparisons evaluate all operands and
combine through selects -- pinned language semantics), real branches for conditional expressions, StaticFor
templates for for-loops and comprehensions (a comprehension target lives in its own frame overlay and never leaks;
its outermost iterable is evaluated in the enclosing scope), and one canonical exit per unit (return = store +
jump). Non-local name reads resolve at build time through the code object (the snapshot doctrine); a Missing name
becomes a Fail terminator so a dead branch can stay dead. Everything the subset excludes is a located rejection.
"""

import ast
import inspect
import logging
import textwrap
import types
import typing
from dataclasses import dataclass

from ..._errors import SourceUnavailable, UnsupportedConstruct
from ..._util import RelationalOp
from .._ast_support import UNROLL_THRESHOLD
from ._ir import (
    BindingId,
    Block,
    BlockId,
    Branch,
    BuildList,
    BuildTuple,
    Fail,
    FunctionUnit,
    Jump,
    LoadConst,
    LoadPlace,
    Local,
    Op,
    Origin,
    OriginStack,
    Place,
    PyAttr,
    PyBin,
    PyCall,
    PyCompare,
    PyLen,
    PyNot,
    PySelect,
    PyStoreAttr,
    PySubscript,
    PyTruth,
    PyUn,
    ReturnPlace,
    SelectMode,
    StaticFor,
    StorePlace,
    UnbindPlace,
    UnitExit,
    verify,
)
from ._opsem import BinOp, UnOp
from ._resolve import (
    Builtin,
    Free,
    Global,
    Local as ResolvedLocal,
    Missing,
    NameResolver,
    UnboundCell,
    comprehension_only_targets,
)
from ._value import admit, admit_ref

_logger = logging.getLogger(__name__)


class BuildRejection(UnsupportedConstruct):
    """A located refusal: the construct is outside the supported subset (or plainly wrong Python)."""

    def __init__(self, message: str, origin: OriginStack) -> None:
        frame = origin[0]
        super().__init__(f"{frame.function}:{frame.line}:{frame.column}: {message}")
        self.message = message
        self.origin = origin


_BIN_OPS: dict[type[ast.operator], BinOp] = {
    ast.Add: BinOp.ADD,
    ast.Sub: BinOp.SUB,
    ast.Mult: BinOp.MUL,
    ast.Div: BinOp.DIV,
    ast.FloorDiv: BinOp.FLOORDIV,
    ast.Mod: BinOp.MOD,
    ast.Pow: BinOp.POW,
    ast.MatMult: BinOp.MATMUL,
}

_COMPARE_OPS: dict[type[ast.cmpop], RelationalOp] = {
    ast.Eq: RelationalOp.EQ,
    ast.NotEq: RelationalOp.NE,
    ast.Lt: RelationalOp.LT,
    ast.LtE: RelationalOp.LE,
    ast.Gt: RelationalOp.GT,
    ast.GtE: RelationalOp.GE,
}


def build_unit(fn: object) -> FunctionUnit:
    """Builds the FIR template of one live Python function (or bound method); verified before it is returned."""
    bound_self: object | None = None
    if isinstance(fn, types.MethodType):
        bound_self = fn.__self__
        fn = fn.__func__
    if not isinstance(fn, types.FunctionType):
        raise SourceUnavailable(
            f"the synthesis target must be a Python function or bound method, not {type(fn).__name__}"
        )
    # The callable is built AS-IS: a wraps-style decorator may add behavior, so silently unwrapping would diverge.
    # Reading source via the CODE OBJECT (never the function) keeps inspect from following __wrapped__; a variadic
    # wrapper then rejects on its own parameters, which is the honest answer.
    try:
        source_lines, first_line = inspect.getsourcelines(fn.__code__)
    except (OSError, TypeError) as error:
        raise SourceUnavailable(f"could not retrieve source for {fn.__code__.co_qualname}: {error}") from None
    module = ast.parse(textwrap.dedent("".join(source_lines)))
    fndef = module.body[0]
    if not isinstance(fndef, ast.FunctionDef):
        raise BuildRejection("only plain functions can be kernels", (Origin(fn.__code__.co_qualname, first_line, 0),))
    assert fndef.name == fn.__code__.co_name, "source and code object must describe the same function"
    resolver = NameResolver(fn, comprehension_only=comprehension_only_targets(fndef))
    indent = len(source_lines[0]) - len(source_lines[0].lstrip())
    # Parameter bool-ness comes from the RESOLVED annotation object (``fn.__annotations__``), not the AST spelling:
    # ``x: builtins.bool`` and ``x: Alias`` are bool just as ``x: bool`` is. String annotations that fail to resolve
    # simply fall back to float, which is the safe default.
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {
            name: value for name, value in getattr(fn, "__annotations__", {}).items() if not isinstance(value, str)
        }
    builder = _Builder(fn.__code__.co_qualname, first_line, indent, resolver, bound_self, hints)
    return builder.build(fndef)


@dataclass(frozen=True, slots=True)
class _LoopContext:
    header: BlockId
    exit_target: BlockId


class _Builder:
    def __init__(
        self,
        qualname: str,
        first_line: int,
        indent: int,
        resolver: NameResolver,
        bound_self: object | None,
        hints: dict[str, object],
    ) -> None:
        self._qualname = qualname
        self._line_offset = first_line - 1
        self._column_offset = indent  # ast columns are post-dedent; diagnostics report source-file columns
        self._resolver = resolver
        self._bound_self = bound_self
        self._hints = hints
        self._blocks: dict[BlockId, Block] = {}
        self._current = self._new_block()
        self._serial = 0
        self._temp_serial = 0
        self._frames: list[dict[str, BindingId]] = [{}]
        self._loops: list[_LoopContext] = []
        self._comprehension_frames = 0  # running total of nested generators, bounded to keep recursion finite
        self._self_spelling: str | None = None  # the receiver parameter's runtime spelling, once known
        self._params_bound = False  # gate so the receiver's own initial binding is not mistaken for a rebinding
        self._fn_origin: OriginStack = ()

    def build(self, fndef: ast.FunctionDef) -> FunctionUnit:
        origin = self._origin(fndef)
        self._fn_origin = origin
        args = fndef.args
        if args.vararg or args.kwarg:
            raise BuildRejection("variadic parameters are not supported", origin)
        declared = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        if self._bound_self is not None and not declared:
            raise BuildRejection("a bound method must declare a receiver parameter", origin)
        if self._bound_self is not None:
            self._self_spelling = self._resolver.runtime_spelling(declared[0].arg)
        # A bound method's leading ``self`` is the instance receiver, not a datapath input, so it is exempt from
        # annotation validation; every other parameter needs an explicit scalar annotation (there is no implicit
        # float default). An array (jaxtyping/ndarray) annotation is admitted here and deferred at its point of use.
        for arg in declared[1:] if self._bound_self is not None else declared:
            hint = self._hints.get(self._resolver.runtime_spelling(arg.arg))
            if hint is None:
                raise BuildRejection(
                    f"parameter {arg.arg!r} requires an explicit type annotation (float or bool)", self._origin(arg)
                )
            if hint in (int, str, bytes, complex):
                raise BuildRejection(
                    f"unsupported parameter annotation for {arg.arg!r}: expected float or bool", self._origin(arg)
                )
        params = [self._bind(arg.arg) for arg in declared]
        self._params_bound = True  # any further store to the receiver name is now a rebinding, not the initial bind
        # Annotation keys are CPython-mangled just like the param slots (``__enabled`` -> ``_Klass__enabled``), so
        # both the hint lookup and the bool_params entry use the runtime spelling.
        bool_params = frozenset(
            spelled for arg in declared if self._hints.get(spelled := self._resolver.runtime_spelling(arg.arg)) is bool
        )
        declared_return_bool = self._validate_return_annotation(fndef, origin)
        entry = self._current.id
        exit_block = self._new_block()
        exit_block.terminator = UnitExit(origin)
        for statement in fndef.body:
            self._statement(statement, exit_block.id)
        if self._current.terminator is None:  # implicit `return None` on fall-off
            none_temp = self._temp()
            self._emit(LoadConst(none_temp, admit_ref(None), origin))
            self._emit(StorePlace(ReturnPlace(), none_temp, origin))
            self._current.terminator = Jump(exit_block.id, origin)
        unit = FunctionUnit(
            name=self._qualname,
            params=params,
            blocks=self._blocks,
            entry=entry,
            exit=exit_block.id,
            bound_self=self._bound_self,
            bool_params=bool_params,
            declared_return_bool=declared_return_bool,
        )
        verify(unit)
        return unit

    def _validate_return_annotation(self, fndef: ast.FunctionDef, origin: OriginStack) -> bool | None:
        # The declared return type is mandatory and drives the output-port type; a mismatch against the inferred value
        # type is caught at emission. ``None`` here means void or an aggregate (deferred), so no scalar check applies.
        if fndef.returns is None:
            raise BuildRejection("the return type must be explicitly annotated (float or bool)", origin)
        hint = self._hints.get("return")
        # ``float | None`` (Optional): the None arm is the implicit fall-off of an early-return kernel, so unwrap it.
        args = typing.get_args(hint)
        if typing.get_origin(hint) in (typing.Union, types.UnionType) and type(None) in args:
            remainder = [arg for arg in args if arg is not type(None)]
            hint = remainder[0] if len(remainder) == 1 else hint
        if hint is float:
            return False
        if hint is bool:
            return True
        if hint is type(None) or typing.get_origin(hint) in (tuple, list):
            return None
        raise BuildRejection(
            f"unsupported return annotation {getattr(hint, '__name__', hint)}: expected float, bool, or None", origin
        )

    # ---------------------------------------- machinery ----------------------------------------

    def _new_block(self) -> Block:
        block = Block(BlockId(len(self._blocks)))
        self._blocks[block.id] = block
        return block

    def _start_block(self, block: Block) -> None:
        self._current = block

    def _emit(self, op: Op) -> None:
        assert self._current.terminator is None, "op emitted after terminator"
        self._current.ops.append(op)

    def _temp(self) -> BindingId:
        binding = BindingId(f"%{self._temp_serial}", self._temp_serial)
        self._temp_serial += 1
        return binding

    def _bind(self, name: str) -> BindingId:
        """
        The base-frame binding of a source name, under its runtime spelling. Named stores always land in the
        function scope: statements cannot occur under a comprehension overlay, and a walrus target binds in the
        function scope even inside one (PEP 572). Only comprehension targets live in overlay frames.
        """
        name = self._resolver.runtime_spelling(name)
        if self._params_bound and name == self._self_spelling:
            raise BuildRejection("reassigning the instance parameter is not supported", self._fn_origin)
        if name in self._frames[0]:
            return self._frames[0][name]
        binding = BindingId(name, self._serial)
        self._serial += 1
        self._frames[0][name] = binding
        return binding

    def _origin(self, node: ast.AST) -> OriginStack:
        line = getattr(node, "lineno", 1) + self._line_offset
        column = getattr(node, "col_offset", 0) + self._column_offset
        return (Origin(self._qualname, line, column),)

    def _reject_walrus(self, node: ast.expr, position: str) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.NamedExpr):
                raise BuildRejection(f"assignment expression is not supported in {position}", self._origin(sub))

    # ---------------------------------------- statements ----------------------------------------

    def _statement(self, node: ast.stmt, exit_target: BlockId) -> None:
        origin = self._origin(node)
        match node:
            case ast.Assign(targets=targets, value=value):
                source = self._expression(value)
                for target in targets:
                    self._assign_target(target, source, origin)
            case ast.AnnAssign(target=target, value=value):
                if value is not None:
                    self._assign_target(target, self._expression(value), origin)
                else:
                    match target:  # Python evaluates the receiver (and index) of a bare non-name annotation
                        case ast.Attribute(value=receiver):
                            self._expression(receiver)
                        case ast.Subscript(value=receiver, slice=index):
                            self._expression(receiver)
                            self._expression(index)
                        case _:
                            pass
            case ast.AugAssign(target=target, op=op, value=value):
                bin_op = _BIN_OPS.get(type(op))
                if bin_op is None:
                    raise BuildRejection(f"operator {type(op).__name__} is not supported", origin)
                match target:
                    case ast.Name(id=name):
                        binding = self._bind(name)
                        current = self._temp()
                        self._emit(LoadPlace(current, Local(binding), origin))
                        operand = self._expression(value)
                        result = self._temp()
                        self._emit(PyBin(result, bin_op, current, operand, True, origin))
                        self._emit(StorePlace(Local(binding), result, origin))
                    case ast.Attribute(value=obj, attr=attr):
                        obj_temp = self._expression(obj)  # evaluated once, exactly as Python does
                        spelled = self._resolver.runtime_spelling(attr)
                        current = self._temp()
                        self._emit(PyAttr(current, obj_temp, spelled, origin))
                        operand = self._expression(value)
                        result = self._temp()
                        self._emit(PyBin(result, bin_op, current, operand, True, origin))
                        self._emit(PyStoreAttr(obj_temp, spelled, result, origin))
                    case _:
                        raise BuildRejection(
                            f"augmented assignment to {type(target).__name__} is not supported", origin
                        )
            case ast.Return(value=value):
                result = self._expression(value) if value is not None else self._none(origin)
                self._emit(StorePlace(ReturnPlace(), result, origin))
                self._current.terminator = Jump(exit_target, origin)
                self._start_block(self._new_block())
            case ast.If(test=test, body=body, orelse=orelse):
                condition = self._truth(test)
                then_block, else_block, join = self._new_block(), self._new_block(), self._new_block()
                self._current.terminator = Branch(condition, then_block.id, else_block.id, origin)
                self._start_block(then_block)
                for inner in body:
                    self._statement(inner, exit_target)
                if self._current.terminator is None:
                    self._current.terminator = Jump(join.id, origin)
                self._start_block(else_block)
                for inner in orelse:
                    self._statement(inner, exit_target)
                if self._current.terminator is None:
                    self._current.terminator = Jump(join.id, origin)
                self._start_block(join)
            case ast.While(test=test, body=body, orelse=orelse):
                if orelse:
                    raise BuildRejection("while-else is not supported", origin)
                header, body_block, after = self._new_block(), self._new_block(), self._new_block()
                self._current.terminator = Jump(header.id, origin)
                self._start_block(header)
                condition = self._truth(test)  # re-evaluated per iteration, so a walrus here binds on every test
                self._current.terminator = Branch(condition, body_block.id, after.id, origin)
                self._start_block(body_block)
                self._loops.append(_LoopContext(header.id, after.id))
                for inner in body:
                    self._statement(inner, exit_target)
                self._loops.pop()
                if self._current.terminator is None:
                    self._current.terminator = Jump(header.id, origin)
                self._start_block(after)
            case ast.For(target=target, iter=iterable, body=body, orelse=orelse):
                if orelse:
                    raise BuildRejection("for-else is not supported", origin)
                if not isinstance(target, ast.Name):
                    raise BuildRejection("only a plain name is supported as a for-loop target", origin)
                iterable_temp = self._expression(iterable)
                header, body_block, after = self._new_block(), self._new_block(), self._new_block()
                self._current.terminator = Jump(header.id, origin)
                self._start_block(body_block)
                self._loops.append(_LoopContext(header.id, after.id))
                for inner in body:
                    self._statement(inner, exit_target)
                self._loops.pop()
                if self._current.terminator is None:
                    self._current.terminator = Jump(header.id, origin)
                self._start_block(after)
                members = frozenset(block_id for block_id in self._blocks if block_id.index >= body_block.id.index) - {
                    header.id,
                    after.id,
                }
                header.terminator = StaticFor(
                    Local(self._bind(target.id)), iterable_temp, body_block.id, after.id, members, origin
                )
            case ast.Break():
                if not self._loops:
                    raise BuildRejection("break outside a loop", origin)
                self._current.terminator = Jump(self._loops[-1].exit_target, origin)
                self._start_block(self._new_block())
            case ast.Continue():
                if not self._loops:
                    raise BuildRejection("continue outside a loop", origin)
                self._current.terminator = Jump(self._loops[-1].header, origin)
                self._start_block(self._new_block())
            case ast.Expr(value=value):
                self._expression(value)
            case ast.Pass():
                pass
            case ast.Delete(targets=targets):
                for target in targets:
                    if not isinstance(target, ast.Name):
                        raise BuildRejection("only plain names can be deleted", origin)
                    self._emit(UnbindPlace(Local(self._bind(target.id)), True, origin))
            case ast.Raise(exc=exc):
                self._current.terminator = Fail(self._raise_message(exc), origin)
                self._start_block(self._new_block())
            case ast.Assert():
                # Accepted and ignored wholesale: the test is never lowered, mirroring Python under -O. Any effect the
                # test would have had (a walrus binding, a call) is dropped with it, so an assert must be side-effect-
                # free; a later read of a name a dropped assert would have bound is a normal unbound-name rejection.
                _logger.info("assert at %s has no effect in Holoso and is dropped", origin[0])
            case ast.FunctionDef() | ast.AsyncFunctionDef() | ast.ClassDef():
                raise BuildRejection("nested function and class definitions are not supported in kernels", origin)
            case ast.Import() | ast.ImportFrom():
                raise BuildRejection("imports inside a kernel are not supported", origin)
            case ast.Global() | ast.Nonlocal():
                raise BuildRejection("global/nonlocal declarations are not supported in kernels", origin)
            case _:
                raise BuildRejection(f"statement {type(node).__name__} is not supported", origin)

    def _assign_target(self, target: ast.expr, source: BindingId, origin: OriginStack) -> None:
        match target:
            case ast.Name(id=name):
                self._emit(StorePlace(Local(self._bind(name)), source, origin))
            case ast.Attribute(value=obj, attr=attr):
                obj_temp = self._expression(obj)
                self._emit(PyStoreAttr(obj_temp, self._resolver.runtime_spelling(attr), source, origin))
            case ast.Tuple(elts=elts) | ast.List(elts=elts):
                if any(isinstance(elt, ast.Starred) for elt in elts):
                    raise BuildRejection("starred unpacking targets are not supported", origin)
                # The arity constraint must be encoded here: downstream, a prefix read is indistinguishable from
                # an honest too-many-values mistake that Python would refuse with ValueError.
                length = self._temp()
                self._emit(PyLen(length, source, origin))
                expected = self._temp()
                arity = admit(len(elts))
                assert arity is not None
                self._emit(LoadConst(expected, arity, origin))
                matches = self._temp()
                self._emit(PyCompare(matches, RelationalOp.EQ, length, expected, origin))
                arity_ok = self._temp()
                self._emit(PyTruth(arity_ok, matches, origin))
                unpack_block, mismatch = self._new_block(), self._new_block()
                self._current.terminator = Branch(arity_ok, unpack_block.id, mismatch.id, origin)
                mismatch.terminator = Fail(f"cannot unpack: expected {len(elts)} values", origin)
                self._start_block(unpack_block)
                element_temps = []
                for index in range(len(elts)):
                    index_temp = self._temp()
                    index_value = admit(index)
                    assert index_value is not None
                    self._emit(LoadConst(index_temp, index_value, origin))
                    element = self._temp()
                    self._emit(PySubscript(element, source, index_temp, origin))
                    element_temps.append(element)
                for elt, element in zip(elts, element_temps):  # all reads precede all writes: a, b = b, a swaps
                    self._assign_target(elt, element, origin)
            case ast.Subscript():
                raise BuildRejection("assignment to a subscript is not supported (aggregates are immutable)", origin)
            case _:
                raise BuildRejection(f"assignment target {type(target).__name__} is not supported", origin)

    def _raise_message(self, exc: ast.expr | None) -> str:
        match exc:
            case ast.Call(args=[ast.Constant(value=str(message)), *_]):
                return message
            case _:
                return "raise"

    # ---------------------------------------- expressions ----------------------------------------

    def _none(self, origin: OriginStack) -> BindingId:
        temp = self._temp()
        self._emit(LoadConst(temp, admit_ref(None), origin))
        return temp

    def _truth(self, node: ast.expr) -> BindingId:
        value = self._expression(node)
        temp = self._temp()
        self._emit(PyTruth(temp, value, self._origin(node)))
        return temp

    def _expression(self, node: ast.expr) -> BindingId:
        origin = self._origin(node)
        match node:
            case ast.Constant(value=value):
                admitted = admit(value)
                if admitted is None and value is not None:
                    raise BuildRejection(f"constant of type {type(value).__name__} is not supported", origin)
                temp = self._temp()
                self._emit(LoadConst(temp, admitted if admitted is not None else admit_ref(None), origin))
                return temp
            case ast.Name(id=name):
                return self._load_name(name, origin)
            case ast.NamedExpr(target=ast.Name(id=name), value=value):
                result = self._expression(value)
                self._emit(StorePlace(Local(self._bind(name)), result, origin))
                return result
            case ast.BinOp(left=left, op=op, right=right):
                bin_op = _BIN_OPS.get(type(op))
                if bin_op is None:
                    raise BuildRejection(f"operator {type(op).__name__} is not supported", origin)
                lhs = self._expression(left)
                rhs = self._expression(right)
                temp = self._temp()
                self._emit(PyBin(temp, bin_op, lhs, rhs, False, origin))
                return temp
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                return self._unary(UnOp.NEG, operand, origin)
            case ast.UnaryOp(op=ast.UAdd(), operand=operand):
                return self._unary(UnOp.POS, operand, origin)
            case ast.UnaryOp(op=ast.Not(), operand=operand):
                negated = self._expression(operand)
                temp = self._temp()
                self._emit(PyNot(temp, negated, origin))
                return temp
            case ast.UnaryOp(op=ast.Invert()):
                raise BuildRejection("operator Invert is not supported", origin)
            case ast.BoolOp(op=op, values=values):
                mode = SelectMode.AND if isinstance(op, ast.And) else SelectMode.OR
                for operand in values[1:]:
                    self._reject_walrus(operand, "a short-circuit position of and/or")
                accumulated = self._expression(values[0])
                for operand in values[1:]:
                    rhs = self._expression(operand)  # eager: pinned language semantics
                    condition = self._temp()
                    self._emit(PyTruth(condition, accumulated, origin))
                    combined = self._temp()
                    self._emit(PySelect(combined, mode, condition, accumulated, rhs, origin))
                    accumulated = combined
                return accumulated
            case ast.Compare(left=left, ops=ops, comparators=comparators):
                for tail in comparators[1:]:
                    self._reject_walrus(tail, "a chained-comparison tail")
                operands = [self._expression(left)] + [self._expression(comparator) for comparator in comparators]
                links = []
                for index, cmp_op in enumerate(ops):
                    rel = _COMPARE_OPS.get(type(cmp_op))
                    if rel is None:
                        raise BuildRejection(f"comparison {type(cmp_op).__name__} is not supported", origin)
                    link = self._temp()
                    self._emit(PyCompare(link, rel, operands[index], operands[index + 1], origin))
                    links.append(link)
                accumulated = links[0]
                for link in links[1:]:
                    condition = self._temp()
                    self._emit(PyTruth(condition, accumulated, origin))
                    combined = self._temp()
                    self._emit(PySelect(combined, SelectMode.AND, condition, accumulated, link, origin))
                    accumulated = combined
                return accumulated
            case ast.IfExp(test=test, body=body, orelse=orelse):
                self._reject_walrus(body, "a conditional-expression arm")
                self._reject_walrus(orelse, "a conditional-expression arm")
                condition = self._truth(test)
                result = BindingId(f"ifexp@{origin[0].line}", self._serial)
                self._serial += 1
                then_block, else_block, join = self._new_block(), self._new_block(), self._new_block()
                self._current.terminator = Branch(condition, then_block.id, else_block.id, origin)
                self._start_block(then_block)
                then_value = self._expression(body)
                self._emit(StorePlace(Local(result), then_value, origin))
                self._current.terminator = Jump(join.id, origin)
                self._start_block(else_block)
                else_value = self._expression(orelse)
                self._emit(StorePlace(Local(result), else_value, origin))
                self._current.terminator = Jump(join.id, origin)
                self._start_block(join)
                temp = self._temp()
                self._emit(LoadPlace(temp, Local(result), origin))
                return temp
            case ast.Call(func=func, args=args, keywords=keywords):
                if any(isinstance(arg, ast.Starred) for arg in args) or any(kw.arg is None for kw in keywords):
                    raise BuildRejection("argument unpacking in calls is not supported", origin)
                callee = self._expression(func)
                arg_temps = tuple(self._expression(arg) for arg in args)
                kwarg_temps = tuple((kw.arg, self._expression(kw.value)) for kw in keywords if kw.arg is not None)
                temp = self._temp()
                self._emit(PyCall(temp, callee, arg_temps, kwarg_temps, origin))
                return temp
            case ast.Attribute(value=obj, attr=attr):
                obj_temp = self._expression(obj)
                temp = self._temp()
                self._emit(PyAttr(temp, obj_temp, self._resolver.runtime_spelling(attr), origin))
                return temp
            case ast.Subscript(value=obj, slice=index):
                if isinstance(index, ast.Slice):
                    raise BuildRejection("slicing is not supported", origin)
                obj_temp = self._expression(obj)
                index_temp = self._expression(index)
                temp = self._temp()
                self._emit(PySubscript(temp, obj_temp, index_temp, origin))
                return temp
            case ast.Tuple(elts=elts):
                items = tuple(self._expression(elt) for elt in elts)
                temp = self._temp()
                self._emit(BuildTuple(temp, items, origin))
                return temp
            case ast.List(elts=elts):
                items = tuple(self._expression(elt) for elt in elts)
                temp = self._temp()
                self._emit(BuildList(temp, items, origin))
                return temp
            case ast.ListComp(elt=elt, generators=generators):
                return self._comprehension(elt, generators, origin)
            case ast.GeneratorExp() | ast.SetComp() | ast.DictComp():
                raise BuildRejection("only list comprehensions are supported", origin)
            case ast.Lambda():
                raise BuildRejection("lambda expressions are not supported in kernels", origin)
            case _:
                raise BuildRejection(f"expression {type(node).__name__} is not supported", origin)

    def _unary(self, op: UnOp, operand: ast.expr, origin: OriginStack) -> BindingId:
        value = self._expression(operand)
        temp = self._temp()
        self._emit(PyUn(temp, op, value, origin))
        return temp

    def _load_name(self, name: str, origin: OriginStack) -> BindingId:
        name = self._resolver.runtime_spelling(name)
        for frame in reversed(self._frames):
            if name in frame:
                temp = self._temp()
                self._emit(LoadPlace(temp, Local(frame[name]), origin))
                return temp
        try:
            resolution = self._resolver.resolve(name)
        except UnboundCell:
            self._current.terminator = Fail(f"cannot access free variable '{name}'", origin)
            self._start_block(self._new_block())
            return self._none(origin)
        match resolution:
            case ResolvedLocal(name=runtime_name):
                temp = self._temp()
                self._emit(LoadPlace(temp, Local(self._bind(runtime_name)), origin))
                return temp
            case Free(value=value) | Global(value=value) | Builtin(value=value):
                admitted = admit(value)
                temp = self._temp()
                self._emit(LoadConst(temp, admitted if admitted is not None else admit_ref(value), origin))
                return temp
            case Missing(name=runtime_name):
                # Not a build-time rejection: Python raises only if the read executes, so a dead branch stays dead.
                self._current.terminator = Fail(f"name '{runtime_name}' is not defined", origin)
                self._start_block(self._new_block())
                return self._none(origin)

    def _comprehension(self, elt: ast.expr, generators: list[ast.comprehension], origin: OriginStack) -> BindingId:
        # One expansion frame per generator, and a nested comprehension's frames sit atop this one's, so bounding the
        # running total bounds the build recursion (a deeply nested comprehension would otherwise overflow the stack).
        self._comprehension_frames += len(generators)
        if self._comprehension_frames > UNROLL_THRESHOLD:
            raise BuildRejection(f"comprehension nesting expands more than {UNROLL_THRESHOLD} generators", origin)
        try:
            return self._comprehension_body(elt, generators, origin)
        finally:
            self._comprehension_frames -= len(generators)

    def _comprehension_body(self, elt: ast.expr, generators: list[ast.comprehension], origin: OriginStack) -> BindingId:
        accumulator = BindingId(f"listcomp@{origin[0].line}", self._serial)
        self._serial += 1
        empty = self._temp()
        self._emit(BuildList(empty, (), origin))
        self._emit(StorePlace(Local(accumulator), empty, origin))
        frame: dict[str, BindingId] = {}
        for generator in generators:
            if not isinstance(generator.target, ast.Name):
                raise BuildRejection("only a plain name is supported as a comprehension target", origin)
            name = self._resolver.runtime_spelling(generator.target.id)
            if name not in frame:
                frame[name] = BindingId(name, self._serial)
                self._serial += 1
        self._frames.append(frame)
        self._comprehension_level(elt, generators, 0, accumulator, origin)
        self._frames.pop()
        result = self._temp()
        self._emit(LoadPlace(result, Local(accumulator), origin))
        return result

    def _comprehension_level(
        self,
        elt: ast.expr,
        generators: list[ast.comprehension],
        level: int,
        accumulator: BindingId,
        origin: OriginStack,
    ) -> None:
        if level == len(generators):
            element = self._expression(elt)
            single = self._temp()
            self._emit(BuildList(single, (element,), origin))
            previous = self._temp()
            self._emit(LoadPlace(previous, Local(accumulator), origin))
            extended = self._temp()
            self._emit(PyBin(extended, BinOp.ADD, previous, single, False, origin))
            self._emit(StorePlace(Local(accumulator), extended, origin))
            return
        generator = generators[level]
        if generator.is_async:
            raise BuildRejection("async comprehensions are not supported", origin)
        assert isinstance(generator.target, ast.Name)
        # The outermost iterable is evaluated in the enclosing scope: the frame overlay must not capture its reads.
        # Inner iterables evaluate inside the frame, exactly like Python.
        frame = self._frames[-1]
        if level == 0:
            self._frames.pop()
        iterable_temp = self._expression(generator.iter)
        if level == 0:
            self._frames.append(frame)
            for binding in frame.values():  # a fresh scope per execution: targets start unbound, as Python's do
                self._emit(UnbindPlace(Local(binding), False, origin))
        target = frame[self._resolver.runtime_spelling(generator.target.id)]
        header, body_block, after = self._new_block(), self._new_block(), self._new_block()
        self._current.terminator = Jump(header.id, origin)
        self._start_block(body_block)
        for condition_expr in generator.ifs:
            condition = self._truth(condition_expr)
            keep_block = self._new_block()
            self._current.terminator = Branch(condition, keep_block.id, header.id, self._origin(condition_expr))
            self._start_block(keep_block)
        self._comprehension_level(elt, generators, level + 1, accumulator, origin)
        if self._current.terminator is None:
            self._current.terminator = Jump(header.id, origin)
        self._start_block(after)
        members = frozenset(block_id for block_id in self._blocks if block_id.index >= body_block.id.index) - {
            header.id,
            after.id,
        }
        header.terminator = StaticFor(Local(target), iterable_temp, body_block.id, after.id, members, origin)

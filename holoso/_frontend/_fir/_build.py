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

import annotationlib
import ast
import inspect
import logging
import types
import typing
from dataclasses import dataclass

from ..._errors import SourceUnavailable, UnsupportedConstruct
from ..._util import RelationalOp
from .._ast_support import UNROLL_THRESHOLD, indexed_names
from ._signature import (
    RecordParameter,
    ArrayParameter,
    ContractError,
    ParameterContract,
    ReturnContract,
    parameter_contract,
    return_contract,
)
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
    LoadRef,
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
    StoreRole,
    UnbindPlace,
    UnitExit,
    LocatedRejection,
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
from ._value import admit

_logger = logging.getLogger(__name__)


class BuildRejection(LocatedRejection, UnsupportedConstruct):
    """A located refusal: the construct is outside the supported subset (or plainly wrong Python)."""


_BIN_OPS: dict[type[ast.operator], BinOp] = {
    ast.Add: BinOp.ADD,
    ast.Sub: BinOp.SUB,
    ast.Mult: BinOp.MUL,
    ast.Div: BinOp.DIV,
    ast.FloorDiv: BinOp.FLOORDIV,
    ast.Mod: BinOp.MOD,
    ast.Pow: BinOp.POW,
    ast.MatMult: BinOp.MATMUL,
    ast.LShift: BinOp.LSHIFT,
    ast.RShift: BinOp.RSHIFT,
    ast.BitAnd: BinOp.BITAND,
    ast.BitOr: BinOp.BITOR,
    ast.BitXor: BinOp.BITXOR,
}

_COMPARE_OPS: dict[type[ast.cmpop], RelationalOp] = {
    ast.Eq: RelationalOp.EQ,
    ast.NotEq: RelationalOp.NE,
    ast.Lt: RelationalOp.LT,
    ast.LtE: RelationalOp.LE,
    ast.Gt: RelationalOp.GT,
    ast.GtE: RelationalOp.GE,
}


def _contract_cells(base: str, contract: "ParameterContract") -> list[str]:
    match contract:
        case ArrayParameter(shape=shape):
            return indexed_names(base, shape)
        case RecordParameter(fields=fields):
            return [cell for name, sub in fields for cell in _contract_cells(f"{base}_{name}", sub)]
        case _:
            return [base]


def build_unit(fn: object, root: bool = False) -> FunctionUnit:
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
    source = "".join(source_lines)
    indent = len(source_lines[0]) - len(source_lines[0].lstrip())
    # An indented kernel (a method) parses inside a synthetic block rather than being textually dedented: dedent
    # would also strip the interior of multiline string literals, and a column-zero line inside a docstring makes
    # it a no-op that leaves the def unparseable. The wrapper keeps every literal intact and yields true source
    # columns, so diagnostics need no column compensation.
    if indent:
        module = ast.parse("if 1:\n" + source)
        ast.increment_lineno(module, -1)
        wrapper = module.body[0]
        assert isinstance(wrapper, ast.If)
        body = wrapper.body
    else:
        module = ast.parse(source)
        body = list(module.body)
    fndef = body[0]
    if not isinstance(fndef, ast.FunctionDef):
        raise BuildRejection(
            "only plain functions can be kernels",
            (Origin(fn.__code__.co_qualname, first_line, 0, fn.__code__.co_filename),),
        )
    assert fndef.name == fn.__code__.co_name, "source and code object must describe the same function"
    resolver = NameResolver(fn, comprehension_only=comprehension_only_targets(fndef))
    # Parameter bool-ness comes from the RESOLVED annotation object (``fn.__annotations__``), not the AST spelling:
    # ``x: builtins.bool`` and ``x: Alias`` are bool just as ``x: bool`` is. Unresolvable STRING annotations are
    # dropped (the parameter then rejects as unannotated). Only the root kernel's annotations are contracts —
    # a callee's are documentation (never evaluated by Python either), so hints are not even computed for one.
    hints: dict[str, object] = {}
    if root:
        try:
            hints = typing.get_type_hints(fn)
        except Exception:
            # PEP 649 evaluates annotations only when read, so a typo'd one raises here, not at definition time.
            # FORWARDREF never raises: the failing names come back as ForwardRef proxies, and only the
            # annotations the port boundary actually consumes reject — the receiver's stays documentation,
            # exactly as Python (which never evaluates it either) behaves.
            hints = dict(annotationlib.get_annotations(fn, format=annotationlib.Format.FORWARDREF))
    builder = _Builder(fn.__code__.co_qualname, fn.__code__.co_filename, first_line, resolver, bound_self, hints, root)
    return builder.build(fndef)


@dataclass(frozen=True, slots=True)
class _LoopContext:
    header: BlockId
    exit_target: BlockId


class _Builder:
    def __init__(
        self,
        qualname: str,
        file: str,
        first_line: int,
        resolver: NameResolver,
        bound_self: object | None,
        hints: dict[str, object],
        root: bool,
    ) -> None:
        self._qualname = qualname
        self._file = file
        self._line_offset = first_line - 1
        self._resolver = resolver
        self._bound_self = bound_self
        self._hints = hints
        self._root = root
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
        # annotation validation; every other parameter needs an explicit annotation (there is no implicit float
        # default). Annotation keys are CPython-mangled just like the param slots (``__enabled`` ->
        # ``_Klass__enabled``), so the hint lookup and the contract entry use the runtime spelling.
        param_contracts: dict[str, ParameterContract] = {}
        port_claims: dict[str, str] = {}
        for arg in declared[1:] if self._bound_self is not None else declared:
            if not self._root:
                continue  # a callee's facts flow from its call sites; annotations are documentation, if present
            spelled = self._resolver.runtime_spelling(arg.arg)
            hint = self._hints.get(spelled) if spelled is not None else None
            if hint is None:
                raise BuildRejection(
                    f"parameter {arg.arg!r} requires an explicit type annotation (float or bool)", self._origin(arg)
                )
            if isinstance(hint, (str, annotationlib.ForwardRef)):
                unresolved = getattr(hint, "__forward_arg__", hint)
                raise BuildRejection(
                    f"the annotation of parameter {arg.arg!r} does not resolve ({unresolved})", self._origin(arg)
                )
            try:
                contract = parameter_contract(hint)
            except ContractError as error:
                raise BuildRejection(
                    f"unsupported parameter annotation for {arg.arg!r}: {error}", self._origin(arg)
                ) from None
            assert spelled is not None
            param_contracts[spelled] = contract
            for cell in _contract_cells(spelled, contract):
                if cell in port_claims:
                    # Includes SELF-collisions: the underscore join is not injective (a field 'a_b' beside a
                    # nested 'a.b' renders the same cell), so an ambiguous record refuses here instead of
                    # emitting duplicate ports that fail deep in synthesis.
                    raise BuildRejection(
                        f"parameter {spelled!r} decomposes onto input port '{cell}', which "
                        f"{port_claims[cell]!r} already claims (rename one of them)",
                        self._origin(arg),
                    )
                port_claims[cell] = spelled
        params = [self._bind(arg.arg) for arg in declared]
        self._params_bound = True  # any further store to the receiver name is now a rebinding, not the initial bind
        declared_return: ReturnContract | None = None
        if self._root:  # the port boundary; a callee's return remaps to a caller local, its annotation untouched
            if fndef.returns is None:
                raise BuildRejection("the return type must be explicitly annotated (float or bool)", origin)
            return_hint = self._hints.get("return")
            if isinstance(return_hint, (str, annotationlib.ForwardRef)):
                unresolved = getattr(return_hint, "__forward_arg__", return_hint)
                raise BuildRejection(f"the return annotation does not resolve ({unresolved})", origin)
            try:
                declared_return = return_contract(return_hint)
            except ContractError as error:
                raise BuildRejection(f"unsupported return annotation: {error}", origin)
        entry = self._current.id
        exit_block = self._new_block()
        exit_block.terminator = UnitExit(origin)
        for statement in fndef.body:
            self._statement(statement, exit_block.id)
        if self._current.terminator is None:  # implicit `return None` on fall-off
            none_temp = self._temp()
            self._emit(LoadRef(none_temp, None, origin))
            self._emit(StorePlace(ReturnPlace(), none_temp, origin, StoreRole.RETURN))
            self._current.terminator = Jump(exit_block.id, origin)
        unit = FunctionUnit(
            name=self._qualname,
            file=self._file,
            params=params,
            blocks=self._blocks,
            entry=entry,
            exit=exit_block.id,
            bound_self=self._bound_self,
            param_contracts=param_contracts,
            return_contract=declared_return,
        )
        verify(unit)
        return unit

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
        return (Origin(self._qualname, line, getattr(node, "col_offset", 0), self._file),)

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
                        self._emit(StorePlace(Local(binding), result, origin, StoreRole.SOURCE))
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
                self._emit(StorePlace(ReturnPlace(), result, origin, StoreRole.RETURN))
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
                self._current.terminator = Fail(self._raise_parts(exc), origin)
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
                self._emit(StorePlace(Local(self._bind(name)), source, origin, StoreRole.SOURCE))
            case ast.Attribute(value=obj, attr=attr):
                obj_temp = self._expression(obj)
                self._emit(PyStoreAttr(obj_temp, self._resolver.runtime_spelling(attr), source, origin))
            case ast.Tuple(elts=elts) | ast.List(elts=elts):
                for elt in elts:
                    if isinstance(elt, ast.Starred):
                        raise BuildRejection(
                            "a starred element is not supported in an assignment target", self._origin(elt)
                        )
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
                mismatch.terminator = Fail((f"cannot unpack: expected {len(elts)} values",), origin)
                self._start_block(unpack_block)
                element_temps = []
                for position in range(len(elts)):
                    element = self._temp()
                    index = self._temp()
                    admitted = admit(position)
                    assert admitted is not None
                    self._emit(LoadConst(index, admitted, origin))
                    self._emit(PySubscript(element, source, index, origin))
                    element_temps.append(element)
                for elt, element in zip(elts, element_temps):  # all reads precede all writes: a, b = b, a swaps
                    self._assign_target(elt, element, origin)
            case ast.Subscript():
                raise BuildRejection("assignment to a subscript is not supported (aggregates are immutable)", origin)
            case _:
                raise BuildRejection(f"assignment target {type(target).__name__} is not supported", origin)

    def _raise_parts(self, exc: ast.expr | None) -> tuple[str | BindingId, ...]:
        match exc:
            case ast.Call(args=[ast.Constant(value=str(message)), *_]):
                return (message,)
            case ast.Call(args=[ast.JoinedStr(values=pieces), *_]):
                parts: list[str | BindingId] = []
                for piece in pieces:
                    match piece:
                        case ast.Constant(value=str(text)):
                            parts.append(text)
                        case ast.FormattedValue(value=value, conversion=-1, format_spec=None):
                            parts.append(self._expression(value))
                        case _:
                            return ("raise",)
                return tuple(parts)
            case _:
                return ("raise",)

    # ---------------------------------------- expressions ----------------------------------------

    def _none(self, origin: OriginStack) -> BindingId:
        temp = self._temp()
        self._emit(LoadRef(temp, None, origin))
        return temp

    def _slice_value(self, index: ast.Slice, origin: OriginStack) -> BindingId:
        # Slice syntax desugars to the vetted slice() constructor: static bounds fold to a slice VALUE
        # (StaticSlice), which the subscript transfer consumes structurally; runtime bounds reject at the
        # call exactly like an explicit slice(a, b) spelling would.
        slice_ref = self._temp()
        self._emit(LoadRef(slice_ref, slice, origin))
        bounds = tuple(
            self._expression(bound) if bound is not None else self._none(origin)
            for bound in (index.lower, index.upper, index.step)
        )
        temp = self._temp()
        self._emit(PyCall(temp, slice_ref, bounds, (), origin))
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
                if admitted is not None:
                    self._emit(LoadConst(temp, admitted, origin))
                else:
                    self._emit(LoadRef(temp, None, origin))
                return temp
            case ast.Name(id=name):
                return self._load_name(name, origin)
            case ast.NamedExpr(target=ast.Name(id=name), value=value):
                result = self._expression(value)
                self._emit(StorePlace(Local(self._bind(name)), result, origin, StoreRole.SOURCE))
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
                self._emit(StorePlace(Local(result), then_value, origin, StoreRole.MERGE))
                self._current.terminator = Jump(join.id, origin)
                self._start_block(else_block)
                else_value = self._expression(orelse)
                self._emit(StorePlace(Local(result), else_value, origin, StoreRole.MERGE))
                self._current.terminator = Jump(join.id, origin)
                self._start_block(join)
                temp = self._temp()
                self._emit(LoadPlace(temp, Local(result), origin))
                return temp
            case ast.Call(func=func, args=args, keywords=keywords):
                if any(kw.arg is None for kw in keywords):
                    raise BuildRejection("dictionary argument unpacking (**kwargs) is not supported", origin)
                callee = self._expression(func)
                arg_temps = tuple(self._expression(arg.value if isinstance(arg, ast.Starred) else arg) for arg in args)
                stars = tuple(isinstance(arg, ast.Starred) for arg in args)
                kwarg_temps = tuple((kw.arg, self._expression(kw.value)) for kw in keywords if kw.arg is not None)
                temp = self._temp()
                self._emit(PyCall(temp, callee, arg_temps, kwarg_temps, origin, starred=stars if any(stars) else ()))
                return temp
            case ast.Attribute(value=obj, attr=attr):
                obj_temp = self._expression(obj)
                temp = self._temp()
                self._emit(PyAttr(temp, obj_temp, self._resolver.runtime_spelling(attr), origin))
                return temp
            case ast.Subscript(value=obj, slice=index):
                obj_temp = self._expression(obj)
                if isinstance(index, ast.Tuple) and any(isinstance(elt, ast.Slice) for elt in index.elts):
                    # A slice inside a multi-axis key (m[:, 1]) desugars element-wise, so the key becomes an
                    # ordinary tuple of values the subscript transfer consumes structurally.
                    for elt in index.elts:
                        if isinstance(elt, ast.Starred):
                            raise BuildRejection(
                                "a starred element is not supported in a list or tuple display", self._origin(elt)
                            )
                    items = tuple(
                        self._slice_value(elt, origin) if isinstance(elt, ast.Slice) else self._expression(elt)
                        for elt in index.elts
                    )
                    index_temp = self._temp()
                    self._emit(BuildTuple(index_temp, items, origin))
                elif isinstance(index, ast.Slice):
                    index_temp = self._slice_value(index, origin)
                else:
                    index_temp = self._expression(index)
                temp = self._temp()
                self._emit(PySubscript(temp, obj_temp, index_temp, origin))
                return temp
            case ast.Tuple(elts=elts):
                items = self._display_items(elts)
                temp = self._temp()
                self._emit(BuildTuple(temp, items, origin))
                return temp
            case ast.List(elts=elts):
                items = self._display_items(elts)
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

    def _display_items(self, elts: list[ast.expr]) -> tuple[BindingId, ...]:
        for elt in elts:
            if isinstance(elt, ast.Starred):
                raise BuildRejection("a starred element is not supported in a list or tuple display", self._origin(elt))
        return tuple(self._expression(elt) for elt in elts)

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
            self._current.terminator = Fail((f"cannot access free variable '{name}'",), origin)
            self._start_block(self._new_block())
            return self._none(origin)
        match resolution:
            case ResolvedLocal(name=runtime_name):
                temp = self._temp()
                self._emit(LoadPlace(temp, Local(self._bind(runtime_name)), origin))
                return temp
            case Free(value=value) | Global(value=value) | Builtin(value=value):
                if type(value).__name__ == "ndarray" and getattr(value, "ndim", None) == 0:
                    # Detected structurally (numpy stays out of the builder), matching the admission refusal.
                    raise BuildRejection("a 0-dimensional array is not supported; use the scalar directly", origin)
                admitted = admit(value)
                temp = self._temp()
                if admitted is not None:
                    self._emit(LoadConst(temp, admitted, origin))
                else:
                    self._emit(LoadRef(temp, value, origin))
                return temp
            case Missing(name=runtime_name):
                # Not a build-time rejection: Python raises only if the read executes, so a dead branch stays dead.
                self._current.terminator = Fail((f"name '{runtime_name}' is not defined",), origin)
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
        self._emit(StorePlace(Local(accumulator), empty, origin, StoreRole.ACCUMULATOR))
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
            self._emit(StorePlace(Local(accumulator), extended, origin, StoreRole.ACCUMULATOR))
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

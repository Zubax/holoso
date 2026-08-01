"""
The specializing interpreter: Eel -> residual Eel. Binding time is its whole product; the rest follows
CPython or the ratified rulings, of which the non-obvious ones are recorded here.

Static folds go through the selected HIR operator's own ``evaluate``, and a fold naming no number
residualizes with constant operands: one expression, one answer, refusal survivor-based in HIR. No algebra
over residual operands -- that is HIR strength reduction's. Laziness is CPython's: a dead arm resolves no
names and desugars no callees; captures are judged at use, never at capture.

A name bound on only one side of a residual branch drops (CPython would raise UnboundLocalError); divergent
unmergeable values stay bound to a marker so the read, in any spelling, rejects truthfully. ``raise`` is
judged within its own function: unconditional there is a compile-time diagnostic even under a caller's
residual arm. Library stubs bind positionally only. All pow spellings share the ``**`` lowering; a fully
static non-integer power folds THROUGH the pow_ stub so the answer cannot depend on binding time. Recorded
edge, do not "fix": a power chain over a static base saturates to infinity where CPython raises
OverflowError, identical to HIR folding the same chain, covered by the fastmath charter.

Ownership events ride value flow: a desugared temp is a linear conduit, so the first read of an aggregate
temp MOVES it and any re-read is the second handle, judged at the read itself regardless of where the value
lands; local-name reads stay innocent, their events firing where the value gains a persistent place.
Sequence-to-tensor conversions and sequence slices copy scalar leaves (exact); tensor-to-tensor derivations
conservatively share -- except np.array of an array, an independent copy on the host and the A5 explicit-copy
spelling, which stays unique. A scalar answers rank zero so library stubs can interrogate rank. Annotation
promotion rebinds leaves along the SAME allocations: a fresh tree would sever the handle relationship and
admit a mutation CPython observes through the original name.
"""

import dataclasses
import inspect
import logging
import types
import typing
from dataclasses import dataclass

from ..._errors import SourceLocation, SynthesisError
from .._desugar import desugar
from .._ir import *
from .._lib import Conversion, Factory, Intrinsic, Library, matmul_, resolve, transpose_
from .._names import indexed_names
from . import _aggregate, _ops
from ._ownership import blame, escape, mutable, share
from ._reject import reattribute, reject
from ._snapshot import Snapshotter, describe_opaque as _describe_opaque, nan_payload, tensor_of
from ._values import (
    Allocation,
    AllocationState,
    ExpansionBudget,
    Opaque,
    ResidualScalar,
    Scalar,
    SequenceValue,
    StaticScalar,
    TensorMethod,
    TensorValue,
    Value,
    same,
)

_logger = logging.getLogger(__name__)

type _EnvKey = str | int

_AGGREGATE = (SequenceValue, TensorValue)


@dataclass(frozen=True, slots=True)
class _Unjoinable:
    """
    A binding whose two branch values cannot merge (captured objects with different identities, aggregates
    of divergent shape). The binding itself stays in the environment so that a later read draws a truthful
    located rejection at the read site -- CPython would have carried either value happily, so the honest
    report is "the compiler cannot merge these", never "unbound".
    """

    description: str


@dataclass(frozen=True, slots=True)
class _Moved:
    """A spent temp conduit: the value went on to a persistent place, so a re-read is a second handle."""

    value: SequenceValue | TensorValue


@dataclass(frozen=True, slots=True)
class _StoreSite:
    spelled: str
    chain: list[Allocation]
    spine: list[tuple[SequenceValue, int]]
    holder: TensorValue
    slot: int
    slot_value: Value


type _Env = dict[_EnvKey, Value | _Unjoinable | _Moved]


@dataclass(slots=True)
class _Frame:
    """One function under interpretation; ``result`` is where a non-root ``return`` deposits the call's value."""

    fn: types.FunctionType
    annotations: dict[str, object]
    env: _Env
    root: bool
    result: Value | None = None


_MISSING = object()

_SCALAR_ANNOTATIONS: list[tuple[type, ScalarType]] = [
    (bool, ScalarType.BOOL),
    (int, ScalarType.INT),
    (float, ScalarType.FLOAT),
]


def partial_evaluate(eel: EelFunction, fn: types.FunctionType) -> EelFunction:
    return _Interpreter(fn).run(eel)


def _annotation_stype(annotation: object) -> ScalarType | None:
    return next((stype for ty, stype in _SCALAR_ANNOTATIONS if annotation is ty), None)


def _key_order(key: _EnvKey) -> tuple[bool, str]:
    return isinstance(key, str), str(key)


def _rechained(node: object, frames: tuple[CallFrame, ...]) -> object:
    """
    An inlined callee's tree with every origin carrying the frame chain, so residual nodes and rejections
    alike re-attribute; shape-generic over the frozen node dataclasses so IR growth needs no maintenance here.
    """
    if isinstance(node, Origin):
        return Origin(node.location, frames)
    if isinstance(node, SourceLocation):
        return node
    if isinstance(node, tuple):
        return tuple(_rechained(item, frames) for item in node)
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return dataclasses.replace(
            node, **{field.name: _rechained(getattr(node, field.name), frames) for field in dataclasses.fields(node)}
        )
    return node


class _Interpreter:
    def __init__(self, fn: types.FunctionType) -> None:
        self._fn = fn
        self._budget = ExpansionBudget()
        self._next_temp = 0
        self._outputs: tuple[OutputDecl, ...] = ()
        self._return_values: tuple[Atom, ...] = ()
        self._eels: dict[types.FunctionType, EelFunction] = {}
        self._meta: dict[types.FunctionType, dict[str, object]] = {}
        self._inlining: set[types.FunctionType] = {fn}
        self._snapshot = Snapshotter()
        anchor = resolve(pow)
        assert isinstance(anchor, Library), "the builtin pow key anchors the ** lowering"
        self._pow_stub = anchor.stub
        assert isinstance(matmul_, types.FunctionType) and isinstance(transpose_, types.FunctionType)
        self._matmul_stub: types.FunctionType = matmul_
        self._transpose_stub: types.FunctionType = transpose_

    def run(self, eel: EelFunction) -> EelFunction:
        frame = _Frame(fn=self._fn, annotations=self._annotations_of(eel.origin, self._fn), env={}, root=True)
        params: list[Param] = []
        for param in eel.params:
            params.extend(self._param(param, frame))
        names = [param.name for param in params]
        if len(set(names)) != len(names):
            collision = next(name for name in names if names.count(name) > 1)
            reject(eel.origin, f"the decomposed parameter names collide on {collision!r}; rename the parameters")
        sink: list[Stmt] = []
        returned = self._block(eel.body, frame, sink, in_branch=False)
        if not returned:
            self._conclude_bare(eel.origin, frame)
        sink.append(ResidualReturn(eel.origin, self._return_values))
        body, live = _prune(sink)
        assert not live, "a residual temp is read before any assignment"
        residual = EelFunction(eel.origin, eel.name, tuple(params), tuple(body), slots=(), outputs=self._outputs)
        _logger.info(
            "%s: partial evaluation: %d residual statement(s), %d output(s), %d budget unit(s) spent",
            eel.name,
            len(body),
            len(self._outputs),
            self._budget.spent,
        )
        return residual

    def _desugared(self, fn: types.FunctionType) -> EelFunction:
        found = self._eels.get(fn)
        if found is None:
            found = desugar(fn)
            self._eels[fn] = found
        return found

    def _annotations_of(self, origin: Origin, fn: types.FunctionType) -> dict[str, object]:
        found = self._meta.get(fn)
        if found is None:
            try:
                # eval_str accepts quoted annotations; lazy PEP 649 annotations execute user expressions
                # here, so any exception is the user's -- the broad catch is sanctioned alongside the shadow's.
                found = inspect.get_annotations(fn, eval_str=True)
            except Exception as error:
                reject(origin, f"the type annotations cannot be evaluated: {error}")
            self._meta[fn] = found
        return found

    def _param(self, param: Param, frame: _Frame) -> list[Param]:
        annotation = frame.annotations.get(param.name, _MISSING)
        if annotation is _MISSING:
            reject(param.origin, f"the parameter {param.name!r} requires a type annotation")
        stype = _annotation_stype(annotation)
        if stype is not None:
            frame.env[param.name] = ResidualScalar(stype, LocalRef(param.origin, param.name))
            return [Param(param.origin, param.name, param.kind, stype)]
        annotated = _aggregate.array_annotation_shape(annotation, param.origin, f"parameter {param.name!r}")
        if annotated is None:
            reject(param.origin, f"the annotation of parameter {param.name!r} is not supported yet")
        shape, family = annotated
        leaf_names = indexed_names(param.name, shape)
        leaves = tuple(ResidualScalar(family, LocalRef(param.origin, leaf)) for leaf in leaf_names)
        frame.env[param.name] = TensorValue(shape, family, leaves, Allocation(AllocationState.ESCAPED))
        return [Param(param.origin, leaf, param.kind, family) for leaf in leaf_names]

    def _fresh(self) -> int:
        index = self._next_temp
        self._next_temp += 1
        return index

    # ------------------------------------------------------------------ statements

    def _block(self, stmts: tuple[Stmt, ...], frame: _Frame, sink: list[Stmt], in_branch: bool) -> bool:
        for stmt in stmts:
            match stmt:
                case Assign(target=target, value=value):
                    result = self._expr(value, frame, sink)
                    key = self._binding_key(target)
                    if isinstance(result, _AGGREGATE) and isinstance(value, LocalRef) and value.name != key:
                        share(result)
                    frame.env[key] = result
                case AugAssign(origin=origin, target=target, op=op, value=value):
                    current = frame.env.get(target.name)
                    if current is None:
                        reject(origin, f"the local name {target.name!r} is not bound on every path reaching this read")
                    current_value = self._readable(current, origin)
                    rhs = self._expr(value, frame, sink)
                    if isinstance(current_value, _AGGREGATE):
                        updated = self._aug_aggregate(origin, target.name, current_value, op, rhs, sink)
                    else:
                        updated = self._binary(origin, op, current_value, rhs, sink)
                    frame.env[target.name] = updated
                case Unpack(origin=origin, targets=targets, value=value):
                    source = self._expr(value, frame, sink)
                    items = _aggregate.unpack_items(origin, source, len(targets))
                    for target, item in zip(targets, items, strict=True):
                        frame.env[self._binding_key(target)] = item
                case If():
                    if self._if(stmt, frame, sink, in_branch):
                        return True
                case Return(origin=origin):
                    if in_branch:
                        reject(origin, "a return inside a runtime branch is not supported yet")
                    if frame.root:
                        self._conclude(stmt, frame, sink)
                    elif stmt.value is not None:
                        result = self._expr(stmt.value, frame, sink)
                        annotation = frame.annotations.get("return", _MISSING)
                        if annotation is not _MISSING:
                            # At the return itself, so a mismatch points into the callee.
                            result = self._conform_value(result, annotation, origin, sink, "the returned value")
                        frame.result = result
                    return True
                case Raise():
                    self._raise(stmt, frame, sink, in_branch)
                case While(origin=origin) | For(origin=origin):
                    reject(origin, "loops are not supported yet")
                case Break(origin=origin) | Continue(origin=origin):
                    reject(origin, "loops are not supported yet")
                case Store() | AugStore():
                    self._store(stmt, frame, sink)
                case _:
                    raise AssertionError(stmt)
        return False

    def _binding_key(self, binding: Binding) -> _EnvKey:
        match binding:
            case LocalBind(name=name):
                return name
            case TempBind(index=index):
                return index

    def _if(self, stmt: If, frame: _Frame, sink: list[Stmt], in_branch: bool) -> bool:
        cond_value = self._expr(stmt.cond, frame, sink)
        if isinstance(cond_value, _AGGREGATE):
            reject(stmt.origin, "the truthiness of an aggregate is not supported")
        cond = self._scalar(cond_value, stmt.origin)
        if cond.stype is not ScalarType.BOOL:
            reject(stmt.origin, "the branch condition must be a bool; Python truthiness is not supported")
        if isinstance(cond, StaticScalar):
            decided = _ops.const_value(cond.const)
            assert isinstance(decided, bool)
            taken = stmt.then if decided else stmt.orelse
            return self._block(taken, frame, sink, in_branch)
        then_frame = _Frame(frame.fn, frame.annotations, dict(frame.env), frame.root)
        else_frame = _Frame(frame.fn, frame.annotations, dict(frame.env), frame.root)
        then_sink: list[Stmt] = []
        else_sink: list[Stmt] = []
        returned = self._block(stmt.then, then_frame, then_sink, in_branch=True)
        assert not returned
        returned = self._block(stmt.orelse, else_frame, else_sink, in_branch=True)
        assert not returned
        joined = self._join(stmt.origin, then_frame.env, then_sink, else_frame.env, else_sink)
        sink.append(If(stmt.origin, self._materialize(cond, stmt.origin), tuple(then_sink), tuple(else_sink)))
        frame.env.clear()
        frame.env.update(joined)
        return False

    def _join(
        self, origin: Origin, then_env: _Env, then_sink: list[Stmt], else_env: _Env, else_sink: list[Stmt]
    ) -> _Env:
        joined: _Env = {}
        for key in sorted(set(then_env) & set(else_env), key=_key_order):
            a_bound, b_bound = then_env[key], else_env[key]
            moved = isinstance(a_bound, _Moved) or isinstance(b_bound, _Moved)
            a = a_bound.value if isinstance(a_bound, _Moved) else a_bound
            b = b_bound.value if isinstance(b_bound, _Moved) else b_bound
            if isinstance(a, _Unjoinable) or isinstance(b, _Unjoinable):
                joined[key] = _Unjoinable(self._describe_key(key))
                continue
            if same(a, b):
                joined[key] = _Moved(a) if moved and isinstance(a, _AGGREGATE) else a
                continue
            if isinstance(a, _AGGREGATE) and isinstance(b, _AGGREGATE):
                merged = self._join_aggregates(origin, key, a, b, then_sink, else_sink)
                joined[key] = _Moved(merged) if moved and isinstance(merged, _AGGREGATE) else merged
                continue
            if isinstance(a, (StaticScalar, ResidualScalar)) and isinstance(b, (StaticScalar, ResidualScalar)):
                joined[key] = self._join_scalars(origin, self._describe_key(key), a, b, then_sink, else_sink)
                continue
            joined[key] = _Unjoinable(self._describe_key(key))
        return joined

    def _join_scalars(
        self, origin: Origin, described: str, a: Scalar, b: Scalar, then_sink: list[Stmt], else_sink: list[Stmt]
    ) -> Scalar:
        if {a.stype, b.stype} == {ScalarType.INT, ScalarType.FLOAT}:
            a = self._as_float(a, origin, then_sink)
            b = self._as_float(b, origin, else_sink)
        elif a.stype is not b.stype:
            reject(origin, f"the branches bind {described} to incompatible types ({a.stype.value} vs {b.stype.value})")
        if same(a, b):
            return a
        stype = a.stype
        index = self._fresh()
        then_sink.append(Assign(origin, TempBind(origin, index), self._materialize(a, origin), stype))
        else_sink.append(Assign(origin, TempBind(origin, index), self._materialize(b, origin), stype))
        return ResidualScalar(stype, TempRef(origin, index))

    def _join_aggregates(
        self, origin: Origin, key: _EnvKey, a: Value, b: Value, then_sink: list[Stmt], else_sink: list[Stmt]
    ) -> Value | _Unjoinable:
        keep = _allocations_match(a, b)
        merged = self._merge(origin, key, a, b, then_sink, else_sink, keep)
        if merged is None:
            return _Unjoinable(self._describe_key(key))
        if not keep:
            # CPython would carry ONE of the two objects; a mutation through the merged name must therefore
            # reject, and so must a mutation through either source, hence all three trees share.
            share(a)
            share(b)
            share(merged)
        return merged

    def _merge(
        self,
        origin: Origin,
        key: _EnvKey,
        a: Value,
        b: Value,
        then_sink: list[Stmt],
        else_sink: list[Stmt],
        keep: bool,
    ) -> Value | None:
        match a, b:
            case SequenceValue(), SequenceValue():
                if len(a.items) != len(b.items):
                    return None
                items: list[Value] = []
                for x, y in zip(a.items, b.items, strict=True):
                    item = self._merge(origin, key, x, y, then_sink, else_sink, keep)
                    if item is None:
                        return None
                    items.append(item)
                return SequenceValue(tuple(items), a.allocation if keep else Allocation())
            case TensorValue(), TensorValue():
                if a.shape != b.shape:
                    return None
                leaves: list[Scalar | Opaque] = []
                for x, y in zip(a.leaves, b.leaves, strict=True):
                    leaf = self._merge(origin, key, x, y, then_sink, else_sink, keep)
                    if leaf is None:
                        return None
                    assert isinstance(leaf, (StaticScalar, ResidualScalar, Opaque))
                    leaves.append(leaf)
                family = next((leaf.stype for leaf in leaves if not isinstance(leaf, Opaque)), a.family)
                return TensorValue(a.shape, family, tuple(leaves), a.allocation if keep else Allocation())
            case (StaticScalar() | ResidualScalar()), (StaticScalar() | ResidualScalar()):
                return self._join_scalars(origin, self._describe_key(key), a, b, then_sink, else_sink)
            case Opaque(), Opaque():
                return a if same(a, b) else None
            case _:
                return None

    def _describe_key(self, key: _EnvKey) -> str:
        return f"the local {key!r}" if isinstance(key, str) else "the conditional result"

    def _readable(self, binding: Value | _Unjoinable | _Moved, origin: Origin) -> Value:
        if isinstance(binding, _Unjoinable):
            reject(
                origin,
                f"{binding.description} holds branch values the compiler cannot merge; "
                "only bool, int, and float values join branches",
            )
        if isinstance(binding, _Moved):
            share(binding.value)
            return binding.value
        return binding

    def _raise(self, stmt: Raise, frame: _Frame, sink: list[Stmt], in_branch: bool) -> None:
        if in_branch:
            reject(stmt.origin, "a raise on a data-dependent path (a runtime branch arm) cannot be lowered")
        pieces: list[str] = []
        for part in stmt.parts:
            if isinstance(part, str):
                pieces.append(part)
                continue
            value = self._expr(part, frame, sink)
            if isinstance(value, StaticScalar):
                pieces.append(format(_ops.const_value(value.const)))
            elif isinstance(value, Opaque) and type(value.value) is str:
                pieces.append(value.value)
            else:
                reject(stmt.origin, "the raise message interpolates a value that is not a compile-time constant")
        text = "".join(pieces)
        reject(stmt.origin, f"{stmt.exc_type}: {text}" if text else stmt.exc_type)

    # ------------------------------------------------------------------ the module boundary

    def _conclude(self, stmt: Return, frame: _Frame, sink: list[Stmt]) -> None:
        annotation = frame.annotations.get("return", _MISSING)
        if annotation is _MISSING:
            reject(stmt.origin, "the return type annotation is required")
        if stmt.value is None:
            if annotation is not None:
                reject(stmt.origin, "the kernel returns no value but its annotation declares one")
            return
        value = self._expr(stmt.value, frame, sink)
        if isinstance(value, Opaque):
            reject(stmt.origin, f"the captured object {value.name!r} cannot be returned")
        conformed = self._conform_value(value, annotation, stmt.origin, sink, "the returned value", root=True)
        match conformed:
            case StaticScalar() | ResidualScalar():
                self._outputs = (OutputDecl((0,), conformed.stype),)
                self._return_values = (self._materialize(conformed, stmt.origin),)
            case SequenceValue() | TensorValue():
                escape(conformed)
                leaves = _aggregate.flatten(stmt.origin, conformed)
                self._outputs = tuple(OutputDecl(path, leaf.stype) for path, leaf in leaves)
                self._return_values = tuple(self._materialize(leaf, stmt.origin) for _, leaf in leaves)
            case _:
                raise AssertionError(conformed)

    def _conclude_bare(self, origin: Origin, frame: _Frame) -> None:
        annotation = frame.annotations.get("return", _MISSING)
        if annotation is _MISSING:
            reject(origin, "the return type annotation is required")
        if annotation is not None:
            reject(origin, "the kernel can complete without returning a value but its annotation declares one")

    def _conform_value(
        self, value: Value, annotation: object, origin: Origin, sink: list[Stmt], what: str, *, root: bool = False
    ) -> Value:
        """
        The one annotation-conformance rule for both module boundaries: the root interface demands a
        recognized annotation, while an inlined frame skips unrecognized ones (library stubs annotate with
        shapeless types the subset cannot check).
        """
        stype = _annotation_stype(annotation)
        if stype is not None:
            return self._conform(value, stype, origin, sink, what)
        shaped = typing.get_origin(annotation)
        if shaped is tuple:
            args = typing.get_args(annotation)
            if not isinstance(value, SequenceValue):
                reject(origin, f"{what} is not a sequence")
            if len(args) == 2 and args[1] is Ellipsis:
                items = tuple(self._conform_value(item, args[0], origin, sink, what, root=root) for item in value.items)
                return dataclasses.replace(value, items=items)
            if len(args) != len(value.items):
                reject(origin, f"{what} has {len(value.items)} element(s) where the annotation declares {len(args)}")
            items = tuple(
                self._conform_value(item, arg, origin, sink, what, root=root)
                for item, arg in zip(value.items, args, strict=True)
            )
            return dataclasses.replace(value, items=items)
        if shaped is list:
            reject(origin, "list annotations are not supported; annotate a tuple")
        annotated = _aggregate.array_annotation_shape(annotation, origin, what)
        if annotated is not None:
            shape, family = annotated
            if not isinstance(value, TensorValue):
                reject(origin, f"{what} is not an array")
            if value.shape != shape:
                reject(origin, f"{what} has shape {value.shape} where the annotation declares {shape}")
            if value.family is ScalarType.INT and family is ScalarType.FLOAT:
                promoted = tuple(
                    leaf if isinstance(leaf, Opaque) else self._as_float(leaf, origin, sink) for leaf in value.leaves
                )
                return dataclasses.replace(value, family=ScalarType.FLOAT, leaves=promoted)
            if value.family is not family:
                reject(origin, f"{what} is a {value.family.value} array where the annotation declares {family.value}")
            return value
        if not root:
            return value
        if isinstance(value, (StaticScalar, ResidualScalar)):
            reject(origin, "the return annotation does not match the returned scalar")
        reject(
            origin,
            "the return annotation is not supported yet; "
            "annotate with scalars, tuple[...], or a fixed-shape jaxtyping array",
        )

    def _conform(self, value: Value, declared: ScalarType, origin: Origin, sink: list[Stmt], what: str) -> Scalar:
        if isinstance(value, Opaque):
            reject(origin, _describe_opaque(value))
        if not isinstance(value, (StaticScalar, ResidualScalar)):
            reject(origin, f"{what} is not a {declared.value} scalar")
        if value.stype is declared:
            return value
        if value.stype is ScalarType.INT and declared is ScalarType.FLOAT:
            return self._as_float(value, origin, sink)
        reject(origin, f"{what} has type {value.stype.value} where the annotation declares {declared.value}")

    # ------------------------------------------------------------------ expressions

    def _expr(self, expr: Expr, frame: _Frame, sink: list[Stmt]) -> Value:
        match expr:
            case Const(value=value):
                return StaticScalar(_ops.make_const(value))
            case TempRef(origin=origin, index=index):
                bound = frame.env.get(index)
                assert bound is not None, "desugar binds every temp before its first read"
                read = self._readable(bound, origin)
                if isinstance(read, _AGGREGATE) and not isinstance(bound, _Moved):
                    frame.env[index] = _Moved(read)
                return read
            case LocalRef(origin=origin, name=name):
                found = frame.env.get(name)
                if found is None:
                    reject(origin, f"the local name {name!r} is not bound on every path reaching this read")
                return self._readable(found, origin)
            case EnvRead():
                return self._env_read(expr, frame)
            case Unary(origin=origin, op=op, operand=operand):
                return self._unary(origin, op, self._expr(operand, frame, sink), sink)
            case Binary(origin=origin, op=op, left=left, right=right):
                lv = self._expr(left, frame, sink)
                rv = self._expr(right, frame, sink)
                return self._binary(origin, op, lv, rv, sink)
            case Compare(origin=origin, op=op, left=left, right=right):
                lv = self._expr(left, frame, sink)
                rv = self._expr(right, frame, sink)
                return self._compare(origin, op, lv, rv, sink)
            case Call():
                return self._call(expr, frame, sink)
            case TupleExpr(origin=origin, items=items):
                return self._display(origin, items, frame, sink)
            case ListExpr(origin=origin, items=items):
                return self._display(origin, items, frame, sink)
            case AttrRead(origin=origin, base=base, attr=attr):
                return self._attr_read(origin, self._expr(base, frame, sink), attr, sink)
            case IndexRead(origin=origin, base=base, index=index):
                base_value = self._expr(base, frame, sink)
                index_value = self._expr(index, frame, sink)
                return _aggregate.index_read(origin, base_value, index_value)
            case SliceRead(origin=origin, base=base, lo=lo, hi=hi):
                base_value = self._expr(base, frame, sink)
                lo_bound = self._slice_bound(lo, frame, sink)
                hi_bound = self._slice_bound(hi, frame, sink)
                return _aggregate.slice_read(origin, base_value, lo_bound, hi_bound)
            case MultiIndexRead(origin=origin, base=base, axes=axes):
                base_value = self._expr(base, frame, sink)
                resolved: list[_aggregate.ResolvedAxis] = []
                for axis in axes:
                    if isinstance(axis, SliceSel):
                        resolved.append(
                            (self._slice_bound(axis.lo, frame, sink), self._slice_bound(axis.hi, frame, sink))
                        )
                    else:
                        resolved.append(self._expr(axis, frame, sink))
                return _aggregate.multi_index_read(origin, base_value, tuple(resolved))
            case Comp(origin=origin):
                reject(origin, "comprehensions are not supported yet")
            case _:
                raise AssertionError(expr)

    def _slice_bound(self, atom: Atom | None, frame: _Frame, sink: list[Stmt]) -> int | None:
        if atom is None:
            return None
        return _aggregate.static_index(atom.origin, self._expr(atom, frame, sink), "a slice bound")

    def _display(
        self, origin: Origin, items: tuple[Atom | StarArg, ...], frame: _Frame, sink: list[Stmt]
    ) -> SequenceValue:
        collected: list[Value] = []
        for item in items:
            if isinstance(item, StarArg):
                source = self._expr(item.value, frame, sink)
                collected.extend(_aggregate.splice_items(origin, source))
            else:
                value = self._expr(item, frame, sink)
                if isinstance(value, _AGGREGATE) and isinstance(item, LocalRef):
                    share(value)
                collected.append(value)
        self._budget.spend(max(len(collected), 1), origin, "the sequence display")
        return SequenceValue(tuple(collected), Allocation())

    def _env_read(self, node: EnvRead, frame: _Frame) -> Value:
        if node.free:
            code = frame.fn.__code__
            assert node.name in code.co_freevars
            closure = frame.fn.__closure__
            assert closure is not None
            cell = closure[code.co_freevars.index(node.name)]
            try:
                raw = cell.cell_contents
            except ValueError:
                reject(node.origin, f"the free name {node.name!r} is unbound in its enclosing scope")
        else:
            if node.name in frame.fn.__globals__:
                raw = frame.fn.__globals__[node.name]
            else:
                # The function's own builtins mapping, which is what CPython would consult -- absent from
                # typeshed's FunctionType, hence the getattr.
                builtins_ns = getattr(frame.fn, "__builtins__")
                assert isinstance(builtins_ns, dict)
                if node.name not in builtins_ns:
                    reject(node.origin, f"the name {node.name!r} is not defined")
                raw = builtins_ns[node.name]
        return self._snapshot.admit(node.name, raw)

    # ------------------------------------------------------------------ attribute reads

    def _attr_read(self, origin: Origin, base_value: Value, attr: str, sink: list[Stmt]) -> Value:
        match base_value:
            case TensorValue():
                return self._tensor_attr(origin, base_value, attr, sink)
            case SequenceValue():
                if attr in ("shape", "ndim", "T", "flatten"):
                    reject(
                        origin,
                        f"`.{attr}` on a Python sequence is not supported; " "build a numpy array with np.array([...])",
                    )
                reject(origin, f"a sequence has no supported attribute {attr!r}")
            case StaticScalar() | ResidualScalar():
                if attr == "ndim":
                    return StaticScalar(_ops.make_const(0))
                if attr == "shape":
                    return SequenceValue((), Allocation())
                reject(origin, f"a scalar has no supported attribute {attr!r}")
            case TensorMethod():
                reject(origin, "a bound array method can only be called")
            case Opaque(name=name, value=value) if isinstance(value, (types.ModuleType, type)):
                # A module/class attribute is a metadata read (class access unwraps staticmethod and
                # plain functions); an instance never runs live descriptors, below.
                try:
                    raw = getattr(value, attr)
                except AttributeError:
                    reject(origin, f"{name!r} has no attribute {attr!r}")
                return self._snapshot.admit(f"{name}.{attr}", raw)
            case Opaque():
                return self._instance_attr(origin, base_value, attr, sink)
            case _:
                raise AssertionError(base_value)

    def _tensor_attr(self, origin: Origin, tensor: TensorValue, attr: str, sink: list[Stmt]) -> Value:
        if attr == "ndim":
            return StaticScalar(_ops.make_const(len(tensor.shape)))
        if attr == "shape":
            dims = tuple(StaticScalar(_ops.make_const(dim)) for dim in tensor.shape)
            return SequenceValue(dims, Allocation())
        if attr == "T":
            share(tensor)
            return self._inline(origin, "np.transpose", self._transpose_stub, [tensor], {}, sink, positional_only=True)
        if attr == "flatten":
            share(tensor)
            return TensorMethod(tensor, "flatten")
        reject(origin, f"an array has no supported attribute {attr!r}")

    def _instance_attr(self, origin: Origin, base: Opaque, attr: str, sink: list[Stmt]) -> Value:
        """
        CPython's attribute precedence over class metadata, never calling into the object: a data descriptor
        wins (only a plain property is readable -- its getter is inlined as ordinary code), then the instance
        dict, then non-data class attributes.
        """
        found: object = next(
            (klass.__dict__[attr] for klass in type(base.value).__mro__ if attr in klass.__dict__), _MISSING
        )
        if isinstance(found, property) and isinstance(found.fget, types.FunctionType):
            return self._inline(origin, f"{base.name}.{attr}", found.fget, [base], {}, sink, positional_only=False)
        if found is not _MISSING and inspect.isdatadescriptor(found):
            reject(origin, f"the attribute {attr!r} is a descriptor the compiler cannot read")
        instance_dict = getattr(base.value, "__dict__", None)
        if isinstance(instance_dict, dict) and attr in instance_dict:
            return self._snapshot.admit(f"{base.name}.{attr}", instance_dict[attr])
        if found is _MISSING:
            reject(origin, f"{base.name!r} has no attribute {attr!r}")
        if isinstance(found, staticmethod):
            return self._snapshot.admit(f"{base.name}.{attr}", found.__func__)
        if isinstance(found, types.FunctionType):
            return self._snapshot.admit(f"{base.name}.{attr}", types.MethodType(found, base.value))
        if isinstance(found, classmethod):
            return self._snapshot.admit(f"{base.name}.{attr}", types.MethodType(found.__func__, type(base.value)))
        return self._snapshot.admit(f"{base.name}.{attr}", found)

    # ------------------------------------------------------------------ scalars and operators

    def _scalar(self, value: Value, origin: Origin) -> Scalar:
        match value:
            case StaticScalar() | ResidualScalar():
                return value
            case Opaque():
                reject(origin, _describe_opaque(value))
            case SequenceValue() | TensorValue():
                reject(origin, f"{_aggregate.a_kind(value)} cannot be used as a scalar here")
            case TensorMethod():
                reject(origin, "a bound array method can only be called")

    def _materialize(self, scalar: Scalar, origin: Origin) -> Atom:
        match scalar:
            case StaticScalar(const=const):
                return Const(origin, _ops.const_value(const))
            case ResidualScalar(atom=atom):
                return atom

    def _apply(self, operator: _ops.Operator, operands: list[Scalar], origin: Origin, sink: list[Stmt]) -> Scalar:
        if all(isinstance(operand, StaticScalar) for operand in operands):
            consts = [operand.const for operand in operands if isinstance(operand, StaticScalar)]
            try:
                return StaticScalar(operator.evaluate(consts))
            except _ops.NoNumber:
                pass  # the graph re-derives the fault and the survivor sweep judges it; never convict here
        stype = _ops.result_stype(operator)
        atoms = tuple(self._materialize(operand, origin) for operand in operands)
        index = self._fresh()
        sink.append(Assign(origin, TempBind(origin, index), IntrinsicCall(origin, operator, atoms), stype))
        return ResidualScalar(stype, TempRef(origin, index))

    def _as_float(self, scalar: Scalar, origin: Origin, sink: list[Stmt]) -> Scalar:
        if scalar.stype is ScalarType.FLOAT:
            return scalar
        assert scalar.stype is ScalarType.INT
        return self._apply(_ops.CONVERT[(ScalarType.INT, ScalarType.FLOAT)], [scalar], origin, sink)

    def _unary(self, origin: Origin, op: UnaryOp, value: Value, sink: list[Stmt]) -> Value:
        if isinstance(value, TensorValue) and op in (UnaryOp.NEG, UnaryOp.POS):
            leaves = tuple(self._unary_leaf(origin, op, leaf, sink) for leaf in value.leaves)
            self._budget.spend(len(leaves), origin, "the elementwise operation")
            return TensorValue(value.shape, value.family, leaves, Allocation())
        if isinstance(value, _AGGREGATE):
            if op is UnaryOp.NOT:
                reject(origin, "the truthiness of an aggregate is not supported")
            reject(origin, f"`{op.value}` is not supported on {_aggregate.a_kind(value)}")
        operand = self._scalar(value, origin)
        match op:
            case UnaryOp.NEG | UnaryOp.POS:
                if operand.stype is ScalarType.BOOL:
                    reject(origin, "booleans take no part in arithmetic; cast explicitly with int(...) or float(...)")
                if op is UnaryOp.POS:
                    return operand
                negate = _ops.INT_NEG if operand.stype is ScalarType.INT else _ops.FLOAT_NEG
                return self._apply(negate, [operand], origin, sink)
            case UnaryOp.NOT:
                if operand.stype is not ScalarType.BOOL:
                    reject(origin, "`not` requires a bool operand; Python truthiness is not supported")
                return self._apply(_ops.BOOL_NOT, [operand], origin, sink)
            case UnaryOp.INVERT:
                if operand.stype is not ScalarType.INT:
                    reject(origin, "`~` is integer-only")
                return self._apply(_ops.INT_NOT, [operand], origin, sink)

    def _unary_leaf(self, origin: Origin, op: UnaryOp, leaf: Scalar | Opaque, sink: list[Stmt]) -> Scalar | Opaque:
        if op is UnaryOp.POS:
            return leaf
        result = self._unary(origin, op, self._scalar_leaf(origin, leaf), sink)
        assert isinstance(result, (StaticScalar, ResidualScalar))
        return result

    def _operand_arguments(self, node: Call, display: str, frame: _Frame, sink: list[Stmt]) -> list[Value]:
        """Arguments for a callee that retains no handle: no aliasing event, unlike parameter binding."""
        values: list[Value] = []
        for arg in node.args:
            match arg:
                case PosArg(value=value):
                    values.append(self._expr(value, frame, sink))
                case StarArg(value=value):
                    values.extend(_aggregate.splice_items(node.origin, self._expr(value, frame, sink)))
                case KwArg():
                    reject(node.origin, f"{display}() takes no keyword arguments")
        return values

    def _scalar_leaf(self, origin: Origin, leaf: Scalar | Opaque) -> Scalar:
        if isinstance(leaf, Opaque):
            reject(origin, _describe_opaque(leaf))
        return leaf

    def _binary(self, origin: Origin, op: BinaryOp, lv: Value, rv: Value, sink: list[Stmt]) -> Value:
        if op is BinaryOp.MATMUL:
            if isinstance(lv, SequenceValue) or isinstance(rv, SequenceValue):
                reject(origin, "`@` on a Python list/tuple is not supported; build a numpy array with np.array([...])")
            for operand in (lv, rv):
                if isinstance(operand, _AGGREGATE):
                    share(operand)
            return self._inline(origin, "np.matmul", self._matmul_stub, [lv, rv], {}, sink, positional_only=True)
        tensors = isinstance(lv, TensorValue) or isinstance(rv, TensorValue)
        sequences = isinstance(lv, SequenceValue) or isinstance(rv, SequenceValue)
        if (tensors or sequences) and op in (BinaryOp.AND, BinaryOp.OR):
            reject(origin, "the truthiness of an aggregate is not supported")
        if tensors and sequences:
            reject(origin, "cannot mix an array with a Python list/tuple; convert with np.array([...])")
        if tensors:
            return self._elementwise(origin, op, lv, rv, sink)
        if sequences:
            return self._sequence_binary(origin, op)
        return self._binary_scalars(origin, op, lv, rv, sink)

    def _elementwise(self, origin: Origin, op: BinaryOp, lv: Value, rv: Value, sink: list[Stmt]) -> TensorValue:
        if op not in (BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV):
            reject(origin, f"the operator `{op.value}` is not supported on arrays yet")
        pairs: list[tuple[Scalar, Scalar]]
        shape: tuple[int, ...]
        if isinstance(lv, TensorValue) and isinstance(rv, TensorValue):
            if lv.shape != rv.shape:
                reject(origin, f"array shapes {lv.shape} and {rv.shape} do not match; broadcasting is not supported")
            shape = lv.shape
            pairs = [
                (self._scalar_leaf(origin, a), self._scalar_leaf(origin, b))
                for a, b in zip(lv.leaves, rv.leaves, strict=True)
            ]
        elif isinstance(lv, TensorValue):
            scalar = self._scalar(rv, origin)
            shape = lv.shape
            pairs = [(self._scalar_leaf(origin, a), scalar) for a in lv.leaves]
        else:
            assert isinstance(rv, TensorValue)
            scalar = self._scalar(lv, origin)
            shape = rv.shape
            pairs = [(scalar, self._scalar_leaf(origin, b)) for b in rv.leaves]
        leaves: list[Scalar | Opaque] = []
        for a, b in pairs:
            leaves.append(self._binary_scalars(origin, op, a, b, sink))
        self._budget.spend(len(leaves), origin, "the elementwise operation")
        family = next(leaf.stype for leaf in leaves if not isinstance(leaf, Opaque))
        return TensorValue(shape, family, tuple(leaves), Allocation())

    def _sequence_binary(self, origin: Origin, op: BinaryOp) -> Value:
        reject(origin, f"the operator `{op.value}` is not supported on a sequence; build a numpy array instead")

    def _binary_scalars(self, origin: Origin, op: BinaryOp, lv: Value, rv: Value, sink: list[Stmt]) -> Scalar:
        left = self._scalar(lv, origin)
        right = self._scalar(rv, origin)
        both_bool = left.stype is ScalarType.BOOL and right.stype is ScalarType.BOOL
        both_int = left.stype is ScalarType.INT and right.stype is ScalarType.INT
        match op:
            case BinaryOp.AND | BinaryOp.OR:
                if not both_bool:
                    reject(origin, "the boolean gates require bool operands; Python truthiness is not supported")
                return self._apply(_ops.BOOL_GATES[op], [left, right], origin, sink)
            case BinaryOp.BITAND | BinaryOp.BITOR | BinaryOp.BITXOR:
                if both_bool:
                    return self._apply(_ops.BOOL_GATES[op], [left, right], origin, sink)
                if both_int:
                    return self._apply(_ops.INT_BINARY[op], [left, right], origin, sink)
                reject(origin, f"the bitwise operator `{op.value}` requires two ints or two bools")
            case BinaryOp.POW:
                return self._pow(origin, left, right, sink)
            case _:
                pass
        if ScalarType.BOOL in (left.stype, right.stype):
            reject(origin, "booleans take no part in arithmetic; cast explicitly with int(...) or float(...)")
        match op:
            case BinaryOp.FLOORDIV | BinaryOp.MOD | BinaryOp.LSHIFT | BinaryOp.RSHIFT:
                if not both_int:
                    reject(origin, f"the operator `{op.value}` is integer-only")
                return self._apply(_ops.INT_BINARY[op], [left, right], origin, sink)
            case BinaryOp.DIV:
                dividend = self._as_float(left, origin, sink)
                divisor = self._as_float(right, origin, sink)
                return self._apply(_ops.FLOAT_DIV, [dividend, divisor], origin, sink)
            case BinaryOp.ADD | BinaryOp.SUB | BinaryOp.MUL:
                if both_int:
                    return self._apply(_ops.INT_BINARY[op], [left, right], origin, sink)
                left = self._as_float(left, origin, sink)
                right = self._as_float(right, origin, sink)
                if op is BinaryOp.SUB:
                    negated = self._apply(_ops.FLOAT_NEG, [right], origin, sink)
                    return self._apply(_ops.FLOAT_ADD, [left, negated], origin, sink)
                operator = _ops.FLOAT_ADD if op is BinaryOp.ADD else _ops.FLOAT_MUL
                return self._apply(operator, [left, right], origin, sink)
            case _:
                raise AssertionError(op)

    def _compare(self, origin: Origin, op: CompareOp, lv: Value, rv: Value, sink: list[Stmt]) -> Scalar:
        if isinstance(lv, _AGGREGATE) or isinstance(rv, _AGGREGATE):
            reject(origin, "aggregate comparison is not supported")
        left = self._scalar(lv, origin)
        right = self._scalar(rv, origin)
        if left.stype is ScalarType.BOOL and right.stype is ScalarType.BOOL:
            if op is CompareOp.EQ:
                distinct = self._apply(_ops.BOOL_XOR, [left, right], origin, sink)
                return self._apply(_ops.BOOL_NOT, [distinct], origin, sink)
            if op is CompareOp.NE:
                return self._apply(_ops.BOOL_XOR, [left, right], origin, sink)
            reject(origin, "ordering comparisons of booleans are not supported")
        if ScalarType.BOOL in (left.stype, right.stype):
            reject(origin, "a boolean cannot be compared with a number; cast explicitly")
        if left.stype is ScalarType.INT and right.stype is ScalarType.INT:
            return self._apply(_ops.int_relational(op), [left, right], origin, sink)
        left = self._as_float(left, origin, sink)
        right = self._as_float(right, origin, sink)
        return self._apply(_ops.float_relational(op), [left, right], origin, sink)

    # ------------------------------------------------------------------ calls

    def _call(self, node: Call, frame: _Frame, sink: list[Stmt]) -> Value:
        callee = self._expr(node.callee, frame, sink)
        if isinstance(callee, TensorMethod):
            return self._tensor_method(node, callee, frame, sink)
        if not isinstance(callee, Opaque):
            reject(node.origin, "the callee is not a callable object")
        raw = callee.value
        match resolve(raw):
            case Intrinsic(operator=operator):
                return self._intrinsic(node, callee.name, operator, frame, sink)
            case Library(stub=stub):
                values = self._positional_arguments(node, callee.name, frame, sink)
                if stub is self._pow_stub:
                    # All pow spellings share the ** lowering, so the staged int/int rejection cannot be
                    # dodged by spelling.
                    if len(values) != 2:
                        reject(node.origin, f"{callee.name}() takes 2 argument(s), got {len(values)}")
                    base = self._scalar(values[0], node.origin)
                    exponent = self._scalar(values[1], node.origin)
                    return self._pow(node.origin, base, exponent, sink)
                return self._inline(node.origin, callee.name, stub, values, {}, sink, positional_only=True)
            case Factory() as match:
                return self._factory(node, callee.name, match, frame, sink)
            case Conversion(copies=copies):
                values = self._operand_arguments(node, callee.name, frame, sink)
                if len(values) != 1:
                    reject(node.origin, f"{callee.name}() takes a single array-like argument")
                return self._to_tensor(node.origin, callee.name, values[0], sink, copies=copies)
            case None:
                pass
        if raw is float or raw is int or raw is bool:
            target = raw
            assert isinstance(target, type)
            return self._cast(node, target, frame, sink)
        if raw is len:
            return self._len(node, frame, sink)
        if raw is list or raw is tuple:
            return self._rebuild_sequence(node, callee.name, frame, sink)
        if isinstance(raw, types.FunctionType):
            # Ingested like user code wherever it lives: substitution is the registry's decision alone
            # (maintainer ruling), and the interpreter knows no library names.
            positional, keywords = self._signature_arguments(node, frame, sink)
            return self._inline(node.origin, callee.name, raw, positional, keywords, sink, positional_only=False)
        if inspect.ismethod(raw):
            reject(node.origin, "bound methods are not supported yet")
        if callable(raw):
            reject(node.origin, f"calls to {callee.name!r} are not supported yet")
        reject(node.origin, f"the captured object {callee.name!r} is not callable")

    def _argument(self, atom: Atom, frame: _Frame, sink: list[Stmt]) -> Value:
        value = self._expr(atom, frame, sink)
        if isinstance(value, _AGGREGATE) and isinstance(atom, LocalRef):
            share(value)
        return value

    def _positional_arguments(self, node: Call, display: str, frame: _Frame, sink: list[Stmt]) -> list[Value]:
        values: list[Value] = []
        for arg in node.args:
            match arg:
                case PosArg(value=value):
                    values.append(self._argument(value, frame, sink))
                case StarArg(value=value):
                    values.extend(_aggregate.splice_items(node.origin, self._expr(value, frame, sink)))
                case KwArg():
                    reject(node.origin, f"{display}() takes no keyword arguments")
        return values

    def _signature_arguments(self, node: Call, frame: _Frame, sink: list[Stmt]) -> tuple[list[Value], dict[str, Value]]:
        positional: list[Value] = []
        keywords: dict[str, Value] = {}
        for arg in node.args:
            match arg:
                case PosArg(value=value):
                    positional.append(self._argument(value, frame, sink))
                case StarArg(value=value):
                    positional.extend(_aggregate.splice_items(node.origin, self._expr(value, frame, sink)))
                case KwArg(name=name, value=value):
                    keywords[name] = self._argument(value, frame, sink)
        return positional, keywords

    def _intrinsic(self, node: Call, display: str, operator: _ops.Operator, frame: _Frame, sink: list[Stmt]) -> Value:
        values = self._operand_arguments(node, display, frame, sink)
        stypes = _ops.operand_stypes(operator)
        if len(values) != len(stypes):
            reject(node.origin, f"{display}() takes {len(stypes)} argument(s), got {len(values)}")
        operands = [
            self._conform(value, stype, node.origin, sink, f"argument {i + 1} of {display}()")
            for i, (value, stype) in enumerate(zip(values, stypes, strict=True))
        ]
        return self._apply(operator, operands, node.origin, sink)

    def _tensor_method(self, node: Call, method: TensorMethod, frame: _Frame, sink: list[Stmt]) -> Value:
        assert method.name == "flatten"
        if node.args:
            reject(node.origin, ".flatten() takes no arguments")
        return _aggregate.flattened(method.receiver)

    def _factory(self, node: Call, display: str, match: Factory, frame: _Frame, sink: list[Stmt]) -> Value:
        values = self._operand_arguments(node, display, frame, sink)
        host = [self._static_argument(node.origin, display, value) for value in values]
        try:
            built = match.build(*host)
        except Exception as error:  # a registered library builder, not user code; its refusal is a diagnostic
            reject(node.origin, f"{display}() rejects its arguments: {error}")
        tensor = tensor_of(built, display)
        if tensor is None:
            reject(node.origin, f"{display}() must build a non-empty 1-D or 2-D numeric array")
        self._budget.spend(len(tensor.leaves), node.origin, "the array factory")
        return tensor

    def _static_argument(self, origin: Origin, display: str, value: Value) -> object:
        match value:
            case StaticScalar(const=const):
                return _ops.const_value(const)
            case SequenceValue(items=items):
                return tuple(self._static_argument(origin, display, item) for item in items)
            case _:
                reject(origin, f"the arguments of {display}() must be compile-time constants")

    def _len(self, node: Call, frame: _Frame, sink: list[Stmt]) -> Value:
        values = self._operand_arguments(node, "len", frame, sink)
        if len(values) != 1:
            reject(node.origin, f"len() takes exactly one argument, got {len(values)}")
        match values[0]:
            case SequenceValue(items=items):
                return StaticScalar(_ops.make_const(len(items)))
            case TensorValue(shape=shape):
                return StaticScalar(_ops.make_const(shape[0]))
            case _:
                reject(node.origin, f"len() requires an aggregate, not {_aggregate.a_kind(values[0])}")

    def _rebuild_sequence(self, node: Call, display: str, frame: _Frame, sink: list[Stmt]) -> Value:
        values = self._operand_arguments(node, display, frame, sink)
        if len(values) != 1:
            reject(node.origin, f"{display}() takes exactly one aggregate argument here")
        source = values[0]
        if not isinstance(source, _AGGREGATE):
            reject(node.origin, f"{display}() requires an aggregate argument")
        children = _aggregate.splice_items(node.origin, source)
        self._budget.spend(max(len(children), 1), node.origin, "the sequence conversion")
        return SequenceValue(tuple(children), Allocation())

    def _store(self, stmt: Store | AugStore, frame: _Frame, sink: list[Stmt]) -> None:
        """
        The one mutation gate: resolve the target path with the store-prefix exemption (walked over the value
        tree, never through expression reads), judge admission at the store step after the RHS and index
        temps, and rebind new frozen values along the unchanged allocation spine.
        """
        origin = stmt.origin
        rhs = self._expr(stmt.value, frame, sink)
        match stmt.root:
            case EnvRead(name=name):
                reject(
                    origin,
                    f"cannot mutate {name!r}: it was captured from outside the kernel, "
                    "so the change would be observable there",
                )
            case LocalRef(name=name):
                bound = frame.env.get(name)
                if bound is None:
                    reject(origin, f"the local name {name!r} is not bound on every path reaching this read")
                root_value = self._readable(bound, origin)
        if not isinstance(root_value, _AGGREGATE):
            if isinstance(stmt.path[0], AttrSel):
                reject(origin, "attribute stores are not supported yet")
            reject(origin, f"{_aggregate.a_kind(root_value)} does not support item assignment")
        site = self._store_site(origin, name, root_value, stmt.path, frame, sink)
        for allocation in site.chain:
            if not mutable(allocation):
                reject(origin, f"cannot store into {site.spelled}: {blame(allocation)}")
        old = site.slot_value
        if isinstance(stmt, AugStore):
            if isinstance(old, Opaque):
                reject(origin, _describe_opaque(old))
            value = self._binary(origin, stmt.op, old, rhs, sink)
        else:
            value = rhs
        if isinstance(value, (SequenceValue, TensorValue, TensorMethod)):
            reject(
                origin, "storing an aggregate into a container is not supported; rebind or build with a comprehension"
            )
        assert isinstance(value, (StaticScalar, ResidualScalar, Opaque))
        leaf = self._tensor_leaf(origin, site.holder.family, value, sink)
        new_leaves = tuple(leaf if position == site.slot else kept for position, kept in enumerate(site.holder.leaves))
        rebuilt: SequenceValue | TensorValue = dataclasses.replace(site.holder, leaves=new_leaves)
        for container, position in reversed(site.spine):
            rebuilt = dataclasses.replace(
                container,
                items=tuple(rebuilt if index == position else kept for index, kept in enumerate(container.items)),
            )
        frame.env[name] = rebuilt

    def _store_site(
        self,
        origin: Origin,
        root_name: str,
        root_value: SequenceValue | TensorValue,
        path: tuple[Selector, ...],
        frame: _Frame,
        sink: list[Stmt],
    ) -> "_StoreSite":
        spelled = root_name
        chain: list[Allocation] = []
        spine: list[tuple[SequenceValue, int]] = []
        current: Value = root_value
        tensor: TensorValue | None = None
        axes: list[int] = []
        assert path, "desugar never emits an empty store path"
        for step, selector in enumerate(path):
            terminal = step == len(path) - 1
            match selector:
                case AttrSel():
                    reject(origin, "attribute stores are not supported yet")
                case IndexSel(index=atom):
                    index_value = self._expr(atom, frame, sink)
            if tensor is not None:
                added = self._store_axes(origin, tensor, len(axes), index_value)
                axes.extend(added)
                spelled += self._spell_axes(added)
                continue
            match current:
                case SequenceValue(items=items):
                    chain.append(current.allocation)
                    position = _aggregate.static_index(origin, index_value, "a subscript index")
                    resolved = position + len(items) if position < 0 else position
                    if not 0 <= resolved < len(items):
                        reject(
                            origin,
                            f"index {position} is out of bounds for a "
                            f"{_aggregate.kind_label(current)} of length {len(items)}",
                        )
                    spelled += f"[{position}]"
                    if terminal:
                        reject(
                            origin,
                            f"cannot store into {spelled}: sequences are immutable; build a numpy array instead",
                        )
                    spine.append((current, resolved))
                    current = items[resolved]
                case TensorValue():
                    chain.append(current.allocation)
                    tensor = current
                    added = self._store_axes(origin, tensor, 0, index_value)
                    axes.extend(added)
                    spelled += self._spell_axes(added)
                case _:
                    reject(origin, f"{_aggregate.a_kind(current)} is not subscriptable")
        if tensor is None:
            raise AssertionError("the loop returns at the terminal sequence slot")
        rank = len(tensor.shape)
        if len(axes) < rank:
            reject(origin, "a store into an array must write one scalar element; rebind the whole array instead")
        flat = axes[0] * tensor.shape[1] + axes[1] if rank == 2 else axes[0]
        return _StoreSite(spelled, chain, spine, tensor, flat, tensor.leaves[flat])

    def _store_axes(self, origin: Origin, tensor: TensorValue, taken: int, index_value: Value) -> list[int]:
        if isinstance(index_value, SequenceValue):
            reject(origin, "a sequence index on an array is not supported; spell the axes directly (m[i, j])")
        rank = len(tensor.shape)
        if taken >= rank:
            plural = "index" if rank == 1 else "indices"
            reject(origin, f"a {rank}-D array store takes at most {rank} {plural}")
        dim = tensor.shape[taken]
        position = _aggregate.static_index(origin, index_value, "a subscript index")
        wrapped = position + dim if position < 0 else position
        if not 0 <= wrapped < dim:
            reject(origin, f"index {position} is out of bounds for an axis of length {dim}")
        return [wrapped]

    def _spell_axes(self, axes: list[int]) -> str:
        return "".join(f"[{axis}]" for axis in axes)

    def _aug_aggregate(
        self,
        origin: Origin,
        name: str,
        current: SequenceValue | TensorValue,
        op: BinaryOp,
        rhs: Value,
        sink: list[Stmt],
    ) -> Value:
        match current:
            case SequenceValue():
                reject(
                    origin, f"the operator `{op.value}=` is not supported on a sequence; build a numpy array instead"
                )
            case TensorValue():
                if op not in (BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV):
                    reject(origin, f"the operator `{op.value}=` is not supported on arrays yet")
                if isinstance(rhs, SequenceValue):
                    reject(origin, "cannot mix an array with a Python list/tuple; convert with np.array([...])")
                if not mutable(current.allocation):
                    reject(origin, f"cannot update {name!r} in place: {blame(current.allocation)}")
                computed = self._elementwise(origin, op, current, rhs, sink)
                if computed.family is not current.family:
                    reject(
                        origin,
                        f"an in-place operation cannot change the array's element family from "
                        f"{current.family.value} to {computed.family.value}; rebind with `{name} = {name} "
                        f"{op.value} ...` instead",
                    )
                return TensorValue(current.shape, current.family, computed.leaves, current.allocation)
            case _:
                raise AssertionError(current)

    def _to_tensor(self, origin: Origin, display: str, value: Value, sink: list[Stmt], *, copies: bool) -> TensorValue:
        match value:
            case TensorValue():
                result = TensorValue(value.shape, value.family, value.leaves, Allocation())
                if not copies:
                    share(value)
                    share(result)
                return result
            case SequenceValue(items=items):
                if not items:
                    reject(origin, f"{display}() of an empty sequence is not supported")
                shape, leaves = self._tensor_rows(origin, display, items)
                for leaf in leaves:
                    if not isinstance(leaf, Opaque) and leaf.stype is ScalarType.BOOL:
                        reject(origin, "an array must hold numbers, not booleans")
                if any(isinstance(leaf, Opaque) or leaf.stype is ScalarType.FLOAT for leaf in leaves):
                    family = ScalarType.FLOAT
                else:
                    family = ScalarType.INT
                conformed = [self._tensor_leaf(origin, family, leaf, sink) for leaf in leaves]
                self._budget.spend(len(conformed), origin, "the array conversion")
                return TensorValue(shape, family, tuple(conformed), Allocation())
            case _:
                reject(origin, f"{display}() requires a sequence or array argument, not {_aggregate.a_kind(value)}")

    def _tensor_rows(
        self, origin: Origin, display: str, items: tuple[Value, ...]
    ) -> tuple[tuple[int, ...], list[Scalar | Opaque]]:
        """The scalar leaves are copied, so the source keeps sole ownership of itself."""
        if all(isinstance(item, (StaticScalar, ResidualScalar, Opaque)) for item in items):
            flat = [item for item in items if isinstance(item, (StaticScalar, ResidualScalar, Opaque))]
            return (len(items),), flat
        rows: list[list[Scalar | Opaque]] = []
        for item in items:
            match item:
                case SequenceValue(items=row_items):
                    row: list[Scalar | Opaque] = []
                    for element in row_items:
                        if not isinstance(element, (StaticScalar, ResidualScalar, Opaque)):
                            reject(origin, f"{display}() supports only 1-D and 2-D rectangular constructions")
                        row.append(element)
                    rows.append(row)
                case TensorValue(shape=shape, leaves=leaves) if len(shape) == 1:
                    rows.append(list(leaves))
                case _:
                    reject(origin, f"{display}() supports only 1-D and 2-D rectangular constructions")
        widths = {len(row) for row in rows}
        if len(widths) != 1 or 0 in widths:
            reject(origin, f"{display}() requires rectangular rows of equal nonzero length")
        return (len(rows), widths.pop()), [leaf for row in rows for leaf in row]

    def _tensor_leaf(
        self, origin: Origin, family: ScalarType, leaf: Scalar | Opaque, sink: list[Stmt]
    ) -> Scalar | Opaque:
        if isinstance(leaf, Opaque):
            if family is not ScalarType.FLOAT or not nan_payload(leaf.value):
                reject(origin, _describe_opaque(leaf))
            return leaf
        if leaf.stype is ScalarType.BOOL:
            reject(origin, "an array must hold numbers, not booleans")
        if family is ScalarType.FLOAT:
            return self._as_float(leaf, origin, sink)
        if leaf.stype is ScalarType.FLOAT:
            reject(origin, "storing a float into an integer array truncates on the host; rebind a float array instead")
        return leaf

    def _inline(
        self,
        origin: Origin,
        display: str,
        fn: types.FunctionType,
        positional: list[Value],
        keywords: dict[str, Value],
        sink: list[Stmt],
        *,
        positional_only: bool,
    ) -> Value:
        site = Origin(origin.location, origin.frames + (CallFrame(display, origin.location),))
        if fn in self._inlining:
            reject(site, "recursive inlining is not supported")
        try:
            eel = self._desugared(fn)
        except SynthesisError as error:
            reattribute(site, error)
        annotations = self._annotations_of(site, fn)
        self._budget.spend(1, site, "the inlined call")
        callee = _rechained(eel, site.frames)
        assert isinstance(callee, EelFunction)
        if positional_only:
            assert not keywords
            if len(positional) != len(callee.params):
                reject(site, f"{display}() takes {len(callee.params)} argument(s), got {len(positional)}")
            bindings = {param.name: value for param, value in zip(callee.params, positional, strict=True)}
        else:
            bindings = self._bind_signature(site, fn, callee.params, positional, keywords)
        env: _Env = {}
        for param in callee.params:
            value = bindings[param.name]
            declared = annotations.get(param.name, _MISSING)
            if declared is not _MISSING:
                value = self._conform_value(value, declared, site, sink, f"the argument {param.name!r}")
            env[param.name] = value
        inner = _Frame(fn=fn, annotations=annotations, env=env, root=False)
        self._inlining.add(fn)
        try:
            returned = self._block(callee.body, inner, sink, in_branch=False)
        finally:
            self._inlining.discard(fn)
        if not returned or inner.result is None:
            reject(site, "the call returns no value, so it cannot be used in an expression")
        return inner.result

    def _bind_signature(
        self,
        site: Origin,
        fn: types.FunctionType,
        params: tuple[Param, ...],
        positional: list[Value],
        keywords: dict[str, Value],
    ) -> dict[str, Value]:
        signature = inspect.signature(fn)
        assert [param.name for param in params] == list(signature.parameters), "desugared params mirror the signature"
        try:
            bound = signature.bind(*positional, **keywords)
        except TypeError as error:
            reject(site, f"the arguments do not bind: {error}")
        bound.apply_defaults()
        bindings: dict[str, Value] = {}
        for name, value in bound.arguments.items():
            if isinstance(value, (StaticScalar, ResidualScalar, Opaque, SequenceValue, TensorValue, TensorMethod)):
                bindings[name] = value
            else:
                bindings[name] = self._snapshot.admit(name, value)  # an injected default: a plain Python object
        return bindings

    def _cast(self, node: Call, target: type, frame: _Frame, sink: list[Stmt]) -> Value:
        if len(node.args) != 1 or not isinstance(node.args[0], PosArg):
            reject(node.origin, f"{target.__name__}() takes exactly one positional argument here")
        operand = self._scalar(self._expr(node.args[0].value, frame, sink), node.origin)
        declared = _annotation_stype(target)
        assert declared is not None
        if operand.stype is declared:
            return operand
        return self._apply(_ops.CONVERT[(operand.stype, declared)], [operand], node.origin, sink)

    # ------------------------------------------------------------------ powers

    def _pow(self, origin: Origin, base: Scalar, exponent: Scalar, sink: list[Stmt]) -> Scalar:
        if ScalarType.BOOL in (base.stype, exponent.stype):
            reject(origin, "booleans take no part in arithmetic; cast explicitly with int(...) or float(...)")
        if isinstance(exponent, StaticScalar) and exponent.stype is ScalarType.INT:
            n = _ops.const_value(exponent.const)
            assert isinstance(n, int)
            if isinstance(base, StaticScalar):
                if (folded := self._fold_pow(base, n)) is not None:
                    return folded
            return self._pow_chain(origin, base, n, sink)
        if base.stype is ScalarType.INT and exponent.stype is ScalarType.INT:
            reject(
                origin,
                "a runtime integer exponent of an integer base is not supported yet; "
                "use a float base or a compile-time exponent",
            )
        promoted: list[Value] = [self._as_float(base, origin, sink), self._as_float(exponent, origin, sink)]
        value = self._inline(origin, "pow", self._pow_stub, promoted, {}, sink, positional_only=True)
        return self._scalar(value, origin)

    def _fold_pow(self, base: StaticScalar, n: int) -> StaticScalar | None:
        """
        The whole-power host fold for a fully static power with an integer exponent; an int power is exact at
        arbitrary precision (the budget carve-out demands better than a linear chain of huge folds). A refused
        value (overflow, zero to a negative power) falls back to the chain so the graph re-derives the fault
        survivor-based.
        """
        value = _ops.const_value(base.const)
        assert isinstance(value, (int, float))
        try:
            folded = value**n
        except (OverflowError, ZeroDivisionError, ValueError):
            return None
        assert type(folded) in (int, float), "an integer exponent cannot yield a complex power"
        return StaticScalar(_ops.make_const(folded))

    def _pow_chain(self, origin: Origin, base: Scalar, n: int, sink: list[Stmt]) -> Scalar:
        if n < 0:
            base = self._as_float(base, origin, sink)
            chain = self._chain(origin, base, -n, sink)
            one = StaticScalar(_ops.make_const(1.0))
            return self._apply(_ops.FLOAT_DIV, [one, chain], origin, sink)
        if n == 0:
            return StaticScalar(_ops.make_const(1 if base.stype is ScalarType.INT else 1.0))
        return self._chain(origin, base, n, sink)

    def _chain(self, origin: Origin, base: Scalar, n: int, sink: list[Stmt]) -> Scalar:
        assert n >= 1
        multiply = _ops.INT_BINARY[BinaryOp.MUL] if base.stype is ScalarType.INT else _ops.FLOAT_MUL
        result = base
        for _ in range(n - 1):
            self._budget.spend(1, origin, "the power chain")
            result = self._apply(multiply, [result, base], origin, sink)
        return result


def _allocations_match(a: Value, b: Value) -> bool:
    match a, b:
        case SequenceValue(), SequenceValue():
            return (
                a.allocation is b.allocation
                and len(a.items) == len(b.items)
                and all(_allocations_match(x, y) for x, y in zip(a.items, b.items))
            )
        case TensorValue(), TensorValue():
            return a.allocation is b.allocation
        case (SequenceValue() | TensorValue(), _) | (_, SequenceValue() | TensorValue()):
            return False
        case _:
            return True


# ---------------------------------------------------------------------- residual pruning


def _prune(stmts: list[Stmt] | tuple[Stmt, ...], live_after: set[int] | None = None) -> tuple[list[Stmt], set[int]]:
    """
    One reverse-liveness sweep; returns the surviving statements and the temp reads they need live-in.
    A branch whose arms both empty is dropped whole, de-rooting its condition so the condition's producer can
    disappear too. An arm is swept against the post-branch live set, so a join temp it assigns stays exactly
    when something after the branch reads it.
    """
    kept_reversed: list[Stmt] = []
    live = set(live_after) if live_after is not None else set()
    for stmt in reversed(stmts):
        match stmt:
            case Assign(target=TempBind(index=index), value=value):
                if index not in live:
                    continue
                live.discard(index)
                live |= _reads(value)
                kept_reversed.append(stmt)
            case If(origin=origin, cond=cond, then=then, orelse=orelse):
                then_kept, then_live = _prune(then, live)
                else_kept, else_live = _prune(orelse, live)
                if not then_kept and not else_kept:
                    continue
                live = then_live | else_live | _reads(cond)
                kept_reversed.append(If(origin, cond, tuple(then_kept), tuple(else_kept)))
            case ResidualReturn(values=values):
                for atom in values:
                    live |= _reads(atom)
                kept_reversed.append(stmt)
            case _:
                raise AssertionError(stmt)
    return list(reversed(kept_reversed)), live


def _reads(expr: Expr) -> set[int]:
    match expr:
        case TempRef(index=index):
            return {index}
        case IntrinsicCall(args=args):
            reads: set[int] = set()
            for arg in args:
                if isinstance(arg, TempRef):
                    reads.add(arg.index)
            return reads
        case LocalRef() | Const():
            return set()
        case _:
            raise AssertionError(expr)

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
residual arm. Library stubs bind positionally only. All pow spellings share the ``**`` lowering: a fully
static integer-exponent power folds exactly with host arithmetic, a fully static non-integer power folds
THROUGH the pow_ stub so the answer cannot depend on binding time, a static int exponent over a residual
base expands a float multiply chain, and everything else promotes to float through the pow_ stub. Exact
int results exist only for a nonnegative-int-base fold -- every non-folded power is float. A static power
that raises on the host surfaces as a compile-time diagnostic, never masked into a runtime value; the
raise is certain wherever the expression is evaluated, under the subset's own eager-gate semantics.

Ownership events ride value flow: a desugared temp is a linear conduit, so the first read of an aggregate
temp MOVES it and any re-read is the second handle, judged at the read itself regardless of where the value
lands; local-name reads stay innocent, their events firing where the value gains a persistent place.
Sequence-to-tensor conversions and sequence slices copy scalar leaves (exact); tensor-to-tensor derivations
conservatively share over the SOURCE allocation (a storage-equivalence token) -- except np.array of an
array, an independent copy on the host and the A5 explicit-copy spelling, which stays unique. A scalar
answers rank zero so library stubs can interrogate rank. An annotation never converts an array family: the
runtime object would keep its dtype, so a mismatch rejects rather than modeling a fiction.

Persistent state: slots are environment bindings keyed by the attribute path, initialized eagerly from
SlotRead temps so a write under one residual arm joins the entry value from the other. An aggregate state
read aliases only where its value gains a persistent place, and only when the value IS the current slot
tree -- state-slot provenance, never syntax. A root return terminates its path: the annotation fixes the
output table and each site commits its own outputs and slot live-outs, so a partially-returning branch
simply continues in the surviving arm. Installing a branch-JOINED aggregate rejects: the join mints a fresh
allocation for what is one of two runtime objects, erasing the provenance the disjointness checks need.
"""

import dataclasses
import inspect
import logging
import types
import typing
from dataclasses import dataclass

from ..._errors import SynthesisError
from .._desugar import desugar
from .._ir import *
from .._lib import Conversion, Factory, Intrinsic, Library, matmul_, resolve, transpose_
from .._names import indexed_names, public_slot, state_port_name
from . import _aggregate, _ops
from ._ownership import allocations, blame, borrow, escape, mutable, release, share
from ._reject import reattribute, reject
from ._residual import assigned_names, drop_return_rows, prune, rechained
from ._snapshot import Snapshotter, describe_opaque as _describe_opaque, nan_payload, tensor_of
from ._state import (
    ScalarSpec,
    SequenceSpec,
    Spec,
    StateModel,
    TensorSpec,
    descriptor_guard,
    environment_aggregates,
    spec_leaves,
)
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
    allocations_match,
    same,
    same_structure,
    tensor_occurrences,
    tree_leaves,
    tree_rebuild,
)

_logger = logging.getLogger(__name__)

type _SlotKey = tuple[str, ...]

type _EnvKey = str | int | _SlotKey

_AGGREGATE = (SequenceValue, TensorValue)

_RETURN_KEY = "<return>"


@dataclass(frozen=True, slots=True)
class _Ctx:
    """Where interpretation stands: a residual branch arm rejects raise; a residual loop also rejects return."""

    branch: bool = False
    loop: bool = False

    def arm(self) -> "_Ctx":
        return _Ctx(branch=True, loop=self.loop)


class _PromoteRun(Exception):
    """Internal driver signal: an INT slot leaf met a FLOAT live-out; re-run with that leaf promoted."""

    def __init__(self, path: SlotPath) -> None:
        super().__init__(path)
        self.path = path


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
class _SlotAlias:
    """
    A temp holding an aggregate state read: the tree's persistent home is the slot environment, so unlike an
    ordinary linear conduit the landing site must fire the aliasing share; reads are innocent like a local's.
    """

    value: SequenceValue | TensorValue


@dataclass(frozen=True, slots=True)
class _StoreSite:
    spelled: str
    chain: list[Allocation]
    spine: list[tuple[SequenceValue, int]]
    holder: TensorValue
    slot: int
    slot_value: Value


type _Env = dict[_EnvKey, Value | _Unjoinable | _Moved | _SlotAlias]


@dataclass(slots=True)
class _Frame:
    """
    One function under interpretation; ``result`` is where a non-root ``return`` deposits the call's value.
    ``slots`` is the environment holding the slot keys: the frame's own env for the root and its branch-arm
    forks, the caller's transitively for inlined helpers, so a helper reads the current post-write state.
    """

    fn: types.FunctionType
    annotations: dict[str, object]
    env: _Env
    root: bool
    slots: _Env
    result: Value | None = None


_MISSING = object()

_SCALAR_ANNOTATIONS: list[tuple[type, ScalarType]] = [
    (bool, ScalarType.BOOL),
    (int, ScalarType.INT),
    (float, ScalarType.FLOAT),
]


def partial_evaluate(eel: EelFunction, fn: types.FunctionType, instance: object | None = None) -> EelFunction:
    """
    The A2 trim driver: run with the seeded assumed-state set, shrink to the attributes whose write was
    actually reached, promote an INT slot that met a FLOAT live-out, and re-run until stable -- each run a
    fresh interpreter over the same immutable tree. A rejection under the conservative assumption is final,
    annotated with the pinning writes.
    """
    if instance is None:
        return _Interpreter(fn, None, None).run(eel)
    assert eel.params, "a bound method always has a receiver parameter"
    seed = assigned_names(eel.body, eel.params[0].name)[2]
    if not seed:
        return _Interpreter(fn, instance, None).run(eel)
    descriptor_guard(instance, seed)
    model = StateModel(instance, seed, environment_aggregates(fn))
    while True:  # terminates: each pass either finishes, strictly shrinks S, or promotes a fresh leaf
        interpreter = _Interpreter(fn, instance, model)
        try:
            residual = interpreter.run(eel)
        except _PromoteRun as promotion:
            model.promote(promotion.path)
            _logger.info("%s: the state leaf %s promotes to float; re-running", eel.name, promotion.path)
            continue
        except SynthesisError as error:
            raise type(error)(error.message + model.note(), error.location) from None
        if model.trim(interpreter.written_attrs):
            _logger.info(
                "%s: the assumed-state set shrinks to {%s}; re-running", eel.name, ", ".join(sorted(model.assumed))
            )
            if not model.assumed:
                return _Interpreter(fn, instance, None).run(eel)
            continue
        return residual


def _annotation_stype(annotation: object) -> ScalarType | None:
    return next((stype for ty, stype in _SCALAR_ANNOTATIONS if annotation is ty), None)


def _key_order(key: _EnvKey) -> tuple[int, str]:
    return (0 if isinstance(key, int) else 1 if isinstance(key, str) else 2), str(key)


class _Interpreter:
    def __init__(self, fn: types.FunctionType, instance: object | None, state: StateModel | None) -> None:
        self._fn = fn
        self._instance = instance
        self._state = state
        self._budget = ExpansionBudget()
        self._next_temp = 0
        self._outputs: tuple[OutputDecl, ...] | None = None
        self._elide: list[SlotPath | None] = []
        self._specs: dict[str, Spec] = {}
        self._decls: tuple[SlotDecl, ...] = ()
        self._slot_keys: list[_SlotKey] = []
        self._receiver_name: str | None = None
        self._state_owners: dict[Allocation, str] = {}
        self.written_attrs: set[str] = set()
        self._eels: dict[types.FunctionType, EelFunction] = {}
        self._meta: dict[types.FunctionType, dict[str, object]] = {}
        self._inlining: set[types.FunctionType] = {fn}
        self._snapshot = Snapshotter(state.check_capture if state is not None else None)
        anchor = resolve(pow)
        assert isinstance(anchor, Library), "the builtin pow key anchors the ** lowering"
        self._pow_stub = anchor.stub
        assert isinstance(matmul_, types.FunctionType) and isinstance(transpose_, types.FunctionType)
        self._matmul_stub: types.FunctionType = matmul_
        self._transpose_stub: types.FunctionType = transpose_

    def run(self, eel: EelFunction) -> EelFunction:
        env: _Env = {}
        frame = _Frame(
            fn=self._fn, annotations=self._annotations_of(eel.origin, self._fn), env=env, root=True, slots=env
        )
        remaining = eel.params
        if self._instance is not None:
            # The bound receiver is a snapshot root: frozen-attribute reads fold, state attributes read and
            # write their slot bindings, and calling its methods inlines them over the same receiver.
            assert remaining, "a bound method always has a receiver parameter"
            receiver = remaining[0]
            self._receiver_name = receiver.name
            frame.env[receiver.name] = self._snapshot.admit(receiver.name, self._instance, eel.origin)
            remaining = remaining[1:]
        params: list[Param] = []
        for param in remaining:
            params.extend(self._param(param, frame))
        names = [param.name for param in params]
        if len(set(names)) != len(names):
            collision = next(name for name in names if names.count(name) > 1)
            reject(eel.origin, f"the decomposed parameter names collide on {collision!r}; rename the parameters")
        sink: list[Stmt] = []
        if self._state is not None:
            self._specs = self._state.prepare()
            self._decls = self._state.decls()
            self._slot_keys = [(attr,) for attr in sorted(self._specs)]
            for attr in sorted(self._specs):
                frame.env[(attr,)] = self._init_slot(attr, self._specs[attr], eel.origin, sink)
            ports = names + [
                state_port_name(decl.slot) for decl in self._decls if not str(decl.slot[0]).startswith("_")
            ]
            if len(set(ports)) != len(ports):
                collision = next(name for name in ports if ports.count(name) > 1)
                reject(eel.origin, f"a state port collides with a parameter on {collision!r}; rename one of them")
        if not self._block(eel.body, frame, sink, _Ctx()):
            self._bare_site(eel.origin, frame, sink)
        assert self._outputs is not None, "every path commits a return site"
        keep = [candidate is None for candidate in self._elide]
        outputs = tuple(row for row, kept in zip(self._outputs, keep, strict=True) if kept)
        statements = sink if all(keep) else drop_return_rows(sink, keep)
        body, live = prune(statements)
        assert not live, "a residual temp is read before any assignment"
        residual = EelFunction(eel.origin, eel.name, tuple(params), tuple(body), slots=self._decls, outputs=outputs)
        _logger.info(
            "%s: partial evaluation: %d residual statement(s), %d output(s), %d slot(s), %d budget unit(s) spent",
            eel.name,
            len(body),
            len(outputs),
            len(self._decls),
            self._budget.spent,
        )
        return residual

    def _init_slot(self, attr: str, spec: Spec, origin: Origin, sink: list[Stmt]) -> Value:
        match spec:
            case ScalarSpec(path=path, stype=stype):
                index = self._fresh()
                sink.append(Assign(origin, TempBind(origin, index), SlotRead(origin, path), stype))
                return ResidualScalar(stype, TempRef(origin, index))
            case SequenceSpec(items=items):
                sequence = SequenceValue(
                    tuple(self._init_slot(attr, item, origin, sink) for item in items), Allocation()
                )
                self._state_owners[sequence.allocation] = attr
                return sequence
            case TensorSpec(shape=shape, family=family, leaves=leaves):
                scalars: list[Scalar | Opaque] = []
                for leaf in leaves:
                    index = self._fresh()
                    sink.append(Assign(origin, TempBind(origin, index), SlotRead(origin, leaf.path), leaf.stype))
                    scalars.append(ResidualScalar(leaf.stype, TempRef(origin, index)))
                tensor = TensorValue(shape, family, tuple(scalars), Allocation())
                self._state_owners[tensor.allocation] = attr
                return tensor

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

    def _block(self, stmts: tuple[Stmt, ...], frame: _Frame, sink: list[Stmt], ctx: _Ctx) -> bool:
        for stmt in stmts:
            match stmt:
                case Assign(target=target, value=value):
                    self._assign(target, value, frame, sink)
                case AugAssign(origin=origin, target=target, op=op, value=value):
                    current = frame.env.get(target.name)
                    if current is None:
                        reject(origin, f"the local name {target.name!r} is not bound on every path reaching this read")
                    current_value = self._readable(current, origin)
                    rhs = self._expr(value, frame, sink)
                    if isinstance(current_value, _AGGREGATE):
                        updated = self._aug_aggregate(origin, target.name, current_value, op, rhs, frame, sink)
                    else:
                        updated = self._binary(origin, op, current_value, rhs, frame, sink)
                    frame.env[target.name] = updated
                case Unpack(origin=origin, targets=targets, value=value):
                    source = self._expr(value, frame, sink)
                    items = _aggregate.unpack_items(origin, source, len(targets))
                    for target, item in zip(targets, items, strict=True):
                        frame.env[self._binding_key(target)] = item
                case If():
                    if self._if(stmt, frame, sink, ctx):
                        return True
                case Return(origin=origin):
                    if ctx.loop:
                        reject(origin, "a return inside a data-dependent loop is not supported yet")
                    if frame.root:
                        self._return_site(stmt, frame, sink)
                        return True
                    if stmt.value is not None:
                        result = self._expr(stmt.value, frame, sink)
                        annotation = frame.annotations.get("return", _MISSING)
                        if annotation is not _MISSING:
                            # At the return itself, so a mismatch points into the callee.
                            result = self._conform_value(result, annotation, origin, sink, "the returned value")
                        if isinstance(result, _AGGREGATE) and self._alias_conduit(frame, stmt.value):
                            share(result)
                        frame.result = result
                    return True
                case Raise():
                    self._raise(stmt, frame, sink, ctx)
                case While():
                    if self._while(stmt, frame, sink, ctx):
                        return True
                case For():
                    if self._for(stmt, frame, sink, ctx):
                        return True
                case Break(origin=origin):
                    reject(origin, "`break` is not supported yet")
                case Continue(origin=origin):
                    reject(origin, "`continue` is not supported yet")
                case Store() | AugStore():
                    self._store(stmt, frame, sink, ctx)
                case _:
                    raise AssertionError(stmt)
        return False

    def _assign(self, target: Binding, value: Expr, frame: _Frame, sink: list[Stmt]) -> None:
        result = self._expr(value, frame, sink)
        key = self._binding_key(target)
        if isinstance(result, _AGGREGATE):
            if isinstance(value, LocalRef) and value.name != key:
                share(result)
            elif (isinstance(value, AttrRead) and self._slot_tree(frame, result)) or self._alias_conduit(frame, value):
                if isinstance(target, TempBind):
                    frame.env[key] = _SlotAlias(result)
                    return
                share(result)
        frame.env[key] = result

    def _slot_tree(self, frame: _Frame, value: Value) -> bool:
        return any(frame.slots.get(key) is value for key in self._slot_keys)

    def _alias_conduit(self, frame: _Frame, atom: Expr) -> bool:
        return isinstance(atom, TempRef) and isinstance(frame.env.get(atom.index), _SlotAlias)

    def _binding_key(self, binding: Binding) -> _EnvKey:
        match binding:
            case LocalBind(name=name):
                return name
            case TempBind(index=index):
                return index

    def _if(self, stmt: If, frame: _Frame, sink: list[Stmt], ctx: _Ctx) -> bool:
        """The branch structure itself is the return join; an inlined frame value-joins all-arms returns only."""
        cond = self._condition(stmt.origin, self._expr(stmt.cond, frame, sink))
        if isinstance(cond, StaticScalar):
            decided = _ops.const_value(cond.const)
            assert isinstance(decided, bool)
            taken = stmt.then if decided else stmt.orelse
            return self._block(taken, frame, sink, ctx)
        then_env: _Env = dict(frame.env)
        else_env: _Env = dict(frame.env)
        then_frame = _Frame(
            frame.fn, frame.annotations, then_env, frame.root, slots=then_env if frame.root else frame.slots
        )
        else_frame = _Frame(
            frame.fn, frame.annotations, else_env, frame.root, slots=else_env if frame.root else frame.slots
        )
        then_sink: list[Stmt] = []
        else_sink: list[Stmt] = []
        then_done = self._block(stmt.then, then_frame, then_sink, ctx.arm())
        else_done = self._block(stmt.orelse, else_frame, else_sink, ctx.arm())
        if not frame.root and (then_done or else_done):
            if not (then_done and else_done):
                reject(stmt.origin, "a return on only one branch path of an inlined function is not supported yet")
            frame.result = self._join_results(stmt.origin, then_frame.result, else_frame.result, then_sink, else_sink)
        joined: _Env | None = None
        if not then_done and not else_done:
            joined = self._join(stmt.origin, then_frame.env, then_sink, else_frame.env, else_sink)
        sink.append(If(stmt.origin, self._materialize(cond, stmt.origin), tuple(then_sink), tuple(else_sink)))
        if then_done and else_done:
            return True
        frame.env.clear()
        frame.env.update(joined if joined is not None else else_frame.env if then_done else then_frame.env)
        return False

    def _join_results(
        self, origin: Origin, a: Value | None, b: Value | None, then_sink: list[Stmt], else_sink: list[Stmt]
    ) -> Value | None:
        if a is None or b is None:
            if a is not b:
                reject(origin, "one branch returns a value and the other does not")
            return None
        merged = self._merge(origin, _RETURN_KEY, a, b, then_sink, else_sink, allocations_match(a, b))
        if merged is None:
            reject(origin, "the branches return values the compiler cannot merge")
        if not allocations_match(a, b) and isinstance(merged, _AGGREGATE):
            share(a)
            share(b)
            share(merged)
        return merged

    def _join(
        self, origin: Origin, then_env: _Env, then_sink: list[Stmt], else_env: _Env, else_sink: list[Stmt]
    ) -> _Env:
        joined: _Env = {}
        for key in sorted(set(then_env) & set(else_env), key=_key_order):
            a_bound, b_bound = then_env[key], else_env[key]
            moved = isinstance(a_bound, _Moved) or isinstance(b_bound, _Moved)
            aliased = isinstance(a_bound, _SlotAlias) and isinstance(b_bound, _SlotAlias)
            a = a_bound.value if isinstance(a_bound, (_Moved, _SlotAlias)) else a_bound
            b = b_bound.value if isinstance(b_bound, (_Moved, _SlotAlias)) else b_bound
            if isinstance(a, _Unjoinable) or isinstance(b, _Unjoinable):
                joined[key] = _Unjoinable(self._describe_key(key))
                continue
            if same(a, b):
                if isinstance(a, _AGGREGATE) and (moved or aliased):
                    joined[key] = _Moved(a) if moved else _SlotAlias(a)
                else:
                    joined[key] = a
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
        keep = allocations_match(a, b)
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
                return SequenceValue(tuple(items), a.allocation if keep else self._minted(a, b))
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
                return TensorValue(a.shape, family, tuple(leaves), a.allocation if keep else self._minted(a, b))
            case (StaticScalar() | ResidualScalar()), (StaticScalar() | ResidualScalar()):
                return self._join_scalars(origin, self._describe_key(key), a, b, then_sink, else_sink)
            case Opaque(), Opaque():
                return a if same(a, b) else None
            case _:
                return None

    def _minted(self, a: Value, b: Value) -> Allocation:
        """A keep=False merge product is ONE of the two runtime objects; state-ness rides along with it."""
        assert isinstance(a, _AGGREGATE) and isinstance(b, _AGGREGATE)
        minted = Allocation(joined=True)
        owner = self._state_owners.get(a.allocation) or self._state_owners.get(b.allocation)
        if owner is not None:
            self._state_owners[minted] = owner
        return minted

    def _describe_key(self, key: _EnvKey) -> str:
        if key == _RETURN_KEY:
            return "the returned value"
        if isinstance(key, tuple):
            return f"the state attribute self.{key[0]}"
        return f"the local {key!r}" if isinstance(key, str) else "the conditional result"

    def _readable(self, binding: Value | _Unjoinable | _Moved | _SlotAlias, origin: Origin) -> Value:
        if isinstance(binding, _Unjoinable):
            reject(
                origin,
                f"{binding.description} holds branch values the compiler cannot merge; "
                "only bool, int, and float values join branches",
            )
        if isinstance(binding, _SlotAlias):
            return binding.value
        if isinstance(binding, _Moved):
            share(binding.value)
            return binding.value
        return binding

    def _raise(self, stmt: Raise, frame: _Frame, sink: list[Stmt], ctx: _Ctx) -> None:
        if ctx.branch or ctx.loop or (frame.root and self._outputs is not None):
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

    # ------------------------------------------------------------------ loops

    def _for(self, stmt: For, frame: _Frame, sink: list[Stmt], ctx: _Ctx) -> bool:
        iterable = self._expr(stmt.iterable, frame, sink)
        if not isinstance(iterable, _AGGREGATE):
            reject(stmt.origin, f"{_aggregate.a_kind(iterable)} is not iterable")
        items = _aggregate.splice_items(stmt.origin, iterable)
        held = borrow(iterable)
        try:
            for item in items:
                self._budget.spend(1, stmt.origin, "the unrolled loop")
                frame.env[stmt.target.name] = item
                if self._block(stmt.body, frame, sink, ctx):
                    return True
        finally:
            release(held)
        return False

    def _while(self, stmt: While, frame: _Frame, sink: list[Stmt], ctx: _Ctx) -> bool:
        """
        Static tests execute trip by trip, splicing each header evaluation (CPython runs the failing test's
        header too); the first residual test rolls the env back to before its header and residualizes the
        remaining loop. Ownership events fired by the discarded header evaluation persist -- the recorded
        speculative-evaluation conservatism.
        """
        while True:
            saved = dict(frame.env)
            header_sink: list[Stmt] = []
            returned = self._block(stmt.header, frame, header_sink, ctx)
            assert not returned, "a test expression cannot return"
            cond = self._condition(stmt.origin, self._expr(stmt.cond, frame, header_sink))
            if isinstance(cond, ResidualScalar):
                frame.env.clear()
                frame.env.update(saved)
                self._residual_while(stmt, frame, sink)
                return False
            sink.extend(header_sink)
            decided = _ops.const_value(cond.const)
            assert isinstance(decided, bool)
            if not decided:
                return False
            self._budget.spend(1, stmt.origin, "the unrolled loop")
            if self._block(stmt.body, frame, sink, ctx):
                return True

    def _condition(self, origin: Origin, value: Value) -> Scalar:
        if isinstance(value, _AGGREGATE):
            reject(origin, "the truthiness of an aggregate is not supported")
        cond = self._scalar(value, origin)
        if cond.stype is not ScalarType.BOOL:
            reject(origin, "the branch condition must be a bool; Python truthiness is not supported")
        return cond

    def _residual_while(self, stmt: While, frame: _Frame, sink: list[Stmt]) -> None:
        """
        The carried set is the syntactic assigned-name set of header+body, restricted to names bound at entry
        -- each becomes a scalar phi. One symbolic header+body pass computes the back-edge values; the stype
        assumptions join per the one rule (a FLOAT phi converts an INT back value in place; an INT assumption
        meeting a FLOAT back value promotes and re-runs, monotone and bounded by the carried count). The
        post-loop env is the env after the final pass's HEADER: the header runs on the failing test too, so
        its bindings survive the loop (emit-side the header dominates the exit), while body-only names drop
        (a zero-trip loop never binds them).
        """
        origin = stmt.origin
        carried: list[tuple[_EnvKey, list[int]]] = []
        stypes: dict[tuple[_EnvKey, int], ScalarType] = {}
        rebound, stored, loop_attrs = assigned_names((*stmt.header, *stmt.body), self._receiver_name)
        for name in sorted(rebound | stored):
            bound = frame.env.get(name)
            if bound is None:
                continue
            assert not isinstance(bound, (_Moved, _SlotAlias)), "only temps ride the conduits"
            value = self._readable(bound, origin)
            if isinstance(value, (StaticScalar, ResidualScalar)):
                if name in rebound:
                    carried.append((name, [self._fresh()]))
                    stypes[(name, 0)] = value.stype
                continue
            if name in rebound or isinstance(value, _AGGREGATE):
                reject(
                    origin,
                    f"{name!r} is {_aggregate.a_kind(value)}; only bool, int, and float values can be "
                    "carried across the iterations of a data-dependent loop",
                )
            # A store through a non-aggregate root can never be admitted; the body pass draws the precise
            # rejection at the store itself (an attribute store on a snapshot, an item store on a scalar).
        if frame.root:
            for attr in sorted(set(loop_attrs) & set(self._specs)):
                slot_key: _SlotKey = (attr,)
                entry_tree = self._readable(frame.env[slot_key], origin)
                leaves = tree_leaves(entry_tree)
                for position, leaf in enumerate(leaves):
                    if isinstance(leaf, Opaque):
                        reject(origin, _describe_opaque(leaf))
                    stypes[(slot_key, position)] = leaf.stype
                carried.append((slot_key, [self._fresh() for _ in leaves]))
        while True:
            loop_env: _Env = dict(frame.env)
            loop_frame = _Frame(
                frame.fn, frame.annotations, loop_env, frame.root, slots=loop_env if frame.root else frame.slots
            )
            for key, indices in carried:
                phi_leaves: list[Scalar] = [
                    ResidualScalar(stypes[(key, position)], TempRef(origin, index))
                    for position, index in enumerate(indices)
                ]
                if isinstance(key, tuple):
                    entry_tree = self._readable(frame.env[key], origin)
                    loop_frame.env[key] = tree_rebuild(entry_tree, iter(phi_leaves))
                else:
                    (only,) = phi_leaves
                    loop_frame.env[key] = only
            header_sink: list[Stmt] = []
            loop_ctx = _Ctx(branch=True, loop=True)
            returned = self._block(stmt.header, loop_frame, header_sink, loop_ctx)
            assert not returned, "a test expression cannot return"
            cond = self._condition(origin, self._expr(stmt.cond, loop_frame, header_sink))
            assert isinstance(cond, ResidualScalar), "a residual test cannot re-fold under residual carries"
            post_header_env = dict(loop_frame.env)
            body_sink: list[Stmt] = []
            returned = self._block(stmt.body, loop_frame, body_sink, loop_ctx)
            assert not returned, "the loop context rejects returns before they unwind"
            promoted = False
            backs: dict[tuple[_EnvKey, int], Scalar] = {}
            for key, indices in carried:
                back_bound = loop_frame.env.get(key)
                assert back_bound is not None, "an entry-bound key cannot vanish"
                back_value = self._readable(back_bound, origin)
                back_leaves = self._back_leaves(key, back_value, frame.env, origin)
                assert len(back_leaves) == len(indices)
                for position, back in enumerate(back_leaves):
                    assumed = stypes[(key, position)]
                    if back.stype is assumed:
                        backs[(key, position)] = back
                    elif assumed is ScalarType.FLOAT and back.stype is ScalarType.INT:
                        backs[(key, position)] = self._as_float(back, origin, body_sink)
                    elif assumed is ScalarType.INT and back.stype is ScalarType.FLOAT:
                        if isinstance(key, tuple):
                            assert len(indices) == 1 and isinstance(
                                self._specs[key[0]], ScalarSpec
                            ), "aggregate slot leaves cannot change type inside the loop"
                        stypes[(key, position)] = ScalarType.FLOAT
                        promoted = True
                    else:
                        reject(
                            origin,
                            f"the loop rebinds {self._describe_key(key)} from {assumed.value} to "
                            f"{back.stype.value} across iterations; only int-with-float joins promote",
                        )
            if not promoted:
                break
        phis: list[LoopPhi] = []
        for key, indices in carried:
            entry_value = self._readable(frame.env[key], origin)
            entry_leaves = tree_leaves(entry_value)
            assert len(entry_leaves) == len(indices)
            assert not any(isinstance(leaf, Opaque) for leaf in entry_leaves), "the carry setup rejected opaques"
            for position, index in enumerate(indices):
                entry_leaf = entry_leaves[position]
                assert isinstance(entry_leaf, (StaticScalar, ResidualScalar))
                entry: Scalar = entry_leaf
                if stypes[(key, position)] is ScalarType.FLOAT and entry.stype is ScalarType.INT:
                    entry = self._as_float(entry, origin, sink)
                assert entry.stype is stypes[(key, position)]
                back_atom = self._materialize(backs[(key, position)], origin)
                phis.append(
                    LoopPhi(origin, index, stypes[(key, position)], self._materialize(entry, origin), back_atom)
                )
        _logger.debug(
            "%s: the loop at %s residualizes with %d carried phi(s): %s",
            frame.fn.__qualname__,
            origin.location,
            len(phis),
            ", ".join(str(key) for key, _ in carried),
        )
        sink.append(
            ResidualWhile(origin, tuple(phis), tuple(header_sink), self._materialize(cond, origin), tuple(body_sink))
        )
        frame.env.clear()
        frame.env.update(post_header_env)

    def _back_leaves(self, key: _EnvKey, back_value: Value, entry_env: _Env, origin: Origin) -> list[Scalar]:
        """The per-leaf back-edge values of one carried key; a slot tree must keep its structure and identity."""
        if isinstance(key, str):
            if not isinstance(back_value, (StaticScalar, ResidualScalar)):
                reject(
                    origin,
                    f"{key!r} is rebound to {_aggregate.a_kind(back_value)} inside a data-dependent loop; "
                    "only bool, int, and float values can be carried across its iterations",
                )
            return [back_value]
        entry_tree = self._readable(entry_env[key], origin)
        if not same_structure(entry_tree, back_value):
            reject(
                origin,
                f"installing a new aggregate into {self._describe_key(key)} inside a data-dependent loop "
                "is not supported yet; store its elements instead",
            )
        leaves: list[Scalar] = []
        for leaf in tree_leaves(back_value):
            if isinstance(leaf, Opaque):
                reject(origin, _describe_opaque(leaf))
            leaves.append(leaf)
        return leaves

    def _comp(self, node: Comp, frame: _Frame, sink: list[Stmt]) -> SequenceValue:
        iterable = self._expr(node.iterable, frame, sink)
        if not isinstance(iterable, _AGGREGATE):
            reject(node.origin, f"{_aggregate.a_kind(iterable)} is not iterable")
        items = _aggregate.splice_items(node.origin, iterable)
        held = borrow(iterable)
        collected: list[Value] = []
        try:
            for item in items:
                self._budget.spend(1, node.origin, "the comprehension")
                frame.env[node.target] = item
                returned = self._block(node.body, frame, sink, _Ctx(branch=True))
                assert not returned, "a comprehension body holds no return"
                value = self._expr(node.element, frame, sink)
                if isinstance(value, _AGGREGATE) and (
                    isinstance(node.element, LocalRef) or self._alias_conduit(frame, node.element)
                ):
                    share(value)
                collected.append(value)
        finally:
            release(held)
        return SequenceValue(tuple(collected), Allocation())

    # ------------------------------------------------------------------ the module boundary

    def _return_site(self, stmt: Return, frame: _Frame, sink: list[Stmt]) -> None:
        annotation = frame.annotations.get("return", _MISSING)
        if annotation is _MISSING:
            reject(stmt.origin, "the return type annotation is required")
        if stmt.value is None:
            if annotation is not None:
                reject(stmt.origin, "the kernel returns no value but its annotation declares one")
            self._commit_site(stmt.origin, frame, sink, [])
            return
        value = self._expr(stmt.value, frame, sink)
        if isinstance(value, Opaque):
            reject(stmt.origin, f"the captured object {value.name!r} cannot be returned")
        conformed = self._conform_value(value, annotation, stmt.origin, sink, "the returned value", root=True)
        match conformed:
            case StaticScalar() | ResidualScalar():
                self._commit_site(stmt.origin, frame, sink, [((0,), conformed)])
            case SequenceValue() | TensorValue():
                for allocation in allocations(conformed):
                    owner = self._state_owners.get(allocation)
                    if owner is not None:
                        reject(
                            stmt.origin,
                            f"returning the state attribute self.{owner} would hand out a live alias of its "
                            "storage in Python, which hardware cannot honor; return an explicit copy "
                            "(np.array(...) or a fresh sequence) instead",
                        )
                escape(conformed)
                self._commit_site(stmt.origin, frame, sink, _aggregate.flatten(stmt.origin, conformed))
            case _:
                raise AssertionError(conformed)

    def _bare_site(self, origin: Origin, frame: _Frame, sink: list[Stmt]) -> None:
        annotation = frame.annotations.get("return", _MISSING)
        if annotation is _MISSING:
            reject(origin, "the return type annotation is required")
        if annotation is not None:
            reject(origin, "the kernel can complete without returning a value but its annotation declares one")
        self._commit_site(origin, frame, sink, [])

    def _commit_site(
        self, origin: Origin, frame: _Frame, sink: list[Stmt], rows: list[tuple[_aggregate.LeafPath, Scalar]]
    ) -> None:
        """A leaf matching the same public, unpromoted slot's live-out at EVERY site is elided after the run."""
        decls = tuple(OutputDecl(path, leaf.stype) for path, leaf in rows)
        first = self._outputs is None
        if first:
            self._outputs = decls
        elif decls != self._outputs:
            reject(origin, "this return does not match the kernel's other return sites in shape or type")
        slot_values: dict[SlotPath, Scalar] = {}
        for attr in sorted(self._specs):
            tree = self._readable(frame.slots[(attr,)], origin)
            for spec_leaf, leaf in zip(spec_leaves(self._specs[attr]), tree_leaves(tree), strict=True):
                if isinstance(leaf, Opaque):
                    reject(origin, _describe_opaque(leaf))
                committed = self._slot_conform(attr, spec_leaf, leaf, origin, sink)
                slot_values[spec_leaf.path] = committed
                sink.append(SlotWrite(origin, spec_leaf.path, self._materialize(committed, origin)))
        promoted = self._state.promoted_paths() if self._state is not None else frozenset()
        public = {path: scalar for path, scalar in slot_values.items() if public_slot(path) and path not in promoted}
        ordered = sorted(public)
        if first:
            self._elide = [next((path for path in ordered if same(leaf, public[path])), None) for _, leaf in rows]
        else:
            for position, (_, leaf) in enumerate(rows):
                candidate = self._elide[position]
                if candidate is not None and not same(leaf, public[candidate]):
                    self._elide[position] = None
        sink.append(ResidualReturn(origin, tuple(self._materialize(leaf, origin) for _, leaf in rows)))

    def _slot_conform(self, attr: str, spec: ScalarSpec, leaf: Scalar, origin: Origin, sink: list[Stmt]) -> Scalar:
        if leaf.stype is spec.stype:
            return leaf
        if spec.stype is ScalarType.FLOAT and leaf.stype is ScalarType.INT:
            return self._as_float(leaf, origin, sink)
        if spec.stype is ScalarType.INT and leaf.stype is ScalarType.FLOAT:
            raise _PromoteRun(spec.path)
        reject(
            origin,
            f"the state attribute self.{attr} would change type from {spec.stype.value} to {leaf.stype.value}; "
            "bool state joins only with bool",
        )

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
            if value.family is not family:
                reject(
                    origin,
                    f"{what} is a {value.family.value} array where the annotation declares {family.value}; "
                    "an annotation does not convert an array -- build it from matching elements",
                )
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
                if isinstance(read, _AGGREGATE) and not isinstance(bound, (_Moved, _SlotAlias)):
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
                return self._binary(origin, op, lv, rv, frame, sink)
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
                return self._attr_read(origin, self._expr(base, frame, sink), attr, frame, sink)
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
            case Comp():
                return self._comp(expr, frame, sink)
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
                if isinstance(value, _AGGREGATE) and (isinstance(item, LocalRef) or self._alias_conduit(frame, item)):
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
        return self._snapshot.admit(node.name, raw, node.origin)

    # ------------------------------------------------------------------ attribute reads

    def _attr_read(self, origin: Origin, base_value: Value, attr: str, frame: _Frame, sink: list[Stmt]) -> Value:
        match base_value:
            case TensorValue():
                return self._tensor_attr(origin, base_value, attr, frame, sink)
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
                return self._snapshot.admit(f"{name}.{attr}", raw, origin)
            case Opaque():
                return self._instance_attr(origin, base_value, attr, frame, sink)
            case _:
                raise AssertionError(base_value)

    def _tensor_attr(self, origin: Origin, tensor: TensorValue, attr: str, frame: _Frame, sink: list[Stmt]) -> Value:
        if attr == "ndim":
            return StaticScalar(_ops.make_const(len(tensor.shape)))
        if attr == "shape":
            dims = tuple(StaticScalar(_ops.make_const(dim)) for dim in tensor.shape)
            return SequenceValue(dims, Allocation())
        if attr == "T":
            share(tensor)
            return self._transposed(origin, tensor, frame, sink)
        if attr == "flatten":
            share(tensor)
            return TensorMethod(tensor, "flatten")
        reject(origin, f"an array has no supported attribute {attr!r}")

    def _instance_attr(self, origin: Origin, base: Opaque, attr: str, frame: _Frame, sink: list[Stmt]) -> Value:
        """
        CPython's attribute precedence over class metadata, never calling into the object: a state attribute
        of the entry receiver reads its current slot binding, then a data descriptor wins (only a plain
        property is readable -- its getter is inlined as ordinary code), then the instance dict, then
        non-data class attributes.
        """
        if base.value is self._instance and (attr,) in frame.slots:
            bound = frame.slots[(attr,)]
            assert not isinstance(bound, (_Moved, _SlotAlias)), "slot bindings never ride the conduits"
            return self._readable(bound, origin)
        found: object = next(
            (klass.__dict__[attr] for klass in type(base.value).__mro__ if attr in klass.__dict__), _MISSING
        )
        if isinstance(found, property) and isinstance(found.fget, types.FunctionType):
            return self._inline(
                origin, f"{base.name}.{attr}", found.fget, [base], {}, frame, sink, positional_only=False
            )
        if found is not _MISSING and inspect.isdatadescriptor(found):
            reject(origin, f"the attribute {attr!r} is a descriptor the compiler cannot read")
        instance_dict = getattr(base.value, "__dict__", None)
        if isinstance(instance_dict, dict) and attr in instance_dict:
            return self._snapshot.admit(f"{base.name}.{attr}", instance_dict[attr], origin)
        if found is _MISSING:
            reject(origin, f"{base.name!r} has no attribute {attr!r}")
        if isinstance(found, staticmethod):
            return self._snapshot.admit(f"{base.name}.{attr}", found.__func__, origin)
        if isinstance(found, types.FunctionType):
            return self._snapshot.admit(f"{base.name}.{attr}", types.MethodType(found, base.value), origin)
        if isinstance(found, classmethod):
            return self._snapshot.admit(
                f"{base.name}.{attr}", types.MethodType(found.__func__, type(base.value)), origin
            )
        return self._snapshot.admit(f"{base.name}.{attr}", found, origin)

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

    def _binary(self, origin: Origin, op: BinaryOp, lv: Value, rv: Value, frame: _Frame, sink: list[Stmt]) -> Value:
        if op is BinaryOp.MATMUL:
            if isinstance(lv, SequenceValue) or isinstance(rv, SequenceValue):
                reject(origin, "`@` on a Python list/tuple is not supported; build a numpy array with np.array([...])")
            for operand in (lv, rv):
                if isinstance(operand, _AGGREGATE):
                    share(operand)
            return self._inline(origin, "np.matmul", self._matmul_stub, [lv, rv], {}, frame, sink, positional_only=True)
        tensors = isinstance(lv, TensorValue) or isinstance(rv, TensorValue)
        sequences = isinstance(lv, SequenceValue) or isinstance(rv, SequenceValue)
        if (tensors or sequences) and op in (BinaryOp.AND, BinaryOp.OR):
            reject(origin, "the truthiness of an aggregate is not supported")
        if tensors and sequences:
            reject(origin, "cannot mix an array with a Python list/tuple; convert with np.array([...])")
        if tensors:
            return self._elementwise(origin, op, lv, rv, frame, sink)
        if sequences:
            return self._sequence_binary(origin, op)
        return self._binary_scalars(origin, op, lv, rv, frame, sink)

    def _elementwise(
        self, origin: Origin, op: BinaryOp, lv: Value, rv: Value, frame: _Frame, sink: list[Stmt]
    ) -> TensorValue:
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
            leaves.append(self._binary_scalars(origin, op, a, b, frame, sink))
        self._budget.spend(len(leaves), origin, "the elementwise operation")
        family = next(leaf.stype for leaf in leaves if not isinstance(leaf, Opaque))
        return TensorValue(shape, family, tuple(leaves), Allocation())

    def _sequence_binary(self, origin: Origin, op: BinaryOp) -> Value:
        reject(origin, f"the operator `{op.value}` is not supported on a sequence; build a numpy array instead")

    def _binary_scalars(
        self, origin: Origin, op: BinaryOp, lv: Value, rv: Value, frame: _Frame, sink: list[Stmt]
    ) -> Scalar:
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
                return self._pow(origin, left, right, frame, sink)
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
            return self._tensor_method(node, callee)
        if not isinstance(callee, Opaque):
            reject(node.origin, "the callee is not a callable object")
        raw = callee.value
        match resolve(raw):
            case Intrinsic(operator=operator):
                return self._intrinsic(node, callee.name, operator, frame, sink)
            case Library(stub=stub):
                values = self._positional_arguments(node, callee.name, frame, sink)
                if stub is self._transpose_stub:
                    if len(values) != 1 or not isinstance(values[0], TensorValue):
                        return self._inline(
                            node.origin, callee.name, stub, values, {}, frame, sink, positional_only=True
                        )
                    return self._transposed(node.origin, values[0], frame, sink)
                if stub is self._pow_stub:
                    # All pow spellings share the ** lowering.
                    if len(values) != 2:
                        reject(node.origin, f"{callee.name}() takes 2 argument(s), got {len(values)}")
                    base = self._scalar(values[0], node.origin)
                    exponent = self._scalar(values[1], node.origin)
                    return self._pow(node.origin, base, exponent, frame, sink)
                return self._inline(node.origin, callee.name, stub, values, {}, frame, sink, positional_only=True)
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
        if raw is range:
            return self._range(node, frame, sink)
        if raw is list or raw is tuple:
            return self._rebuild_sequence(node, callee.name, frame, sink)
        if isinstance(raw, types.FunctionType):
            # Ingested like user code wherever it lives: substitution is the registry's decision alone
            # (maintainer ruling), and the interpreter knows no library names.
            positional, keywords = self._signature_arguments(node, frame, sink)
            return self._inline(node.origin, callee.name, raw, positional, keywords, frame, sink, positional_only=False)
        if inspect.ismethod(raw):
            # A helper method of any snapshotted object: the receiver rides as the first argument, sharing
            # the memoized identity, so its frozen attributes fold and its state reads see the current slots;
            # a receiver write inside the helper draws the helper-write rejection.
            inner = raw.__func__
            if not isinstance(inner, types.FunctionType):
                reject(node.origin, f"calls to {callee.name!r} are not supported yet")
            receiver = self._snapshot.admit(callee.name, raw.__self__, node.origin)
            positional, keywords = self._signature_arguments(node, frame, sink)
            return self._inline(
                node.origin,
                callee.name,
                inner,
                [receiver, *positional],
                keywords,
                frame,
                sink,
                positional_only=False,
            )
        if callable(raw):
            if not isinstance(raw, type) and isinstance(getattr(raw, "__dict__", None), dict):
                reject(
                    node.origin,
                    f"cannot call {callee.name!r}: it is a separate component instance "
                    f"({type(raw).__name__}); hierarchical state is not supported yet",
                )
            reject(node.origin, f"calls to {callee.name!r} are not supported yet")
        reject(node.origin, f"the captured object {callee.name!r} is not callable")

    def _argument(self, atom: Atom, frame: _Frame, sink: list[Stmt]) -> Value:
        value = self._expr(atom, frame, sink)
        if isinstance(value, _AGGREGATE) and (isinstance(atom, LocalRef) or self._alias_conduit(frame, atom)):
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

    def _transposed(self, origin: Origin, tensor: TensorValue, frame: _Frame, sink: list[Stmt]) -> Value:
        result = self._inline(
            origin, "np.transpose", self._transpose_stub, [tensor], {}, frame, sink, positional_only=True
        )
        if isinstance(result, TensorValue):
            result = dataclasses.replace(result, allocation=tensor.allocation)
        return result

    def _tensor_method(self, node: Call, method: TensorMethod) -> Value:
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

    def _range(self, node: Call, frame: _Frame, sink: list[Stmt]) -> SequenceValue:
        values = self._operand_arguments(node, "range", frame, sink)
        if not 1 <= len(values) <= 3:
            reject(node.origin, f"range() takes 1 to 3 arguments, got {len(values)}")
        bounds = [_aggregate.static_index(node.origin, value, "a range argument") for value in values]
        try:
            span = range(*bounds)
        except ValueError as error:
            reject(node.origin, f"range() rejects its arguments: {error}")
        # The length in exact arithmetic: len() of a huge range overflows Py_ssize_t, and the charge must
        # land as the located budget rejection before any materialization.
        length = max(0, (span.stop - span.start + span.step - (1 if span.step > 0 else -1)) // span.step)
        self._budget.spend(max(length, 1), node.origin, "the range materialization")
        return SequenceValue(tuple(StaticScalar(_ops.make_const(i)) for i in span), Allocation())

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

    def _store(self, stmt: Store | AugStore, frame: _Frame, sink: list[Stmt], ctx: _Ctx) -> None:
        """
        The one mutation gate: resolve the target path with the store-prefix exemption (walked over the value
        tree, never through expression reads), judge admission at the store step after the RHS and index
        temps, and rebind new frozen values along the unchanged allocation spine. A store rooted at the
        entry method's receiver is a state write.
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
        if isinstance(root_value, Opaque) and self._instance is not None and root_value.value is self._instance:
            self._receiver_store(stmt, name, rhs, frame, sink, ctx)
            return
        if not isinstance(root_value, _AGGREGATE):
            if isinstance(stmt.path[0], AttrSel):
                reject(origin, "an attribute store is only supported on the entry method's receiver")
            reject(origin, f"{_aggregate.a_kind(root_value)} does not support item assignment")
        frame.env[name] = self._element_store(stmt, name, root_value, stmt.path, rhs, frame, sink)

    def _receiver_store(
        self, stmt: Store | AugStore, name: str, rhs: Value, frame: _Frame, sink: list[Stmt], ctx: _Ctx
    ) -> None:
        origin = stmt.origin
        if not frame.root:
            reject(origin, "a helper method cannot write attributes of the receiver; only the entry method may")
        assert self._receiver_name is not None
        if name != self._receiver_name:
            reject(origin, f"the receiver may only be written through its parameter name {self._receiver_name!r}")
        first = stmt.path[0]
        if not isinstance(first, AttrSel):
            reject(origin, "the receiver itself does not support item assignment; store into an attribute")
        attr = first.name
        assert self._state is not None and attr in self._specs, "the syntactic seed covers every receiver store"
        self.written_attrs.add(attr)
        assert frame.env is frame.slots, "root and arm frames resolve slots in their own environment"
        key: _SlotKey = (attr,)
        slot_value = self._readable(frame.slots[key], origin)
        rest = stmt.path[1:]
        if rest:
            if isinstance(rest[0], AttrSel):
                reject(origin, "an attribute store through a nested object is not supported")
            if not isinstance(slot_value, _AGGREGATE):
                reject(origin, f"self.{attr} is a scalar state attribute; it has no elements")
            frame.env[key] = self._element_store(stmt, f"self.{attr}", slot_value, rest, rhs, frame, sink)
            return
        spec = self._specs[attr]
        if isinstance(stmt, AugStore):
            if isinstance(slot_value, Opaque):
                reject(origin, _describe_opaque(slot_value))
            if isinstance(slot_value, _AGGREGATE):
                frame.env[key] = self._aug_aggregate(origin, f"self.{attr}", slot_value, stmt.op, rhs, frame, sink)
                return
            value = self._binary(origin, stmt.op, slot_value, rhs, frame, sink)
        else:
            value = rhs
        match value:
            case TensorMethod():
                reject(origin, "a bound array method cannot be stored")
            case SequenceValue() | TensorValue():
                if same(value, slot_value):
                    return  # rebinding the attribute to its own current tree is Python's no-op
                if ctx.loop:
                    reject(
                        origin,
                        f"installing a new aggregate into the state attribute self.{attr} inside a "
                        "data-dependent loop is not supported yet; store its elements instead",
                    )
                self._install(origin, attr, spec, value)
                frame.env[key] = value
            case StaticScalar() | ResidualScalar():
                if not isinstance(spec, ScalarSpec):
                    reject(
                        origin,
                        f"self.{attr} is an aggregate state attribute; a scalar cannot replace it -- "
                        "store its elements instead",
                    )
                frame.env[key] = value
            case Opaque():
                if not isinstance(spec, ScalarSpec):
                    reject(origin, _describe_opaque(value))
                frame.env[key] = value  # judged at use or at the commit, like any captured scalar

    def _install(self, origin: Origin, attr: str, spec: Spec, value: SequenceValue | TensorValue) -> None:
        """A5 clause (1), plus clause (3) over the value model (tensor-only: in-model sequences are immutable)."""
        for allocation in allocations(value):
            if allocation.joined:
                reject(
                    origin,
                    f"cannot install this aggregate into self.{attr}: it was merged across runtime branches, "
                    "so its identity is branch-dependent; install it in each branch arm instead",
                )
            owner = self._state_owners.get(allocation)
            if owner is not None:
                reject(
                    origin,
                    f"cannot install this aggregate into self.{attr}: it backs (or backed) the state attribute "
                    f"self.{owner}; state trees must be disjoint -- store an explicit copy "
                    "(list(...) or np.array(...)) instead",
                )
            if allocation.state is AllocationState.ESCAPED:
                reject(origin, f"cannot install this aggregate into self.{attr}: {blame(allocation)}")
        occurrences = [id(allocation) for allocation in tensor_occurrences(value)]
        if len(occurrences) != len(set(occurrences)):
            reject(
                origin,
                f"cannot install this aggregate into self.{attr}: the same array is reachable through more "
                "than one path within it; build independent elements (e.g. with a comprehension)",
            )
        self._install_conform(origin, attr, spec, value)
        escape(value)
        for allocation in allocations(value):
            self._state_owners.setdefault(allocation, attr)

    def _install_conform(self, origin: Origin, attr: str, spec: Spec, value: Value) -> None:
        """Structure and array family; scalar leaf types are judged at the commit, like any slot value."""
        match spec, value:
            case ScalarSpec(), (StaticScalar() | ResidualScalar() | Opaque()):
                pass
            case SequenceSpec(items=items), SequenceValue(items=value_items) if len(items) == len(value_items):
                for item_spec, item in zip(items, value_items, strict=True):
                    self._install_conform(origin, attr, item_spec, item)
            case TensorSpec(shape=shape, family=family), TensorValue() if shape == value.shape:
                if value.family is not family:
                    reject(
                        origin,
                        f"cannot install this array into self.{attr}: its element family ({value.family.value}) "
                        f"differs from the reset value's ({family.value}), and the dtype is part of the state "
                        "attribute's identity; build the array from matching elements",
                    )
            case _:
                reject(
                    origin,
                    f"cannot install this value into self.{attr}: its structure does not match the reset "
                    "value's (the kind, length, and shape of every node must agree)",
                )

    def _element_store(
        self,
        stmt: Store | AugStore,
        spelled_root: str,
        root_value: SequenceValue | TensorValue,
        path: tuple[Selector, ...],
        rhs: Value,
        frame: _Frame,
        sink: list[Stmt],
    ) -> SequenceValue | TensorValue:
        origin = stmt.origin
        site = self._store_site(origin, spelled_root, root_value, path, frame, sink)
        for allocation in site.chain:
            if not mutable(allocation):
                owner = self._state_owners.get(allocation)
                if owner is not None and allocation.state is AllocationState.ESCAPED:
                    reject(
                        origin,
                        f"cannot store into {site.spelled}: it was installed into the state attribute "
                        f"self.{owner} this transaction; construct the aggregate fully, then install it once",
                    )
                advice = (
                    "; if the sharing comes from an extracting read, the augmented (+=) and multi-index "
                    "(m[i, j]) spellings avoid it"
                    if spelled_root.startswith("self.") and allocation.state is AllocationState.SHARED
                    else ""
                )
                reject(origin, f"cannot store into {site.spelled}: {blame(allocation)}{advice}")
        old = site.slot_value
        if isinstance(stmt, AugStore):
            if isinstance(old, Opaque):
                reject(origin, _describe_opaque(old))
            value = self._binary(origin, stmt.op, old, rhs, frame, sink)
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
        return rebuilt

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
                    reject(origin, "an attribute cannot be stored on a sequence or array")
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
        frame: _Frame,
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
                computed = self._elementwise(origin, op, current, rhs, frame, sink)
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
                if copies:
                    return TensorValue(value.shape, value.family, value.leaves, Allocation())
                # The non-copying conversion reuses the source allocation as a storage-equivalence token,
                # so the state-install disjointness checks see the derivation for free.
                share(value)
                return TensorValue(value.shape, value.family, value.leaves, value.allocation)
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
        frame: _Frame,
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
        callee = rechained(eel, site.frames)
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
        inner = _Frame(fn=fn, annotations=annotations, env=env, root=False, slots=frame.slots)
        self._inlining.add(fn)
        try:
            returned = self._block(callee.body, inner, sink, _Ctx())
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
                bindings[name] = self._snapshot.admit(name, value, site)  # an injected default: a plain Python object
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

    def _pow(self, origin: Origin, base: Scalar, exponent: Scalar, frame: _Frame, sink: list[Stmt]) -> Scalar:
        if ScalarType.BOOL in (base.stype, exponent.stype):
            reject(origin, "booleans take no part in arithmetic; cast explicitly with int(...) or float(...)")
        if isinstance(exponent, StaticScalar) and exponent.stype is ScalarType.INT:
            n = _ops.const_value(exponent.const)
            assert isinstance(n, int)
            if isinstance(base, StaticScalar):
                return self._fold_pow(origin, base, n)
            return self._pow_chain(origin, self._as_float(base, origin, sink), n, sink)
        if isinstance(base, StaticScalar) and isinstance(exponent, StaticScalar):
            # The host as the raise oracle only: the folded VALUE still comes from the stub body, so the
            # answer cannot depend on binding time. A complex host result is no raise; the stub's log2
            # domain owns that refusal.
            b = _ops.const_value(base.const)
            e = _ops.const_value(exponent.const)
            assert isinstance(b, (int, float)) and isinstance(e, (int, float))
            try:
                b**e
            except (OverflowError, ZeroDivisionError) as error:
                reject(origin, f"this power always raises on the host: {type(error).__name__}: {error}")
        promoted: list[Value] = [self._as_float(base, origin, sink), self._as_float(exponent, origin, sink)]
        value = self._inline(origin, "pow", self._pow_stub, promoted, {}, frame, sink, positional_only=True)
        return self._scalar(value, origin)

    def _fold_pow(self, origin: Origin, base: StaticScalar, n: int) -> StaticScalar:
        """
        Exact host arithmetic; the result stays an int only for a nonnegative int base — every power the
        compiler cannot fold is float, so a negative base folds to its float image.
        """
        value = _ops.const_value(base.const)
        assert isinstance(value, (int, float))
        try:
            folded = value**n
        except (OverflowError, ZeroDivisionError) as error:
            reject(origin, f"this power always raises on the host: {type(error).__name__}: {error}")
        assert type(folded) in (int, float), "an integer exponent cannot yield a complex power"
        if isinstance(folded, int) and value < 0:
            try:
                folded = float(folded)
            except OverflowError:
                reject(origin, "the folded power of a negative base does not fit a binary64 float")
        return StaticScalar(_ops.make_const(folded))

    def _pow_chain(self, origin: Origin, base: Scalar, n: int, sink: list[Stmt]) -> Scalar:
        assert base.stype is ScalarType.FLOAT
        if n < 0:
            chain = self._chain(origin, base, -n, sink)
            one = StaticScalar(_ops.make_const(1.0))
            return self._apply(_ops.FLOAT_DIV, [one, chain], origin, sink)
        if n == 0:
            return StaticScalar(_ops.make_const(1.0))
        return self._chain(origin, base, n, sink)

    def _chain(self, origin: Origin, base: Scalar, n: int, sink: list[Stmt]) -> Scalar:
        assert n >= 1
        result = base
        for _ in range(n - 1):
            self._budget.spend(1, origin, "the power chain")
            result = self._apply(_ops.FLOAT_MUL, [result, base], origin, sink)
        return result

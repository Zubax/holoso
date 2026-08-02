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

Every other exit is a PENDING LANE, not a terminator: a break, continue, or callee return holds its branch
open while the surviving lane's continuation nests into the other arm, and the exit's one consumer -- the
enclosing loop driver for break/continue, the frame boundary for a callee return -- joins the lanes under
the one join rule and seals the region flat. A residual loop consumes its own break and continue lanes into
bare terminators over join temps; a callee return crossing the callee's own residual loop cannot rejoin its
siblings as arms, so the whole callee region wraps into a frame whose sites converge at the frame exit, and
until then that one loop stays open for the boundary fold to write into. Root returns discharge inside
residual loop bodies too (the site's edge leaves before the latch); a body whose every path leaves the loop
has no back edge, cannot iterate, and rejects.
"""

import dataclasses
import enum
import inspect
import logging
import types
import typing
from dataclasses import dataclass

from ..._errors import SynthesisError
from .._desugar import desugar
from .._ir import *
from .._names import indexed_names, public_slot, state_port_name
from . import _aggregate, _express, _mutate, _ops
from ._ownership import allocations, borrow, escape, release, share
from ._reject import reattribute, reject
from ._residual import assigned_names, drop_return_rows, prune, rechained
from ._snapshot import Snapshotter, describe_opaque as _describe_opaque
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
    TensorValue,
    Value,
    allocations_match,
    same,
    same_structure,
    tree_leaves,
    tree_rebuild,
)

_logger = logging.getLogger(__name__)

type _SlotKey = tuple[str, ...]

type _EnvKey = str | int | _SlotKey

_AGGREGATE = (SequenceValue, TensorValue)

_RETURN_KEY = "<return>"


@dataclass(frozen=True, slots=True)
class Ctx:
    """
    Where interpretation stands: a residual branch arm rejects raise; a residual loop also rejects aggregate
    state installs. ``inline`` starts a fresh context, so the flags describe the frame's own position, never
    the caller's.
    """

    branch: bool = False
    loop: bool = False

    def arm(self) -> Ctx:
        return Ctx(branch=True, loop=self.loop)


type Sink = list[Stmt | _OpenIf | _OpenWhile]


@dataclass(slots=True)
class _OpenIf:
    """
    Arms still receiving statements while an exit lane is pending; frozen into the immutable ``If`` at
    publish, so the printer, pruner, and emitter never see one.
    """

    origin: Origin
    cond: Atom
    then: Sink
    orelse: Sink


@dataclass(slots=True)
class _OpenWhile:
    """
    A residual loop published with a return lane still pending inside its body; the frame boundary folds
    the lane, terminates its arms, and freezes the loop with the rest of the callee region.
    """

    origin: Origin
    phis: tuple[LoopPhi, ...]
    header: Sink
    cond: Atom
    body: Sink


def _frozen(sink: Sink) -> tuple[Stmt, ...]:
    out: list[Stmt] = []
    for item in sink:
        match item:
            case _OpenIf():
                out.append(If(item.origin, item.cond, _frozen(item.then), _frozen(item.orelse)))
            case _OpenWhile():
                out.append(ResidualWhile(item.origin, item.phis, _frozen(item.header), item.cond, _frozen(item.body)))
            case _:
                out.append(item)
    return tuple(out)


class _ExitKind(enum.Enum):
    BREAK = "break"
    CONTINUE = "continue"
    RETURN = "return"


@dataclass(slots=True)
class _Exit:
    """
    A pending exit lane, joined at the one consumer its kind names. ``env`` is a SNAPSHOT: a statement-level
    exit ends a lane whose frame dict a sibling lane's fold may later revive and keep mutating in place, so
    capturing by reference would hand the consumer the survivor's values. ``crossed`` marks a return lane
    that left a residual loop: its arms lie inside the published loop and never reconverge with any seal
    region, so folds keep it on the union of sinks until the frame boundary wraps the region.
    """

    kind: _ExitKind
    origin: Origin
    env: _Env
    sinks: list[Sink]
    result: Value | None = None
    crossed: bool = False


@dataclass(slots=True)
class _Flow:
    """
    Exit lanes stay in creation (source) order -- fold order is part of the residual text. The fall lane's
    environment is the frame's, in place; every later statement must reach ALL of its open sinks.
    """

    exits: list[_Exit]
    fall: list[Sink] | None


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


type _Env = dict[_EnvKey, Value | _Unjoinable | _Moved | _SlotAlias]


@dataclass(slots=True)
class Frame:
    """
    One function under interpretation; a non-root ``return`` rides a pending exit lane to the frame boundary.
    ``slots`` is the environment holding the slot keys: the frame's own env for the root and its branch-arm
    forks, the caller's transitively for inlined helpers, so a helper reads the current post-write state.
    """

    fn: types.FunctionType
    annotations: dict[str, object]
    env: _Env
    root: bool
    slots: _Env


_MISSING = object()


def partial_evaluate(eel: EelFunction, fn: types.FunctionType, instance: object | None = None) -> EelFunction:
    """
    The A2 trim driver: run with the seeded assumed-state set, shrink to the attributes whose write was
    actually reached, promote an INT slot that met a FLOAT live-out, and re-run until stable -- each run a
    fresh interpreter over the same immutable tree. A rejection under the conservative assumption is final,
    annotated with the pinning writes.
    """
    if instance is None:
        return Interpreter(fn, None, None).run(eel)
    assert eel.params, "a bound method always has a receiver parameter"
    seed = assigned_names(eel.body, eel.params[0].name)[2]
    if not seed:
        return Interpreter(fn, instance, None).run(eel)
    descriptor_guard(instance, seed)
    model = StateModel(instance, seed, environment_aggregates(fn))
    while True:  # terminates: each pass either finishes, strictly shrinks S, or promotes a fresh leaf
        interpreter = Interpreter(fn, instance, model)
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
                return Interpreter(fn, instance, None).run(eel)
            continue
        return residual


def _key_order(key: _EnvKey) -> tuple[int, str]:
    return (0 if isinstance(key, int) else 1 if isinstance(key, str) else 2), str(key)


class Interpreter:
    def __init__(self, fn: types.FunctionType, instance: object | None, state: StateModel | None) -> None:
        self._fn = fn
        self.instance = instance
        self.state = state
        self.budget = ExpansionBudget()
        self._next_temp = 0
        self._outputs: tuple[OutputDecl, ...] | None = None
        self._elide: list[SlotPath | None] = []
        self.specs: dict[str, Spec] = {}
        self._decls: tuple[SlotDecl, ...] = ()
        self._slot_keys: list[_SlotKey] = []
        self.receiver_name: str | None = None
        self.state_owners: dict[Allocation, str] = {}
        self.written_attrs: set[str] = set()
        self._eels: dict[types.FunctionType, EelFunction] = {}
        self._meta: dict[types.FunctionType, dict[str, object]] = {}
        self._inlining: set[types.FunctionType] = {fn}
        self.snapshot = Snapshotter(state.check_capture if state is not None else None)

    def run(self, eel: EelFunction) -> EelFunction:
        env: _Env = {}
        frame = Frame(
            fn=self._fn, annotations=self._annotations_of(eel.origin, self._fn), env=env, root=True, slots=env
        )
        remaining = eel.params
        if self.instance is not None:
            # The bound receiver is a snapshot root: frozen-attribute reads fold, state attributes read and
            # write their slot bindings, and calling its methods inlines them over the same receiver.
            assert remaining, "a bound method always has a receiver parameter"
            receiver = remaining[0]
            self.receiver_name = receiver.name
            frame.env[receiver.name] = self.snapshot.admit(receiver.name, self.instance, eel.origin)
            remaining = remaining[1:]
        params: list[Param] = []
        for param in remaining:
            params.extend(self._param(param, frame))
        names = [param.name for param in params]
        if len(set(names)) != len(names):
            collision = next(name for name in names if names.count(name) > 1)
            reject(eel.origin, f"the decomposed parameter names collide on {collision!r}; rename the parameters")
        sink: Sink = []
        if self.state is not None:
            self.specs = self.state.prepare()
            self._decls = self.state.decls()
            self._slot_keys = [(attr,) for attr in sorted(self.specs)]
            for attr in sorted(self.specs):
                frame.env[(attr,)] = self._init_slot(attr, self.specs[attr], eel.origin, sink)
            ports = names + [
                state_port_name(decl.slot) for decl in self._decls if not str(decl.slot[0]).startswith("_")
            ]
            if len(set(ports)) != len(ports):
                collision = next(name for name in ports if ports.count(name) > 1)
                reject(eel.origin, f"a state port collides with a parameter on {collision!r}; rename one of them")
        flow = self._block(eel.body, frame, [sink], Ctx())
        assert not flow.exits, "no exit lane escapes the root frame"
        if flow.fall is not None:
            piece = self._piece(flow.fall)
            self._bare_site(eel.origin, frame, piece)
            self._spread(flow.fall, piece)
        assert self._outputs is not None, "every path commits a return site"
        keep = [candidate is None for candidate in self._elide]
        outputs = tuple(row for row, kept in zip(self._outputs, keep, strict=True) if kept)
        frozen = _frozen(sink)
        statements = frozen if all(keep) else drop_return_rows(frozen, keep)
        body, live = prune(statements)
        assert not live, "a residual temp is read before any assignment"
        residual = EelFunction(eel.origin, eel.name, tuple(params), tuple(body), slots=self._decls, outputs=outputs)
        _logger.info(
            "%s: partial evaluation: %d residual statement(s), %d output(s), %d slot(s), %d budget unit(s) spent",
            eel.name,
            len(body),
            len(outputs),
            len(self._decls),
            self.budget.spent,
        )
        return residual

    def _init_slot(self, attr: str, spec: Spec, origin: Origin, sink: Sink) -> Value:
        match spec:
            case ScalarSpec(path=path, stype=stype):
                index = self.fresh()
                sink.append(Assign(origin, TempBind(origin, index), SlotRead(origin, path), stype))
                return ResidualScalar(stype, TempRef(origin, index))
            case SequenceSpec(items=items):
                sequence = SequenceValue(
                    tuple(self._init_slot(attr, item, origin, sink) for item in items), Allocation()
                )
                self.state_owners[sequence.allocation] = attr
                return sequence
            case TensorSpec(shape=shape, family=family, leaves=leaves):
                scalars: list[Scalar | Opaque] = []
                for leaf in leaves:
                    index = self.fresh()
                    sink.append(Assign(origin, TempBind(origin, index), SlotRead(origin, leaf.path), leaf.stype))
                    scalars.append(ResidualScalar(leaf.stype, TempRef(origin, index)))
                tensor = TensorValue(shape, family, tuple(scalars), Allocation())
                self.state_owners[tensor.allocation] = attr
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

    def _param(self, param: Param, frame: Frame) -> list[Param]:
        annotation = frame.annotations.get(param.name, _MISSING)
        if annotation is _MISSING:
            reject(param.origin, f"the parameter {param.name!r} requires a type annotation")
        stype = _express.annotation_stype(annotation)
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

    def fresh(self) -> int:
        index = self._next_temp
        self._next_temp += 1
        return index

    # ------------------------------------------------------------------ statements

    def _block(self, stmts: tuple[Stmt, ...], frame: Frame, sinks: list[Sink], ctx: Ctx) -> _Flow:
        exits: list[_Exit] = []
        current: list[Sink] | None = sinks
        for stmt in stmts:
            if current is None:
                break  # CPython-unreachable suffix: every lane already exited
            piece = self._piece(current)
            match stmt:
                case Assign(target=target, value=value):
                    self._assign(target, value, frame, piece)
                case AugAssign(origin=origin, target=target, op=op, value=value):
                    bound = frame.env.get(target.name)
                    if bound is None:
                        reject(origin, f"the local name {target.name!r} is not bound on every path reaching this read")
                    current_value = self.readable(bound, origin)
                    rhs = self.expr(value, frame, piece)
                    if isinstance(current_value, _AGGREGATE):
                        updated = _mutate.aug_aggregate(self, origin, target.name, current_value, op, rhs, frame, piece)
                    else:
                        updated = _express.binary(self, origin, op, current_value, rhs, frame, piece)
                    frame.env[target.name] = updated
                case Unpack(origin=origin, targets=targets, value=value):
                    source = self.expr(value, frame, piece)
                    items = _aggregate.unpack_items(origin, source, len(targets))
                    for target, item in zip(targets, items, strict=True):
                        frame.env[self._binding_key(target)] = item
                case If():
                    flow = self._if(stmt, frame, current, ctx)
                    exits.extend(flow.exits)
                    current = flow.fall
                    continue
                case Return(origin=origin):
                    if frame.root:
                        self._return_site(stmt, frame, piece)
                        self._spread(current, piece)
                        current = None
                        continue
                    result: Value | None = None
                    if stmt.value is not None:
                        result = self.expr(stmt.value, frame, piece)
                        annotation = frame.annotations.get("return", _MISSING)
                        if annotation is not _MISSING:
                            # At the return itself, so a mismatch points into the callee.
                            result = self._conform_value(result, annotation, origin, piece, "the returned value")
                        if isinstance(result, _AGGREGATE) and self.alias_conduit(frame, stmt.value):
                            share(result)
                    self._spread(current, piece)
                    exits.append(_Exit(_ExitKind.RETURN, origin, dict(frame.env), current, result))
                    current = None
                    continue
                case Raise():
                    self._raise(stmt, frame, piece, ctx)
                case While():
                    flow = self._while(stmt, frame, current, ctx)
                    exits.extend(flow.exits)
                    current = flow.fall
                    continue
                case For():
                    flow = self._for(stmt, frame, current, ctx)
                    exits.extend(flow.exits)
                    current = flow.fall
                    continue
                case Break(origin=origin):
                    exits.append(_Exit(_ExitKind.BREAK, origin, dict(frame.env), current))
                    current = None
                    continue
                case Continue(origin=origin):
                    exits.append(_Exit(_ExitKind.CONTINUE, origin, dict(frame.env), current))
                    current = None
                    continue
                case Store() | AugStore():
                    _mutate.store(self, stmt, frame, piece, ctx)
                case _:
                    raise AssertionError(stmt)
            self._spread(current, piece)
        return _Flow(exits, current)

    def _piece(self, sinks: list[Sink]) -> Sink:
        return sinks[0] if len(sinks) == 1 else []

    def _spread(self, sinks: list[Sink], piece: Sink) -> None:
        if piece is not sinks[0]:
            for sink in sinks:
                sink.extend(piece)

    def _assign(self, target: Binding, value: Expr, frame: Frame, sink: Sink) -> None:
        result = self.expr(value, frame, sink)
        key = self._binding_key(target)
        if isinstance(result, _AGGREGATE):
            if isinstance(value, LocalRef) and value.name != key:
                share(result)
            elif (isinstance(value, AttrRead) and self._slot_tree(frame, result)) or self.alias_conduit(frame, value):
                if isinstance(target, TempBind):
                    frame.env[key] = _SlotAlias(result)
                    return
                share(result)
        frame.env[key] = result

    def _slot_tree(self, frame: Frame, value: Value) -> bool:
        return any(frame.slots.get(key) is value for key in self._slot_keys)

    def alias_conduit(self, frame: Frame, atom: Expr) -> bool:
        return isinstance(atom, TempRef) and isinstance(frame.env.get(atom.index), _SlotAlias)

    def _binding_key(self, binding: Binding) -> _EnvKey:
        match binding:
            case LocalBind(name=name):
                return name
            case TempBind(index=index):
                return index

    def _if(self, stmt: If, frame: Frame, sinks: list[Sink], ctx: Ctx) -> _Flow:
        """
        The branch structure itself is every join; a pending exit holds the If open, and the surviving
        lane's continuation then nests into its arm until the exit's consumer joins and seals.
        """
        piece = self._piece(sinks)
        cond = self._condition(stmt.origin, self.expr(stmt.cond, frame, piece))
        self._spread(sinks, piece)
        if isinstance(cond, StaticScalar):
            decided = _ops.const_value(cond.const)
            assert isinstance(decided, bool)
            taken = stmt.then if decided else stmt.orelse
            return self._block(taken, frame, sinks, ctx)
        then_env: _Env = dict(frame.env)
        else_env: _Env = dict(frame.env)
        then_frame = Frame(
            frame.fn, frame.annotations, then_env, frame.root, slots=then_env if frame.root else frame.slots
        )
        else_frame = Frame(
            frame.fn, frame.annotations, else_env, frame.root, slots=else_env if frame.root else frame.slots
        )
        then_sink: Sink = []
        else_sink: Sink = []
        then_flow = self._block(stmt.then, then_frame, [then_sink], ctx.arm())
        else_flow = self._block(stmt.orelse, else_frame, [else_sink], ctx.arm())
        pending = then_flow.exits + else_flow.exits
        joined: _Env | None = None
        if then_flow.fall is not None and else_flow.fall is not None:
            joined = self._join(stmt.origin, then_env, then_flow.fall, else_env, else_flow.fall)
        cond_atom = _express.materialize(cond, stmt.origin)
        if not pending:
            node: Stmt | _OpenIf = If(stmt.origin, cond_atom, _frozen(then_sink), _frozen(else_sink))
        else:
            node = _OpenIf(stmt.origin, cond_atom, then_sink, else_sink)
        for sink in sinks:
            sink.append(node)
        if then_flow.fall is None and else_flow.fall is None:
            if len(pending) > 1 and all(exit.kind is _ExitKind.RETURN for exit in pending):
                arms = [
                    self._fold_returns(stmt.origin, flow.exits, [arm])
                    for flow, arm in ((then_flow, then_sink), (else_flow, else_sink))
                    if flow.exits
                ]
                return _Flow([self._fold_returns(stmt.origin, arms, sinks)], None)
            return _Flow(pending, None)
        frame.env.clear()
        frame.env.update(joined if joined is not None else else_env if then_flow.fall is None else then_env)
        if not pending:
            return _Flow([], sinks)
        falls = [arm_fall for arm_fall in (then_flow.fall, else_flow.fall) if arm_fall is not None]
        return _Flow(pending, [sink for arm_fall in falls for sink in arm_fall])

    def _fold_returns(self, origin: Origin, lanes: list[_Exit], seal: list[Sink]) -> _Exit:
        """
        Intermediate merges stay on the UNION of the constituent sinks -- sealing early would place a later
        materialization flat where an unfolded lane's path could read it undefined; only the final lane,
        which covers every path of the region, continues at ``seal``. A crossed lane never seals: its arms
        do not reconverge with the seal region, so it stays on the union for the frame boundary to wrap.
        """
        folded = self._fold_union(origin, lanes)
        if folded.crossed:
            return folded
        return _Exit(folded.kind, folded.origin, folded.env, seal, folded.result)

    def _fold_union(self, origin: Origin, lanes: list[_Exit]) -> _Exit:
        assert lanes and all(lane.kind is _ExitKind.RETURN for lane in lanes)
        folded = lanes[0]
        for lane in lanes[1:]:
            merged = self._join_results(origin, folded.result, lane.result, folded.sinks, lane.sinks)
            folded = _Exit(
                _ExitKind.RETURN,
                origin,
                folded.env,
                folded.sinks + lane.sinks,
                merged,
                crossed=folded.crossed or lane.crossed,
            )
        return folded

    def _join_results(
        self, origin: Origin, a: Value | None, b: Value | None, then_sinks: list[Sink], else_sinks: list[Sink]
    ) -> Value | None:
        if a is None or b is None:
            if a is not b:
                reject(origin, "one branch returns a value and the other does not")
            return None
        merged = self._merge(origin, _RETURN_KEY, a, b, then_sinks, else_sinks, allocations_match(a, b))
        if merged is None:
            reject(origin, "the branches return values the compiler cannot merge")
        if not allocations_match(a, b) and isinstance(merged, _AGGREGATE):
            share(a)
            share(b)
            share(merged)
        return merged

    def _join(
        self, origin: Origin, then_env: _Env, then_sinks: list[Sink], else_env: _Env, else_sinks: list[Sink]
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
                merged = self._join_aggregates(origin, key, a, b, then_sinks, else_sinks)
                joined[key] = _Moved(merged) if moved and isinstance(merged, _AGGREGATE) else merged
                continue
            if isinstance(a, (StaticScalar, ResidualScalar)) and isinstance(b, (StaticScalar, ResidualScalar)):
                joined[key] = self._join_scalars(origin, self._describe_key(key), a, b, then_sinks, else_sinks)
                continue
            joined[key] = _Unjoinable(self._describe_key(key))
        return joined

    def _join_scalars(
        self, origin: Origin, described: str, a: Scalar, b: Scalar, then_sinks: list[Sink], else_sinks: list[Sink]
    ) -> Scalar:
        if {a.stype, b.stype} == {ScalarType.INT, ScalarType.FLOAT}:
            a = self._float_in(a, origin, then_sinks)
            b = self._float_in(b, origin, else_sinks)
        elif a.stype is not b.stype:
            reject(origin, f"the branches bind {described} to incompatible types ({a.stype.value} vs {b.stype.value})")
        if same(a, b):
            return a
        stype = a.stype
        index = self.fresh()
        for sink in then_sinks:
            sink.append(Assign(origin, TempBind(origin, index), _express.materialize(a, origin), stype))
        for sink in else_sinks:
            sink.append(Assign(origin, TempBind(origin, index), _express.materialize(b, origin), stype))
        return ResidualScalar(stype, TempRef(origin, index))

    def _join_aggregates(
        self, origin: Origin, key: _EnvKey, a: Value, b: Value, then_sinks: list[Sink], else_sinks: list[Sink]
    ) -> Value | _Unjoinable:
        keep = allocations_match(a, b)
        merged = self._merge(origin, key, a, b, then_sinks, else_sinks, keep)
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
        then_sinks: list[Sink],
        else_sinks: list[Sink],
        keep: bool,
    ) -> Value | None:
        match a, b:
            case SequenceValue(), SequenceValue():
                if len(a.items) != len(b.items):
                    return None
                items: list[Value] = []
                for x, y in zip(a.items, b.items, strict=True):
                    item = self._merge(origin, key, x, y, then_sinks, else_sinks, keep)
                    if item is None:
                        return None
                    items.append(item)
                return SequenceValue(tuple(items), a.allocation if keep else self._minted(a, b))
            case TensorValue(), TensorValue():
                if a.shape != b.shape:
                    return None
                leaves: list[Scalar | Opaque] = []
                for x, y in zip(a.leaves, b.leaves, strict=True):
                    leaf = self._merge(origin, key, x, y, then_sinks, else_sinks, keep)
                    if leaf is None:
                        return None
                    assert isinstance(leaf, (StaticScalar, ResidualScalar, Opaque))
                    leaves.append(leaf)
                family = next((leaf.stype for leaf in leaves if not isinstance(leaf, Opaque)), a.family)
                return TensorValue(a.shape, family, tuple(leaves), a.allocation if keep else self._minted(a, b))
            case (StaticScalar() | ResidualScalar()), (StaticScalar() | ResidualScalar()):
                return self._join_scalars(origin, self._describe_key(key), a, b, then_sinks, else_sinks)
            case Opaque(), Opaque():
                return a if same(a, b) else None
            case _:
                return None

    def _minted(self, a: Value, b: Value) -> Allocation:
        """A keep=False merge product is ONE of the two runtime objects; state-ness rides along with it."""
        assert isinstance(a, _AGGREGATE) and isinstance(b, _AGGREGATE)
        minted = Allocation(joined=True)
        owner = self.state_owners.get(a.allocation) or self.state_owners.get(b.allocation)
        if owner is not None:
            self.state_owners[minted] = owner
        return minted

    def _describe_key(self, key: _EnvKey) -> str:
        if key == _RETURN_KEY:
            return "the returned value"
        if isinstance(key, tuple):
            return f"the state attribute self.{key[0]}"
        return f"the local {key!r}" if isinstance(key, str) else "the conditional result"

    def readable(self, binding: Value | _Unjoinable | _Moved | _SlotAlias, origin: Origin) -> Value:
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

    def _raise(self, stmt: Raise, frame: Frame, sink: Sink, ctx: Ctx) -> None:
        if ctx.branch or ctx.loop or (frame.root and self._outputs is not None):
            reject(stmt.origin, "a raise on a data-dependent path (a runtime branch arm) cannot be lowered")
        pieces: list[str] = []
        for part in stmt.parts:
            if isinstance(part, str):
                pieces.append(part)
                continue
            value = self.expr(part, frame, sink)
            if isinstance(value, StaticScalar):
                pieces.append(format(_ops.const_value(value.const)))
            elif isinstance(value, Opaque) and type(value.value) is str:
                pieces.append(value.value)
            else:
                reject(stmt.origin, "the raise message interpolates a value that is not a compile-time constant")
        text = "".join(pieces)
        reject(stmt.origin, f"{stmt.exc_type}: {text}" if text else stmt.exc_type)

    # ------------------------------------------------------------------ loops

    def _for(self, stmt: For, frame: Frame, sinks: list[Sink], ctx: Ctx) -> _Flow:
        piece = self._piece(sinks)
        iterable = self.expr(stmt.iterable, frame, piece)
        self._spread(sinks, piece)
        if not isinstance(iterable, _AGGREGATE):
            reject(stmt.origin, f"{_aggregate.a_kind(iterable)} is not iterable")
        items = _aggregate.splice_items(stmt.origin, iterable)
        held = borrow(iterable)
        breaks: list[_Exit] = []
        escaped: list[_Exit] = []
        current: list[Sink] | None = sinks
        try:
            for item in items:
                if current is None:
                    break
                self.budget.spend(1, stmt.origin, "the unrolled loop")
                frame.env[stmt.target.name] = item
                current = self._trip(stmt.origin, stmt.body, frame, current, ctx, breaks, escaped)
        finally:
            release(held)
        current = self._meet_lanes(stmt.origin, frame, breaks, current, sinks if not escaped else None)
        return _Flow(escaped, current)

    def _trip(
        self,
        origin: Origin,
        body: tuple[Stmt, ...],
        frame: Frame,
        entry: list[Sink],
        ctx: Ctx,
        breaks: list[_Exit],
        escaped: list[_Exit],
    ) -> list[Sink] | None:
        """
        Trip-completion lanes fold BEFORE the next trip expands -- expanding each lane separately would
        nest the remaining trips per lane and go exponential.
        """
        flow = self._block(body, frame, entry, ctx)
        escaped.extend(exit for exit in flow.exits if exit.kind is _ExitKind.RETURN)
        breaks.extend(exit for exit in flow.exits if exit.kind is _ExitKind.BREAK)
        continues = [exit for exit in flow.exits if exit.kind is _ExitKind.CONTINUE]
        return self._meet_lanes(origin, frame, continues, flow.fall, entry if not (breaks or escaped) else None)

    def _meet_lanes(
        self, origin: Origin, frame: Frame, lanes: list[_Exit], fall: list[Sink] | None, seal: list[Sink] | None
    ) -> list[Sink] | None:
        """
        The merged lane continues flat at ``seal`` only when the region has nothing pending any more --
        flat continuation would run on a still-pending path; otherwise it stays on the union of the
        constituent sinks and later statements broadcast.
        """
        if not lanes:
            return fall
        parts = [(lane.env, lane.sinks) for lane in lanes]
        if fall is not None:
            parts.append((frame.env, fall))
        env, sinks = self._meet_envs(origin, parts)
        if frame.env is not env:
            frame.env.clear()
            frame.env.update(env)
        return seal if seal is not None else sinks

    def _meet_envs(self, origin: Origin, parts: list[tuple[_Env, list[Sink]]]) -> tuple[_Env, list[Sink]]:
        """
        Folded pairwise against the GROWING sink union, so each join's materializations reach every lane
        already folded in -- a join written only into the newest pair's sinks would leave the earlier lanes
        reading it undefined.
        """
        env, sinks = parts[0]
        for next_env, next_sinks in parts[1:]:
            env = self._join(origin, env, sinks, next_env, next_sinks)
            sinks = sinks + next_sinks
        return env, sinks

    def _while(self, stmt: While, frame: Frame, sinks: list[Sink], ctx: Ctx) -> _Flow:
        """
        Static tests execute trip by trip, splicing each header evaluation (CPython runs the failing test's
        header too); the first residual test rolls the env back to before its header and residualizes the
        remaining loop. Ownership events fired by the discarded header evaluation persist -- the recorded
        speculative-evaluation conservatism.
        """
        breaks: list[_Exit] = []
        escaped: list[_Exit] = []
        current: list[Sink] | None = sinks
        while current is not None:
            saved = dict(frame.env)
            header_piece: Sink = []
            header_flow = self._block(stmt.header, frame, [header_piece], ctx)
            assert not header_flow.exits and header_flow.fall is not None, "a test expression cannot exit"
            cond = self._condition(stmt.origin, self.expr(stmt.cond, frame, header_piece))
            if isinstance(cond, ResidualScalar):
                frame.env.clear()
                frame.env.update(saved)
                piece = self._piece(current)
                escaped.extend(self._residual_while(stmt, frame, piece))
                self._spread(current, piece)
                break
            self._spread(current, header_piece)
            decided = _ops.const_value(cond.const)
            assert isinstance(decided, bool)
            if not decided:
                break
            self.budget.spend(1, stmt.origin, "the unrolled loop")
            current = self._trip(stmt.origin, stmt.body, frame, current, ctx, breaks, escaped)
        current = self._meet_lanes(stmt.origin, frame, breaks, current, sinks if not escaped else None)
        return _Flow(escaped, current)

    def _condition(self, origin: Origin, value: Value) -> Scalar:
        if isinstance(value, _AGGREGATE):
            reject(origin, "the truthiness of an aggregate is not supported")
        cond = _express.scalar(value, origin)
        if cond.stype is not ScalarType.BOOL:
            reject(origin, "the branch condition must be a bool; Python truthiness is not supported")
        return cond

    def _residual_while(self, stmt: While, frame: Frame, sink: Sink) -> list[_Exit]:
        """
        The carried set is the syntactic assigned-name set of header+body, restricted to names bound at entry
        -- each becomes a scalar phi. One symbolic header+body pass computes the back-edge values; the stype
        assumptions join per the one rule (a FLOAT phi converts an INT back value in place; an INT assumption
        meeting a FLOAT back value promotes and re-runs, monotone and bounded by the carried count).

        The loop consumes its own break and continue lanes. Continue lanes meet the fall lane at the latch,
        so ``phi.back`` is the joined value and back-edge conversions broadcast to every back sink (a direct
        continue never reaches the flat body tail). Break lanes meet the normal exit after promotion settles;
        the normal lane's join temps land at HEADER END, the only region that runs on the normal exit path
        and on no break path (assignments on non-final tests are dead). The post-loop env is that meet -- or,
        with no breaks, the env after the final pass's header: the header runs on the failing test too, so
        its bindings survive the loop, while body-only names drop (a zero-trip loop never binds them).
        """
        origin = stmt.origin
        carried: list[tuple[_EnvKey, list[int]]] = []
        stypes: dict[tuple[_EnvKey, int], ScalarType] = {}
        rebound, stored, loop_attrs = assigned_names((*stmt.header, *stmt.body), self.receiver_name)
        for name in sorted(rebound | stored):
            bound = frame.env.get(name)
            if bound is None:
                continue
            assert not isinstance(bound, (_Moved, _SlotAlias)), "only temps ride the conduits"
            value = self.readable(bound, origin)
            if isinstance(value, (StaticScalar, ResidualScalar)):
                if name in rebound:
                    carried.append((name, [self.fresh()]))
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
            for attr in sorted(set(loop_attrs) & set(self.specs)):
                slot_key: _SlotKey = (attr,)
                entry_tree = self.readable(frame.env[slot_key], origin)
                leaves = tree_leaves(entry_tree)
                for position, leaf in enumerate(leaves):
                    if isinstance(leaf, Opaque):
                        reject(origin, _describe_opaque(leaf))
                    stypes[(slot_key, position)] = leaf.stype
                carried.append((slot_key, [self.fresh() for _ in leaves]))
        while True:
            outputs_before = self._outputs
            elide_before = list(self._elide)
            loop_env: _Env = dict(frame.env)
            loop_frame = Frame(
                frame.fn, frame.annotations, loop_env, frame.root, slots=loop_env if frame.root else frame.slots
            )
            for key, indices in carried:
                phi_leaves: list[Scalar] = [
                    ResidualScalar(stypes[(key, position)], TempRef(origin, index))
                    for position, index in enumerate(indices)
                ]
                if isinstance(key, tuple):
                    entry_tree = self.readable(frame.env[key], origin)
                    loop_frame.env[key] = tree_rebuild(entry_tree, iter(phi_leaves))
                else:
                    (only,) = phi_leaves
                    loop_frame.env[key] = only
            header_sink: Sink = []
            loop_ctx = Ctx(branch=True, loop=True)
            header_flow = self._block(stmt.header, loop_frame, [header_sink], loop_ctx)
            assert not header_flow.exits and header_flow.fall is not None, "a test expression cannot exit"
            cond = self._condition(origin, self.expr(stmt.cond, loop_frame, header_sink))
            assert isinstance(cond, ResidualScalar), "a residual test cannot re-fold under residual carries"
            post_header_env = dict(loop_frame.env)
            body_sink: Sink = []
            body_flow = self._block(stmt.body, loop_frame, [body_sink], loop_ctx)
            returns = [exit for exit in body_flow.exits if exit.kind is _ExitKind.RETURN]
            assert not (frame.root and returns), "a root return discharges at its site"
            continues = [exit for exit in body_flow.exits if exit.kind is _ExitKind.CONTINUE]
            breaks = [exit for exit in body_flow.exits if exit.kind is _ExitKind.BREAK]
            if body_flow.fall is None and not continues:
                how = "leaves the loop" if breaks else "returns"
                reject(
                    origin,
                    f"the body of this data-dependent loop {how} on every path, so the loop cannot iterate; "
                    "write `if` instead",
                )
            latch_sinks = self._meet_lanes(origin, loop_frame, continues, body_flow.fall, None)
            assert latch_sinks is not None, "a body with no back edge was rejected above"
            promoted = False
            backs: dict[tuple[_EnvKey, int], Scalar] = {}
            for key, indices in carried:
                back_bound = loop_frame.env.get(key)
                assert back_bound is not None, "an entry-bound key cannot vanish"
                back_value = self.readable(back_bound, origin)
                back_leaves = self._back_leaves(key, back_value, frame.env, origin)
                assert len(back_leaves) == len(indices)
                for position, back in enumerate(back_leaves):
                    assumed = stypes[(key, position)]
                    if back.stype is assumed:
                        backs[(key, position)] = back
                    elif assumed is ScalarType.FLOAT and back.stype is ScalarType.INT:
                        backs[(key, position)] = self._float_in(back, origin, latch_sinks)
                    elif assumed is ScalarType.INT and back.stype is ScalarType.FLOAT:
                        if isinstance(key, tuple):
                            assert len(indices) == 1 and isinstance(
                                self.specs[key[0]], ScalarSpec
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
            self._outputs = outputs_before
            self._elide = elide_before
        phis: list[LoopPhi] = []
        for key, indices in carried:
            entry_value = self.readable(frame.env[key], origin)
            entry_leaves = tree_leaves(entry_value)
            assert len(entry_leaves) == len(indices)
            assert not any(isinstance(leaf, Opaque) for leaf in entry_leaves), "the carry setup rejected opaques"
            for position, index in enumerate(indices):
                entry_leaf = entry_leaves[position]
                assert isinstance(entry_leaf, (StaticScalar, ResidualScalar))
                entry: Scalar = entry_leaf
                if stypes[(key, position)] is ScalarType.FLOAT and entry.stype is ScalarType.INT:
                    entry = self.as_float(entry, origin, sink)
                assert entry.stype is stypes[(key, position)]
                back_atom = _express.materialize(backs[(key, position)], origin)
                phis.append(
                    LoopPhi(origin, index, stypes[(key, position)], _express.materialize(entry, origin), back_atom)
                )
        _logger.debug(
            "%s: the loop at %s residualizes with %d carried phi(s), %d break(s), %d continue(s): %s",
            frame.fn.__qualname__,
            origin.location,
            len(phis),
            len(breaks),
            len(continues),
            ", ".join(str(key) for key, _ in carried),
        )
        exit_env = post_header_env
        if breaks:
            parts = [(lane.env, lane.sinks) for lane in breaks] + [(post_header_env, [header_sink])]
            exit_env, _ = self._meet_envs(origin, parts)
        for lane in breaks:
            for arm in lane.sinks:
                arm.append(ResidualBreak(lane.origin))
        for lane in continues:
            for arm in lane.sinks:
                arm.append(ResidualContinue(lane.origin))
        cond_atom = _express.materialize(cond, origin)
        if returns:
            for lane in returns:
                lane.crossed = True
            sink.append(_OpenWhile(origin, tuple(phis), header_sink, cond_atom, body_sink))
        else:
            sink.append(ResidualWhile(origin, tuple(phis), _frozen(header_sink), cond_atom, _frozen(body_sink)))
        frame.env.clear()
        frame.env.update(exit_env)
        return returns

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
        entry_tree = self.readable(entry_env[key], origin)
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

    def _comp(self, node: Comp, frame: Frame, sink: Sink) -> SequenceValue:
        iterable = self.expr(node.iterable, frame, sink)
        if not isinstance(iterable, _AGGREGATE):
            reject(node.origin, f"{_aggregate.a_kind(iterable)} is not iterable")
        items = _aggregate.splice_items(node.origin, iterable)
        held = borrow(iterable)
        collected: list[Value] = []
        try:
            for item in items:
                self.budget.spend(1, node.origin, "the comprehension")
                frame.env[node.target] = item
                body_flow = self._block(node.body, frame, [sink], Ctx(branch=True))
                assert not body_flow.exits and body_flow.fall is not None, "a comprehension body holds no exit"
                value = self.expr(node.element, frame, sink)
                if isinstance(value, _AGGREGATE) and (
                    isinstance(node.element, LocalRef) or self.alias_conduit(frame, node.element)
                ):
                    share(value)
                collected.append(value)
        finally:
            release(held)
        return SequenceValue(tuple(collected), Allocation())

    # ------------------------------------------------------------------ the module boundary

    def _return_site(self, stmt: Return, frame: Frame, sink: Sink) -> None:
        annotation = frame.annotations.get("return", _MISSING)
        if annotation is _MISSING:
            reject(stmt.origin, "the return type annotation is required")
        if stmt.value is None:
            if annotation is not None:
                reject(stmt.origin, "the kernel returns no value but its annotation declares one")
            self._commit_site(stmt.origin, frame, sink, [])
            return
        value = self.expr(stmt.value, frame, sink)
        if isinstance(value, Opaque):
            reject(stmt.origin, f"the captured object {value.name!r} cannot be returned")
        conformed = self._conform_value(value, annotation, stmt.origin, sink, "the returned value", root=True)
        match conformed:
            case StaticScalar() | ResidualScalar():
                self._commit_site(stmt.origin, frame, sink, [((0,), conformed)])
            case SequenceValue() | TensorValue():
                for allocation in allocations(conformed):
                    owner = self.state_owners.get(allocation)
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

    def _bare_site(self, origin: Origin, frame: Frame, sink: Sink) -> None:
        annotation = frame.annotations.get("return", _MISSING)
        if annotation is _MISSING:
            reject(origin, "the return type annotation is required")
        if annotation is not None:
            reject(origin, "the kernel can complete without returning a value but its annotation declares one")
        self._commit_site(origin, frame, sink, [])

    def _commit_site(
        self, origin: Origin, frame: Frame, sink: Sink, rows: list[tuple[_aggregate.LeafPath, Scalar]]
    ) -> None:
        """A leaf matching the same public, unpromoted slot's live-out at EVERY site is elided after the run."""
        decls = tuple(OutputDecl(path, leaf.stype) for path, leaf in rows)
        first = self._outputs is None
        if first:
            self._outputs = decls
        elif decls != self._outputs:
            reject(origin, "this return does not match the kernel's other return sites in shape or type")
        slot_values: dict[SlotPath, Scalar] = {}
        for attr in sorted(self.specs):
            tree = self.readable(frame.slots[(attr,)], origin)
            for spec_leaf, leaf in zip(spec_leaves(self.specs[attr]), tree_leaves(tree), strict=True):
                if isinstance(leaf, Opaque):
                    reject(origin, _describe_opaque(leaf))
                committed = self._slot_conform(attr, spec_leaf, leaf, origin, sink)
                slot_values[spec_leaf.path] = committed
                sink.append(SlotWrite(origin, spec_leaf.path, _express.materialize(committed, origin)))
        promoted = self.state.promoted_paths() if self.state is not None else frozenset()
        public = {path: scalar for path, scalar in slot_values.items() if public_slot(path) and path not in promoted}
        ordered = sorted(public)
        if first:
            self._elide = [next((path for path in ordered if same(leaf, public[path])), None) for _, leaf in rows]
        else:
            for position, (_, leaf) in enumerate(rows):
                candidate = self._elide[position]
                if candidate is not None and not same(leaf, public[candidate]):
                    self._elide[position] = None
        sink.append(ResidualReturn(origin, tuple(_express.materialize(leaf, origin) for _, leaf in rows)))

    def _slot_conform(self, attr: str, spec: ScalarSpec, leaf: Scalar, origin: Origin, sink: Sink) -> Scalar:
        if leaf.stype is spec.stype:
            return leaf
        if spec.stype is ScalarType.FLOAT and leaf.stype is ScalarType.INT:
            return self.as_float(leaf, origin, sink)
        if spec.stype is ScalarType.INT and leaf.stype is ScalarType.FLOAT:
            raise _PromoteRun(spec.path)
        reject(
            origin,
            f"the state attribute self.{attr} would change type from {spec.stype.value} to {leaf.stype.value}; "
            "bool state joins only with bool",
        )

    def _conform_value(
        self, value: Value, annotation: object, origin: Origin, sink: Sink, what: str, *, root: bool = False
    ) -> Value:
        """
        The one annotation-conformance rule for both module boundaries: the root interface demands a
        recognized annotation, while an inlined frame skips unrecognized ones (library stubs annotate with
        shapeless types the subset cannot check).
        """
        stype = _express.annotation_stype(annotation)
        if stype is not None:
            return self.conform(value, stype, origin, sink, what)
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

    def conform(self, value: Value, declared: ScalarType, origin: Origin, sink: Sink, what: str) -> Scalar:
        if isinstance(value, Opaque):
            reject(origin, _describe_opaque(value))
        if not isinstance(value, (StaticScalar, ResidualScalar)):
            reject(origin, f"{what} is not a {declared.value} scalar")
        if value.stype is declared:
            return value
        if value.stype is ScalarType.INT and declared is ScalarType.FLOAT:
            return self.as_float(value, origin, sink)
        reject(origin, f"{what} has type {value.stype.value} where the annotation declares {declared.value}")

    # ------------------------------------------------------------------ expressions

    def expr(self, expr: Expr, frame: Frame, sink: Sink) -> Value:
        match expr:
            case Const(value=value):
                return StaticScalar(_ops.make_const(value))
            case TempRef(origin=origin, index=index):
                bound = frame.env.get(index)
                assert bound is not None, "desugar binds every temp before its first read"
                read = self.readable(bound, origin)
                if isinstance(read, _AGGREGATE) and not isinstance(bound, (_Moved, _SlotAlias)):
                    frame.env[index] = _Moved(read)
                return read
            case LocalRef(origin=origin, name=name):
                found = frame.env.get(name)
                if found is None:
                    reject(origin, f"the local name {name!r} is not bound on every path reaching this read")
                return self.readable(found, origin)
            case EnvRead():
                return self._env_read(expr, frame)
            case Unary(origin=origin, op=op, operand=operand):
                return _express.unary(self, origin, op, self.expr(operand, frame, sink), sink)
            case Binary(origin=origin, op=op, left=left, right=right):
                lv = self.expr(left, frame, sink)
                rv = self.expr(right, frame, sink)
                return _express.binary(self, origin, op, lv, rv, frame, sink)
            case Compare(origin=origin, op=op, left=left, right=right):
                lv = self.expr(left, frame, sink)
                rv = self.expr(right, frame, sink)
                return _express.compare(self, origin, op, lv, rv, sink)
            case Call():
                return _express.call(self, expr, frame, sink)
            case TupleExpr(origin=origin, items=items):
                return self._display(origin, items, frame, sink)
            case ListExpr(origin=origin, items=items):
                return self._display(origin, items, frame, sink)
            case AttrRead(origin=origin, base=base, attr=attr):
                return _express.attr_read(self, origin, self.expr(base, frame, sink), attr, frame, sink)
            case IndexRead(origin=origin, base=base, index=index):
                base_value = self.expr(base, frame, sink)
                index_value = self.expr(index, frame, sink)
                return _aggregate.index_read(origin, base_value, index_value)
            case SliceRead(origin=origin, base=base, lo=lo, hi=hi):
                base_value = self.expr(base, frame, sink)
                lo_bound = self._slice_bound(lo, frame, sink)
                hi_bound = self._slice_bound(hi, frame, sink)
                return _aggregate.slice_read(origin, base_value, lo_bound, hi_bound)
            case MultiIndexRead(origin=origin, base=base, axes=axes):
                base_value = self.expr(base, frame, sink)
                resolved: list[_aggregate.ResolvedAxis] = []
                for axis in axes:
                    if isinstance(axis, SliceSel):
                        resolved.append(
                            (self._slice_bound(axis.lo, frame, sink), self._slice_bound(axis.hi, frame, sink))
                        )
                    else:
                        resolved.append(self.expr(axis, frame, sink))
                return _aggregate.multi_index_read(origin, base_value, tuple(resolved))
            case Comp():
                return self._comp(expr, frame, sink)
            case _:
                raise AssertionError(expr)

    def _slice_bound(self, atom: Atom | None, frame: Frame, sink: Sink) -> int | None:
        if atom is None:
            return None
        return _aggregate.static_index(atom.origin, self.expr(atom, frame, sink), "a slice bound")

    def _display(self, origin: Origin, items: tuple[Atom | StarArg, ...], frame: Frame, sink: Sink) -> SequenceValue:
        collected: list[Value] = []
        for item in items:
            if isinstance(item, StarArg):
                source = self.expr(item.value, frame, sink)
                collected.extend(_aggregate.splice_items(origin, source))
            else:
                value = self.expr(item, frame, sink)
                if isinstance(value, _AGGREGATE) and (isinstance(item, LocalRef) or self.alias_conduit(frame, item)):
                    share(value)
                collected.append(value)
        self.budget.spend(max(len(collected), 1), origin, "the sequence display")
        return SequenceValue(tuple(collected), Allocation())

    def _env_read(self, node: EnvRead, frame: Frame) -> Value:
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
        return self.snapshot.admit(node.name, raw, node.origin)

    def instance_attr(self, origin: Origin, base: Opaque, attr: str, frame: Frame, sink: Sink) -> Value:
        """
        CPython's attribute precedence over class metadata, never calling into the object: a state attribute
        of the entry receiver reads its current slot binding, then a data descriptor wins (only a plain
        property is readable -- its getter is inlined as ordinary code), then the instance dict, then
        non-data class attributes.
        """
        if base.value is self.instance and (attr,) in frame.slots:
            bound = frame.slots[(attr,)]
            assert not isinstance(bound, (_Moved, _SlotAlias)), "slot bindings never ride the conduits"
            return self.readable(bound, origin)
        found: object = next(
            (klass.__dict__[attr] for klass in type(base.value).__mro__ if attr in klass.__dict__), _MISSING
        )
        if isinstance(found, property) and isinstance(found.fget, types.FunctionType):
            return self.inline(
                origin, f"{base.name}.{attr}", found.fget, [base], {}, frame, sink, positional_only=False
            )
        if found is not _MISSING and inspect.isdatadescriptor(found):
            reject(origin, f"the attribute {attr!r} is a descriptor the compiler cannot read")
        instance_dict = getattr(base.value, "__dict__", None)
        if isinstance(instance_dict, dict) and attr in instance_dict:
            return self.snapshot.admit(f"{base.name}.{attr}", instance_dict[attr], origin)
        if found is _MISSING:
            reject(origin, f"{base.name!r} has no attribute {attr!r}")
        if isinstance(found, staticmethod):
            return self.snapshot.admit(f"{base.name}.{attr}", found.__func__, origin)
        if isinstance(found, types.FunctionType):
            return self.snapshot.admit(f"{base.name}.{attr}", types.MethodType(found, base.value), origin)
        if isinstance(found, classmethod):
            return self.snapshot.admit(
                f"{base.name}.{attr}", types.MethodType(found.__func__, type(base.value)), origin
            )
        return self.snapshot.admit(f"{base.name}.{attr}", found, origin)

    def as_float(self, scalar: Scalar, origin: Origin, sink: Sink) -> Scalar:
        if scalar.stype is ScalarType.FLOAT:
            return scalar
        assert scalar.stype is ScalarType.INT
        return _express.apply(self, _ops.CONVERT[(ScalarType.INT, ScalarType.FLOAT)], [scalar], origin, sink)

    def _float_in(self, scalar: Scalar, origin: Origin, sinks: list[Sink]) -> Scalar:
        piece = self._piece(sinks)
        result = self.as_float(scalar, origin, piece)
        self._spread(sinks, piece)
        return result

    def inline(
        self,
        origin: Origin,
        display: str,
        fn: types.FunctionType,
        positional: list[Value],
        keywords: dict[str, Value],
        frame: Frame,
        sink: Sink,
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
        self.budget.spend(1, site, "the inlined call")
        callee = rechained(eel, site.frames)
        assert isinstance(callee, EelFunction)
        if positional_only:
            assert not keywords
            if len(positional) != len(callee.params):
                reject(site, f"{display}() takes {len(callee.params)} argument(s), got {len(positional)}")
            bindings = {param.name: value for param, value in zip(callee.params, positional, strict=True)}
        else:
            bindings = _express.bind_signature(self, site, fn, callee.params, positional, keywords)
        env: _Env = {}
        for param in callee.params:
            value = bindings[param.name]
            declared = annotations.get(param.name, _MISSING)
            if declared is not _MISSING:
                value = self._conform_value(value, declared, site, sink, f"the argument {param.name!r}")
            env[param.name] = value
        inner = Frame(fn=fn, annotations=annotations, env=env, root=False, slots=frame.slots)
        start = len(sink)
        self._inlining.add(fn)
        try:
            flow = self._block(callee.body, inner, [sink], Ctx())
        finally:
            self._inlining.discard(fn)
        assert all(exit.kind is _ExitKind.RETURN for exit in flow.exits), "loop exits cannot escape a frame"
        if flow.fall is not None and flow.exits:
            reject(site, "the call can complete without returning a value, so it cannot be used in an expression")
        if flow.fall is not None or not flow.exits:
            reject(site, "the call returns no value, so it cannot be used in an expression")
        folded = self._fold_returns(site, flow.exits, [sink])
        if folded.result is None:
            reject(site, "the call returns no value, so it cannot be used in an expression")
        if folded.crossed:
            return self._wrap_frame(site, folded, sink, start)
        return folded.result

    def _wrap_frame(self, site: Origin, folded: _Exit, sink: Sink, start: int) -> Value:
        """
        A return that crossed the callee's own residual loop cannot rejoin its siblings as arms, so the
        callee region converges at a frame exit instead: every lane's arm ends with one terminator carrying
        the residual result leaves, and the caller reads them back through the frame rows. Statically
        uniform leaves never leave the value model, and the rebuilt tree keeps the fold's allocations.
        """
        assert folded.result is not None
        assert len({id(arm) for arm in folded.sinks}) == len(folded.sinks), "the union holds each lane region once"
        rows: list[FrameRow] = []
        atoms: list[Atom] = []
        leaves: list[Scalar | Opaque] = []
        for leaf in tree_leaves(folded.result):
            if isinstance(leaf, (StaticScalar, Opaque)):
                leaves.append(leaf)
                continue
            index = self.fresh()
            rows.append(FrameRow(site, index, leaf.stype))
            atoms.append(_express.materialize(leaf, site))
            leaves.append(ResidualScalar(leaf.stype, TempRef(site, index)))
        terminator = ResidualFrameReturn(site, tuple(atoms))
        for arm in folded.sinks:
            arm.append(terminator)
        body = _frozen(sink[start:])
        del sink[start:]
        sink.append(ResidualFrame(site, tuple(rows), body))
        return tree_rebuild(folded.result, iter(leaves))

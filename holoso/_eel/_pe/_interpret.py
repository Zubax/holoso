"""
The specializing interpreter: Eel -> residual Eel. Binding time is its whole product; the rest follows
CPython or the subset's own rules, of which the non-obvious ones are recorded here.

Static folds go through the selected HIR operator's own `evaluate`, and a fold naming no number
residualizes with constant operands: one expression, one answer, refusal survivor-based in HIR. No algebra
over residual operands -- that is HIR strength reduction's. Laziness is CPython's: a dead arm resolves no
names and desugars no callees; captures are judged at use, never at capture.

A name bound on only one side of a residual branch drops (CPython would raise UnboundLocalError); divergent
unmergeable values stay bound to a marker so the read, in any spelling, rejects truthfully. `raise` is
judged within its own function: unconditional there is a compile-time diagnostic even under a caller's
residual arm. Library stubs bind positionally only. An operator whose lowering the registry holds -- `**` and
`@` -- reaches it exactly as a spelled call does, so the two cannot drift; which lowering serves is the
registry's per-position domain rule, not a decision taken here, and the host is never consulted for a value.

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

from ..._errors import HolosoError, SynthesisError
from .._annotations import accepted_stypes, annotation_stype, unaliased
from .._desugar import desugar
from .._ir import *
from .._names import indexed_names, public_slot, state_port_name
from . import _aggregate, _express, _mutate, _ops
from ._ownership import allocations, borrow, escape, release, share
from ._reject import reattribute, reject
from ._residual import assigned_names, drop_return_rows, prune, rechained
from ._snapshot import Snapshotter, describe_opaque as _describe_opaque
from ._state import (
    READ_PROTOCOLS,
    ComponentTree,
    ScalarSpec,
    SequenceSpec,
    Spec,
    StateKey,
    StateModel,
    TensorSpec,
    build_component_tree,
    environment_aggregates,
    mro_attr,
    overridden_protocol,
    resolve_chain,
    seed_state,
    spec_leaves,
    spell_state,
)
from ._values import (
    Allocation,
    AllocationState,
    ExpansionBudget,
    Opaque,
    RangeValue,
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

type _EnvKey = str | int | StateKey

type _LoopKey = tuple[Origin, int]

_AGGREGATE = (SequenceValue, TensorValue)

_RETURN_KEY = "<return>"


@dataclass(frozen=True, slots=True)
class Ctx:
    """
    Where interpretation stands: a residual branch arm rejects raise; a residual loop also rejects aggregate
    state installs. `inline` starts a fresh context, so the flags describe the frame's own position, never
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
    Arms still receiving statements while an exit lane is pending; frozen into the immutable `If` at
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


def _unaliased(annotation: object, origin: Origin, what: str) -> object:
    try:
        return unaliased(annotation)
    except Exception as error:
        reject(origin, f"{what}: the type alias cannot be evaluated: {error}")


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
    A pending exit lane, joined at the one consumer its kind names. `env` and `state` are SNAPSHOTS: a
    statement-level exit ends a lane whose frame dicts a sibling lane's fold may later revive and keep
    mutating in place, so capturing by reference would hand the consumer the survivor's values. `crossed`
    marks a return lane that left a residual loop: its arms lie inside the published loop and never
    reconverge with any seal region, so folds keep it on the union of sinks until the frame boundary wraps
    the region.
    """

    kind: _ExitKind
    origin: Origin
    env: _Env
    sinks: list[Sink]
    result: Value | None = None
    crossed: bool = False

    @classmethod
    def snap(
        cls, kind: _ExitKind, origin: Origin, frame: "Frame", sinks: list[Sink], result: "Value | None" = None
    ) -> "_Exit":
        return cls(kind, origin, dict(frame.env), sinks, result)


@dataclass(slots=True)
class _Flow:
    """
    Exit lanes stay in creation (source) order -- fold order is part of the residual text. The fall lane's
    environment is the frame's, in place; every later statement must reach ALL of its open sinks.
    """

    exits: list[_Exit]
    fall: list[Sink] | None


class _PromoteRun(HolosoError):
    """Internal driver signal: an INT slot leaf met a FLOAT live-out; re-run with that leaf promoted."""

    def __init__(self, path: SlotPath) -> None:
        super().__init__(path)
        self.path = path


class _CarryGrow(HolosoError):
    """
    Internal driver signal: a state store was reached inside a residual loop pass whose carried set lacks
    the key, paired with every such enclosing loop. Lean-first is safe where maximal-first is not, since
    residualization only grows with the carried set.
    """

    def __init__(self, pairs: list[tuple[_LoopKey, StateKey]]) -> None:
        super().__init__(pairs)
        self.pairs = pairs


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
    A temp holding an aggregate state read: the tree's persistent home is the state environment, so unlike
    an ordinary linear conduit the landing site must fire the aliasing share; reads are innocent like a
    local's. Consuming it after an element mutation of the root's tree rejects, since the leaves it holds
    are the ones CPython's live handle no longer shows.
    """

    value: SequenceValue | TensorValue
    root: StateKey
    epoch: int


type _Env = dict[_EnvKey, Value | _Unjoinable | _Moved | _SlotAlias]


@dataclass(slots=True)
class Frame:
    """
    One function under interpretation; a non-root `return` rides a pending exit lane to the frame boundary.
    The environment holds locals and temps beside the persistent-state bindings, keyed by state root path --
    a key family nothing else mints; an inlined helper's fresh environment inherits the caller's state
    entries, and the return fold hands the joined state back.
    """

    fn: types.FunctionType
    annotations: dict[str, object]
    env: _Env
    root: bool
    marks: dict[int, int] = dataclasses.field(default_factory=dict)

    def fork(self) -> "Frame":
        """Marks are shared, not copied: an AugMark/AugStore pair is suite-local to one function."""
        return Frame(self.fn, self.annotations, dict(self.env), self.root, marks=self.marks)


_MISSING = object()


def _rewrap(
    value: SequenceValue | TensorValue,
    moved: bool,
    a_alias: _SlotAlias | None,
    b_alias: _SlotAlias | None,
    shares: tuple[Value, ...],
) -> Value | _Moved | _SlotAlias:
    """
    A same-root alias pair keeps the older (min) epoch, the conservative one; a mismatched alias provenance
    shares instead, so a later mutation through the join refuses.
    """
    if moved:
        return _Moved(value)
    if a_alias is not None and b_alias is not None and a_alias.root == b_alias.root:
        return _SlotAlias(value, a_alias.root, min(a_alias.epoch, b_alias.epoch))
    if a_alias is not None or b_alias is not None:
        for tree in shares:
            share(tree)
    return value


def _state_view(env: _Env) -> _Env:
    """The persistent-state projection of an environment: what survives a return fold into the caller."""
    return {key: value for key, value in env.items() if isinstance(key, tuple)}


def partial_evaluate(
    eel: EelFunction, fn: types.FunctionType, instance: object | None, unroll_max_trips: int
) -> EelFunction:
    """
    The A2 trim driver: run with the seeded assumed-state set (the syntactic may-write scan over the whole
    component tree), shrink it to the paths whose write was actually reached, and re-run on every other
    repair a run signals, until stable -- each run a fresh interpreter over the same immutable tree. A
    rejection under the conservative assumption is final, annotated with the pinning writes.
    """
    if instance is None:
        return Interpreter(fn, None, unroll_max_trips).run(eel)
    assert eel.params, "a bound method always has a receiver parameter"
    tree = build_component_tree(instance)
    model = StateModel(tree, seed_state(tree, eel), environment_aggregates(fn))
    extra_carries: dict[_LoopKey, frozenset[StateKey]] = {}
    while True:  # terminates: each pass finishes, strictly shrinks S, promotes a leaf, or grows a carry set
        interpreter = Interpreter(fn, model, unroll_max_trips, extra_carries)
        try:
            residual = interpreter.run(eel)
        except _PromoteRun as promotion:
            model.promote(promotion.path)
            _logger.info("%s: the state leaf %s promotes to float; re-running", eel.name, promotion.path)
            continue
        except _CarryGrow as grow:
            for loop_key, state_key in grow.pairs:
                extra_carries[loop_key] = extra_carries.get(loop_key, frozenset()) | {state_key}
            _logger.info("%s: a residual loop's carried set grows by %s; re-running", eel.name, grow.pairs)
            continue
        except SynthesisError as error:
            raise type(error)(error.message + model.note(), error.location) from None
        if model.trim(interpreter.written_attrs):
            _logger.info(
                "%s: the assumed-state set shrinks to {%s}; re-running",
                eel.name,
                ", ".join(str(key) for key in sorted(model.assumed)),
            )
            continue
        return residual


def _key_order(key: _EnvKey) -> tuple[int, str]:
    return (0 if isinstance(key, int) else 1 if isinstance(key, str) else 2), str(key)


class Interpreter:
    def __init__(
        self,
        fn: types.FunctionType,
        state: StateModel | None,
        unroll_max_trips: int,
        extra_carries: dict[_LoopKey, frozenset[StateKey]] | None = None,
    ) -> None:
        self._fn = fn
        self.tree = state.tree if state is not None else None
        self.instance = self.tree.objects[()] if self.tree is not None else None
        self.state = state
        self.unroll_max_trips = unroll_max_trips
        self.extra_carries = extra_carries or {}
        self.budget = ExpansionBudget()
        self._next_temp = 0
        self._outputs: tuple[OutputDecl, ...] | None = None
        self._elide: list[SlotPath | None] = []
        self.specs: dict[StateKey, Spec] = {}
        self._decls: tuple[SlotDecl, ...] = ()
        self.state_owners: dict[Allocation, StateKey] = {}
        self.written_attrs: set[StateKey] = set()
        self.state_writes = 0
        self.root_epochs: dict[StateKey, int] = {}
        self.active_loops: list[tuple[_LoopKey, frozenset[StateKey]]] = []
        self._loop_occurrences: dict[Origin, int] = {}
        self._eels: dict[types.FunctionType, EelFunction] = {}
        self._meta: dict[types.FunctionType, dict[str, object]] = {}
        self._inlining: set[types.FunctionType] = {fn}
        self.snapshot = Snapshotter(state.check_capture if state is not None else None)

    def state_prefix(self, raw: object) -> StateKey | None:
        return self.tree.prefix_of(raw) if self.tree is not None else None

    def carry_gate(self, key: StateKey) -> None:
        """A state store inside a residual loop pass must find its key carried at EVERY crossed back edge."""
        missing = [(loop_key, key) for loop_key, carried in self.active_loops if key not in carried]
        if missing:
            raise _CarryGrow(missing)

    def check_mark(self, frame: Frame, mark: int, origin: Origin) -> None:
        recorded = frame.marks.get(mark)
        assert recorded is not None, "desugar pairs every AugStore with a preceding AugMark in the same suite"
        if recorded != self.state_writes:
            reject(
                origin,
                "the right-hand side of this augmented assignment writes persistent state, so the target's "
                "old value may be read after a write CPython reads it before (the check does not track which "
                "slot was written); split into an explicit read and store",
            )

    def note_state_write(self) -> None:
        self.state_writes += 1

    def bump_root_epoch(self, key: StateKey) -> None:
        self.root_epochs[key] = self.root_epochs.get(key, 0) + 1

    def run(self, eel: EelFunction) -> EelFunction:
        env: _Env = {}
        frame = Frame(fn=self._fn, annotations=self._annotations_of(eel.origin, self._fn), env=env, root=True)
        remaining = eel.params
        if self.instance is not None:
            # The bound receiver is a snapshot root: frozen-attribute reads fold, state attributes read and
            # write their slot bindings, and calling its methods inlines them over the same receiver.
            assert remaining, "a bound method always has a receiver parameter"
            receiver = remaining[0]
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
            for key in sorted(self.specs):
                frame.env[key] = self._init_slot(key, self.specs[key], eel.origin, sink)
            ports = names + [state_port_name(decl.slot) for decl in self._decls if public_slot(decl.slot)]
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

    def _init_slot(self, key: StateKey, spec: Spec, origin: Origin, sink: Sink) -> Value:
        match spec:
            case ScalarSpec(path=path, stype=stype):
                index = self.fresh()
                sink.append(Assign(origin, TempBind(origin, index), SlotRead(origin, path), stype))
                return ResidualScalar(stype, TempRef(origin, index))
            case SequenceSpec(items=items):
                sequence = SequenceValue(
                    tuple(self._init_slot(key, item, origin, sink) for item in items), Allocation()
                )
                self.state_owners[sequence.allocation] = key
                return sequence
            case TensorSpec(shape=shape, family=family, leaves=leaves):
                scalars: list[Scalar | Opaque] = []
                for leaf in leaves:
                    index = self.fresh()
                    sink.append(Assign(origin, TempBind(origin, index), SlotRead(origin, leaf.path), leaf.stype))
                    scalars.append(ResidualScalar(leaf.stype, TempRef(origin, index)))
                tensor = TensorValue(shape, family, tuple(scalars), Allocation())
                self.state_owners[tensor.allocation] = key
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
                # eval_str accepts quoted annotations; lazy PEP 649 annotations and PEP 695 alias bodies execute
                # user expressions here, so any exception is the user's, and containing it is the only way to
                # locate it. Unaliasing here means every consumer of these annotations sees the named annotation.
                found = {
                    name: unaliased(annotation)
                    for name, annotation in inspect.get_annotations(fn, eval_str=True).items()
                }
            except Exception as error:
                reject(origin, f"the type annotations cannot be evaluated: {error}")
            self._meta[fn] = found
        return found

    def _param(self, param: Param, frame: Frame) -> list[Param]:
        annotation = frame.annotations.get(param.name, _MISSING)
        if annotation is _MISSING:
            reject(param.origin, f"the parameter {param.name!r} requires a type annotation")
        stype = annotation_stype(annotation)
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
                    source = _aggregate.decay(self.budget, self.expr(value, frame, piece), origin)
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
                    exits.append(_Exit.snap(_ExitKind.RETURN, origin, frame, current, result))
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
                    exits.append(_Exit.snap(_ExitKind.BREAK, origin, frame, current))
                    current = None
                    continue
                case Continue(origin=origin):
                    exits.append(_Exit.snap(_ExitKind.CONTINUE, origin, frame, current))
                    current = None
                    continue
                case AugMark(mark=mark):
                    frame.marks[mark] = self.state_writes
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
            else:
                alias = self._alias_of(frame, value, result)
                if alias is not None:
                    if isinstance(target, TempBind):
                        frame.env[key] = alias
                        return
                    share(result)
        frame.env[key] = result

    def _alias_of(self, frame: Frame, value: Expr, result: SequenceValue | TensorValue) -> _SlotAlias | None:
        if isinstance(value, AttrRead):
            root = self._slot_tree(frame, result)
            if root is not None:
                return _SlotAlias(result, root, self.root_epochs.get(root, 0))
        if isinstance(value, TempRef):
            bound = frame.env.get(value.index)
            if isinstance(bound, _SlotAlias):
                return _SlotAlias(result, bound.root, bound.epoch)
        return None

    def _slot_tree(self, frame: Frame, value: Value) -> StateKey | None:
        return next((key for key in self.specs if frame.env.get(key) is value), None)

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
        then_frame = frame.fork()
        else_frame = frame.fork()
        then_sink: Sink = []
        else_sink: Sink = []
        then_flow = self._block(stmt.then, then_frame, [then_sink], ctx.arm())
        else_flow = self._block(stmt.orelse, else_frame, [else_sink], ctx.arm())
        pending = then_flow.exits + else_flow.exits
        joined: _Env | None = None
        if then_flow.fall is not None and else_flow.fall is not None:
            joined = self._join(stmt.origin, then_frame.env, then_flow.fall, else_frame.env, else_flow.fall)
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
        frame.env.update(joined if joined is not None else else_frame.env if then_flow.fall is None else then_frame.env)
        if not pending:
            return _Flow([], sinks)
        falls = [arm_fall for arm_fall in (then_flow.fall, else_flow.fall) if arm_fall is not None]
        return _Flow(pending, [sink for arm_fall in falls for sink in arm_fall])

    def _fold_returns(self, origin: Origin, lanes: list[_Exit], seal: list[Sink]) -> _Exit:
        """
        Intermediate merges stay on the UNION of the constituent sinks -- sealing early would place a later
        materialization flat where an unfolded lane's path could read it undefined; only the final lane,
        which covers every path of the region, continues at `seal`. A crossed lane never seals: its arms
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
            merged_env = self._join(origin, _state_view(folded.env), folded.sinks, _state_view(lane.env), lane.sinks)
            folded = _Exit(
                _ExitKind.RETURN,
                origin,
                merged_env,
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
            a_alias = a_bound if isinstance(a_bound, _SlotAlias) else None
            b_alias = b_bound if isinstance(b_bound, _SlotAlias) else None
            a = a_bound.value if isinstance(a_bound, (_Moved, _SlotAlias)) else a_bound
            b = b_bound.value if isinstance(b_bound, (_Moved, _SlotAlias)) else b_bound
            if isinstance(a, _Unjoinable) or isinstance(b, _Unjoinable):
                joined[key] = _Unjoinable(self._describe_key(key))
                continue
            if same(a, b):
                if isinstance(a, _AGGREGATE):
                    joined[key] = _rewrap(a, moved, a_alias, b_alias, (a,))
                else:
                    joined[key] = a
                continue
            if isinstance(a, _AGGREGATE) and isinstance(b, _AGGREGATE):
                merged = self._join_aggregates(origin, key, a, b, then_sinks, else_sinks)
                if isinstance(merged, _AGGREGATE):
                    joined[key] = _rewrap(merged, moved, a_alias, b_alias, (a, b, merged))
                else:
                    joined[key] = merged
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
            case RangeValue(), RangeValue():
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
            return f"the state attribute {spell_state(key)}"
        return f"the local {key!r}" if isinstance(key, str) else "the conditional result"

    def readable(self, binding: Value | _Unjoinable | _Moved | _SlotAlias, origin: Origin) -> Value:
        if isinstance(binding, _Unjoinable):
            reject(
                origin,
                f"{binding.description} holds branch values the compiler cannot merge: scalars join freely, and "
                "aggregates join only when every arm agrees in kind, length, and shape",
            )
        if isinstance(binding, _SlotAlias):
            if self.root_epochs.get(binding.root, 0) != binding.epoch:
                reject(
                    origin,
                    f"this handle of the state attribute {spell_state(binding.root)} was taken before a call "
                    "that mutated its elements, so it no longer shows what Python's live handle shows; "
                    "reload the handle after the call",
                )
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
        if isinstance(iterable, RangeValue):
            span = _aggregate.static_range(iterable)
            if span is None or _aggregate.range_length(span) > self.unroll_max_trips:
                crossed = self._counted(stmt, iterable, span, frame, piece)
                self._spread(sinks, piece)
                return _Flow(crossed, sinks)
            iterable = _aggregate.decay(self.budget, iterable, stmt.origin)
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

    def _counted(self, stmt: For, iterable: RangeValue, span: range | None, frame: Frame, sink: Sink) -> list[_Exit]:
        """
        The counted back-edge lowering: the hidden counter/stop/test locals capture the range by value, so
        neither a later rebind of a bound's source local nor a target that shadows one can reach the header.
        A static range here is nonempty, so Python's first trip always kills the target's entry binding and
        the replacement seed is exact.
        """
        origin = stmt.origin
        assert span is None or _aggregate.range_length(span) > self.unroll_max_trips >= 0
        k = self.fresh()
        counter, stop, test = f"for${k}n", f"for${k}s", f"for${k}t"
        assert stmt.target.name not in (counter, stop, test), "the hidden names are unspellable in source"
        frame.env[counter] = iterable.start
        frame.env[stop] = iterable.stop
        if span is not None:
            frame.env[stmt.target.name] = iterable.start
        header = Assign(
            origin,
            LocalBind(origin, test),
            Compare(
                origin,
                CompareOp.LT if iterable.step > 0 else CompareOp.GT,
                LocalRef(origin, counter),
                LocalRef(origin, stop),
            ),
        )
        advance = Assign(
            origin,
            LocalBind(origin, counter),
            Binary(origin, BinaryOp.ADD, LocalRef(origin, counter), Const(origin, iterable.step)),
        )
        bind_target = Assign(origin, stmt.target, LocalRef(origin, counter))
        synth = While(origin, (header,), LocalRef(origin, test), (bind_target, advance, *stmt.body))
        escaped = self._residual_while(synth, frame, sink)
        for name in (counter, stop, test):
            del frame.env[name]
        return escaped

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
        The merged lane continues flat at `seal` only when the region has nothing pending any more --
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
        so `phi.back` is the joined value and back-edge conversions broadcast to every back sink (a direct
        continue never reaches the flat body tail). Break lanes meet the normal exit after promotion settles;
        the normal lane's join temps land at HEADER END, the only region that runs on the normal exit path
        and on no break path (assignments on non-final tests are dead). The post-loop env is that meet -- or,
        with no breaks, the env after the final pass's header: the header runs on the failing test too, so
        its bindings survive the loop, while body-only names drop (a zero-trip loop never binds them).
        """
        origin = stmt.origin
        occurrence = self._loop_occurrences.get(origin, 0)
        self._loop_occurrences[origin] = occurrence + 1
        loop_key: _LoopKey = (origin, occurrence)
        carried: list[tuple[_EnvKey, list[int]]] = []
        stypes: dict[tuple[_EnvKey, int], ScalarType] = {}
        rebound, stored, attr_writes = assigned_names((*stmt.header, *stmt.body))
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
        # Lean-first is deliberate: carrying an untouched leaf residualizes reads that may be static (a
        # constant written before the loop), and a maximal pass can reject before any repair could run.
        carried_keys: set[StateKey] = set(self.extra_carries.get(loop_key, frozenset())) & set(self.specs)
        for root, chain in attr_writes:
            bound = frame.env.get(root)
            if isinstance(bound, Opaque) and self.tree is not None:
                prefix = self.state_prefix(bound.value)
                if prefix is not None:
                    resolved = resolve_chain(self.tree, prefix, chain)
                    if resolved is not None and resolved in self.specs:
                        carried_keys.add(resolved)
        for slot_key in sorted(carried_keys):
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
            budget_before = self.budget.mark()
            loop_frame = frame.fork()
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
            self.active_loops.append((loop_key, frozenset(carried_keys)))
            try:
                header_flow = self._block(stmt.header, loop_frame, [header_sink], loop_ctx)
                assert not header_flow.exits and header_flow.fall is not None, "a test expression cannot exit"
                cond = self._condition(origin, self.expr(stmt.cond, loop_frame, header_sink))
                assert isinstance(cond, ResidualScalar), "a residual test cannot re-fold under residual carries"
                post_header_env = dict(loop_frame.env)
                body_sink: Sink = []
                body_flow = self._block(stmt.body, loop_frame, [body_sink], loop_ctx)
            finally:
                self.active_loops.pop()
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
            for spec_key in self.specs:
                if spec_key not in carried_keys:
                    assert (
                        loop_frame.env[spec_key] is frame.env[spec_key]
                    ), "an uncarried slot changed across a residual loop pass; the store gate missed it"
            promoted = False
            backs: dict[tuple[_EnvKey, int], Scalar] = {}
            for key, indices in carried:
                back_value = self.readable(loop_frame.env[key], origin)
                back_leaves = self._back_leaves(key, back_value, frame, origin)
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
                                self.specs[key], ScalarSpec
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
            # Charging the discarded pass would let the mere presence of a promotion reject a body that fits.
            self._outputs = outputs_before
            self._elide = elide_before
            self.budget.rewind(budget_before)
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
            parts = [(lane.env, lane.sinks) for lane in breaks]
            parts.append((post_header_env, [header_sink]))
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

    def _back_leaves(self, key: _EnvKey, back_value: Value, frame: Frame, origin: Origin) -> list[Scalar]:
        """The per-leaf back-edge values of one carried key; a slot tree must keep its structure and identity."""
        if isinstance(key, str):
            if not isinstance(back_value, (StaticScalar, ResidualScalar)):
                reject(
                    origin,
                    f"{key!r} is rebound to {_aggregate.a_kind(back_value)} inside a data-dependent loop; "
                    "only bool, int, and float values can be carried across its iterations",
                )
            return [back_value]
        assert isinstance(key, tuple)
        entry_tree = self.readable(frame.env[key], origin)
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
        iterable = _aggregate.decay(self.budget, self.expr(node.iterable, frame, sink), node.origin)
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
                value = _aggregate.decay(self.budget, self.expr(node.element, frame, sink), node.origin)
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
        value = _aggregate.decay(self.budget, self.expr(stmt.value, frame, sink), stmt.origin)
        if isinstance(value, Opaque):
            if value.value is None:
                if annotation is None:
                    # `return self.void_helper()` in a `-> None` kernel: None is the bare-return shape.
                    self._commit_site(stmt.origin, frame, sink, [])
                    return
                reject(stmt.origin, "the kernel returns no value (None) but its annotation declares one")
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
                            f"returning the state attribute {spell_state(owner)} would hand out a live alias of its "
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
        for key in sorted(self.specs):
            tree = self.readable(frame.env[key], origin)
            for spec_leaf, leaf in zip(spec_leaves(self.specs[key]), tree_leaves(tree), strict=True):
                if isinstance(leaf, Opaque):
                    reject(origin, _describe_opaque(leaf))
                committed = self._slot_conform(key, spec_leaf, leaf, origin, sink)
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

    def _slot_conform(self, key: StateKey, spec: ScalarSpec, leaf: Scalar, origin: Origin, sink: Sink) -> Scalar:
        if leaf.stype is spec.stype:
            return leaf
        if spec.stype is ScalarType.FLOAT and leaf.stype is ScalarType.INT:
            return self.as_float(leaf, origin, sink)
        if spec.stype is ScalarType.INT and leaf.stype is ScalarType.FLOAT:
            raise _PromoteRun(spec.path)
        reject(
            origin,
            f"the state attribute {spell_state(key)} would change type from {spec.stype.value} to "
            f"{leaf.stype.value}; bool state joins only with bool",
        )

    def _conform_value(
        self, value: Value, annotation: object, origin: Origin, sink: Sink, what: str, *, root: bool = False
    ) -> Value:
        """
        The one annotation-conformance rule for both module boundaries: the root interface demands a
        recognized annotation, while an inlined frame skips unrecognized ones (library stubs annotate with
        shapeless types the subset cannot check).
        """
        annotation = _unaliased(annotation, origin, what)
        stype = annotation_stype(annotation)
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
        if value.stype not in accepted_stypes(declared):
            reject(origin, f"{what} has type {value.stype.value} where the annotation declares {declared.value}")
        return value if value.stype is declared else self.as_float(value, origin, sink)

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
            case IsNone(operand=operand, negated=negated):
                probed = self.expr(operand, frame, sink)
                answer = isinstance(probed, Opaque) and probed.value is None
                return StaticScalar(_ops.make_const(answer != negated))
            case Call():
                return _express.call(self, expr, frame, sink)
            case TupleExpr(origin=origin, items=items):
                return self._display(origin, items, frame, sink)
            case ListExpr(origin=origin, items=items):
                return self._display(origin, items, frame, sink)
            case AttrRead(origin=origin, base=base, attr=attr):
                return _express.attr_read(self, origin, self.expr(base, frame, sink), attr, frame, sink)
            case IndexRead(origin=origin, base=base, index=index):
                base_value = _aggregate.decay(self.budget, self.expr(base, frame, sink), origin)
                index_value = self.expr(index, frame, sink)
                return _aggregate.index_read(origin, base_value, index_value)
            case SliceRead(origin=origin, base=base, lo=lo, hi=hi):
                base_value = _aggregate.decay(self.budget, self.expr(base, frame, sink), origin)
                lo_bound = self._slice_bound(lo, frame, sink)
                hi_bound = self._slice_bound(hi, frame, sink)
                return _aggregate.slice_read(origin, base_value, lo_bound, hi_bound)
            case MultiIndexRead(origin=origin, base=base, axes=axes):
                base_value = _aggregate.decay(self.budget, self.expr(base, frame, sink), origin)
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
                source = _aggregate.decay(self.budget, self.expr(item.value, frame, sink), origin)
                collected.extend(_aggregate.splice_items(origin, source))
            else:
                value = _aggregate.decay(self.budget, self.expr(item, frame, sink), origin)
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
        non-data class attributes. Structural resolution is faithful only while the attribute protocol is
        CPython's, so an overridden one is a rejection rather than a value read past it.
        """
        protocol = overridden_protocol(base.value, READ_PROTOCOLS)
        if protocol is not None:
            reject(
                origin,
                f"the class {type(base.value).__name__} overrides {protocol}, so reading {attr!r} runs host "
                "code; the compiler resolves attributes structurally and would answer past it",
            )
        prefix = self.state_prefix(base.value)
        if prefix is not None and (*prefix, attr) in frame.env:
            bound = frame.env[(*prefix, attr)]
            assert not isinstance(bound, (_Moved, _SlotAlias)), "slot bindings never ride the conduits"
            return self.readable(bound, origin)
        found: object = mro_attr(type(base.value), attr, _MISSING)
        if isinstance(found, property) and isinstance(found.fget, types.FunctionType):
            return self.inline(
                origin, f"{base.name}.{attr}", found.fget, [base], {}, frame, sink, positional_only=False
            )
        if isinstance(found, types.MemberDescriptorType):
            # Named apart from the general descriptor refusal: __slots__ made an ordinary attribute a descriptor
            # behind the user's back, so the generic message would name nothing they wrote.
            reject(
                origin,
                f"the class {type(base.value).__name__} defines __slots__, so {attr!r} has no instance "
                "__dict__ entry to read; __slots__ is not supported yet",
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
        if hasattr(type(found), "__get__"):
            # Any remaining descriptor kind: CPython would route the read through __get__, so admitting the
            # raw class attribute would answer past it (and a callable one would then inline the wrong body).
            reject(origin, f"the attribute {attr!r} is a descriptor the compiler cannot read")
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
            assert not getattr(fn, "__kwdefaults__", None), "registry stubs bind positionally"
            low = len(callee.params) - len(fn.__defaults__ or ())
            if not low <= len(positional) <= len(callee.params):
                expected = str(low) if low == len(callee.params) else f"{low} to {len(callee.params)}"
                reject(site, f"{display}() takes {expected} argument(s), got {len(positional)}")
        bindings = _express.bind_signature(self, site, fn, callee.params, positional, keywords)
        env: _Env = {}
        for param in callee.params:
            value = bindings[param.name]
            declared = annotations.get(param.name, _MISSING)
            if declared is not _MISSING:
                value = self._conform_value(value, declared, site, sink, f"the argument {param.name!r}")
            env[param.name] = value
        env.update(_state_view(frame.env))
        inner = Frame(fn=fn, annotations=annotations, env=env, root=False)
        entry_state = _state_view(frame.env)
        start = len(sink)
        self._inlining.add(fn)
        try:
            flow = self._block(callee.body, inner, [sink], Ctx())
        finally:
            self._inlining.discard(fn)
        assert all(exit.kind is _ExitKind.RETURN for exit in flow.exits), "loop exits cannot escape a frame"
        if flow.fall is not None:
            if any(lane.result is not None for lane in flow.exits):
                reject(
                    site,
                    "the call can complete without returning a value; give every path a return, or none at all",
                )
            # A void callee (a state-writing procedure): the fall lane folds like a bare return, and the
            # call's value is Python's None, judged at use like any captured object.
            flow.exits.append(_Exit.snap(_ExitKind.RETURN, site, inner, flow.fall))
        if all(lane.result is None for lane in flow.exits):
            self._void_conform(annotations, site)
        folded = self._fold_returns(site, flow.exits, [sink])
        if folded.crossed:
            return self._wrap_frame(site, display, folded, frame, entry_state, sink, start)
        frame.env.update(_state_view(folded.env))
        return folded.result if folded.result is not None else Opaque(display, None)

    def _void_conform(self, annotations: dict[str, object], site: Origin) -> None:
        """A callee that returns no value on any path must not DECLARE one the subset recognizes."""
        declared = annotations.get("return", _MISSING)
        if declared is _MISSING or declared is None:
            return
        declared = _unaliased(declared, site, "the return annotation")
        if declared is None:
            return
        recognized = (
            annotation_stype(declared) is not None
            or typing.get_origin(declared) in (tuple, list)
            or _aggregate.array_annotation_shape(declared, site, "the return annotation") is not None
        )
        if recognized:
            reject(site, "the call returns no value on any path, but its return annotation declares one")

    def _wrap_frame(
        self, site: Origin, display: str, folded: _Exit, frame: Frame, entry_state: _Env, sink: Sink, start: int
    ) -> Value:
        """
        A return that crossed the callee's own residual loop cannot rejoin its siblings as arms, so the
        callee region converges at a frame exit instead: every lane's arm ends with one terminator carrying
        the residual result leaves AND the residual leaves of every state root the region changed, and the
        caller reads them back through the frame rows. Statically uniform leaves never leave the value
        model, and the rebuilt trees keep the fold's allocations.
        """
        if isinstance(folded.result, RangeValue):
            reject(site, "a range returned across a data-dependent region is not supported; build it at the use site")
        assert len({id(arm) for arm in folded.sinks}) == len(folded.sinks), "the union holds each lane region once"
        rows: list[FrameRow] = []
        atoms: list[Atom] = []

        def through_rows(value: Value) -> Value:
            leaves: list[Scalar | Opaque] = []
            for leaf in tree_leaves(value):
                if isinstance(leaf, (StaticScalar, Opaque)):
                    leaves.append(leaf)
                    continue
                index = self.fresh()
                rows.append(FrameRow(site, index, leaf.stype))
                atoms.append(_express.materialize(leaf, site))
                leaves.append(ResidualScalar(leaf.stype, TempRef(site, index)))
            return tree_rebuild(value, iter(leaves))

        result = Opaque(display, None) if folded.result is None else through_rows(folded.result)
        folded_state = _state_view(folded.env)
        state_updates: dict[_EnvKey, Value | _Unjoinable | _Moved | _SlotAlias] = {}
        for key in sorted(folded_state, key=_key_order):
            after = folded_state[key]
            before = entry_state.get(key)
            if isinstance(after, _Unjoinable):
                state_updates[key] = after
                continue
            assert not isinstance(after, (_Moved, _SlotAlias)), "slot bindings never ride the conduits"
            if before is not None and not isinstance(before, (_Unjoinable, _Moved, _SlotAlias)) and same(after, before):
                continue
            state_updates[key] = through_rows(after)
        terminator = ResidualFrameReturn(site, tuple(atoms))
        for arm in folded.sinks:
            arm.append(terminator)
        body = _frozen(sink[start:])
        del sink[start:]
        sink.append(ResidualFrame(site, tuple(rows), body))
        frame.env.update(state_updates)
        return result

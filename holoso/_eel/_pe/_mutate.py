"""The mutation side of the specializing interpreter: element stores and state installs."""

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._ir import *
from . import _aggregate, _express
from ._ownership import allocations, blame, escape, mutable
from ._reject import reject
from ._snapshot import describe_opaque as _describe_opaque
from ._state import (
    READ_PROTOCOLS,
    ScalarSpec,
    SequenceSpec,
    Spec,
    TensorSpec,
    attribute_dict,
    overridden_protocol,
    shadowed_edge,
    spell_state,
)
from ._values import (
    Allocation,
    AllocationState,
    BoundMethod,
    Opaque,
    ResidualScalar,
    SequenceValue,
    StaticScalar,
    TensorValue,
    Value,
    same,
    tensor_occurrences,
)

if TYPE_CHECKING:
    from ._interpret import Ctx, Frame, Interpreter, Sink

_AGGREGATE = (SequenceValue, TensorValue)


@dataclass(frozen=True, slots=True)
class _StoreSite:
    spelled: str
    chain: list[Allocation]
    spine: list[tuple[SequenceValue, int]]
    holder: TensorValue
    slot: int
    slot_value: Value


def store(interp: Interpreter, stmt: Store | AugStore, frame: Frame, sink: Sink, ctx: Ctx) -> None:
    """
    The one mutation gate: resolve the target path with the store-prefix exemption (walked over the value
    tree, never through expression reads), judge admission at the store step after the RHS and index
    temps, and rebind new frozen values along the unchanged allocation spine. A store whose root resolves
    by identity into the receiver's component tree is a state write, whatever name it travelled through.
    """
    origin = stmt.origin
    rhs = _aggregate.decay(interp.budget, interp.expr(stmt.value, frame, sink), origin)
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
            root_value = interp.readable(bound, origin)
    if isinstance(root_value, Opaque):
        prefix = interp.state_prefix(root_value.value)
        if prefix is not None:
            _state_store(interp, stmt, prefix, rhs, frame, sink, ctx)
            return
    if not isinstance(root_value, _AGGREGATE):
        if isinstance(stmt.path[0], AttrSel):
            reject(origin, "an attribute store is only supported on the kernel's own component objects")
        reject(origin, f"{_aggregate.a_kind(root_value)} does not support item assignment")
    frame.env[name] = _element_store(interp, stmt, name, root_value, stmt.path, rhs, frame, sink)


def _state_store(
    interp: Interpreter,
    stmt: Store | AugStore,
    prefix: tuple[str, ...],
    rhs: Value,
    frame: Frame,
    sink: Sink,
    ctx: Ctx,
) -> None:
    origin = stmt.origin
    if not isinstance(stmt.path[0], AttrSel):
        reject(origin, "a component object does not support item assignment; store into an attribute")
    assert interp.tree is not None
    obj = interp.tree.objects[prefix]
    selectors = list(stmt.path)
    key: tuple[str, ...] | None = None
    while selectors:
        selector = selectors[0]
        if not isinstance(selector, AttrSel):
            break
        if len(selectors) > 1 and isinstance(selectors[1], AttrSel):
            shadow = shadowed_edge(obj, selector.name)
            if shadow is not None:
                reject(origin, shadow)
            protocol = overridden_protocol(obj, READ_PROTOCOLS)
            if protocol is not None:
                # Structural descent past a read override would diverge; refuse, as instance_attr does for reads.
                reject(
                    origin,
                    f"the class {type(obj).__name__} overrides {protocol}, so reading {selector.name!r} on the "
                    "way to this store runs host code; the compiler resolves the chain structurally and would "
                    "answer past it",
                )
            child = attribute_dict(obj).get(selector.name)
            child_prefix = interp.tree.prefix_of(child) if child is not None else None
            if child_prefix is not None:
                obj = child
                prefix = child_prefix
                selectors.pop(0)
                continue
        key = (*prefix, selector.name)
        selectors.pop(0)
        break
    assert key is not None, "the leading selector is an attribute, so the walk names a leaf"
    rest = tuple(selectors)
    if rest and isinstance(rest[0], AttrSel):
        reject(origin, f"an attribute store through {spell_state(key)} is not supported: it is not a component")
    if rest and interp.tree.objects.get(key) is not None:
        reject(origin, "a component object does not support item assignment; store into an attribute")
    if key not in interp.specs:
        poison = interp.state.poisoned.get(key) if interp.state is not None else None
        if poison is not None:
            reject(origin, poison)
        reject(
            origin,
            f"the write to {spell_state(key)} was not statically visible when state was seeded; "
            "spell it in a plain method of the component that owns the attribute (a property getter is "
            "not scanned), or through the component's own attribute path",
        )
    interp.written_attrs.add(key)
    interp.carry_gate(key)
    if isinstance(stmt, AugStore):
        interp.check_mark(frame, stmt.mark, origin)
    slot_value = interp.readable(frame.env[key], origin)
    if rest:
        if not isinstance(slot_value, _AGGREGATE):
            reject(origin, f"{spell_state(key)} is a scalar state attribute; it has no elements")
        frame.env[key] = _element_store(interp, stmt, spell_state(key), slot_value, rest, rhs, frame, sink)
        interp.note_state_write()
        interp.bump_root_epoch(key)
        return
    spec = interp.specs[key]
    if isinstance(stmt, AugStore):
        if isinstance(slot_value, Opaque):
            reject(origin, _describe_opaque(slot_value))
        if isinstance(slot_value, _AGGREGATE):
            frame.env[key] = aug_aggregate(interp, origin, spell_state(key), slot_value, stmt.op, rhs, frame, sink)
            interp.note_state_write()
            interp.bump_root_epoch(key)
            return
        value = _express.binary(interp, origin, stmt.op, slot_value, rhs, frame, sink)
    else:
        value = rhs
    match value:
        case BoundMethod():
            reject(origin, f"{_aggregate.a_kind(value)} cannot be stored")
        case SequenceValue() | TensorValue():
            if same(value, slot_value):
                return  # rebinding the attribute to its own current tree is Python's no-op
            if ctx.loop:
                reject(
                    origin,
                    f"installing a new aggregate into the state attribute {spell_state(key)} inside a "
                    "data-dependent loop is not supported yet; store its elements instead",
                )
            _install(interp, origin, key, spec, value)
            frame.env[key] = value
            interp.note_state_write()
        case StaticScalar() | ResidualScalar():
            if not isinstance(spec, ScalarSpec):
                reject(
                    origin,
                    f"{spell_state(key)} is an aggregate state attribute; a scalar cannot replace it -- "
                    "store its elements instead",
                )
            frame.env[key] = value
            interp.note_state_write()
        case Opaque():
            if not isinstance(spec, ScalarSpec):
                reject(origin, _describe_opaque(value))
            frame.env[key] = value  # judged at use or at the commit, like any captured scalar
            interp.note_state_write()


def _install(
    interp: Interpreter, origin: Origin, key: tuple[str, ...], spec: Spec, value: SequenceValue | TensorValue
) -> None:
    """The install gate: an aggregate becomes state only if nothing else can still reach its storage."""
    for allocation in allocations(value):
        if allocation.joined:
            reject(
                origin,
                f"cannot install this aggregate into {spell_state(key)}: it was merged across runtime "
                "branches, so its identity is branch-dependent; install it in each branch arm instead",
            )
        owner = interp.state_owners.get(allocation)
        if owner is not None:
            reject(
                origin,
                f"cannot install this aggregate into {spell_state(key)}: it backs (or backed) the state "
                f"attribute {spell_state(owner)}; state trees must be disjoint -- store an explicit copy "
                "(list(...) or np.array(...)) instead",
            )
        if allocation.state is AllocationState.ESCAPED:
            reject(origin, f"cannot install this aggregate into {spell_state(key)}: {blame(allocation)}")
    occurrences = [id(allocation) for allocation in tensor_occurrences(value)]
    if len(occurrences) != len(set(occurrences)):
        reject(
            origin,
            f"cannot install this aggregate into {spell_state(key)}: the same array is reachable through "
            "more than one path within it; build independent elements (e.g. with a comprehension)",
        )
    _install_conform(origin, key, spec, value)
    escape(value)
    for allocation in allocations(value):
        interp.state_owners.setdefault(allocation, key)


def _install_conform(origin: Origin, key: tuple[str, ...], spec: Spec, value: Value) -> None:
    """Structure and array family; scalar leaf types are judged at the commit, like any slot value."""
    match spec, value:
        case ScalarSpec(), (StaticScalar() | ResidualScalar() | Opaque()):
            pass
        case SequenceSpec(items=items), SequenceValue(items=value_items) if len(items) == len(value_items):
            for item_spec, item in zip(items, value_items, strict=True):
                _install_conform(origin, key, item_spec, item)
        case TensorSpec(shape=shape, family=family), TensorValue() if shape == value.shape:
            if value.family is not family:
                reject(
                    origin,
                    f"cannot install this array into {spell_state(key)}: its element family "
                    f"({value.family.value}) differs from the reset value's ({family.value}), and the dtype "
                    "is part of the state attribute's identity; build the array from matching elements",
                )
        case _:
            reject(
                origin,
                f"cannot install this value into {spell_state(key)}: its structure does not match the reset "
                "value's (the kind, length, and shape of every node must agree)",
            )


def _element_store(
    interp: Interpreter,
    stmt: Store | AugStore,
    spelled_root: str,
    root_value: SequenceValue | TensorValue,
    path: tuple[Selector, ...],
    rhs: Value,
    frame: Frame,
    sink: Sink,
) -> SequenceValue | TensorValue:
    origin = stmt.origin
    site = _store_site(interp, origin, spelled_root, root_value, path, frame, sink)
    for allocation in site.chain:
        if not mutable(allocation):
            owner = interp.state_owners.get(allocation)
            if owner is not None and allocation.state is AllocationState.ESCAPED:
                reject(
                    origin,
                    f"cannot store into {site.spelled}: it was installed into the state attribute "
                    f"{spell_state(owner)} this transaction; construct the aggregate fully, then install it once",
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
        value = _express.binary(interp, origin, stmt.op, old, rhs, frame, sink)
    else:
        value = rhs
    if isinstance(value, BoundMethod):
        reject(origin, f"{_aggregate.a_kind(value)} cannot be stored")
    if isinstance(value, (SequenceValue, TensorValue)):
        reject(origin, "storing an aggregate into a container is not supported; rebind or build with a comprehension")
    assert isinstance(value, (StaticScalar, ResidualScalar, Opaque))
    leaf = _express.tensor_leaf(interp, origin, site.holder.family, value, sink)
    new_leaves = tuple(leaf if position == site.slot else kept for position, kept in enumerate(site.holder.leaves))
    rebuilt: SequenceValue | TensorValue = dataclasses.replace(site.holder, leaves=new_leaves)
    for container, position in reversed(site.spine):
        rebuilt = dataclasses.replace(
            container,
            items=tuple(rebuilt if index == position else kept for index, kept in enumerate(container.items)),
        )
    return rebuilt


def _store_site(
    interp: Interpreter,
    origin: Origin,
    root_name: str,
    root_value: SequenceValue | TensorValue,
    path: tuple[Selector, ...],
    frame: Frame,
    sink: Sink,
) -> _StoreSite:
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
                index_value = interp.expr(atom, frame, sink)
        if tensor is not None:
            added = _store_axes(origin, tensor, len(axes), index_value)
            axes.extend(added)
            spelled += _spell_axes(added)
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
                added = _store_axes(origin, tensor, 0, index_value)
                axes.extend(added)
                spelled += _spell_axes(added)
            case _:
                reject(origin, f"{_aggregate.a_kind(current)} is not subscriptable")
    if tensor is None:
        raise AssertionError("the loop returns at the terminal sequence slot")
    rank = len(tensor.shape)
    if len(axes) < rank:
        reject(origin, "a store into an array must write one scalar element; rebind the whole array instead")
    flat = axes[0] * tensor.shape[1] + axes[1] if rank == 2 else axes[0]
    return _StoreSite(spelled, chain, spine, tensor, flat, tensor.leaves[flat])


def _store_axes(origin: Origin, tensor: TensorValue, taken: int, index_value: Value) -> list[int]:
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


def _spell_axes(axes: list[int]) -> str:
    return "".join(f"[{axis}]" for axis in axes)


def aug_aggregate(
    interp: Interpreter,
    origin: Origin,
    name: str,
    current: SequenceValue | TensorValue,
    op: BinaryOp,
    rhs: Value,
    frame: Frame,
    sink: Sink,
) -> Value:
    match current:
        case SequenceValue():
            reject(origin, f"the operator `{op.value}=` is not supported on a sequence; build a numpy array instead")
        case TensorValue():
            if op not in (BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV):
                reject(origin, f"the operator `{op.value}=` is not supported on arrays yet")
            if isinstance(rhs, SequenceValue):
                reject(origin, "cannot mix an array with a Python list/tuple; convert with np.array([...])")
            if not mutable(current.allocation):
                reject(origin, f"cannot update {name!r} in place: {blame(current.allocation)}")
            computed = _express.elementwise(interp, origin, op, current, rhs, frame, sink)
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

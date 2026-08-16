"""
Persistent-state determination over the receiver's COMPONENT TREE, and the reset snapshot with its
disjointness invariant.

The tree is walked eagerly once, cycle-safe: every plain component instance reachable from the compile
root gets a canonical attribute-path prefix, first registration winning. An edge to an off-stack visited
object records a second address, which convicts at `prepare` only when admitted state lies at or beneath
that object -- naming and privacy would otherwise depend on traversal choice.

S seeds from the syntactic may-write scan of every desugarable plain instance method over every component
class MRO, dead arms included. Seeded chains are canonicalized by walking the actual objects, so a child
method's `self.parent.x` names the root's own slot. Runs re-trim S to the paths whose write was reached --
folding only kills writes, so the loop is monotone.

A seeded path whose snapshot cannot be state (an unrepresentable value, an absent attribute, a component
whose class overrides the write protocol or hides the leaf behind a descriptor) is POISONED rather than
rejected: it gets no slot, reads keep folding frozen, and only a REACHED write rejects, carrying the staged
reason. Poison classification is transactional -- a path that fails spec resolution leaves no claims behind.

Reset values come from instance `__dict__` chains, never through descriptors. Each build checks the state
trees pairwise-disjoint, disjoint from everything the environment could hand the kernel (frozen siblings
and environment aggregates eagerly; lazy captures through `check_capture`, sound one-directionally because
state trees build first), and internally alias-free with exact ndarray storage overlap, self-overlap
through strides included -- the flat slot decomposition assigns one register per leaf path, which is
exactly what an alias would break.
"""

import inspect
import math
import types
from dataclasses import dataclass, field

import numpy as np

from ..._errors import SynthesisError
from .._desugar import desugar
from .._ir import Block, EelFunction, Origin, ScalarType, SlotDecl, SlotPath
from .._names import slot_name
from ._reject import reject
from ._residual import assigned_names

type Reset = bool | int | float

type StateKey = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScalarSpec:
    path: SlotPath
    stype: ScalarType
    reset: Reset


@dataclass(frozen=True, slots=True)
class SequenceSpec:
    items: tuple[Spec, ...]


@dataclass(frozen=True, slots=True)
class TensorSpec:
    shape: tuple[int, ...]
    family: ScalarType
    leaves: tuple[ScalarSpec, ...]


type Spec = ScalarSpec | SequenceSpec | TensorSpec


def _is_nan(raw: object) -> bool:
    return isinstance(raw, (float, np.floating)) and math.isnan(float(raw))


def spell_state(path: SlotPath) -> str:
    assert path and isinstance(path[0], str)
    text = "self." + str(path[0])
    for key in path[1:]:
        text += f"[{key}]" if isinstance(key, int) else f".{key}"
    return text


_VALUE_KINDS = (bool, int, float, complex, str, bytes, list, tuple, dict, set, frozenset, np.ndarray, np.generic)


def attribute_dict(obj: object) -> dict[str, object]:
    """The instance attribute dict, empty for a `__slots__` instance -- never the raw `vars()` TypeError."""
    found = getattr(obj, "__dict__", None)
    return found if isinstance(found, dict) else {}


def _is_component(raw: object) -> bool:
    """A plain instance with its own attribute dict -- never a value kind, a callable, or a type."""
    if isinstance(raw, _VALUE_KINDS) or raw is None:
        return False
    if isinstance(raw, (types.ModuleType, type, types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
        return False
    return isinstance(getattr(raw, "__dict__", None), dict)


@dataclass(slots=True)
class ComponentTree:
    """
    The canonical addressing of every component reachable from the compile root; `objects` also keeps the
    walked instances alive so `id`-keyed registrations cannot be recycled under the build.
    """

    prefixes: dict[int, StateKey] = field(default_factory=dict)
    objects: dict[StateKey, object] = field(default_factory=dict)
    second_paths: dict[int, StateKey] = field(default_factory=dict)

    def prefix_of(self, raw: object) -> StateKey | None:
        return self.prefixes.get(id(raw))


def build_component_tree(instance: object) -> ComponentTree:
    tree = ComponentTree()
    tree.prefixes[id(instance)] = ()
    tree.objects[()] = instance
    stack: list[int] = [id(instance)]

    def walk(obj: object, prefix: StateKey) -> None:
        attributes = attribute_dict(obj)
        for attr in sorted(attributes):
            child = attributes[attr]
            if not _is_component(child):
                continue
            if id(child) in stack:
                continue  # a back-reference to an ancestor: the same object, never a second address
            if id(child) in tree.prefixes:
                tree.second_paths.setdefault(id(child), (*prefix, attr))
                continue
            child_prefix = (*prefix, attr)
            tree.prefixes[id(child)] = child_prefix
            tree.objects[child_prefix] = child
            stack.append(id(child))
            try:
                walk(child, child_prefix)
            finally:
                stack.pop()

    walk(instance, ())
    return tree


def _scan_methods(klass: type) -> list[types.FunctionType]:
    found: list[types.FunctionType] = []
    seen: set[int] = set()
    for base in klass.__mro__:
        for name, member in vars(base).items():
            if name in ("__init__", "__new__") or not isinstance(member, types.FunctionType):
                continue
            if id(member) in seen:
                continue
            seen.add(id(member))
            found.append(member)
    return found


def seed_state(tree: ComponentTree, entry: EelFunction) -> dict[StateKey, Origin]:
    """
    A chain whose intermediate step is not a component attribute is dropped: a reached write through it
    draws the located unseeded-store rejection instead. The entry function rides the class scan too, but a
    bound method whose function lives outside the class MRO would otherwise slip the seed, so it is
    scanned once more explicitly.
    """
    seed: dict[StateKey, Origin] = {}
    for prefix in sorted(tree.objects):
        component = tree.objects[prefix]
        for method in _scan_methods(type(component)):
            try:
                eel = desugar(method)
            except SynthesisError:
                continue  # a method that does not desugar can never inline, so it contributes no reachable write
            if eel.params:
                _seed_from(tree, prefix, eel.body, eel.params[0].name, seed)
    assert entry.params, "a bound method always has a receiver parameter"
    _seed_from(tree, (), entry.body, entry.params[0].name, seed)
    return seed


def _seed_from(tree: ComponentTree, prefix: StateKey, body: Block, receiver: str, seed: dict[StateKey, Origin]) -> None:
    for (root, chain), origin in assigned_names(body)[2].items():
        if root != receiver:
            continue
        key = resolve_chain(tree, prefix, chain)
        if key is not None:
            seed.setdefault(key, origin)


def resolve_chain(tree: ComponentTree, prefix: StateKey, chain: tuple[str, ...]) -> StateKey | None:
    """
    The canonical state key for a written attr chain; the intermediate attrs must step through registered
    components, which canonicalizes back-references.
    """
    assert chain
    holder = tree.objects.get(prefix)
    if holder is None:
        return None
    for step, attr in enumerate(chain):
        if step == len(chain) - 1:
            return (*prefix, attr)
        child = attribute_dict(holder).get(attr)
        if child is None or not _is_component(child):
            return None
        child_prefix = tree.prefix_of(child)
        if child_prefix is None:
            return None
        prefix = child_prefix
        holder = child
    raise AssertionError("the loop returns at the final attr")


_WRITE_PROTOCOLS = ("__setattr__", "__delattr__")

READ_PROTOCOLS = ("__getattribute__", "__getattr__")
"""Overriding one of these hides the value from the structural resolution, so a read through it cannot be lowered."""


def mro_attr(klass: type, name: str, default: object = None) -> object:
    """The undescriptored class attribute -- what a structural walk must judge."""
    return next((k.__dict__[name] for k in klass.__mro__ if name in k.__dict__), default)


def overridden_protocol(instance: object, protocols: tuple[str, ...]) -> str | None:
    """The first of `protocols` the instance's class overrides, or None when it inherits all of them."""
    return next((p for p in protocols if getattr(type(instance), p, None) is not getattr(object, p, None)), None)


def shadowed_edge(holder: object, attr: str) -> str | None:
    """
    A data descriptor on the class outranks the instance dict in CPython, so a chain edge resolved
    structurally through `__dict__` would name a different object than the host routes to.
    """
    found = mro_attr(type(holder), attr)
    if found is not None and inspect.isdatadescriptor(found):
        return (
            f"the attribute {attr!r} of {type(holder).__name__} is shadowed by a property/descriptor, so the "
            "structural resolution of this chain would answer past the host's own routing"
        )
    return None


class StateModel:
    """One model per lowering; `prepare` rebuilds the specs and the raw registry for each run of the trim loop."""

    def __init__(
        self, tree: ComponentTree, seed: dict[StateKey, Origin], environment: list[tuple[str, object]]
    ) -> None:
        self.tree = tree
        self._environment = environment
        self.assumed: dict[StateKey, Origin] = dict(seed)
        self.poisoned: dict[StateKey, str] = {}
        self._promoted: set[SlotPath] = set()
        self._specs: dict[StateKey, Spec] = {}
        self._raw_ids: dict[int, StateKey] = {}
        self._arrays: list[tuple[StateKey, np.ndarray]] = []

    def prepare(self) -> dict[StateKey, Spec]:
        self._specs = {}
        self._raw_ids = {}
        self._arrays = []
        for key in sorted(self.assumed):
            origin = self.assumed[key]
            reason = self._admission_fault(key)
            if reason is None:
                staged_ids, staged_arrays = dict(self._raw_ids), list(self._arrays)
                try:
                    self._specs[key] = self._spec(key, self._holder_value(key), key, origin)
                except SynthesisError as error:
                    self._raw_ids, self._arrays = staged_ids, staged_arrays
                    reason = error.message
            if reason is not None:
                self.poisoned[key] = reason
                del self.assumed[key]
        for key in sorted(self.assumed):
            for step in range(1, len(key)):  # the root stays on the walk stack, so it never gains a second address
                holder = self.tree.objects.get(key[:step])
                if holder is not None and id(holder) in self.tree.second_paths:
                    reject(
                        self.assumed[key],
                        f"the component {spell_state(key[:step])} holding the state attribute "
                        f"{spell_state(key)} is also reachable as "
                        f"{spell_state(self.tree.second_paths[id(holder)])}; a multiply-referenced component "
                        "cannot hold state -- give each reference its own instance",
                    )
        self._sweep_frozen()
        for name, raw in self._environment:
            owner = self._overlap_owner(raw)
            if owner is not None:
                reject(
                    self.assumed[owner],
                    f"the state attribute {spell_state(owner)} overlaps the environment name {name!r}; "
                    "a state update would diverge from Python -- store an explicit copy "
                    "(list(...) or np.array(...)) instead",
                )
        self._check_decl_names()
        return dict(self._specs)

    def _holder_value(self, key: StateKey) -> object:
        return attribute_dict(self.tree.objects[key[:-1]])[key[-1]]

    def _admission_fault(self, key: StateKey) -> str | None:
        """The reasons a seeded path cannot become a slot, short of its value's own spec resolution."""
        holder = self.tree.objects.get(key[:-1])
        if holder is None:
            return f"the holder of {spell_state(key)} is not part of the receiver's component tree"
        if not isinstance(getattr(holder, "__dict__", None), dict):
            return (
                f"the class {type(holder).__name__} has no instance __dict__ (it uses __slots__), "
                "which is not supported yet for stateful kernels"
            )
        # EVERY component along the path, root included: a store chain descends structurally, so an overridden
        # protocol or a shadowing descriptor anywhere on it would diverge from the host's routing.
        for step in range(len(key)):
            component = self.tree.objects.get(key[:step])
            if component is None:
                continue
            protocol = overridden_protocol(component, _WRITE_PROTOCOLS + READ_PROTOCOLS)
            if protocol is not None:
                return (
                    f"the class {type(component).__name__} overrides {protocol}, so attribute access runs host "
                    "code the compiler cannot lower to state operations"
                )
            if step < len(key) - 1:
                fault = shadowed_edge(component, key[step])
                if fault is not None:
                    return fault
        if key[-1] not in attribute_dict(holder):
            return (
                f"the state attribute {spell_state(key)} has no value on the instance at synthesis time; "
                "initialize it in __init__ so the reset value is known"
            )
        found = mro_attr(type(holder), key[-1])
        if found is not None and inspect.isdatadescriptor(found):
            return (
                f"the attribute {key[-1]!r} is a property/descriptor; storing through it runs host code "
                "the compiler cannot lower"
            )
        return None

    def decls(self) -> tuple[SlotDecl, ...]:
        found: list[ScalarSpec] = []
        for key in sorted(self._specs):
            found.extend(spec_leaves(self._specs[key]))
        return tuple(SlotDecl(leaf.path, leaf.stype, leaf.reset) for leaf in found)

    def trim(self, written: set[StateKey]) -> bool:
        surviving = {key: origin for key, origin in self.assumed.items() if key in written}
        assert set(surviving) <= set(self.assumed)
        if set(surviving) == set(self.assumed):
            return False
        self.assumed = surviving
        self._promoted.clear()
        return True

    def promote(self, path: SlotPath) -> None:
        assert isinstance(path[0], str) and path not in self._promoted
        self._promoted.add(path)

    def promoted_paths(self) -> frozenset[SlotPath]:
        return frozenset(self._promoted)

    def note(self) -> str:
        pins = "; ".join(
            f"{spell_state(key)} is treated as persistent state because of the write at {origin.location}"
            for key, origin in sorted(self.assumed.items())
        )
        return f"\nnote: {pins}; if such a write is unreachable, remove it (or the method spelling it)" if pins else ""

    def _sweep_frozen(self) -> None:
        """Every non-state value reachable from any component is a frozen sibling, poisoned paths included."""
        for prefix in sorted(self.tree.objects):
            component = self.tree.objects[prefix]
            attributes = attribute_dict(component)
            for attr in sorted(attributes):
                key = (*prefix, attr)
                if key in self._specs:
                    continue
                raw = attributes[attr]
                if _is_component(raw):
                    continue  # its own attributes are swept as that component's rows
                owner = self._overlap_owner(raw)
                if owner is not None:
                    label = "poisoned" if key in self.poisoned else "frozen"
                    reject(
                        self.assumed[owner],
                        f"the state attribute {spell_state(owner)} shares storage with the {label} attribute "
                        f"{spell_state(key)}; a state update would diverge from Python -- store an explicit "
                        "copy (list(...) or np.array(...)) in one of them",
                    )

    def _overlap_owner(self, raw: object) -> StateKey | None:
        if isinstance(raw, (list, tuple)):
            owner = self._raw_ids.get(id(raw))
            if owner is not None:
                return owner
            return next((found for item in raw if (found := self._overlap_owner(item)) is not None), None)
        if isinstance(raw, np.ndarray):
            owner = self._raw_ids.get(id(raw))
            if owner is None:
                owner = next((key for key, array in self._arrays if np.shares_memory(raw, array)), None)
            return owner
        return None

    def check_capture(self, name: str, raw: object, origin: Origin) -> None:
        """The capture guard: a captured aggregate must not overlap any state tree."""
        owner = self._overlap_owner(raw)
        if owner is not None:
            reject(
                origin,
                f"the captured value of {name!r} overlaps the state attribute {spell_state(owner)}; "
                "a later state update would diverge from Python -- store an explicit copy "
                "(list(...) or np.array(...)) instead",
            )

    def _spec(self, key: StateKey, raw: object, path: SlotPath, origin: Origin) -> Spec:
        scalar = self._scalar_spec(raw, path, origin)
        if scalar is not None:
            return scalar
        if isinstance(raw, (list, tuple)):
            if isinstance(raw, list):
                self._claim(key, raw, path, origin)
            if not raw:
                reject(origin, f"the state attribute {spell_state(path)} is an empty aggregate, which is not supported")
            return SequenceSpec(tuple(self._spec(key, item, (*path, index), origin) for index, item in enumerate(raw)))
        if isinstance(raw, np.ndarray):
            return self._tensor_spec(key, raw, path, origin)
        reject(
            origin,
            f"the state attribute {spell_state(path)} holds {type(raw).__name__!r}, "
            "which the compiler cannot represent as state; only bool/int/float scalars, "
            "sequences, and numpy arrays are supported",
        )

    def _scalar_spec(self, raw: object, path: SlotPath, origin: Origin) -> ScalarSpec | None:
        if _is_nan(raw):
            reject(origin, f"the reset value of {spell_state(path)} is NaN, which the compiler cannot represent")
        if isinstance(raw, (bool, np.bool_)):
            return ScalarSpec(path, ScalarType.BOOL, bool(raw))
        if isinstance(raw, (int, np.integer)):
            if path in self._promoted:
                return ScalarSpec(path, ScalarType.FLOAT, self._float_image(int(raw), path, origin))
            return ScalarSpec(path, ScalarType.INT, int(raw))
        if isinstance(raw, (float, np.floating)):
            return ScalarSpec(path, ScalarType.FLOAT, float(raw))
        return None

    def _tensor_spec(self, key: StateKey, raw: np.ndarray, path: SlotPath, origin: Origin) -> TensorSpec:
        if type(raw) is not np.ndarray:
            reject(origin, f"the state attribute {spell_state(path)} is a numpy subclass, which is not supported")
        if raw.ndim not in (1, 2) or 0 in raw.shape:
            reject(origin, f"the state attribute {spell_state(path)} must be a non-empty 1-D or 2-D array")
        if raw.dtype.kind == "f":
            family = ScalarType.FLOAT
        elif raw.dtype.kind in "iu":
            family = ScalarType.INT
        else:
            reject(origin, f"the state attribute {spell_state(path)} must hold a float or integer dtype")
        if any(promoted[: len(path)] == path for promoted in self._promoted):
            family = ScalarType.FLOAT
        self._check_self_overlap(raw, path, origin)
        self._claim(key, raw, path, origin)
        shape = tuple(raw.shape)
        leaves: list[ScalarSpec] = []
        for position, element in enumerate(raw.flatten().tolist()):
            leaf_path = (*path, *((position,) if len(shape) == 1 else (position // shape[1], position % shape[1])))
            if _is_nan(element):
                reject(
                    origin, f"the reset value of {spell_state(leaf_path)} is NaN, which the compiler cannot represent"
                )
            if family is ScalarType.FLOAT and raw.dtype.kind in "iu":
                reset: Reset = self._float_image(int(element), leaf_path, origin)
            elif family is ScalarType.FLOAT:
                reset = float(element)
            else:
                reset = int(element)
            leaves.append(ScalarSpec(leaf_path, family, reset))
        return TensorSpec(shape, family, tuple(leaves))

    def _float_image(self, reset: int, path: SlotPath, origin: Origin) -> float:
        try:
            image = float(reset)
        except OverflowError:
            image = math.inf
        if image != reset:
            reject(
                origin,
                f"the state attribute {spell_state(path)} promotes to float, but its integer reset value {reset} "
                "has no exact float image; use a float reset or keep the writes integer",
            )
        return image

    def _claim(self, key: StateKey, raw: list[object] | np.ndarray, path: SlotPath, origin: Origin) -> None:
        """The within-tree and cross-tree identity/storage registry, in one pass."""
        owner = self._raw_ids.get(id(raw))
        if owner == key:
            reject(
                origin,
                f"{spell_state(path)} reaches the same mutable object through more than one path within "
                f"{spell_state(key)}; the flat state slots cannot represent the aliasing -- build independent "
                "elements (e.g. with a comprehension)",
            )
        if owner is not None:
            reject(
                origin,
                f"{spell_state(path)} is the same object as part of the state attribute {spell_state(owner)}; "
                "state trees must be disjoint -- store an explicit copy (list(...) or np.array(...)) instead",
            )
        if isinstance(raw, np.ndarray):
            for other_key, array in self._arrays:
                if np.shares_memory(raw, array):
                    if other_key == key:
                        reject(
                            origin,
                            f"{spell_state(path)} overlaps the storage of another array within "
                            f"{spell_state(key)}; the flat state slots cannot represent the aliasing -- "
                            "store independent copies",
                        )
                    reject(
                        origin,
                        f"{spell_state(path)} shares storage with the state attribute {spell_state(other_key)}; "
                        "state trees must be disjoint -- store an explicit np.array(...) copy instead",
                    )
            self._arrays.append((key, raw))
        self._raw_ids[id(raw)] = key

    def _check_self_overlap(self, raw: np.ndarray, path: SlotPath, origin: Origin) -> None:
        """Occupied byte intervals of all elements must be pairwise disjoint (a view can alias itself)."""
        if raw.size < 2:
            return
        starts: list[int] = []
        for index in np.ndindex(raw.shape):
            starts.append(sum(int(i) * int(s) for i, s in zip(index, raw.strides, strict=True)))
        starts.sort()
        if any(a + raw.itemsize > b for a, b in zip(starts, starts[1:], strict=False)):
            reject(
                origin,
                f"the state attribute {spell_state(path)} is a self-overlapping array view; "
                "the flat state slots cannot represent the aliasing -- store an np.array(...) copy instead",
            )

    def _check_decl_names(self) -> None:
        seen: dict[str, SlotPath] = {}
        for key in sorted(self._specs):
            for leaf in spec_leaves(self._specs[key]):
                name = slot_name(leaf.path)
                other = seen.get(name)
                if other is not None:
                    reject(
                        self.assumed[key],
                        f"the state attributes {spell_state(other)} and {spell_state(leaf.path)} decompose to "
                        f"the same slot name {name!r}; rename one of them",
                    )
                seen[name] = leaf.path


def spec_leaves(spec: Spec) -> list[ScalarSpec]:
    match spec:
        case ScalarSpec():
            return [spec]
        case SequenceSpec(items=items):
            return [leaf for item in items for leaf in spec_leaves(item)]
        case TensorSpec(leaves=leaves):
            return list(leaves)


def environment_aggregates(fn: object) -> list[tuple[str, object]]:
    """
    Everything the environment could hand the kernel: module globals, closure cells, and the entry method's
    own default values -- aggregates only, the roots A1 names as entering escaped.
    """
    found: list[tuple[str, object]] = [
        (name, value)
        for name, value in getattr(fn, "__globals__", {}).items()
        if isinstance(value, (list, tuple, np.ndarray))
    ]
    code = getattr(fn, "__code__")
    for name, cell in zip(code.co_freevars, getattr(fn, "__closure__") or (), strict=True):
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            found.append((name, value))
    defaults = list(getattr(fn, "__defaults__") or ()) + list((getattr(fn, "__kwdefaults__") or {}).values())
    found.extend(("a parameter default", value) for value in defaults if isinstance(value, (list, tuple, np.ndarray)))
    return found

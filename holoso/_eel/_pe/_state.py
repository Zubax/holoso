"""
Persistent-state determination (ratified A2) and the reset snapshot with its disjointness invariant
(ratified A5, clauses 2 and 3). S seeds from the syntactic may-write scan of the entry method, dead arms
included; runs re-trim S to the attributes whose write was reached -- folding only kills writes, so the loop
is monotone. A rejection under the conservative assumption is FINAL, annotated with the pinning writes.

Reset values come from the instance ``__dict__``, never through descriptors. Each build checks the state
trees pairwise-disjoint, disjoint from everything the environment could hand the kernel (frozen siblings and
environment aggregates eagerly; lazy captures through ``check_capture``, sound one-directionally because
state trees build first), and internally alias-free with exact ndarray storage overlap, self-overlap through
strides included -- the flat slot decomposition assigns one register per leaf path, which is exactly what an
alias would break. Raw tuples launder nothing only when their subtrees hold no mutable object: ``__init__``
ran arbitrary Python, unlike the in-model sequences the install gate sees.
"""

import inspect
import math
from dataclasses import dataclass

import numpy as np

from .._ir import Origin, ScalarType, SlotDecl, SlotPath
from .._names import slot_name
from ._reject import reject

type Reset = bool | int | float


@dataclass(frozen=True, slots=True)
class ScalarSpec:
    path: SlotPath
    stype: ScalarType
    reset: Reset


@dataclass(frozen=True, slots=True)
class SequenceSpec:
    items: tuple["Spec", ...]


@dataclass(frozen=True, slots=True)
class TensorSpec:
    shape: tuple[int, ...]
    family: ScalarType
    leaves: tuple[ScalarSpec, ...]


type Spec = ScalarSpec | SequenceSpec | TensorSpec


def _is_nan(raw: object) -> bool:
    return isinstance(raw, (float, np.floating)) and math.isnan(float(raw))


def _spell(path: SlotPath) -> str:
    assert path and isinstance(path[0], str)
    return f"self.{path[0]}" + "".join(f"[{key}]" for key in path[1:])


class StateModel:
    """One model per lowering; ``prepare`` rebuilds the specs and the raw registry for each run of the trim loop."""

    def __init__(self, instance: object, seed: dict[str, Origin], environment: list[tuple[str, object]]) -> None:
        assert seed
        self._instance = instance
        self._environment = environment
        self.assumed: dict[str, Origin] = dict(seed)
        self._promoted: set[SlotPath] = set()
        self._specs: dict[str, Spec] = {}
        self._raw_ids: dict[int, str] = {}
        self._arrays: list[tuple[str, np.ndarray]] = []

    def prepare(self) -> dict[str, Spec]:
        self._specs = {}
        self._raw_ids = {}
        self._arrays = []
        attributes = vars(self._instance)
        for attr in sorted(self.assumed):
            origin = self.assumed[attr]
            if attr not in attributes:
                reject(
                    origin,
                    f"the state attribute {attr!r} has no value on the instance at synthesis time; "
                    "initialize it in __init__ so the reset value is known",
                )
            self._specs[attr] = self._spec(attr, attributes[attr], (attr,), origin)
        for attr, raw in attributes.items():
            if attr not in self.assumed:
                self._check_frozen_sibling(attr, raw)
        for name, raw in self._environment:
            owner = self._overlap_owner(raw)
            if owner is not None:
                reject(
                    self.assumed[owner],
                    f"the state attribute self.{owner} overlaps the environment name {name!r}; "
                    "a state update would diverge from Python -- store an explicit copy "
                    "(list(...) or np.array(...)) instead",
                )
        self._check_decl_names()
        return dict(self._specs)

    def decls(self) -> tuple[SlotDecl, ...]:
        found: list[ScalarSpec] = []
        for attr in sorted(self._specs):
            found.extend(spec_leaves(self._specs[attr]))
        return tuple(SlotDecl(leaf.path, leaf.stype, leaf.reset) for leaf in found)

    def trim(self, written: set[str]) -> bool:
        surviving = {attr: origin for attr, origin in self.assumed.items() if attr in written}
        assert set(surviving) <= set(self.assumed)
        if set(surviving) == set(self.assumed):
            return False
        self.assumed = surviving
        self._promoted.clear()
        return True

    def promote(self, path: SlotPath) -> None:
        assert isinstance(path[0], str) and path[0] in self.assumed and path not in self._promoted
        self._promoted.add(path)

    def promoted_paths(self) -> frozenset[SlotPath]:
        return frozenset(self._promoted)

    def note(self) -> str:
        pins = "; ".join(
            f"self.{attr} is treated as persistent state because of the write at {origin.location}"
            for attr, origin in sorted(self.assumed.items())
        )
        return f"\nnote: {pins}; if such a write is unreachable, remove it"

    def _check_frozen_sibling(self, attr: str, raw: object) -> None:
        owner = self._overlap_owner(raw)
        if owner is not None:
            reject(
                self.assumed[owner],
                f"the state attribute self.{owner} shares storage with the frozen attribute self.{attr}; "
                "a state update would diverge from Python -- store an explicit copy "
                "(list(...) or np.array(...)) in one of them",
            )

    def _overlap_owner(self, raw: object) -> str | None:
        if isinstance(raw, (list, tuple)):
            owner = self._raw_ids.get(id(raw))
            if owner is not None:
                return owner
            return next((found for item in raw if (found := self._overlap_owner(item)) is not None), None)
        if isinstance(raw, np.ndarray):
            owner = self._raw_ids.get(id(raw))
            if owner is None:
                owner = next((attr for attr, array in self._arrays if np.shares_memory(raw, array)), None)
            return owner
        return None

    def check_capture(self, name: str, raw: object, origin: Origin) -> None:
        """The A5 clause (2) guard: a captured aggregate must not overlap any state tree."""
        owner = self._overlap_owner(raw)
        if owner is not None:
            reject(
                origin,
                f"the captured value of {name!r} overlaps the state attribute self.{owner}; "
                "a later state update would diverge from Python -- store an explicit copy "
                "(list(...) or np.array(...)) instead",
            )

    def _spec(self, attr: str, raw: object, path: SlotPath, origin: Origin) -> Spec:
        scalar = self._scalar_spec(raw, path, origin)
        if scalar is not None:
            return scalar
        if isinstance(raw, (list, tuple)):
            if isinstance(raw, list):
                self._claim(attr, raw, path, origin)
            if not raw:
                reject(origin, f"the state attribute {_spell(path)} is an empty aggregate, which is not supported")
            return SequenceSpec(tuple(self._spec(attr, item, (*path, index), origin) for index, item in enumerate(raw)))
        if isinstance(raw, np.ndarray):
            return self._tensor_spec(attr, raw, path, origin)
        reject(
            origin,
            f"the state attribute {_spell(path)} holds {type(raw).__name__!r}, "
            "which the compiler cannot represent as state; only bool/int/float scalars, "
            "sequences, and numpy arrays are supported",
        )

    def _scalar_spec(self, raw: object, path: SlotPath, origin: Origin) -> ScalarSpec | None:
        if _is_nan(raw):
            reject(origin, f"the reset value of {_spell(path)} is NaN, which the compiler cannot represent")
        if isinstance(raw, (bool, np.bool_)):
            return ScalarSpec(path, ScalarType.BOOL, bool(raw))
        if isinstance(raw, (int, np.integer)):
            if path in self._promoted:
                return ScalarSpec(path, ScalarType.FLOAT, self._float_image(int(raw), path, origin))
            return ScalarSpec(path, ScalarType.INT, int(raw))
        if isinstance(raw, (float, np.floating)):
            return ScalarSpec(path, ScalarType.FLOAT, float(raw))
        return None

    def _tensor_spec(self, attr: str, raw: np.ndarray, path: SlotPath, origin: Origin) -> TensorSpec:
        if type(raw) is not np.ndarray:
            reject(origin, f"the state attribute {_spell(path)} is a numpy subclass, which is not supported")
        if raw.ndim not in (1, 2) or 0 in raw.shape:
            reject(origin, f"the state attribute {_spell(path)} must be a non-empty 1-D or 2-D array")
        if raw.dtype.kind == "f":
            family = ScalarType.FLOAT
        elif raw.dtype.kind in "iu":
            family = ScalarType.INT
        else:
            reject(origin, f"the state attribute {_spell(path)} must hold a float or integer dtype")
        if any(promoted[: len(path)] == path for promoted in self._promoted):
            family = ScalarType.FLOAT
        self._check_self_overlap(raw, path, origin)
        self._claim(attr, raw, path, origin)
        shape = tuple(raw.shape)
        leaves: list[ScalarSpec] = []
        for position, element in enumerate(raw.flatten().tolist()):
            leaf_path = (*path, *((position,) if len(shape) == 1 else (position // shape[1], position % shape[1])))
            if _is_nan(element):
                reject(origin, f"the reset value of {_spell(leaf_path)} is NaN, which the compiler cannot represent")
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
                f"the state attribute {_spell(path)} promotes to float, but its integer reset value {reset} "
                "has no exact float image; use a float reset or keep the writes integer",
            )
        return image

    def _claim(self, attr: str, raw: list[object] | np.ndarray, path: SlotPath, origin: Origin) -> None:
        """The A5 clause (3) within-tree and clause (2) cross-tree identity/storage registry, in one pass."""
        owner = self._raw_ids.get(id(raw))
        if owner == attr:
            reject(
                origin,
                f"{_spell(path)} reaches the same mutable object through more than one path within "
                f"self.{attr}; the flat state slots cannot represent the aliasing -- build independent "
                "elements (e.g. with a comprehension)",
            )
        if owner is not None:
            reject(
                origin,
                f"{_spell(path)} is the same object as part of the state attribute self.{owner}; "
                "state trees must be disjoint -- store an explicit copy (list(...) or np.array(...)) instead",
            )
        if isinstance(raw, np.ndarray):
            for other_attr, array in self._arrays:
                if np.shares_memory(raw, array):
                    if other_attr == attr:
                        reject(
                            origin,
                            f"{_spell(path)} overlaps the storage of another array within self.{attr}; "
                            "the flat state slots cannot represent the aliasing -- store independent copies",
                        )
                    reject(
                        origin,
                        f"{_spell(path)} shares storage with the state attribute self.{other_attr}; "
                        "state trees must be disjoint -- store an explicit np.array(...) copy instead",
                    )
            self._arrays.append((attr, raw))
        self._raw_ids[id(raw)] = attr

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
                f"the state attribute {_spell(path)} is a self-overlapping array view; "
                "the flat state slots cannot represent the aliasing -- store an np.array(...) copy instead",
            )

    def _check_decl_names(self) -> None:
        seen: dict[str, SlotPath] = {}
        for attr in sorted(self._specs):
            for leaf in spec_leaves(self._specs[attr]):
                name = slot_name(leaf.path)
                other = seen.get(name)
                if other is not None:
                    reject(
                        self.assumed[attr],
                        f"the state attributes {_spell(other)} and {_spell(leaf.path)} decompose to the same "
                        f"slot name {name!r}; rename one of them",
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


READ_PROTOCOLS = ("__getattribute__", "__getattr__")
"""Overriding one of these hides the value from the structural resolution, so a read through it cannot be lowered."""

_WRITE_PROTOCOLS = ("__setattr__", "__delattr__")


def overridden_protocol(instance: object, protocols: tuple[str, ...]) -> str | None:
    """The first of ``protocols`` the instance's class overrides, or None when it inherits all of them."""
    return next((p for p in protocols if getattr(type(instance), p, None) is not getattr(object, p, None)), None)


def descriptor_guard(instance: object, seed: dict[str, Origin]) -> None:
    """
    State reads and writes lower to plain slot operations, so anything that would make CPython run code on
    attribute access instead -- a custom attribute protocol, a data descriptor, a __dict__-less instance --
    is a located rejection at the pinning write.
    """
    anchor = seed[min(seed)]
    for protocol in _WRITE_PROTOCOLS + READ_PROTOCOLS:
        if getattr(type(instance), protocol, None) is not getattr(object, protocol, None):
            reject(
                anchor,
                f"the class {type(instance).__name__} overrides {protocol}, so attribute access runs host "
                "code the compiler cannot lower to state operations",
            )
    if not isinstance(getattr(instance, "__dict__", None), dict):
        reject(
            anchor,
            f"the class {type(instance).__name__} has no instance __dict__ (it uses __slots__), "
            "which is not supported yet for stateful kernels",
        )
    for attr in sorted(seed):
        found = next((klass.__dict__[attr] for klass in type(instance).__mro__ if attr in klass.__dict__), None)
        if found is not None and inspect.isdatadescriptor(found):
            reject(
                seed[attr],
                f"the attribute {attr!r} is a property/descriptor; storing through it runs host code "
                "the compiler cannot lower",
            )

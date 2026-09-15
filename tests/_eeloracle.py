"""
The front-end differential oracle: CPython executing the original kernel against `HirEvaluator` running the
unoptimized HIR lowered from it -- upstream of HIR optimization, so fastmath rewrites cannot muddy the verdict.
The harness is frontend-agnostic: it takes a lowered `Hir` and an identically constructed reference callable.

Protocol, per transaction of the ordered vector sequence:

- The reference runs first, bound signature-aware (the corpus mixes positional-only and keyword-only parameters).
  A vector on which the reference RAISES, or whose observable leaves -- the return leaves and the slot-named
  attribute leaves, not frozen attributes the kernel never lowers -- contain a NaN the compiler refuses to
  represent, lies outside the comparable domain and is discarded; a stateful kernel's remaining sequence goes with
  it, since the instance is no longer trustworthy. At least one transaction must survive comparison.
- The module interface itself is checked: input ports must mirror the reference's parameters in order, under the
  names the hardware spells them, output port names must be unique, every public slot must be observable through its `state_<slot>` port
  -- whose produced value is compared against the post-call attribute leaf, so a mis-wired port diverges from its
  own slot -- and no state port may expose a private slot.
- `out_<path>` ports are compared against the flattened return leaves by name (a `None` return has no leaves).
  Bools and ints are exact -- a bool port demands a genuine bool -- while a float port coerces the reference's
  ints to their float image (a float-typed counter written as `0`; a C-promotion join rounding a large int) and
  matches to `ulps` ULPs with zeros and infinities exact: binary64 against binary64 in identical operation order
  leaves libm-level slack only.
- A return leaf with NO out port must exactly equal (reference against reference) some public slot's post-call
  value: the emitter elides a returned leaf that carries the same VALUE ID as a public state live-out, since that
  wire is already observable through the `state_<slot>` port. An emitter eliding an unrelated leaf fails here;
  an elision whose value coincides with a public slot by accident rather than by identity is the one blind spot,
  accepted because no information is lost through it.
- Every evaluator slot is compared against the instance-attribute leaf of the same name (row-major decomposition,
  the slot naming convention), and the exact SET of slots that changed must equal the exact set of attribute
  leaves that changed -- a missed state write and an invented one both surface.

Comparable-domain edges, accepted: the evaluator has no negative zero, so a kernel observing the sign of zero
(`atan2(-0.0, x)`) diverges; a NaN the reference computes and consumes WITHOUT surfacing it in any leaf (say, a
comparison against `inf - inf`) is undetectable host-side and fails loudly for eye triage. An array kernel
parameter is driven through its decomposed scalar leaves: the vector row stays name -> value with the leaf names
(`v_0`, `v_1`), the expected port set derives from the reference's own jaxtyping annotation (read
structurally, exactly as the frontend reads it), and the binder reassembles the ndarray argument for CPython.
"""

import dataclasses
import inspect
import math
import types
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

from holoso._eel._annotations import unaliased
from holoso._eel._names import indexed_names, spelled
from holoso._hir import Hir, HirEvaluator, NoNumber

from ._modelref import flatten_value, port_name

type Scalar = float | bool | int
type InputRow = Mapping[str, Scalar]

_STATE_PREFIX = "state_"


def floats_match(actual: float, expected: float, ulps: int) -> bool:
    """Exact for zeros and infinities; otherwise within `ulps` ULPs of the larger magnitude."""
    if actual == expected:
        return True
    if math.isinf(actual) or math.isinf(expected) or actual == 0.0 or expected == 0.0:
        return False
    return abs(actual - expected) <= ulps * math.ulp(max(abs(actual), abs(expected)))


def _as_float(context: str, value: object) -> float:
    """
    An int meeting a float lane compares as its float image, rounding included: C-promotion at a join is a
    deliberate type-system deviation, and a wrongly-floated int lane cannot escape the frontend's annotation
    validation.
    """
    numeric = isinstance(value, (float, np.floating)) or (
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
    )
    assert numeric, f"{context}: expected a number, got {type(value).__name__} {value!r}"
    assert isinstance(value, (float, int, np.floating, np.integer))
    return float(value)


def _compare(context: str, actual: Scalar, expected: object, ulps: int) -> None:
    if type(actual) is bool:
        assert isinstance(expected, (bool, np.bool_)), f"{context}: expected a bool, got {expected!r}"
        assert actual == bool(expected), f"{context}: {actual} != {expected}"
    elif type(actual) is int:
        assert isinstance(expected, (int, np.integer)) and not isinstance(
            expected, (bool, np.bool_)
        ), f"{context}: expected an int, got {expected!r}"
        assert actual == int(expected), f"{context}: {actual} != {expected}"
    else:
        assert type(actual) is float
        reference = _as_float(context, expected)
        assert floats_match(actual, reference, ulps), f"{context}: {actual!r} != {reference!r} beyond {ulps} ULPs"


def _plain_component(value: object) -> bool:
    """A nested component instance whose attributes decompose into the parent's slot namespace."""
    if isinstance(value, (bool, int, float, complex, str, bytes, list, tuple, dict, set, frozenset)):
        return False
    if isinstance(value, (np.ndarray, np.generic)) or value is None:
        return False
    if isinstance(value, (types.ModuleType, type, types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
        return False
    return isinstance(getattr(value, "__dict__", None), dict)


def walk_instance_leaves(instance: object) -> list[tuple[str, bool, object]]:
    """
    Every attribute leaf reachable through component instances, as (flattened slot name, private, value):
    scalars keep the dotted-to-underscore name, arrays go row-major, and a leaf is private when any path
    component is underscore-prefixed -- mirroring the compiler's decomposition and privacy exactly.
    """
    found: list[tuple[str, bool, object]] = []
    seen: set[int] = set()

    def walk(obj: object, prefix: str, private: bool) -> None:
        if id(obj) in seen:
            return
        seen.add(id(obj))
        for attr, value in vars(obj).items():
            name = prefix + spelled(attr)
            hidden = private or attr.startswith("_")
            if isinstance(value, (list, tuple, np.ndarray)):
                for path, leaf in flatten_value(value):
                    found.append((name + "".join(f"_{key}" for key in path), hidden, leaf))
            elif _plain_component(value):
                walk(value, name + "_", hidden)
            else:
                found.append((name, hidden, value))

    walk(instance, "", False)
    return found


def instance_leaves(instance: object) -> dict[str, object]:
    """
    Attribute leaves named exactly like decomposed state slots: scalars keep the name, arrays go row-major.
    A name minted twice (nested `child.y` vs flat `child_y`) is dropped rather than silently overwritten, so
    a slot comparison against it fails loudly instead of accepting whichever leaf won the dict race.
    """
    leaves: dict[str, object] = {}
    ambiguous: set[str] = set()
    for name, _, value in walk_instance_leaves(instance):
        if name in leaves:
            ambiguous.add(name)
        leaves[name] = value
    return {name: value for name, value in leaves.items() if name not in ambiguous}


def private_leaf_names(instance: object) -> set[str]:
    return {name for name, private, _ in walk_instance_leaves(instance) if private}


def _annotation_shape(annotation: object) -> tuple[int, ...] | None:
    """A jaxtyping-style fixed shape, read structurally like the frontend reads it; None for scalars."""
    annotation = unaliased(annotation)
    dims = getattr(annotation, "dims", None) if isinstance(annotation, type) else None
    if not isinstance(dims, tuple):
        return None
    sizes = [getattr(dim, "size", None) for dim in dims]
    assert all(isinstance(size, int) for size in sizes), f"non-static array annotation {annotation!r}"
    return tuple(size for size in sizes if isinstance(size, int))


def _parameter_annotations(reference: Callable[..., object]) -> dict[str, object]:
    annotations = inspect.get_annotations(reference, eval_str=True)
    return {name: unaliased(annotations.get(name)) for name in inspect.signature(reference).parameters}


def _field_hints(cls: type) -> dict[str, object]:
    """MRO-merged child-wins, mirroring the frontend's field-annotation resolution."""
    merged: dict[str, object] = {}
    for base in reversed(cls.__mro__):
        merged.update(inspect.get_annotations(base, eval_str=True))
    return {name: unaliased(annotation) for name, annotation in merged.items()}


def _leaf_names(name: str, annotation: object) -> list[str]:
    """The decomposed port lanes of one parameter, mirroring the frontend's record/tuple/array decomposition."""
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        hints = _field_hints(annotation)
        return [
            leaf
            for field in dataclasses.fields(annotation)
            for leaf in _leaf_names(f"{name}_{field.name}", hints[field.name])
        ]
    if typing.get_origin(annotation) is tuple:
        args = typing.get_args(annotation)
        return [leaf for k, arg in enumerate(args) for leaf in _leaf_names(f"{name}_{k}", unaliased(arg))]
    shape = _annotation_shape(annotation)
    return [spelled(name)] if shape is None else indexed_names(spelled(name), shape)


def _leaf_value(name: str, annotation: object, row: InputRow) -> object:
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        hints = _field_hints(annotation)
        return annotation(
            **{
                field.name: _leaf_value(f"{name}_{field.name}", hints[field.name], row)
                for field in dataclasses.fields(annotation)
            }
        )
    if typing.get_origin(annotation) is tuple:
        args = typing.get_args(annotation)
        return tuple(_leaf_value(f"{name}_{k}", unaliased(arg), row) for k, arg in enumerate(args))
    shape = _annotation_shape(annotation)
    if shape is None:
        return row[spelled(name)]
    return np.array([row[leaf] for leaf in indexed_names(spelled(name), shape)]).reshape(shape)


def expected_input_names(reference: Callable[..., object]) -> list[str]:
    names: list[str] = []
    for name, annotation in _parameter_annotations(reference).items():
        names.extend(_leaf_names(name, annotation))
    assert len(set(names)) == len(names), f"duplicate decomposed input names in {names}"
    return names


def binder(reference: Callable[..., object]) -> Callable[[InputRow], tuple[list[object], dict[str, object]]]:
    """Row lookups happen outside the discard-guarded call, so a typo'd vector key crashes instead of discarding."""
    parameters = list(inspect.signature(reference).parameters.values())
    kinds = {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    assert all(parameter.kind in kinds for parameter in parameters), f"unsupported signature {parameters}"
    annotations = _parameter_annotations(reference)

    def bind(row: InputRow) -> tuple[list[object], dict[str, object]]:
        args = [
            _leaf_value(p.name, annotations[p.name], row)
            for p in parameters
            if p.kind is not inspect.Parameter.KEYWORD_ONLY
        ]
        kwargs = {
            p.name: _leaf_value(p.name, annotations[p.name], row)
            for p in parameters
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        }
        return args, kwargs

    return bind


def _is_nan(value: object) -> bool:
    return isinstance(value, (float, np.floating)) and math.isnan(float(value))


def _has_nan(values: Iterable[object]) -> bool:
    return any(_is_nan(value) for value in values)


def _raw_eq(a: object, b: object) -> bool:
    # NaN-to-NaN counts as unchanged: an untouched NaN-holding attribute must not read as a state write.
    return bool(a == b) or (_is_nan(a) and _is_nan(b))


def _kind(value: object) -> type:
    if isinstance(value, (bool, np.bool_)):
        return bool
    if isinstance(value, (int, np.integer)):
        return int
    return float


def _elidable_eq(leaf: object, slot_value: object) -> bool:
    """
    Genuine ValueId elision relates one CPython value to itself, so the kinds agree; without this, a dropped bool
    or int leaf could hide behind an equal-comparing float slot (`False == 0.0` and `1 == 1.0` in Python).
    """
    return _kind(leaf) is _kind(slot_value) and _raw_eq(leaf, slot_value)


def assert_hir_matches_reference(
    hir: Hir,
    reference: Callable[..., object],
    vectors: Sequence[InputRow],
    *,
    label: str,
    ulps: int = 16,
) -> int:
    """Drive the ordered `vectors` through both sides; returns the number of compared transactions."""
    evaluator = HirEvaluator(hir)
    input_names = hir.input_names()
    parameter_names = expected_input_names(reference)
    assert input_names == parameter_names, f"{label}: input ports {input_names} != parameters {parameter_names}"
    out_ports = [out.name for out in hir.outputs]
    assert len(set(out_ports)) == len(out_ports), f"{label}: duplicate output port names in {out_ports}"
    public_slots = {name[len(_STATE_PREFIX) :] for name in out_ports if name.startswith(_STATE_PREFIX)}
    instance = getattr(reference, "__self__", None) if inspect.ismethod(reference) else None
    private_names = private_leaf_names(instance) if instance is not None else set()
    private_ported = sorted(slot for slot in public_slots if slot in private_names)
    assert not private_ported, f"{label}: state ports exposing private slots: {private_ported}"
    portless = [slot for slot in evaluator.state if slot not in private_names and slot not in public_slots]
    assert not portless, f"{label}: public slots without state_ ports: {portless}"
    bind = binder(reference)
    compared = 0
    for index, row in enumerate(vectors):
        context = f"{label}[{index}]"
        before = instance_leaves(instance) if instance is not None else {}
        args, kwargs = bind(row)
        try:
            result = reference(*args, **kwargs)
        except Exception:
            if instance is not None:
                break
            continue
        after = instance_leaves(instance) if instance is not None else {}
        return_leaves: dict[str, object] = (
            {} if result is None else {port_name(path): leaf for path, leaf in flatten_value(result)}
        )
        if _has_nan(return_leaves.values()) or _has_nan(after.get(slot) for slot in evaluator.state):
            if instance is not None:
                break
            continue

        state_before = evaluator.state
        try:
            outputs = evaluator.run(*[row[name] for name in input_names])
        except NoNumber as error:
            raise AssertionError(
                f"{context}: the evaluator names no number ({error}) where CPython succeeded"
            ) from None
        produced = dict(zip(out_ports, outputs, strict=True))

        for name, actual in produced.items():
            if name.startswith(_STATE_PREFIX):
                slot = name[len(_STATE_PREFIX) :]
                assert slot in evaluator.state, f"{context}: port {name} names no slot"
                assert slot in after, f"{context}: port {name} names no instance attribute leaf"
                _compare(f"{context}: {name}", actual, after[slot], ulps)
            else:
                assert name in return_leaves, f"{context}: port {name} has no return leaf among {sorted(return_leaves)}"
                _compare(f"{context}: {name}", actual, return_leaves[name], ulps)
        for leaf_name, leaf in return_leaves.items():
            if leaf_name not in produced:
                assert any(
                    _elidable_eq(leaf, after[slot]) for slot in public_slots
                ), f"{context}: return leaf {leaf_name} has no port and matches no public slot's value"

        state_after = evaluator.state
        for slot, actual in state_after.items():
            assert slot in after, f"{context}: slot {slot} names no instance attribute leaf"
            _compare(f"{context}: state {slot}", actual, after[slot], ulps)
        reference_changed = {name for name, value in after.items() if not _raw_eq(before[name], value)}
        evaluator_changed = {slot for slot, value in state_after.items() if not _raw_eq(state_before[slot], value)}
        assert evaluator_changed == reference_changed, (
            f"{context}: changed-slot sets diverge: "
            f"evaluator {sorted(evaluator_changed)} vs reference {sorted(reference_changed)}"
        )
        compared += 1
    assert compared > 0, f"{label}: no transaction survived to be compared"
    return compared

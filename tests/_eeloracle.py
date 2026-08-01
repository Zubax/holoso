"""
The front-end differential oracle: CPython executing the original kernel against ``HirEvaluator`` running the
unoptimized HIR lowered from it -- upstream of HIR optimization, so fastmath rewrites cannot muddy the verdict.
The harness is frontend-agnostic: it takes a lowered ``Hir`` and an identically constructed reference callable.

Protocol, per transaction of the ordered vector sequence:

- The reference runs first, bound signature-aware (the corpus mixes positional-only and keyword-only parameters).
  A vector on which the reference RAISES, or whose observable leaves -- the return leaves and the slot-named
  attribute leaves, not frozen attributes the kernel never lowers -- contain a NaN the compiler refuses to
  represent, lies outside the comparable domain and is discarded; a stateful kernel's remaining sequence goes with
  it, since the instance is no longer trustworthy. At least one transaction must survive comparison.
- The module interface itself is checked: input ports must mirror the reference's parameters name-for-name in
  order, output port names must be unique, every public slot must be observable through its ``state_<slot>`` port
  -- whose produced value is compared against the post-call attribute leaf, so a mis-wired port diverges from its
  own slot -- and no state port may expose a private slot.
- ``out_<path>`` ports are compared against the flattened return leaves by name (a ``None`` return has no leaves).
  Bools and ints are exact -- a bool port demands a genuine bool -- while a float port coerces the reference's
  ints to their float image (a float-typed counter written as ``0``; a C-promotion join rounding a large int) and
  matches to ``ulps`` ULPs with zeros and infinities exact: binary64 against binary64 in identical operation order
  leaves libm-level slack only.
- A return leaf with NO out port must exactly equal (reference against reference) some public slot's post-call
  value: the emitter elides a returned leaf that carries the same VALUE ID as a public state live-out, since that
  wire is already observable through the ``state_<slot>`` port. An emitter eliding an unrelated leaf fails here;
  an elision whose value coincides with a public slot by accident rather than by identity is the one blind spot,
  accepted because no information is lost through it.
- Every evaluator slot is compared against the instance-attribute leaf of the same name (row-major decomposition,
  the slot naming convention), and the exact SET of slots that changed must equal the exact set of attribute
  leaves that changed -- a missed state write and an invented one both surface.

Comparable-domain edges, accepted: the evaluator has no negative zero, so a kernel observing the sign of zero
(``atan2(-0.0, x)``) diverges; a NaN the reference computes and consumes WITHOUT surfacing it in any leaf (say, a
comparison against ``inf - inf``) is undetectable host-side and fails loudly for eye triage. An array kernel
parameter is driven through its decomposed scalar leaves: the vector row stays name -> value with the leaf names
(``v_0``, ``v_1``), the expected port set derives from the reference's own jaxtyping annotation (read
structurally, exactly as the frontend reads it), and the binder reassembles the ndarray argument for CPython.
"""

import inspect
import math
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

from holoso._eel._names import indexed_names
from holoso._hir import Hir, HirEvaluator, NoNumber

from ._modelref import flatten_value, port_name

type Scalar = float | bool | int
type InputRow = Mapping[str, Scalar]

_STATE_PREFIX = "state_"


def floats_match(actual: float, expected: float, ulps: int) -> bool:
    """Exact for zeros and infinities; otherwise within ``ulps`` ULPs of the larger magnitude."""
    if actual == expected:
        return True
    if math.isinf(actual) or math.isinf(expected) or actual == 0.0 or expected == 0.0:
        return False
    return abs(actual - expected) <= ulps * math.ulp(max(abs(actual), abs(expected)))


def _as_float(context: str, value: object) -> float:
    """
    An int meeting a float lane compares as its float image, rounding included: C-promotion at a join is a ratified
    type-system deviation, and a wrongly-floated int lane cannot escape the frontend's annotation validation.
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


def instance_leaves(instance: object) -> dict[str, object]:
    """Attribute leaves named exactly like decomposed state slots: scalars keep the name, arrays go row-major."""
    leaves: dict[str, object] = {}
    for attr, value in vars(instance).items():
        if isinstance(value, (list, tuple, np.ndarray)):
            for path, leaf in flatten_value(value):
                leaves[attr + "".join(f"_{key}" for key in path)] = leaf
        else:
            leaves[attr] = value
    return leaves


def _annotation_shape(annotation: object) -> tuple[int, ...] | None:
    """A jaxtyping-style fixed shape, read structurally like the frontend reads it; None for scalars."""
    dims = getattr(annotation, "dims", None) if isinstance(annotation, type) else None
    if not isinstance(dims, tuple):
        return None
    sizes = [getattr(dim, "size", None) for dim in dims]
    assert all(isinstance(size, int) for size in sizes), f"non-static array annotation {annotation!r}"
    return tuple(size for size in sizes if isinstance(size, int))


def _parameter_shapes(reference: Callable[..., object]) -> dict[str, tuple[int, ...] | None]:
    annotations = inspect.get_annotations(reference)
    return {name: _annotation_shape(annotations.get(name)) for name in inspect.signature(reference).parameters}


def expected_input_names(reference: Callable[..., object]) -> list[str]:
    shapes = _parameter_shapes(reference)
    names: list[str] = []
    for name, shape in shapes.items():
        names.extend([name] if shape is None else indexed_names(name, shape))
    assert len(set(names)) == len(names), f"duplicate decomposed input names in {names}"
    return names


def _binder(reference: Callable[..., object]) -> Callable[[InputRow], tuple[list[object], dict[str, object]]]:
    """Row lookups happen outside the discard-guarded call, so a typo'd vector key crashes instead of discarding."""
    parameters = list(inspect.signature(reference).parameters.values())
    kinds = {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    assert all(parameter.kind in kinds for parameter in parameters), f"unsupported signature {parameters}"
    shapes = _parameter_shapes(reference)

    def value_of(name: str, row: InputRow) -> object:
        shape = shapes[name]
        if shape is None:
            return row[name]
        return np.array([row[leaf] for leaf in indexed_names(name, shape)]).reshape(shape)

    def bind(row: InputRow) -> tuple[list[object], dict[str, object]]:
        args = [value_of(p.name, row) for p in parameters if p.kind is not inspect.Parameter.KEYWORD_ONLY]
        kwargs = {p.name: value_of(p.name, row) for p in parameters if p.kind is inspect.Parameter.KEYWORD_ONLY}
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
    or int leaf could hide behind an equal-comparing float slot (``False == 0.0`` and ``1 == 1.0`` in Python).
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
    """Drive the ordered ``vectors`` through both sides; returns the number of compared transactions."""
    evaluator = HirEvaluator(hir)
    input_names = hir.input_names()
    parameter_names = expected_input_names(reference)
    assert input_names == parameter_names, f"{label}: input ports {input_names} != parameters {parameter_names}"
    out_ports = [out.name for out in hir.outputs]
    assert len(set(out_ports)) == len(out_ports), f"{label}: duplicate output port names in {out_ports}"
    public_slots = {name[len(_STATE_PREFIX) :] for name in out_ports if name.startswith(_STATE_PREFIX)}
    private_ported = sorted(slot for slot in public_slots if slot.startswith("_"))
    assert not private_ported, f"{label}: state ports exposing private slots: {private_ported}"
    portless = [slot for slot in evaluator.state if not slot.startswith("_") and slot not in public_slots]
    assert not portless, f"{label}: public slots without state_ ports: {portless}"
    instance = getattr(reference, "__self__", None) if inspect.ismethod(reference) else None
    bind = _binder(reference)
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

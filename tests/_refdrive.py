"""
An example's numerical model and its plain-Python reference, driven in lockstep over one input sequence.

Shared so the behavioral suite and the accuracy freeze see one pairing: the freeze measures exactly what the
tolerance check permits, and no lane is guarded by one while invisible to the other.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

import holoso
from holoso import BoolType, FloatFormat, FloatType, IntType

from ._eeloracle import binder, walk_instance_leaves
from ._examples import InputVector
from ._modelref import flatten_value, port_name

_STATE_PREFIX = "state_"


@dataclass(frozen=True, slots=True)
class _Lane:
    """Where in the sequence a comparison came from; the two sides are the family's own, below."""

    age: int
    port: str


@dataclass(frozen=True, slots=True)
class BoolLane(_Lane):
    got: bool
    want: bool


@dataclass(frozen=True, slots=True)
class IntLane(_Lane):
    got: holoso.IntValue
    want: int


@dataclass(frozen=True, slots=True)
class FloatLane(_Lane):
    got: holoso.FloatValue
    want: float


type Comparison = BoolLane | IntLane | FloatLane
"""One output port at one transaction: what the model answered against what plain Python did, in the port's own
declared family, so a reader narrows by matching rather than by asking a tag and asserting the payload."""


def _quantize(value: float | bool | int, fmt: FloatFormat) -> float | bool | int:
    """A float rounded into `fmt`, so the model and the float64 reference get one operand; a bool/int is exact."""
    return value if isinstance(value, (bool, int)) else fmt.decode(fmt.encode(value))


def _lane(where: str, scalar_type: object, age: int, port: str, got: object, want: object) -> Comparison:
    """
    The comparison in the family the PORT declares. Both sides must already carry it, so a bool answered where a
    float was declared is caught here rather than hiding behind a comparison that would coerce it.
    """
    sides = f"{where} {port}: model {got!r} and reference {want!r} against the declared {scalar_type}"
    match scalar_type:
        case BoolType():
            assert isinstance(got, (bool, np.bool_)) and isinstance(want, (bool, np.bool_)), sides
            return BoolLane(age, port, bool(got), bool(want))
        case IntType():
            # Python's `bool` IS an `int` to `isinstance`, so a boolean reference on an integer port would convert
            # to 0 or 1 and pass unnoticed; numpy's owns no such inheritance and needs no exclusion.
            assert isinstance(got, holoso.IntValue), sides
            assert isinstance(want, (int, np.integer)) and not isinstance(want, bool), sides
            return IntLane(age, port, got, int(want))
        case FloatType():
            assert isinstance(got, holoso.FloatValue) and isinstance(want, (float, np.floating)), sides
            return FloatLane(age, port, got, float(want))
    raise AssertionError(scalar_type)


def _state_leaves(instance: object) -> dict[str, Any]:
    """
    Public attribute leaves named exactly as the compiler decomposes state slots (a scalar keeps its name, an
    aggregate flattens row-major, a nested component prefixes its own leaves) -- the forward direction of
    `_eeloracle.walk_instance_leaves`. The name alone cannot prove which attribute a port belongs to (aggregate
    `q` and scalar `q_0` both mint `q_0`), so a name minted twice is dropped and its lookup fails loudly instead
    of comparing against a guess.
    """
    leaves: dict[str, Any] = {}
    ambiguous: set[str] = set()
    for name, private, leaf in walk_instance_leaves(instance):
        if private:
            continue
        if name in leaves:
            ambiguous.add(name)
        leaves[name] = leaf
    return {name: leaf for name, leaf in leaves.items() if name not in ambiguous}


def drive(
    label: str,
    model: holoso.NumericalSimulator,
    reference: Callable[..., object],
    inputs: tuple[str, ...],
    vectors: list[InputVector],
    fmt: FloatFormat,
) -> Iterator[Comparison]:
    """
    Outputs pair BY NAME: a returned leaf maps to its `out_<path>` port via `port_name`, a `state_<attr>` port
    reads the reference instance's attribute, and a returned leaf without a port must equal a public state
    live-out the model folded into its `state_` port. Both sides must carry the port's declared family before any
    coercion, so a bool/float swap cannot hide behind a comparison.

    The structural checks are asserted here, being properties of the pairing itself; what to make of the VALUES is
    the caller's, which is what lets the freeze and the tolerance check share one traversal.
    """
    instance = getattr(reference, "__self__", None)
    assert [port.name for port in model.inputs] == list(inputs), f"{label}: input ports differ from the spec"
    port_names = {port.name for port in model.outputs}
    bind = binder(reference)
    for age, row in enumerate(vectors):
        quantized = {name: _quantize(value, fmt) for name, value in row.items()}
        got = model.run(*[quantized[port.name] for port in model.inputs])
        args, kwargs = bind(quantized)
        result = reference(*args, **kwargs)
        return_leaves = {} if result is None else {port_name(path): leaf for path, leaf in flatten_value(result)}
        state_leaves = {} if instance is None else _state_leaves(instance)
        expected: dict[str, float | bool | int] = {}
        for port in model.outputs:
            if port.name.startswith(_STATE_PREFIX):
                name = port.name[len(_STATE_PREFIX) :]
                assert name in state_leaves, f"{label}: port {port.name} matches no unambiguous public state leaf"
                expected[port.name] = state_leaves[name]
            else:
                assert port.name in return_leaves, f"{label}: port {port.name} has no leaf in {sorted(return_leaves)}"
                expected[port.name] = return_leaves[port.name]
        state_live_outs = [value for name, value in expected.items() if name.startswith(_STATE_PREFIX)]
        for leaf_name, leaf in return_leaves.items():
            if leaf_name not in port_names:
                assert any(
                    isinstance(leaf, bool) == isinstance(value, bool) and leaf == value for value in state_live_outs
                ), f"{label} {row}: return leaf {leaf_name} has no port and matches no public state live-out"
        for port, got_value in zip(model.outputs, got, strict=True):
            yield _lane(f"{label} {row}", port.scalar_type, age, port.name, got_value, expected[port.name])

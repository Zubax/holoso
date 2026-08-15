"""
Frozen-record support: record parameters decomposing to field-path ports, in-kernel structural construction,
record returns flattening to field-path outputs, joins, ownership through record fields, and the admission
gate. Behavior is checked black-box through the public API against the same kernel running under CPython;
the one white-box test covers extraction sharing that field-read sharing masks from any observable store.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from jaxtyping import Float64

import holoso
from holoso import FFromIntOptions, FloatFormat, UnsupportedConstruct

from ._modelref import default_options

_FMT = FloatFormat(11, 52)


@dataclass(frozen=True)
class Kinematics:
    pos: float
    vel: float
    accel: float


@dataclass(frozen=True, slots=True)
class Decision:
    switch: tuple[bool, bool, bool]
    balance: Float64[np.ndarray, "3 1"]
    score: float = 0.0


@dataclass(frozen=True)
class Nested:
    kin: Kinematics
    weight: float


@dataclass(frozen=True)
class Recursive:
    weight: float
    inner: "Recursive"


@dataclass(frozen=True)
class Siblings:
    left: Kinematics
    right: Kinematics


def _synth(fn: Callable[..., object]) -> holoso.SynthesisResult:
    return holoso.synthesize(fn, default_options(_FMT), name="kernel")


def _refused(fn: Callable[..., object], match: str) -> None:
    with pytest.raises(UnsupportedConstruct, match=match):
        _synth(fn)


def test_record_parameter_decomposes_to_field_ports() -> None:
    def kernel(kin: Kinematics, gain: float) -> float:
        return kin.pos * gain + kin.vel - kin.accel

    sim = _synth(kernel).numerical_model.elaborate()
    assert [p.name for p in sim.inputs] == ["kin_pos", "kin_vel", "kin_accel", "gain"]
    for args in ((2.0, -1.5, 0.25, 3.0), (0.0, 1.0, -1.0, 0.5)):
        (got,) = sim.run(*args)
        assert float(got) == kernel(Kinematics(*args[:3]), args[3])


def test_record_return_flattens_to_field_ports() -> None:
    def kernel(kin: Kinematics, gain: float) -> Decision:
        drive = kin.pos * gain + kin.vel
        flags = (drive > 0.0, kin.vel > 0.0, False)
        return Decision(switch=flags, balance=np.array([[drive], [kin.accel], [0.0]]))

    sim = _synth(kernel).numerical_model.elaborate()
    assert [p.name for p in sim.outputs] == [
        "out_switch_0",
        "out_switch_1",
        "out_switch_2",
        "out_balance_0_0",
        "out_balance_1_0",
        "out_balance_2_0",
        "out_score",
    ]
    args = (2.0, -1.5, 0.25, 3.0)
    want = kernel(Kinematics(*args[:3]), args[3])
    got = sim.run(*args)
    assert list(got[:3]) == list(want.switch)
    assert [float(v) for v in got[3:6]] == list(want.balance.flatten())
    assert float(got[6]) == want.score  # the defaulted field


def test_nested_record_through_parameter_and_return() -> None:
    def kernel(n: Nested) -> Nested:
        return Nested(kin=Kinematics(n.kin.pos * n.weight, n.kin.vel, 0.0), weight=n.weight + 1.0)

    sim = _synth(kernel).numerical_model.elaborate()
    assert [p.name for p in sim.inputs] == ["n_kin_pos", "n_kin_vel", "n_kin_accel", "n_weight"]
    assert [p.name for p in sim.outputs] == ["out_kin_pos", "out_kin_vel", "out_kin_accel", "out_weight"]
    want = kernel(Nested(Kinematics(2.0, -1.0, 5.0), 3.0))
    assert [float(v) for v in sim.run(2.0, -1.0, 5.0, 3.0)] == [
        want.kin.pos,
        want.kin.vel,
        want.kin.accel,
        want.weight,
    ]


def test_record_argument_to_an_inlined_helper() -> None:
    def scale(kin: Kinematics, k: float) -> float:
        return kin.pos * k

    def kernel(kin: Kinematics) -> float:
        return scale(kin, 2.0) + kin.vel

    sim = _synth(kernel).numerical_model.elaborate()
    (got,) = sim.run(1.5, -0.5, 0.0)
    assert float(got) == kernel(Kinematics(1.5, -0.5, 0.0))


def test_record_construction_conforms_int_into_a_float_field() -> None:
    def kernel(n: int) -> Kinematics:
        return Kinematics(pos=n, vel=n + 1, accel=0.5)

    base = default_options(_FMT)
    options = dataclasses.replace(base, operator=dataclasses.replace(base.operator, ffromint=FFromIntOptions()))
    sim = holoso.synthesize(kernel, options, name="kernel").numerical_model.elaborate()
    assert [float(v) for v in sim.run(3)] == [3.0, 4.0, 0.5]


def test_records_join_across_runtime_branches() -> None:
    def kernel(x: float) -> Kinematics:
        if x > 0.0:
            found = Kinematics(x, 1.0, 0.0)
        else:
            found = Kinematics(-x, -1.0, 0.0)
        return found

    sim = _synth(kernel).numerical_model.elaborate()
    for x in (2.5, -2.5):
        want = kernel(x)
        assert [float(v) for v in sim.run(x)] == [want.pos, want.vel, want.accel]


def test_record_returned_across_a_residual_loop() -> None:
    def helper(x: float) -> Kinematics:
        while x < 8.0:
            x = x * 2.0
        return Kinematics(x, 0.0, 0.0)

    def kernel(x: float) -> float:
        return helper(x).pos

    sim = _synth(kernel).numerical_model.elaborate()
    for x in (1.0, 3.0, 9.0):
        (got,) = sim.run(x)
        assert float(got) == kernel(x)


def test_kw_only_fields_bind_by_name_and_flatten_in_field_order() -> None:
    @dataclass(frozen=True, kw_only=True)
    class Pair:
        first: float
        second: float

    def kernel(x: float) -> Pair:
        return Pair(second=x, first=x + 1.0)

    sim = _synth(kernel).numerical_model.elaborate()
    assert [p.name for p in sim.outputs] == ["out_first", "out_second"]
    assert [float(v) for v in sim.run(2.0)] == [3.0, 2.0]


def test_captured_record_instances_fold_and_convert() -> None:
    cfg = Kinematics(2.0, 0.5, -1.0)

    def helper(kin: Kinematics) -> float:
        return kin.pos + kin.vel

    def folding(x: float) -> float:  # a captured instance passed to an annotated helper stays fold-friendly
        return helper(cfg) * x

    sim = _synth(folding).numerical_model.elaborate()
    (got,) = sim.run(2.0)
    assert float(got) == folding(2.0)

    def returned(x: float) -> Kinematics:  # a returned captured instance converts and flattens
        return cfg

    sim2 = _synth(returned).numerical_model.elaborate()
    assert [float(v) for v in sim2.run(0.0)] == [2.0, 0.5, -1.0]

    def defaulted(x: float) -> float:  # a captured instance as a construction argument admits structurally
        return Nested(weight=x, kin=cfg).kin.pos

    sim3 = _synth(defaulted).numerical_model.elaborate()
    (got3,) = sim3.run(1.0)
    assert float(got3) == 2.0


def test_tuple_parameters_decompose_through_the_shared_walk() -> None:
    @dataclass(frozen=True)
    class Tagged:
        flags: tuple[bool, bool]
        gain: float

    def kernel(pair: tuple[float, float], t: Tagged) -> float:
        base = pair[0] * t.gain + pair[1]
        return base if t.flags[0] else -base

    sim = _synth(kernel).numerical_model.elaborate()
    assert [p.name for p in sim.inputs] == ["pair_0", "pair_1", "t_flags_0", "t_flags_1", "t_gain"]
    for args in ((1.5, -0.5, True, False, 2.0), (1.5, -0.5, False, True, 2.0)):
        (got,) = sim.run(*args)
        assert float(got) == kernel((args[0], args[1]), Tagged((args[2], args[3]), args[4]))


def test_same_typed_sibling_record_fields_are_not_a_cycle() -> None:
    def kernel(s: Siblings) -> float:
        return s.left.pos + s.right.vel

    sim = _synth(kernel).numerical_model.elaborate()
    assert len(sim.inputs) == 6
    (got,) = sim.run(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert float(got) == 1.0 + 5.0


def test_frozen_instance_attribute_folds_with_no_state_slots() -> None:
    class Filter:
        def __init__(self) -> None:
            self.cfg = Kinematics(4.0, 0.0, 0.0)
            self.acc = 0.0

        def step(self, x: float) -> float:
            self.acc = self.acc + x * self.cfg.pos
            return self.acc

    result = holoso.synthesize(Filter().step, default_options(_FMT), name="kernel")
    sim = result.numerical_model.elaborate()
    assert [p.name for p in sim.inputs] == ["x"]  # cfg contributed no ports and no slots
    assert float(sim.run(1.5)[0]) == 6.0
    assert float(sim.run(1.0)[0]) == 10.0


def test_record_misuse_draws_located_refusals() -> None:
    def arithmetic(kin: Kinematics) -> float:
        return kin + 1.0  # type: ignore[operator]

    _refused(arithmetic, "a record cannot be used as a scalar")

    def truthy(kin: Kinematics) -> float:
        if kin:
            return 1.0
        return 0.0

    _refused(truthy, "truthiness")

    def field_write(kin: Kinematics) -> float:
        kin.pos = 1.0  # type: ignore[misc]
        return kin.pos

    _refused(field_write, "a record is immutable")

    class Holder:
        def __init__(self) -> None:
            self.slot = 0.0

        def step(self, x: float) -> float:
            self.slot = Kinematics(x, x, x)  # type: ignore[assignment]
            return x

    with pytest.raises(UnsupportedConstruct, match="record cannot be installed"):
        holoso.synthesize(Holder().step, default_options(_FMT), name="kernel")


def test_record_admission_gate() -> None:
    @dataclass
    class Unfrozen:
        x: float

    def unfrozen(x: float) -> float:
        return Unfrozen(x).x

    _refused(unfrozen, "not frozen")

    @dataclass(frozen=True)
    class UserInit:
        x: float

        def __init__(self, x: float) -> None:
            object.__setattr__(self, "x", x + 1.0)

    def user_init(x: float) -> float:
        return UserInit(x).x

    _refused(user_init, "its own __init__")

    @dataclass(frozen=True)
    class PostInit:
        x: float

        def __post_init__(self) -> None:
            pass

    def post_init(x: float) -> float:
        return PostInit(x).x

    _refused(post_init, "__post_init__")

    @dataclass(frozen=True)
    class Factory:
        x: float = dataclasses.field(default_factory=float)

    def factory(x: float) -> float:
        return Factory().x

    _refused(factory, "default_factory")

    @dataclass(frozen=True, init=False)
    class NoInitChild(Kinematics):
        pass

    def no_init(x: float) -> float:
        return NoInitChild(x, x, x).pos

    _refused(no_init, "init=False")

    def recursive(r: Recursive) -> float:
        return r.weight

    _refused(recursive, "recursive")


def test_record_ownership() -> None:
    def embed(x: float) -> float:
        arr = np.array([x, x])
        rec = Decision(switch=(True, True, True), balance=np.array([[x], [x], [x]]))
        held = rec.balance
        arr[0] = 1.0  # arr is untouched by the record: still admitted
        held2 = held  # noqa: F841
        return float(rec.balance[0, 0]) + float(arr[0])

    _synth(embed)

    def field_store(x: float) -> float:
        rec = Decision(switch=(True, True, True), balance=np.array([[x], [x], [x]]))
        held = rec.balance
        held[0, 0] = 1.0  # the field read is a second handle of the record's storage
        return float(rec.balance[0, 0])

    _refused(field_store, "shared")

    class Balanced:
        def __init__(self) -> None:
            self.bal = np.zeros(3)

        def step(self, x: float) -> Decision:
            self.bal = self.bal + np.array([x, -x, 0.0])
            return Decision(switch=(True, False, False), balance=self.bal.reshape((3, 1)))

    with pytest.raises(UnsupportedConstruct, match="live alias"):
        holoso.synthesize(Balanced().step, default_options(_FMT), name="kernel")


def test_extraction_of_a_record_from_a_sequence_shares_it() -> None:
    # White-box: field-read sharing masks the extraction share from every observable store, so the allocation
    # states are asserted directly.
    from holoso._eel._pe._values import AllocationState, RecordValue, SequenceValue
    from holoso._eel._pe._aggregate import index_read, slice_read, splice_items
    from holoso._eel._pe._values import Allocation, StaticScalar
    from holoso._eel._pe._ops import make_const
    from holoso._eel._ir import Origin

    def fresh() -> SequenceValue:
        record = RecordValue(Kinematics, (StaticScalar(make_const(1.0)),) * 3, Allocation())
        return SequenceValue((record,), Allocation())

    from holoso._errors import SourceLocation

    origin = Origin(SourceLocation("test", 0, 0), ())
    seq = fresh()
    extracted = index_read(origin, seq, StaticScalar(make_const(0)))
    assert isinstance(extracted, RecordValue) and extracted.allocation.state is AllocationState.SHARED

    seq = fresh()
    sliced = slice_read(origin, seq, None, None)
    assert isinstance(sliced, SequenceValue)
    assert all(
        item.allocation.state is AllocationState.SHARED for item in sliced.items if isinstance(item, RecordValue)
    )

    seq = fresh()
    items = splice_items(origin, seq)
    assert all(item.allocation.state is AllocationState.SHARED for item in items if isinstance(item, RecordValue))


def test_output_name_collisions_are_refused() -> None:
    @dataclass(frozen=True)
    class Inner:
        b: float

    @dataclass(frozen=True)
    class Colliding:
        a_b: float
        a: Inner

    def collides(x: float) -> Colliding:
        return Colliding(a_b=x, a=Inner(b=x))

    _refused(collides, "same port")

    @dataclass(frozen=True)
    class Reserved:
        valid: bool

    def reserved(x: float) -> Reserved:
        return Reserved(valid=x > 0.0)

    _refused(reserved, "reserved port")


def test_a_captured_subclass_is_not_projected_to_the_annotated_base() -> None:
    # CPython returns the Child with both fields; a base-shaped module would silently drop `y`.
    @dataclass(frozen=True)
    class Child(Kinematics):
        extra: float

    child = Child(1.0, 2.0, 3.0, 4.0)

    def kernel(x: float) -> Kinematics:
        return child

    _refused(kernel, "subclass")


def test_captured_record_instances_join_across_runtime_branches() -> None:
    left = Kinematics(1.0, 2.0, 3.0)
    right = Kinematics(4.0, 5.0, 6.0)

    def kernel(flag: bool) -> Kinematics:
        if flag:
            value = left
        else:
            value = right
        return value

    sim = _synth(kernel).numerical_model.elaborate()
    assert [float(v) for v in sim.run(True)] == [1.0, 2.0, 3.0]
    assert [float(v) for v in sim.run(False)] == [4.0, 5.0, 6.0]

    def mixed(x: float) -> Kinematics:  # a captured instance joining a constructed record of the same class
        if x > 0.0:
            value = left
        else:
            value = Kinematics(x, x + 1.0, 0.0)
        return value

    sim2 = _synth(mixed).numerical_model.elaborate()
    assert [float(v) for v in sim2.run(1.0)] == [1.0, 2.0, 3.0]
    assert [float(v) for v in sim2.run(-2.0)] == [-2.0, -1.0, 0.0]

    third = Kinematics(7.0, 8.0, 9.0)

    def three_arms(x: float) -> Kinematics:  # the join is closed over its own record result
        if x > 1.0:
            value = left
        elif x > 0.0:
            value = right
        else:
            value = third
        return value

    sim3 = _synth(three_arms).numerical_model.elaborate()
    assert [float(v) for v in sim3.run(2.0)] == [1.0, 2.0, 3.0]
    assert [float(v) for v in sim3.run(0.5)] == [4.0, 5.0, 6.0]
    assert [float(v) for v in sim3.run(-1.0)] == [7.0, 8.0, 9.0]


@dataclass(frozen=True)
class Labeled:
    label: str  # inadmissible as a value, so admission at a join must poison lazily, not refuse eagerly
    gain: float


def test_an_unreadable_captured_record_join_poisons_lazily() -> None:
    left = Labeled("a", 1.0)
    right = Labeled("b", 2.0)

    def unread(flag: bool, x: float) -> float:
        if flag:
            value = left
        else:
            value = right
        return x

    sim = _synth(unread).numerical_model.elaborate()
    assert float(sim.run(True, 5.0)[0]) == 5.0

    def read(flag: bool) -> float:
        if flag:
            value = left
        else:
            value = right
        return value.gain

    _refused(read, "cannot merge")


def test_per_field_kw_only_records_construct() -> None:
    @dataclass(frozen=True)
    class Mixed:
        first: float = dataclasses.field(kw_only=True)  # the generated signature reorders; binding is by name
        second: float = 0.0

    def kernel(x: float) -> Mixed:
        return Mixed(x, first=x + 1.0)

    sim = _synth(kernel).numerical_model.elaborate()
    assert [p.name for p in sim.outputs] == ["out_first", "out_second"]
    assert [float(v) for v in sim.run(2.0)] == [3.0, 2.0]


def test_void_callee_with_a_record_annotation_is_refused() -> None:
    class Machine:
        def __init__(self) -> None:
            self.acc = 0.0

        def helper(self, x: float) -> Kinematics:  # type: ignore[return]
            self.acc = self.acc + x

        def step(self, x: float) -> float:
            self.helper(x)
            return self.acc

    with pytest.raises(UnsupportedConstruct, match="returns no value"):
        holoso.synthesize(Machine().step, default_options(_FMT), name="kernel")

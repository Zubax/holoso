"""
Acceptance gate + independence guard for the HIR evaluator (``holoso._hir.HirEvaluator``) and the front-end
differential-oracle harness (``tests/_eeloracle.py``).

Before the evaluator can indict a frontend, it must agree with CPython on kernels already known correct: every
bundled example (validated end-to-end by the reference/cosim suites) plus the scheduling corner kernels are lowered
through the front end -- the evaluator is frontend-agnostic -- and driven through full shared vector sequences.
Hand-built builder graphs pin the semantics no lowered kernel reaches: the poison family, the integer vocabulary,
parallel phi snapshots, state carry/reset/commit atomicity, and the runaway bound.
"""

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest

from holoso._eel import lower
from holoso._hir import (
    BoolAnd,
    BoolOr,
    FloatAdd,
    FloatCeil,
    FloatConst,
    FloatDiv,
    FloatExp2,
    FloatFloor,
    FloatLog2,
    FloatMul,
    FloatNeg,
    FloatGreater,
    FloatRound,
    FloatTrunc,
    FloatType,
    Hir,
    HirBuilder,
    HirEvaluator,
    IntAdd,
    IntConst,
    IntDivFloor,
    IntShiftLeft,
    IntToFloat,
    IntType,
    NoNumber,
)

from ._eeloracle import assert_hir_matches_reference
from ._examples import SPECS, ExampleSpec
from ._importguard import forbidden_imports
from ._modelref import (
    ChainedSlots,
    SelectHold,
    SlotSwap,
    bool_phi_swap_computed_loop,
    branch_boundary_kernel,
    branchy_swap_mixed_arm_loop,
    diamond_then_loop_kernel,
    phi_swap_computed_loop,
    phi_swap_loop,
)


def test_evaluator_layering() -> None:
    for forbidden in ("holoso._mir", "holoso._lir", "holoso._eel", "holoso._backend"):
        assert forbidden_imports("holoso._hir._evaluate", forbidden) == []


def _assert_oracle(
    make_kernel: Callable[[], Callable[..., object]], vectors: Sequence[Mapping[str, float | bool]], label: str
) -> None:
    hir = lower(make_kernel()).hir
    compared = assert_hir_matches_reference(hir, make_kernel(), vectors, label=label)
    assert compared == len(vectors)


@pytest.mark.parametrize("spec", SPECS, ids=[spec.name for spec in SPECS])
def test_oracle_on_examples(spec: ExampleSpec) -> None:
    _assert_oracle(spec.make_kernel, spec.reference_vectors(), spec.name)


_CORNERS: list[tuple[str, Callable[[], Callable[..., object]], list[dict[str, float | bool]]]] = [
    (
        "phi_swap",
        lambda: phi_swap_loop,
        [{"x": 1.5, "n": 5.0}, {"x": -2.0, "n": 0.0}, {"x": 0.25, "n": 3.0}],
    ),
    (
        "phi_swap_computed",
        lambda: phi_swap_computed_loop,
        [{"x": 1.0, "n": 4.0}, {"x": -0.5, "n": 7.0}, {"x": 2.0, "n": 1.0}],
    ),
    (
        "bool_phi_swap_computed",
        lambda: bool_phi_swap_computed_loop,
        [{"x": True, "n": 4.0}, {"x": False, "n": 3.0}, {"x": True, "n": 0.0}],
    ),
    (
        "branchy_swap_mixed_arm",
        lambda: branchy_swap_mixed_arm_loop,
        [{"x": 1.0, "d": 2.0, "n": 4.0}, {"x": -1.0, "d": 0.5, "n": 3.0}, {"x": 0.5, "d": 4.0, "n": 0.0}],
    ),
    (
        "branch_boundary",
        lambda: branch_boundary_kernel,
        [{"a": 1.0, "b": 2.0, "c": 3.0}, {"a": -1.0, "b": 0.5, "c": 2.0}, {"a": 0.0, "b": 0.0, "c": 0.0}],
    ),
    (
        "diamond_then_loop",
        lambda: diamond_then_loop_kernel,
        [{"x": 3.0, "y": 1.5}, {"x": 1.0, "y": 4.0}, {"x": 2.0, "y": 2.0}],
    ),
    (
        "slot_swap",
        lambda: SlotSwap().step,
        [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": -1.0}],
    ),
    (
        "chained_slots",
        lambda: ChainedSlots().__call__,
        [{"x": 1.0}, {"x": 2.0}, {"x": 0.75}, {"x": -3.0}],
    ),
    (
        "select_hold",
        lambda: SelectHold().step,
        [{"x": 1.0, "c": 1.0}, {"x": 2.0, "c": -1.0}, {"x": 0.5, "c": 0.0}],
    ),
]


@pytest.mark.parametrize("label,make_kernel,vectors", _CORNERS, ids=[label for label, _, _ in _CORNERS])
def test_oracle_on_corner_kernels(
    label: str, make_kernel: Callable[[], Callable[..., object]], vectors: list[dict[str, float | bool]]
) -> None:
    _assert_oracle(make_kernel, vectors, label)


def _straight_line() -> Hir:
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", FloatType())
    y = builder.input("y", FloatType())
    product = builder.operation(FloatMul(), [x, y])
    builder.output("out_0", builder.operation(FloatAdd(), [product, builder.float_const(1.0)]))
    builder.ret()
    return builder.finish()


def test_straight_line_arithmetic() -> None:
    evaluator = HirEvaluator(_straight_line())
    assert evaluator.run(2.0, 3.0) == [7.0]
    assert evaluator.run(-1.0, 0.5) == [0.5]


def test_operator_reference_poles() -> None:
    """
    Each evaluate follows its own registered np reference at the poles, so a static fold answers the same
    value the RTL and the stub reference produce instead of refusing.
    """
    inf = math.inf
    assert FloatLog2().evaluate([FloatConst(0.0)]) == FloatConst(-inf)
    with pytest.raises(NoNumber):
        FloatLog2().evaluate([FloatConst(-1.0)])
    for operator in (FloatFloor(), FloatCeil(), FloatTrunc(), FloatRound()):
        assert operator.evaluate([FloatConst(inf)]) == FloatConst(inf)
        assert operator.evaluate([FloatConst(-inf)]) == FloatConst(-inf)
    assert FloatRound().evaluate([FloatConst(2.5)]) == FloatConst(2.0)
    assert FloatRound().evaluate([FloatConst(3.5)]) == FloatConst(4.0)


def test_folds_are_immune_to_the_ambient_numpy_error_state() -> None:
    # The compiler runs in-process with user code, and np.seterr(all="raise") is a common defensive setting
    # there; a fold must answer the same values regardless and leak neither warnings nor FloatingPointError.
    with np.errstate(all="raise"):
        assert FloatLog2().evaluate([FloatConst(0.0)]) == FloatConst(-math.inf)
        with pytest.raises(NoNumber):
            FloatLog2().evaluate([FloatConst(-1.0)])
        assert FloatExp2().evaluate([FloatConst(1e30)]) == FloatConst(math.inf)


def test_input_validation() -> None:
    evaluator = HirEvaluator(_straight_line())
    with pytest.raises(ValueError, match="expected 2 inputs"):
        evaluator.run(1.0)
    with pytest.raises(TypeError, match="input 0"):
        evaluator.run(True, 1.0)
    with pytest.raises(TypeError, match="input 1"):
        evaluator.run(1.0, 1)


def test_swap_loop_phis_resolve_in_parallel() -> None:
    builder = HirBuilder()
    entry = builder.block()
    header = builder.block()
    body = builder.block()
    exit_ = builder.block()
    n = builder.input("n", FloatType())
    first = builder.float_const(1.0)
    second = builder.float_const(2.0)
    builder.position_at(entry)
    builder.jump(header)
    builder.position_at(header)
    i = builder.open_phi(FloatType(), (entry, n))
    a = builder.open_phi(FloatType(), (entry, first))
    b = builder.open_phi(FloatType(), (entry, second))
    builder.branch(builder.operation(FloatGreater(), [i, builder.float_const(0.0)]), body, exit_)
    builder.position_at(body)
    dec = builder.operation(FloatAdd(), [i, builder.float_const(-1.0)])
    builder.jump(header)
    builder.set_phi_arms(i, [(entry, n), (body, dec)])
    builder.set_phi_arms(a, [(entry, first), (body, b)])
    builder.set_phi_arms(b, [(entry, second), (body, a)])
    builder.position_at(exit_)
    builder.output("out_0", a)
    builder.output("out_1", b)
    builder.ret()
    evaluator = HirEvaluator(builder.finish())
    assert evaluator.run(0.0) == [1.0, 2.0]
    assert evaluator.run(2.0) == [1.0, 2.0]
    assert evaluator.run(3.0) == [2.0, 1.0]


def test_state_carry_and_reset() -> None:
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", FloatType())
    total = builder.operation(FloatAdd(), [builder.float_state_read("acc"), x])
    builder.state_slot("acc", FloatConst(0.0), total)
    builder.output("state_acc", total)
    builder.ret()
    evaluator = HirEvaluator(builder.finish())
    assert evaluator.run(1.5) == [1.5]
    assert evaluator.run(2.0) == [3.5]
    assert evaluator.state == {"acc": 3.5}
    evaluator.reset()
    assert evaluator.state == {"acc": 0.0}
    assert evaluator.run(0.25) == [0.25]


def test_int_vocabulary() -> None:
    builder = HirBuilder()
    builder.block()
    n = builder.input("n", IntType())
    d = builder.input("d", IntType())
    builder.output("out_0", builder.operation(IntDivFloor(), [n, d]))
    builder.output("out_1", builder.operation(IntShiftLeft(), [n, builder.int_const(3)]))
    builder.ret()
    evaluator = HirEvaluator(builder.finish())
    assert evaluator.run(7, 2) == [3, 56]
    assert evaluator.run(-7, 2) == [-4, -56]
    with pytest.raises(NoNumber, match="out_0"):
        evaluator.run(1, 0)


def test_int_state_and_huge_cast() -> None:
    builder = HirBuilder()
    builder.block()
    count = builder.operation(IntAdd(), [builder.state_read("count", IntType()), builder.int_const(1)])
    builder.state_slot("count", IntConst(0), count)
    builder.output("out_0", builder.operation(IntToFloat(), [builder.int_const(10**400)]))
    builder.ret()
    evaluator = HirEvaluator(builder.finish())
    with pytest.raises(NoNumber, match="out_0"):
        evaluator.run()
    assert evaluator.state == {"count": 0}, "a failed transaction must not advance state"


def test_poison_dead_operation_is_harmless() -> None:
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", FloatType())
    builder.operation(FloatDiv(), [builder.float_const(1.0), builder.float_const(0.0)])
    builder.output("out_0", builder.operation(FloatAdd(), [x, builder.float_const(1.0)]))
    builder.ret()
    assert HirEvaluator(builder.finish()).run(1.0) == [2.0]


def _gated_poison(gate_value: bool) -> Hir:
    """``gate and (1.0/x > 0.0)`` as the eager frontends spell it; poison appears whenever x == 0."""
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", FloatType())
    quotient = builder.operation(FloatDiv(), [builder.float_const(1.0), x])
    positive = builder.operation(FloatGreater(), [quotient, builder.float_const(0.0)])
    builder.output("out_0", builder.operation(BoolAnd(), [builder.bool_const(gate_value), positive]))
    builder.ret()
    return builder.finish()


def test_poison_absorbed_by_declared_absorbing_elements() -> None:
    assert HirEvaluator(_gated_poison(False)).run(0.0) == [False]
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", FloatType())
    quotient = builder.operation(FloatDiv(), [builder.float_const(1.0), x])
    positive = builder.operation(FloatGreater(), [quotient, builder.float_const(0.0)])
    builder.output("out_0", builder.operation(BoolOr(), [builder.bool_const(True), positive]))
    builder.output("out_1", builder.operation(FloatMul(), [builder.float_const(0.0), quotient]))
    builder.ret()
    assert HirEvaluator(builder.finish()).run(0.0) == [True, 0.0]


def test_poison_not_absorbed_by_identity() -> None:
    with pytest.raises(NoNumber, match="out_0"):
        HirEvaluator(_gated_poison(True)).run(0.0)


def test_poison_propagates_through_consumers_to_output() -> None:
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", FloatType())
    quotient = builder.operation(FloatDiv(), [builder.float_const(1.0), x])
    bumped = builder.operation(FloatAdd(), [quotient, builder.float_const(1.0)])
    builder.output("out_0", builder.operation(FloatNeg(), [bumped]))
    builder.ret()
    evaluator = HirEvaluator(builder.finish())
    assert evaluator.run(2.0) == [-1.5]
    with pytest.raises(NoNumber, match="the quotient"):
        evaluator.run(0.0)


def test_poison_at_branch_condition() -> None:
    builder = HirBuilder()
    entry = builder.block()
    then = builder.block()
    other = builder.block()
    merge = builder.block()
    x = builder.input("x", FloatType())
    quotient = builder.operation(FloatDiv(), [builder.float_const(1.0), x])
    positive = builder.operation(FloatGreater(), [quotient, builder.float_const(0.0)])
    builder.position_at(entry)
    builder.branch(positive, then, other)
    builder.position_at(then)
    builder.jump(merge)
    builder.position_at(other)
    builder.jump(merge)
    builder.position_at(merge)
    one = builder.float_const(1.0)
    two = builder.float_const(2.0)
    builder.output("out_0", builder.phi(FloatType(), [(then, one), (other, two)]))
    builder.ret()
    evaluator = HirEvaluator(builder.finish())
    assert evaluator.run(2.0) == [1.0]
    assert evaluator.run(-2.0) == [2.0]
    with pytest.raises(NoNumber, match="branch condition"):
        evaluator.run(0.0)


def test_poison_at_state_live_out_commits_nothing() -> None:
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", FloatType())
    quotient = builder.operation(FloatDiv(), [builder.float_const(1.0), x])
    bumped = builder.operation(FloatAdd(), [builder.float_state_read("b"), builder.float_const(1.0)])
    builder.state_slot("a", FloatConst(0.0), quotient)
    builder.state_slot("b", FloatConst(5.0), bumped)
    builder.ret()
    evaluator = HirEvaluator(builder.finish())
    evaluator.run(2.0)
    assert evaluator.state == {"a": 0.5, "b": 6.0}
    with pytest.raises(NoNumber, match="state slot 'a'"):
        evaluator.run(0.0)
    assert evaluator.state == {"a": 0.5, "b": 6.0}, "a failed transaction must commit no slot at all"
    evaluator.run(4.0)
    assert evaluator.state == {"a": 0.25, "b": 7.0}


def test_known_indeterminate_form_poisons() -> None:
    builder = HirBuilder()
    builder.block()
    inf = builder.float_const(math.inf)
    builder.output("out_0", builder.operation(FloatAdd(), [inf, builder.float_const(-math.inf)]))
    builder.ret()
    with pytest.raises(NoNumber, match="the sum"):
        HirEvaluator(builder.finish()).run()


def test_runaway_loop_bound() -> None:
    builder = HirBuilder()
    entry = builder.block()
    header = builder.block()
    builder.position_at(entry)
    builder.jump(header)
    builder.position_at(header)
    builder.jump(header)
    with pytest.raises(RuntimeError, match="did not reach Ret"):
        HirEvaluator(builder.finish()).run(max_blocks=16)


def _sub_kernel(a: float, b: float) -> float:
    return a - b


def _add_kernel(a: float, b: float) -> float:
    return a + b


def _div_kernel(a: float, b: float) -> float:
    return a / b


def test_reference_nan_discards_vector() -> None:
    hir = lower(_sub_kernel).hir
    vectors = [{"a": math.inf, "b": math.inf}, {"a": 3.0, "b": 1.0}]
    assert assert_hir_matches_reference(hir, _sub_kernel, vectors, label="nan_discard") == 1


def test_reference_raise_discards_vector() -> None:
    hir = lower(_div_kernel).hir
    vectors = [{"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 2.0}]
    assert assert_hir_matches_reference(hir, _div_kernel, vectors, label="raise_discard") == 1


def test_all_vectors_discarded_fails() -> None:
    hir = lower(_div_kernel).hir
    with pytest.raises(AssertionError, match="no transaction survived"):
        assert_hir_matches_reference(hir, _div_kernel, [{"a": 1.0, "b": 0.0}], label="vacuous")


# Conviction pins: each seeds a deliberately divergent HIR/reference pair and requires the oracle to fail, so a
# vacuity rot in any comparison path cannot hide behind an all-agreeing corpus.


class _Gained:
    def __init__(self, gain: float) -> None:
        self._gain = gain
        self._total = 0.0

    def step(self, x: float, /) -> float:
        self._total = self._total + x * self._gain
        return x


class _Sneaky(_Gained):
    def __init__(self) -> None:
        super().__init__(1.0)
        self._extra = 0.0

    def step(self, x: float, /) -> float:
        self._extra = self._extra + 1.0
        return super().step(x)


class _Drift:
    def __init__(self) -> None:
        self._s = 1.0

    def step(self, x: float, /) -> float:
        self._s = self._s + 2e-16
        return x


class _Hold:
    def __init__(self) -> None:
        self._s = 1.0

    def step(self, x: float, /) -> float:
        self._s = self._s
        return x


def test_output_divergence_convicts() -> None:
    with pytest.raises(AssertionError, match="out_0"):
        assert_hir_matches_reference(lower(_sub_kernel).hir, _add_kernel, [{"a": 3.0, "b": 1.0}], label="wrong_output")


def test_state_value_divergence_convicts() -> None:
    hir = lower(_Gained(1.0).step).hir
    with pytest.raises(AssertionError, match="state _total"):
        assert_hir_matches_reference(hir, _Gained(2.0).step, [{"x": 3.0}], label="wrong_state")


def test_missed_state_write_convicts() -> None:
    hir = lower(_Gained(1.0).step).hir
    with pytest.raises(AssertionError, match="changed-slot sets diverge"):
        assert_hir_matches_reference(hir, _Sneaky().step, [{"x": 3.0}], label="missed_write")


def test_change_status_divergence_convicts_within_ulp_tolerance() -> None:
    hir = lower(_Drift().step).hir
    with pytest.raises(AssertionError, match="changed-slot sets diverge"):
        assert_hir_matches_reference(hir, _Hold().step, [{"x": 0.5}], label="drift")


def _pair_kernel(x: float) -> tuple[float, float]:
    return x + 1.0, x * 2.0


def test_dropped_return_leaf_convicts() -> None:
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", FloatType())
    builder.output("out_0", builder.operation(FloatAdd(), [x, builder.float_const(1.0)]))
    builder.ret()
    with pytest.raises(AssertionError, match="out_1 has no port"):
        assert_hir_matches_reference(builder.finish(), _pair_kernel, [{"x": 2.0}], label="dropped_leaf")


def _nan_branch_kernel(x: float) -> float:
    d = x - x
    r = 2.0
    if d != d:
        r = 1.0
    return r


def test_consumed_nan_fails_loudly() -> None:
    """
    The documented comparable-domain edge: CPython consumes a NaN in a comparison without surfacing it in any leaf,
    so the discard rule cannot see it, and the evaluator's poisoned branch condition convicts for eye triage.
    """
    hir = lower(_nan_branch_kernel).hir
    with pytest.raises(AssertionError, match="names no number"):
        assert_hir_matches_reference(hir, _nan_branch_kernel, [{"x": math.inf}], label="consumed_nan")


class _Latch:
    def __init__(self) -> None:
        self.y = 0.0

    def step(self, x: float, /) -> None:
        self.y = x


def _latch_hir(state_port: bool, port_value_of_x: bool) -> Hir:
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", FloatType())
    builder.state_slot("y", FloatConst(0.0), x)
    if state_port:
        builder.output("state_y", x if port_value_of_x else builder.float_const(0.0))
    builder.ret()
    return builder.finish()


def test_miswired_state_port_convicts() -> None:
    with pytest.raises(AssertionError, match="state_y"):
        assert_hir_matches_reference(
            _latch_hir(state_port=True, port_value_of_x=False), _Latch().step, [{"x": 3.0}], label="miswired"
        )


def test_missing_public_state_port_convicts() -> None:
    with pytest.raises(AssertionError, match="public slots without"):
        assert_hir_matches_reference(
            _latch_hir(state_port=False, port_value_of_x=True), _Latch().step, [{"x": 3.0}], label="portless"
        )


def _first_kernel(x: float, y: float) -> float:
    return x


def test_dropped_input_port_convicts() -> None:
    builder = HirBuilder()
    builder.block()
    builder.output("out_0", builder.input("x", FloatType()))
    builder.ret()
    with pytest.raises(AssertionError, match="input ports"):
        assert_hir_matches_reference(builder.finish(), _first_kernel, [{"x": 1.0, "y": 2.0}], label="lost_input")


def test_duplicate_output_ports_convict() -> None:
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", FloatType())
    builder.output("out_0", builder.float_const(999.0))
    builder.output("out_0", builder.operation(FloatAdd(), [x, builder.float_const(1.0)]))
    builder.ret()

    def increment(x: float) -> float:
        return x + 1.0

    with pytest.raises(AssertionError, match="duplicate output port"):
        assert_hir_matches_reference(builder.finish(), increment, [{"x": 2.0}], label="dup_ports")


class _NanConfig:
    def __init__(self) -> None:
        self._unused = math.nan
        self._v = 0.0

    def step(self, x: float, /) -> float:
        self._v = self._v + x
        return x


def test_nan_in_untouched_attribute_is_not_a_discard() -> None:
    """A frozen attribute the kernel never lowers may hold NaN; only the observable surface gates the discard."""
    hir = lower(_NanConfig().step).hir
    vectors: list[dict[str, float | bool]] = [{"x": 1.0}, {"x": 2.0}]
    assert assert_hir_matches_reference(hir, _NanConfig().step, vectors, label="nan_config") == 2


class _Flagged:
    def __init__(self) -> None:
        self.level = 0.0

    def step(self, x: float, /) -> tuple[float, bool]:
        return x + 1.0, x > 100.0


def test_dropped_bool_leaf_convicts_despite_zero_valued_float_slot() -> None:
    """``False == 0.0`` in Python; the elision fallback must not let a dropped bool leaf hide behind it."""
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", FloatType())
    level = builder.float_state_read("level")
    builder.state_slot("level", FloatConst(0.0), level)
    builder.output("out_0", builder.operation(FloatAdd(), [x, builder.float_const(1.0)]))
    builder.output("state_level", level)
    builder.ret()
    with pytest.raises(AssertionError, match="out_1 has no port"):
        assert_hir_matches_reference(builder.finish(), _Flagged().step, [{"x": 2.0}], label="dropped_bool")


class _NoneKernel:
    def step(self, x: float, /) -> None:
        pass


def test_invented_output_port_convicts() -> None:
    builder = HirBuilder()
    builder.block()
    builder.output("out_0", builder.input("x", FloatType()))
    builder.ret()
    with pytest.raises(AssertionError, match="out_0 has no return leaf"):
        assert_hir_matches_reference(builder.finish(), _NoneKernel().step, [{"x": 1.0}], label="invented")


class _SeqNan:
    def __init__(self) -> None:
        self._s = 0.0

    def step(self, x: float, /) -> float:
        self._s = self._s + 1.0
        return x - x


def test_stateful_sequence_ends_at_first_discard() -> None:
    hir = lower(_SeqNan().step).hir
    vectors: list[dict[str, float | bool]] = [{"x": 1.0}, {"x": math.inf}, {"x": 2.0}]
    assert assert_hir_matches_reference(hir, _SeqNan().step, vectors, label="seq_nan") == 1


def _big_kernel() -> int:
    return 2**54 + 1


def test_promoted_big_integer_compares_as_its_float_image() -> None:
    """
    C-promotion at a join is a deliberate type-system deviation, so a float lane meeting an int the reference states
    exactly carries ``float(int)`` -- rounding included -- and that is the faithful value, not a divergence.
    """
    builder = HirBuilder()
    builder.block()
    builder.output("out_0", builder.float_const(float(2**54 + 1)))
    builder.ret()
    assert assert_hir_matches_reference(builder.finish(), _big_kernel, [{}], label="promoted_int") == 1


class _IntLeaf:
    def __init__(self) -> None:
        self.y = 1.0

    def step(self, x: float, /) -> int:
        return 1


def test_dropped_int_leaf_convicts_despite_equal_float_slot() -> None:
    builder = HirBuilder()
    builder.block()
    builder.input("x", FloatType())
    y = builder.float_state_read("y")
    builder.state_slot("y", FloatConst(1.0), y)
    builder.output("state_y", y)
    builder.ret()
    with pytest.raises(AssertionError, match="out_0 has no port"):
        assert_hir_matches_reference(builder.finish(), _IntLeaf().step, [{"x": 0.5}], label="dropped_int")


def test_state_port_exposing_private_slot_convicts() -> None:
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", FloatType())
    total = builder.operation(FloatAdd(), [builder.float_state_read("_s"), x])
    builder.state_slot("_s", FloatConst(0.0), total)
    builder.output("state__s", total)
    builder.ret()
    with pytest.raises(AssertionError, match="private slots"):
        assert_hir_matches_reference(builder.finish(), _Gained(1.0).step, [{"x": 1.0}], label="leaky")


def test_typoed_vector_key_crashes_instead_of_discarding() -> None:
    with pytest.raises(KeyError):
        assert_hir_matches_reference(lower(_add_kernel).hir, _add_kernel, [{"a": 1.0, "WRONG": 2.0}], label="typo")

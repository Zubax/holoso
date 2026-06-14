"""
Public-API, black-box behavioral tests for the cross-block-overlap / cycle-model surface landed by M1-M8.

Every test here drives the compiler ONLY through the public API: ``holoso.synthesize(fn, ops) -> SynthesisResult``,
then ``result.numerical_model.elaborate() -> NumericalSimulator``, then exercises the simulator
(``run`` / ``reset`` / ``set_inputs`` / ``tick`` / ``in_ready`` / ``out_valid`` / ``output_values`` and the typed
``inputs`` / ``outputs`` metadata). Assertions are on OBSERVABLE behavior only -- output values against a Python
reference (floats within a reduced-precision tolerance, bools exact), multi-transaction persistent-state correctness,
and the in_ready/out_valid handshake under back-pressure. No internal LIR structure is inspected, so these survive a
deep refactor of the schedule, register allocation, or block layout. The white-box twins that pin the corners actually
trigger live in test_schedule.py / test_cosim.py; this is the complementary black-box layer.

The genuine gaps these fill (the white-box twins and the stateless overlap kernels are covered elsewhere):
  - cross-block software pipelining (M7) carrying PERSISTENT STATE across many transactions, including back-pressure;
  - the model-level handshake under sustained back-pressure (the cosim has it; nothing exercised it via ``tick`` alone);
  - a two-deep shift register (non-coalesced copy slots) over many transactions plus ``reset`` (model path; cosim-only
    before);
  - nested diamonds (pure -> select, and division-bearing -> real branch) and a mixed select+branch kernel, checked by
    output value rather than by HIR block counts;
  - multi-output mixed float+bool I/O with the typed-port metadata read from the elaborated simulator.
"""

import numpy as np
import pytest

import holoso
from holoso import (
    BoolType,
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FloatFormat,
    FloatType,
    FMulILog2OperatorFamily,
    FMulOperator,
    OpConfig,
)

from ._modelref import default_tolerance, within

FMT = FloatFormat(6, 18)


def _ops() -> OpConfig:
    return OpConfig(
        FAddOperator(FMT), FMulOperator(FMT), FDivOperator(FMT), FMulILog2OperatorFamily(FMT), FCmpOperator(FMT)
    )


def _close(got: float, want: float, op_count: int = 12) -> bool:
    """A reduced-precision agreement for a kernel of about ``op_count`` ZKF ops over operands of order-unity-to-ten."""
    rtol, atol = default_tolerance(FMT, op_count, magnitude=max(1.0, abs(want)))
    return within(got, want, rtol, atol)


# --------------------------------------------------------------------------------------------------------------------
# Cross-block software pipelining (M7) carrying PERSISTENT STATE across many transactions.
#
# Every existing overlap kernel (overlap_spill_kernel, overlap_dead_arm_spill_kernel, overlap_div_err_kernel) is
# STATELESS. The net-new surface is a stateful kernel whose entry/branch block both (a) computes a wide chain that
# spills into single-predecessor arms and (b) feeds a persistent attribute carried across transactions. The wide chain
# ``w`` commits late and spills past the (shrunk) terminator; the unspeculatable division in the else arm keeps the
# diamond a real branch, so the entry block's only successors are the two single-predecessor arms it can shrink into.
# --------------------------------------------------------------------------------------------------------------------


class _OverlapAccumulator:
    """
    A spilling-chain branch whose result feeds a private LEAKY accumulator carried across transactions (-> ``out_0``).
    The accumulator is leaky (a power-of-two decay, which adds essentially no rounding) so its error stays bounded
    across a long stream rather than accumulating without limit -- the tolerance is then a true per-step bound and the
    test stays robust to a value-preserving operator-selection refactor. The overlap still engages: the shrink depends
    on the wide chain, the division arm, and the single-predecessor arms, not on the accumulator form.
    """

    def __init__(self) -> None:
        self._acc = 0.0

    def __call__(self, x, y, z):  # type: ignore[no-untyped-def]
        w = (x * z + y) * z + y  # wide chain: commits late, spills past the shrunk terminator into both arms
        if x < y:
            r = w + 1.0  # then-arm reads the spilled w
        else:
            r = w / (y * y + 1.0)  # else-arm reads w; the division keeps the diamond a real branch
        self._acc = self._acc * 0.5 + r  # the spilled-chain result lands in a persistent (leaky) slot
        return self._acc


def test_overlap_kernel_persistent_state_across_many_vectors() -> None:
    # The accumulator threads the overlapping branch's result across a long random stream; the model must match a fresh
    # Python reference advanced in lockstep, on BOTH branch polarities and across the decision boundary (x == y).
    simulator = holoso.synthesize(
        _OverlapAccumulator().__call__, _ops(), name="overlap_acc"
    ).numerical_model.elaborate()
    reference = _OverlapAccumulator()
    rng = np.random.default_rng(0xA11CE)
    samples = [(0.5, 0.5, 1.0), (2.0, 2.0, 1.0)]  # the x == y decision boundary on both sides
    samples += [tuple(float(rng.uniform(-3.0, 3.0)) for _ in range(3)) for _ in range(60)]
    for x, y, z in samples:
        got = float(simulator.run(x, y, z)[0])
        want = reference(x, y, z)
        assert _close(got, want), f"x={x} y={y} z={z}: {got} vs {want}"


def test_overlap_kernel_reset_restores_initial_state() -> None:
    # The persistent slot resets to its snapshot, so the accumulator restarts after ``reset`` -- the same first output
    # as a fresh elaboration regardless of how far the previous run wandered.
    simulator = holoso.synthesize(
        _OverlapAccumulator().__call__, _ops(), name="overlap_acc_rst"
    ).numerical_model.elaborate()
    reference = _OverlapAccumulator()
    first = float(simulator.run(1.0, 2.0, 0.5)[0])
    assert _close(first, reference(1.0, 2.0, 0.5))
    for x, y, z in [(3.0, 1.0, 2.0), (0.5, 4.0, 1.0), (2.0, 1.0, 1.0)]:
        simulator.run(x, y, z)  # wander the accumulator far from its reset value
    simulator.reset()
    assert float(simulator.run(1.0, 2.0, 0.5)[0]) == first  # bit-identical restart after reset


# --------------------------------------------------------------------------------------------------------------------
# Model-level handshake under back-pressure (the cosim exercises this against RTL; nothing drove it via ``tick`` alone
# at the model level, where test_cycle_model only holds out_ready low BEFORE out_valid, never after).
# --------------------------------------------------------------------------------------------------------------------


def _drive_with_stall(simulator: holoso.NumericalSimulator, inputs: tuple[float, ...], stall: int) -> float:
    """
    Drive one transaction tick-by-tick: present inputs, accept, advance to out_valid, then HOLD with ``out_ready``
    low for ``stall`` cycles (asserting the output is stable and in_ready stays low), then release to accept it.
    """
    simulator.set_inputs(*inputs)
    while not simulator.in_ready:
        simulator.tick(in_valid=False, out_ready=True)
    simulator.tick(in_valid=True, out_ready=False)  # accept the transaction
    while not simulator.out_valid:
        simulator.tick(in_valid=False, out_ready=False)
    held = float(simulator.output_values[0])
    for _ in range(stall):  # sustained back-pressure: the presented output must not move and no new input is accepted
        simulator.tick(in_valid=False, out_ready=False)
        assert simulator.out_valid and not simulator.in_ready
        assert float(simulator.output_values[0]) == held
    simulator.tick(in_valid=False, out_ready=True)  # release: accept the output, advancing the persistent state once
    return held


def test_backpressure_holds_output_and_advances_state_once_through_overlap() -> None:
    # Sustained back-pressure on the overlapping stateful kernel: the output stays frozen and no input is accepted while
    # out_ready is low, and the persistent state advances EXACTLY ONCE per released output -- not once per stalled
    # cycle. Back-to-back transactions with growing stalls thread the carried accumulator correctly.
    simulator = holoso.synthesize(_OverlapAccumulator().__call__, _ops(), name="overlap_bp").numerical_model.elaborate()
    reference = _OverlapAccumulator()
    for index, vector in enumerate([(1.0, 2.0, 0.5), (3.0, 1.0, 2.0), (0.5, 4.0, 1.0), (2.0, 2.0, 1.0)]):
        got = _drive_with_stall(simulator, vector, stall=2 * index)  # 0, 2, 4, 6 cycles of back-pressure
        want = reference(*vector)
        assert _close(got, want), f"vector={vector} stall={2 * index}: {got} vs {want}"


def test_run_drains_partial_overlap_transaction_before_presenting_new_inputs() -> None:
    # The drain-before-present ordering on the overlapping stateful kernel: a transaction accepted by a partial manual
    # ``tick`` must complete with its OWN latched inputs (advancing state) before ``run`` presents the next inputs, so
    # the carried accumulator must reflect BOTH the drained transaction and the freshly-run one.
    simulator = holoso.synthesize(
        _OverlapAccumulator().__call__, _ops(), name="overlap_drain"
    ).numerical_model.elaborate()
    reference = _OverlapAccumulator()
    assert _close(float(simulator.run(1.0, 2.0, 0.5)[0]), reference(1.0, 2.0, 0.5))
    simulator.set_inputs(3.0, 1.0, 2.0)
    simulator.tick(in_valid=True, out_ready=False)  # accept (3, 1, 2); it is now in flight, in_ready is False
    assert not simulator.in_ready
    reference(3.0, 1.0, 2.0)  # the drained transaction advances the reference state too (its output is discarded)
    want = reference(0.5, 4.0, 1.0)  # the transaction run() actually returns
    got = float(simulator.run(0.5, 4.0, 1.0)[0])
    assert _close(got, want), f"drain ordering: {got} vs {want}"


# --------------------------------------------------------------------------------------------------------------------
# State slots: a two-deep shift register (two non-coalesced copy slots) over many transactions plus reset. Only the
# RTL cosim (_ShiftRegister2 in test_cosim.py) exercised this; there was no model-path equivalent.
# --------------------------------------------------------------------------------------------------------------------


class _ShiftRegister2:
    """Two-deep delay line: returns the input from two transactions ago. Both slots are non-coalesced copy slots."""

    def __init__(self) -> None:
        self._a = 0.0
        self._b = 0.0

    def __call__(self, x):  # type: ignore[no-untyped-def]
        out = self._b
        self._b = self._a
        self._a = x
        return out


def test_two_deep_shift_register_delays_by_two_and_resets() -> None:
    # The copy slots advance once per accepted transaction, so the output stream is the input stream delayed by two
    # (exact: every value is representable and only copied, never arithmetically combined). reset reloads both slots
    # to their snapshot, so the delay line restarts emitting the reset value for two samples.
    simulator = holoso.synthesize(_ShiftRegister2().__call__, _ops(), name="shift2").numerical_model.elaborate()
    reference = _ShiftRegister2()
    for x in [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, 0.5]:
        assert float(simulator.run(x)[0]) == reference(x), f"delay mismatch at x={x}"
    simulator.reset()
    fresh = _ShiftRegister2()
    for x in [7.0, 8.0, 9.0, 10.0]:
        assert float(simulator.run(x)[0]) == fresh(x), f"post-reset mismatch at x={x}"


# --------------------------------------------------------------------------------------------------------------------
# Diamond if-conversion vs real branches checked by OUTPUT VALUE (the existing nested/if-conversion tests assert HIR
# block counts; here we confirm the compiled semantics are correct on every path, which is what would survive a
# refactor of the if-conversion heuristic).
# --------------------------------------------------------------------------------------------------------------------


def _nested_pure(x, y):  # type: ignore[no-untyped-def]
    # A two-level pure diamond: every arm is speculatable, so it fully if-converts to selects. All four leaves and the
    # decision boundaries must be exact (additive/subtractive combinations of representable inputs).
    if x > 0.0:
        r = (x + y) if y > 0.0 else (x - y)
    else:
        r = (y - x) if y > 0.0 else (-x - y)
    return r


def test_nested_pure_diamond_output_matches_reference() -> None:
    simulator = holoso.synthesize(_nested_pure, _ops(), name="nested_pure").numerical_model.elaborate()
    for x in (-2.0, -0.5, 0.0, 0.5, 2.0):
        for y in (-2.0, 0.0, 0.5, 2.0):
            got = float(simulator.run(x, y)[0])
            assert got == _nested_pure(x, y), f"x={x} y={y}: {got} vs {_nested_pure(x, y)}"


def _nested_div(x, y):  # type: ignore[no-untyped-def]
    # A nested diamond whose inner arm divides: the division is unspeculatable, so the inner diamond stays a REAL
    # branch (it cannot if-convert). The divisor ``y*y + 1`` is structurally nonzero, so the path is always valid.
    if x > 0.0:
        if y > 0.0:
            r = (x + y) / (y * y + 1.0)
        else:
            r = x - y
    else:
        r = y - x
    return r


def test_nested_division_branch_output_matches_reference() -> None:
    simulator = holoso.synthesize(_nested_div, _ops(), name="nested_div").numerical_model.elaborate()
    for x in (-2.0, -0.5, 0.5, 2.0):
        for y in (-2.0, 0.5, 2.0):
            got = float(simulator.run(x, y)[0])
            want = _nested_div(x, y)
            assert _close(got, want), f"x={x} y={y}: {got} vs {want}"


def _mixed_select_and_branch(x, y):  # type: ignore[no-untyped-def]
    # One kernel mixing an if-converted (select) pure diamond and a real (division-bearing) branch: the max folds to a
    # select, the division gates a real branch. Both the select polarity and the branch decision are crossed.
    m = x if x > y else y  # pure diamond -> select
    if x > 0.0:
        r = m / (x + 1.0)  # division -> real branch (x + 1 > 0 on this arm, structurally nonzero)
    else:
        r = m - 1.0
    return r


def test_mixed_select_and_real_branch_output_matches_reference() -> None:
    simulator = holoso.synthesize(_mixed_select_and_branch, _ops(), name="mixed_sel_branch").numerical_model.elaborate()
    for x in (-2.0, -0.5, 0.5, 2.0):
        for y in (-1.0, 0.5, 1.0, 3.0):
            got = float(simulator.run(x, y)[0])
            want = _mixed_select_and_branch(x, y)
            assert _close(got, want), f"x={x} y={y}: {got} vs {want}"


def _diamond_in_loop(x, n):  # type: ignore[no-untyped-def]
    # A division-bearing diamond INSIDE a real back-edge while loop: a real branch nested in a loop, exercising the
    # loop-header phi merge of the accumulator across a data-dependent trip count. The divisor stays structurally
    # nonzero on the dividing arm.
    acc = 0.0
    i = n
    while i > 0.0:
        if x > 1.0:
            acc = acc + x / (x + 1.0)
        else:
            acc = acc + x
        i = i - 1.0
    return acc


def test_diamond_inside_loop_output_matches_reference() -> None:
    simulator = holoso.synthesize(_diamond_in_loop, _ops(), name="diamond_in_loop").numerical_model.elaborate()
    for x, n in [(2.0, 3.0), (0.5, 4.0), (3.0, 2.0), (1.5, 5.0), (2.0, 0.0)]:
        got = float(simulator.run(x, n)[0])
        want = _diamond_in_loop(x, n)
        assert _close(got, want, op_count=8 * max(1, int(n))), f"x={x} n={n}: {got} vs {want}"


# --------------------------------------------------------------------------------------------------------------------
# Typed ports (M6b): multi-output mixed float+bool I/O, a boolean input, and the scalar-type metadata read from the
# elaborated simulator (the existing typed-port test reads it from the handle; here it is the simulator's own view).
# --------------------------------------------------------------------------------------------------------------------


def _multi_io(flag: bool, x, y):  # type: ignore[no-untyped-def]
    # A boolean input gating a division branch, a tuple return mixing a bool and two floats: the divisor y*y + 1 is
    # structurally nonzero, so the flag-true arm is always valid.
    inside = flag and (x > y)
    if flag:
        d = x / (y * y + 1.0)
    else:
        d = x - y
    return inside, d, x + y


def test_multi_output_mixed_io_metadata_and_values() -> None:
    simulator = holoso.synthesize(_multi_io, _ops(), name="multi_io").numerical_model.elaborate()
    assert [(p.name, p.scalar_type) for p in simulator.inputs] == [
        ("flag", BoolType()),
        ("x", FloatType(FMT)),
        ("y", FloatType(FMT)),
    ]
    assert [(p.name, p.scalar_type) for p in simulator.outputs] == [
        ("out_0", BoolType()),
        ("out_1", FloatType(FMT)),
        ("out_2", FloatType(FMT)),
    ]
    for flag in (True, False):
        for x in (-1.0, 0.5, 2.0):
            for y in (1.0, 3.0):
                got = simulator.run(flag, x, y)
                inside, d, total = _multi_io(flag, x, y)
                assert got[0] is inside, f"flag={flag} x={x} y={y}: bool {got[0]} vs {inside}"
                assert _close(float(got[1]), d), f"flag={flag} x={x} y={y}: {float(got[1])} vs {d}"
                assert float(got[2]) == total, f"flag={flag} x={x} y={y}: {float(got[2])} vs {total}"
    with pytest.raises(TypeError, match="input 0 must be bool"):
        simulator.run(1.0, 2.0, 3.0)  # a float in the boolean lane is rejected by the typed input coercion

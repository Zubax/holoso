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
    """Reduced-precision agreement for a kernel of about ``op_count`` ZKF ops over operands of order unity-to-ten."""
    rtol, atol = default_tolerance(FMT, op_count, magnitude=max(1.0, abs(want)))
    return within(got, want, rtol, atol)


# M7 cross-block software pipelining: the existing overlap kernels are all STATELESS, so the net-new surface is a
# stateful kernel that both spills a wide chain into its arms and feeds a persistent attribute across transactions.


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

    def __call__(self, x: float, y: float, z: float):  # type: ignore[no-untyped-def]
        w = (x * z + y) * z + y  # wide chain: commits late, spills past the shrunk terminator into both arms
        if x < y:
            r = w + 1.0  # then-arm reads the spilled w
        else:
            r = w / (y * y + 1.0)  # else-arm reads w; the division keeps the diamond a real branch
        self._acc = self._acc * 0.5 + r  # the spilled-chain result lands in a persistent (leaky) slot
        return self._acc


def test_overlap_kernel_persistent_state_across_many_vectors() -> None:
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


# Model-level handshake under back-pressure: test_cycle_model only holds out_ready low BEFORE out_valid, never after.


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
        got = _drive_with_stall(simulator, vector, stall=2 * index)
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


def test_two_deep_shift_register_delays_by_two_and_resets() -> None:
    # The copy slots advance once per accepted transaction, so the output stream is the input stream delayed by two
    # (exact: every value is representable and only copied, never arithmetically combined). reset reloads both slots
    # to their snapshot, so the delay line restarts emitting the reset value for two samples.
    class ShiftRegister2:
        """Two-deep delay line: returns the input from two transactions ago. Both slots are non-coalesced copy slots."""

        def __init__(self) -> None:
            self._a = 0.0
            self._b = 0.0

        def __call__(self, x: float):  # type: ignore[no-untyped-def]
            out = self._b
            self._b = self._a
            self._a = x
            return out

    simulator = holoso.synthesize(ShiftRegister2().__call__, _ops(), name="shift2").numerical_model.elaborate()
    reference = ShiftRegister2()
    for x in [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, 0.5]:
        assert float(simulator.run(x)[0]) == reference(x), f"delay mismatch at x={x}"
    simulator.reset()
    fresh = ShiftRegister2()
    for x in [7.0, 8.0, 9.0, 10.0]:
        assert float(simulator.run(x)[0]) == fresh(x), f"post-reset mismatch at x={x}"


def test_nested_pure_diamond_output_matches_reference() -> None:
    def nested_pure(x: float, y: float):  # type: ignore[no-untyped-def]
        # A two-level pure diamond: every arm is speculatable, so it fully if-converts to selects. All four leaves and
        # the decision boundaries must be exact (additive/subtractive combinations of representable inputs).
        if x > 0.0:
            r = (x + y) if y > 0.0 else (x - y)
        else:
            r = (y - x) if y > 0.0 else (-x - y)
        return r

    simulator = holoso.synthesize(nested_pure, _ops(), name="nested_pure").numerical_model.elaborate()
    for x in (-2.0, -0.5, 0.0, 0.5, 2.0):
        for y in (-2.0, 0.0, 0.5, 2.0):
            got = float(simulator.run(x, y)[0])
            assert got == nested_pure(x, y), f"x={x} y={y}: {got} vs {nested_pure(x, y)}"


def test_nested_division_branch_output_matches_reference() -> None:
    def nested_div(x: float, y: float):  # type: ignore[no-untyped-def]
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

    simulator = holoso.synthesize(nested_div, _ops(), name="nested_div").numerical_model.elaborate()
    for x in (-2.0, -0.5, 0.5, 2.0):
        for y in (-2.0, 0.5, 2.0):
            got = float(simulator.run(x, y)[0])
            want = nested_div(x, y)
            assert _close(got, want), f"x={x} y={y}: {got} vs {want}"


def test_spilled_in_branch_condition_is_read_after_it_lands() -> None:
    def spilled_branch_condition(a: float, b: float, c: float, d: float):  # type: ignore[no-untyped-def]
        # A boolean comparison produced in the entry but BRANCHED ON in a single-predecessor successor. The entry
        # overlaps into that successor, and the DEEP condition (a long product, committing late) lands past the entry's
        # shrunk terminator -- so it spills into the successor frame rather than being resident there. The successor's
        # branch must read the condition only AFTER its carried landing: the terminator read-floor folds the spilled-in
        # condition's landing into the issue-side envelope. Without that fold the branch reads a stale condition and
        # takes the wrong arm -- a silent miscompile (the build does not crash, the result is just wrong) the reference
        # comparison catches. The inner arm divides, so both diamonds stay real branches (division is unspeculatable);
        # divisors are structurally nonzero so every path is valid.
        cond = (a * b * c * d) > (a + b + c + d)
        if c > 0.0:
            if cond:
                r = a / (b * b + 1.0)
            else:
                r = a - b
        else:
            r = c / (d * d + 1.0)
        return r

    simulator = holoso.synthesize(spilled_branch_condition, _ops(), name="spilled_cond").numerical_model.elaborate()
    for a, b, c, d in [
        (2.0, 3.0, 1.5, 0.5),
        (0.5, 4.0, 2.0, 1.0),
        (3.0, 2.0, -1.0, 2.0),
        (1.5, 1.5, 0.25, 3.0),
        (4.0, 0.5, 1.0, 2.0),
        (2.0, 2.0, -2.0, 4.0),
        (2.0, 2.0, 2.0, 2.0),  # cond True with c > 0: exercises the inner condition-true arm
    ]:
        got = float(simulator.run(a, b, c, d)[0])
        want = spilled_branch_condition(a, b, c, d)
        assert _close(got, want), f"a={a} b={b} c={c} d={d}: {got} vs {want}"


def test_spilled_wide_mul_operand_is_read_after_it_lands() -> None:
    # The fmul-producer generalization of test_spilled_in_branch_condition: a LATE latency-1 wide product spills past
    # the overlapped entry terminator into the inner arm, where a pooled fadd reads it. Under the latch-free read EVERY
    # latency-1 wide pooled op (fmul / fmul_ilog2 / fcmp -- all latency 1 by default) samples at issue+2, one PC past
    # its control word; without the _issue_side_envelope operand_read_cycle floor the producer fires past the shrunk
    # terminator and is ORPHANED (the model KeyErrors on its register; a wrong arm in general). This guards the WHOLE
    # latency-1-pooled-op spill surface, not only the single fcmp shape -- and the defect is invisible to the
    # interpreter<->model differential, so only a behavioral spill-then-read kernel catches it.
    def spilled_wide_mul(a: float, b: float, c: float, d: float):  # type: ignore[no-untyped-def]
        x = a * b * c * d
        if c > 0.0:
            if d > a:
                r = x + b
            else:
                r = a - b
        else:
            r = c / (d * d + 1.0)
        return r

    simulator = holoso.synthesize(spilled_wide_mul, _ops(), name="spilled_wide_mul").numerical_model.elaborate()
    for a, b, c, d in [
        (2.0, 3.0, 1.5, 0.5),
        (0.5, 4.0, 2.0, 1.0),
        (3.0, 2.0, -1.0, 2.0),
        (1.5, 1.5, 0.25, 3.0),
        (4.0, 0.5, 1.0, 2.0),
        (2.0, 2.0, 2.0, 2.0),
    ]:
        got = float(simulator.run(a, b, c, d)[0])
        want = spilled_wide_mul(a, b, c, d)
        assert _close(got, want), f"a={a} b={b} c={c} d={d}: {got} vs {want}"


def test_mixed_select_and_real_branch_output_matches_reference() -> None:
    def mixed_select_and_branch(x: float, y: float):  # type: ignore[no-untyped-def]
        # One kernel mixing an if-converted (select) pure diamond and a real (division-bearing) branch: the max folds
        # to a select, the division gates a real branch. Both the select polarity and the branch decision are crossed.
        m = x if x > y else y  # pure diamond -> select
        if x > 0.0:
            r = m / (x + 1.0)  # division -> real branch (x + 1 > 0 on this arm, structurally nonzero)
        else:
            r = m - 1.0
        return r

    simulator = holoso.synthesize(mixed_select_and_branch, _ops(), name="mixed_sel_branch").numerical_model.elaborate()
    for x in (-2.0, -0.5, 0.5, 2.0):
        for y in (-1.0, 0.5, 1.0, 3.0):
            got = float(simulator.run(x, y)[0])
            want = mixed_select_and_branch(x, y)
            assert _close(got, want), f"x={x} y={y}: {got} vs {want}"


def test_diamond_inside_loop_output_matches_reference() -> None:
    def diamond_in_loop(x: float, n: float):  # type: ignore[no-untyped-def]
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

    simulator = holoso.synthesize(diamond_in_loop, _ops(), name="diamond_in_loop").numerical_model.elaborate()
    for x, n in [(2.0, 3.0), (0.5, 4.0), (3.0, 2.0), (1.5, 5.0), (2.0, 0.0)]:
        got = float(simulator.run(x, n)[0])
        want = diamond_in_loop(x, n)
        assert _close(got, want, op_count=8 * max(1, int(n))), f"x={x} n={n}: {got} vs {want}"


# Unlike the existing typed-port test (which reads metadata from the handle), this reads it from the simulator's view.


def test_multi_output_mixed_io_metadata_and_values() -> None:
    def multi_io(flag: bool, x: float, y: float):  # type: ignore[no-untyped-def]
        # A boolean input gating a division branch, a tuple return mixing a bool and two floats: the divisor y*y + 1 is
        # structurally nonzero, so the flag-true arm is always valid.
        inside = flag and (x > y)
        if flag:
            d = x / (y * y + 1.0)
        else:
            d = x - y
        return inside, d, x + y

    simulator = holoso.synthesize(multi_io, _ops(), name="multi_io").numerical_model.elaborate()
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
                inside, d, total = multi_io(flag, x, y)
                assert got[0] is inside, f"flag={flag} x={x} y={y}: bool {got[0]} vs {inside}"
                assert _close(float(got[1]), d), f"flag={flag} x={x} y={y}: {float(got[1])} vs {d}"
                assert float(got[2]) == total, f"flag={flag} x={x} y={y}: {float(got[2])} vs {total}"
    with pytest.raises(TypeError, match="input 0 must be bool"):
        simulator.run(1.0, 2.0, 3.0)  # a float in the boolean lane is rejected by the typed input coercion


# Cross-bank cycle-model coverage: the model commits each PC's landings before evaluating that PC's reads, so a wrong
# inline read step or dependency edge would read a stale/not-yet-landed value and diverge.


def test_latching_fault_register_streams_and_resets() -> None:
    # Each channel latches on its first trip and HOLDS until reset; ``any_fault`` summarizes the just-updated channels.
    # Demonstrates the edge guard (a): on the FIRST-TRIP vector (True, False, False) the correct summary is
    # True -- it ORs the channel just latched THIS transaction. A wrong edge that read the channel's stale (pre-update)
    # value would yield False there, which the assertion below would catch. Persistent state across many transactions
    # plus a mid-stream ``reset()`` clearing every sticky latch is the load-bearing observable.
    class LatchingFaultRegister:
        """
        Three independent sticky OR-latches plus a combinational ``any_fault`` summary -- a multi-channel boolean state
        kernel whose new-state producers (``self._x = self._x or x``) read only resident values and so are the entry
        block's cycle-0-eligible inline ops. The summary ORs the THREE freshly-latched channels in the same block, so
        its value depends on the commit ordering being right: a stale read of any channel would drop a just-latched
        fault. Multi-channel bool state with a same-block summary is a shape no other black-box test exercises
        (``_BoolStateMachine`` in test_public_api_behavior is single-channel; ``_ChainedSlots`` is float). A local copy
        rather than examples/latching_fault_register, so the pinned same-block-summary shape stays decoupled from it.
        """

        def __init__(self) -> None:
            self._overcurrent = False
            self._overvoltage = False
            self._overtemp = False

        def __call__(self, overcurrent: bool, overvoltage: bool, overtemp: bool):  # type: ignore[no-untyped-def]
            self._overcurrent = self._overcurrent or overcurrent
            self._overvoltage = self._overvoltage or overvoltage
            self._overtemp = self._overtemp or overtemp
            any_fault = self._overcurrent or self._overvoltage or self._overtemp
            return any_fault, self._overcurrent, self._overvoltage, self._overtemp

    simulator = holoso.synthesize(
        LatchingFaultRegister().__call__, _ops(), name="latching_fault"
    ).numerical_model.elaborate()
    reference = LatchingFaultRegister()
    stream = [
        (False, False, False),  # idle: nothing latches
        (True, False, False),  # overcurrent trips -> latches; any_fault must be True on this very transaction
        (False, False, False),  # transient gone, the latch holds
        (False, True, False),  # overvoltage trips -> both latched
        (False, False, True),  # overtemp trips -> all three latched
        (False, False, False),  # all stay latched (cleared only by reset)
    ]
    for vector in stream:
        got = tuple(bool(value) for value in simulator.run(*vector))
        want = reference(*vector)
        assert got == want, f"vector={vector}: {got} vs {want}"
        # The summary ORs the freshly-latched channels; on the trip cycle this differs from a stale-read OR.
        assert got[0] == (got[1] or got[2] or got[3]), f"any_fault desync at {vector}: {got}"
    simulator.reset()
    fresh = LatchingFaultRegister()
    for vector in [(False, False, False), (False, True, False), (False, False, False)]:
        got = tuple(bool(value) for value in simulator.run(*vector))
        assert got == fresh(*vector), f"post-reset {vector}: {got}"


def test_octave_index_resident_output_drain_only_ret_matches_reference() -> None:
    # The resident-output drain-only Ret shape: the loop body produces the float ``octaves``, which the exit Ret reads
    # resident at its base PC with no boundary drain. Exact ``==`` guards the over-aggressive direction -- a reclaim
    # pushing the boundary BELOW the resident landing would sample ``octaves`` before the loop's final write lands, an
    # off-by-one octave count.
    #
    # The trip count is data-dependent, so the value is the loop's correctness. Inputs are FROZEN to a verified set:
    # |x| >= 1 magnitudes (abs and *0.5 are exact, so the count is unambiguous) and |x| < 1 values comfortably inside
    # an octave (their reciprocal does not round across an octave boundary), so the ZKF count equals the float64 count
    # exactly. Do NOT broaden post-hoc: a value near a power-of-two boundary can flip the rounded count by one.
    def octave_index(x: float):  # type: ignore[no-untyped-def]
        # The order of magnitude of x in octaves: halvings (or doublings, for |x| < 1) to bring |x| into (0.5, 1]. The
        # division-bearing magnitude diamond stays a real branch; its merge is an empty pass-through threaded into the
        # halving loop, whose Ret is a resident-output drain (the float ``octaves`` is produced in the loop body and
        # read combinationally at the Ret's own base PC). A local copy of examples/octave_index (the test must not
        # couple to examples/, and a local kernel preserves the diamond -> merge -> loop -> drain-Ret shape).
        magnitude = abs(x)
        if magnitude >= 1.0:
            scaled = magnitude
        else:
            scaled = 1.0 / magnitude  # the lone non-speculatable op: keeps the diamond a real branch
        octaves = 0.0
        while scaled > 1.0:
            scaled = scaled * 0.5
            octaves = octaves + 1.0
        return octaves

    simulator = holoso.synthesize(octave_index, _ops(), name="octave_drain").numerical_model.elaborate()
    for x in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 3.0, 7.0, -3.0, -16.0, 1.5, 0.5, 0.25, 0.7, 0.1, 0.03):
        got = float(simulator.run(x)[0])
        want = octave_index(x)
        assert got == want, f"octave count at x={x}: {got} vs {want}"


def test_cross_bank_chain_edges_match_reference() -> None:
    # Could-have-failed (a, reasoned): an off-by-one in the inline read step or a dependency edge would let a consumer
    # in this chain read before its producer's value lands, so the model (which commits each PC's landings before its
    # reads) would sample a stale operand -- the boolean ``r``/``t`` would flip and the float ``out`` would diverge from
    # the float64 reference. The chain has no black-box twin (test_arithmetic_behavior's cross-domain test is a single
    # cast; the deep same-bank reduction at test_schedule is white-box and float-only).
    #
    # Inputs are FROZEN to a verified set: ``r`` and ``t`` are comparisons of ROUNDED intermediates (``t = s > 0`` with
    # ``s`` a rounded sum), so a value near a comparison boundary could round differently than float64. The reference
    # bools are derived from the SAME Python expression, and the operands are kept clear of those boundaries.
    import itertools  # noqa: PLC0415

    def cross_bank_chain(a: float, b: float, c: float, d: float):  # type: ignore[no-untyped-def]
        # A deep cross-bank chain on the tight same-bank edge: back-to-back inline ``band``/``bor`` over comparisons, a
        # bool->float cast, a float op consuming the cast, and a float->bool reduction folded into a select. Inline ops
        # are latency 0 and the dependency edge is the unclamped landing-vs-read spacing, so this chain is the most
        # direct exercise of those cross-bank edges through the public API.
        p = a > b
        q = c > d
        r = (p and q) or (a > d)  # back-to-back inline band then bor: the tight same-bank read-first edge
        gate = 1.0 if r else 0.0  # bool -> float cast
        s = gate * (a + b) + c  # a float op consuming the cast result
        t = s > 0.0  # float -> bool reduction
        out = (s - d) if t else (d - s)  # the select consumes ``t`` and ``s``
        return out, r, t

    simulator = holoso.synthesize(cross_bank_chain, _ops(), name="cross_bank").numerical_model.elaborate()
    for a, b, c, d in itertools.product((-2.0, 0.5, 2.0), repeat=4):
        got = simulator.run(a, b, c, d)
        want_out, want_r, want_t = cross_bank_chain(a, b, c, d)
        assert got[1] is want_r, f"r at ({a},{b},{c},{d}): {got[1]} vs {want_r}"
        assert got[2] is want_t, f"t at ({a},{b},{c},{d}): {got[2]} vs {want_t}"
        assert _close(float(got[0]), want_out, op_count=6), f"out at ({a},{b},{c},{d}): {float(got[0])} vs {want_out}"

"""
Public-API, black-box behavioral tests for TIMING-ONLY operator-staging equivalence (round-3 axis A).

Operator pipeline stages change a kernel's LATENCY but must never change its VALUE: the same transaction stream fed
through the minimum-latency operator configuration and through a deeply-staged one must produce BIT-IDENTICAL outputs.
This is the strongest stress on the scheduler, the cross-block software-pipelining / overlap machinery, and the spill
arithmetic, because all of those compute cycle offsets (term_offset, spill landing frames, state-install boundaries)
from operator latencies -- a mis-timed schedule, overlap, or spill at the LONGER latency would diverge from the short
one, and the divergence is caught here without any reference model at all.

Every test drives the compiler ONLY through the public API (``holoso.synthesize(fn, ops).numerical_model.elaborate()``
and the resulting simulator's ``run`` / ``set_inputs`` / ``tick`` / handshake surface) and asserts solely on observable
output BITS -- never on any internal schedule, register, or cycle structure. The two configurations are the no-optional-
stage baseline and a deeply-pipelined fixture (both shared from ``_modelref`` so they track the operator-knob surface):
``default_ops`` has every ``stage_*`` knob at zero; ``staged_ops`` enables them across fadd / fmul / fdiv / fmul_ilog2 /
fcmp. The kernels span the four shapes that exercise distinct timing paths:

  - a branchy diamond (a real, division-bearing branch with a long commit chain),
  - a back-edge while loop (a Newton-Raphson reciprocal with a data-dependent trip count, so the back edge is genuine),
  - a stateful streaming kernel (a persistent accumulator threaded across a multi-transaction sequence, run on the SAME
    sequence under both latencies including a ``reset``),
  - an overlap-spilling kernel (the shared ``overlap_spill_kernel``, whose wide chain spills past a shrunk terminator
    into both arms -- the spill landing frame is latency-derived, so the two staging depths must still agree bit-exact).

The stateful kernel is additionally driven tick-by-tick under sustained back-pressure at BOTH latencies, confirming the
handshake (out stays frozen, no new input accepted, state advances exactly once per released output) is itself latency-
invariant. A divergence under back-pressure would indicate the longer pipeline mis-drains an in-flight transaction.
"""

import numpy as np

import holoso
from holoso import FloatFormat

from ._modelref import default_ops, overlap_spill_kernel, staged_ops

FMT = FloatFormat(6, 18)


def _pair(fn, name: str):  # type: ignore[no-untyped-def]
    """Elaborate the same kernel under the minimum-latency and the deeply-staged operator configurations."""
    short = holoso.synthesize(fn, default_ops(FMT), name=f"{name}_short").numerical_model.elaborate()
    long = holoso.synthesize(fn, staged_ops(FMT), name=f"{name}_long").numerical_model.elaborate()
    return short, long


def _assert_bits_equal(  # type: ignore[no-untyped-def]
    short: holoso.NumericalSimulator, long: holoso.NumericalSimulator, *inputs
) -> int:
    """
    Run one transaction through both simulators and assert every output leaf is bit-identical; return the leaf count.
    """
    out_short = short.run(*inputs)
    out_long = long.run(*inputs)
    assert len(out_short) == len(out_long), f"output arity differs: {len(out_short)} vs {len(out_long)}"
    for index, (a, b) in enumerate(zip(out_short, out_long)):
        if isinstance(a, bool) or isinstance(b, bool):
            assert a is b, f"inputs={inputs} out[{index}] bool {a} vs {b}"
        else:
            assert a.bits == b.bits, (
                f"inputs={inputs} out[{index}] bits differ: short=0x{a.bits:x} ({float(a)}) "
                f"long=0x{b.bits:x} ({float(b)})"
            )
    return len(out_short)


# --------------------------------------------------------------------------------------------------------------------
# Shape 1: a branchy diamond -- a REAL (division-bearing, hence unspeculatable) branch with a long commit chain whose
# result outlives the early comparison. The branch decision and the spill timing both shift with operator latency, so
# the two staging depths agreeing bit-exact (including AT the decision boundary x == c) proves the branchy schedule is
# latency-correct.
# --------------------------------------------------------------------------------------------------------------------


def _branchy_diamond(x, y, c):  # type: ignore[no-untyped-def]
    t = (x * y + c) * y  # a multiply-add-multiply chain that commits late
    if t > c:
        r = t + x  # then-arm reads the late chain
    else:
        r = t / (y * y + 1.0)  # else-arm divides (structurally nonzero) -> the diamond stays a real branch
    return r


def test_branchy_diamond_timing_invariant() -> None:
    short, long = _pair(_branchy_diamond, "branchy")
    rng = np.random.default_rng(0xB47C)
    # Both branch polarities and the exact decision boundary (t == c) on hand-picked vectors plus a random sweep.
    samples = [(1.0, 2.0, 0.0), (0.5, 0.5, 1.0), (-1.0, 3.0, -2.0), (2.0, 1.0, 5.0), (0.0, 4.0, 0.0)]
    samples += [tuple(float(rng.uniform(-3.0, 3.0)) for _ in range(3)) for _ in range(60)]
    for x, y, c in samples:
        _assert_bits_equal(short, long, x, y, c)


# --------------------------------------------------------------------------------------------------------------------
# Shape 2: a back-edge while loop -- a Newton-Raphson reciprocal whose trip count is the data-dependent input ``n``, so
# the loop is unambiguously a genuine back edge (not a constant-trip loop that would fully unroll). The loop-header phi
# merge and the back-edge recurrence are scheduled at the operator latency; the two depths must agree on every (a, n).
# --------------------------------------------------------------------------------------------------------------------


def _newton_reciprocal(a, n):  # type: ignore[no-untyped-def]
    # y_{k+1} = y_k * (2 - a*y_k) converges to 1/a; the loop carries y across a data-dependent number of iterations.
    y = 0.5
    i = n
    while i > 0.0:
        y = y * (2.0 - a * y)
        i = i - 1.0
    return y


def test_back_edge_loop_timing_invariant() -> None:
    short, long = _pair(_newton_reciprocal, "newton")
    # A spread of seeds and trip counts, including n == 0 (zero trips, the loop body never executes).
    for a in (0.7, 1.0, 1.3, 1.5, 1.9):
        for n in (0.0, 1.0, 2.0, 3.0, 5.0):
            _assert_bits_equal(short, long, a, n)


def _cycles_to_out_valid(simulator: holoso.NumericalSimulator, *inputs) -> int:  # type: ignore[no-untyped-def]
    """Count clocks from accepting a transaction until ``out_valid`` -- the realized latency for this configuration."""
    simulator.set_inputs(*inputs)
    count = 1
    simulator.tick(in_valid=True, out_ready=False)  # accept
    while not simulator.out_valid:
        simulator.tick(in_valid=False, out_ready=False)
        count += 1
    simulator.tick(in_valid=False, out_ready=True)  # release, returning the simulator to idle for reuse
    return count


def test_staged_configuration_is_genuinely_deeper() -> None:
    # This guards the WHOLE axis from passing vacuously: the bit-identity tests above only mean something if the two
    # configurations actually differ in latency. If a future change neutralized ``staged_ops`` (or a kernel stopped
    # using any staged operator), the equivalence tests would still pass trivially -- this catches that by requiring
    # the deeply-staged pipeline to take strictly more cycles than the minimum-latency one on a representative kernel.
    short, long = _pair(_newton_reciprocal, "newton_depth")
    short_cycles = _cycles_to_out_valid(short, 1.5, 3.0)
    long_cycles = _cycles_to_out_valid(long, 1.5, 3.0)
    assert long_cycles > short_cycles, f"staged not deeper: short={short_cycles} long={long_cycles}"


# --------------------------------------------------------------------------------------------------------------------
# Shape 3: a stateful streaming kernel -- a persistent leaky accumulator threaded across a multi-transaction sequence.
# Both latencies are driven on the SAME sequence (state is carried, so the comparison is order-sensitive); a ``reset``
# partway must restore both to the same snapshot, after which they must continue to agree. The state-install boundary
# is latency-derived, so the longer pipeline installing the carry on the same logical edge is exactly what is proved.
# --------------------------------------------------------------------------------------------------------------------


class _LeakyAccumulator:
    """A persistent leaky accumulator fed by a branchy diamond -- carried across transactions (-> ``out_0``)."""

    def __init__(self) -> None:
        self._acc = 0.0

    def __call__(self, x, y):  # type: ignore[no-untyped-def]
        d = x * y + x
        if x > y:
            r = d + 1.0
        else:
            r = d / (y * y + 1.0)  # structurally nonzero divisor keeps the diamond a real branch
        self._acc = self._acc * 0.5 + r  # power-of-two decay -> the carry stays bounded over a long stream
        return self._acc


def test_stateful_stream_timing_invariant_with_reset() -> None:
    short, long = _pair(_LeakyAccumulator().__call__, "leaky_acc")
    rng = np.random.default_rng(0x5EED)
    sequence = [(1.0, 2.0), (2.0, 1.0), (0.5, 0.5)]  # both polarities and the x == y boundary
    sequence += [tuple(float(rng.uniform(-3.0, 3.0)) for _ in range(2)) for _ in range(40)]
    for x, y in sequence:
        _assert_bits_equal(short, long, x, y)
    # A reset partway must restore both pipelines to the same snapshot; they must continue bit-identical afterwards.
    short.reset()
    long.reset()
    for x, y in sequence[:20]:
        _assert_bits_equal(short, long, x, y)


def _drain_to_output(simulator: holoso.NumericalSimulator, inputs, stall: int):  # type: ignore[no-untyped-def]
    """Drive one transaction tick-by-tick with ``stall`` cycles of sustained back-pressure; return the held output."""
    simulator.set_inputs(*inputs)
    while not simulator.in_ready:
        simulator.tick(in_valid=False, out_ready=True)
    simulator.tick(in_valid=True, out_ready=False)  # accept
    while not simulator.out_valid:
        simulator.tick(in_valid=False, out_ready=False)
    held = simulator.output_values[0]
    for _ in range(stall):  # back-pressure: the output must not move and no new input may be accepted
        simulator.tick(in_valid=False, out_ready=False)
        assert simulator.out_valid and not simulator.in_ready
        assert simulator.output_values[0].bits == held.bits
    simulator.tick(in_valid=False, out_ready=True)  # release, advancing the persistent state once
    return held


def test_stateful_backpressure_timing_invariant() -> None:
    # Under sustained back-pressure the two latencies must STILL produce bit-identical outputs on the same sequence:
    # the handshake (freeze the output, accept no input, advance state exactly once per release) is latency-invariant.
    short, long = _pair(_LeakyAccumulator().__call__, "leaky_bp")
    for index, vector in enumerate([(1.0, 2.0), (3.0, 1.0), (0.5, 4.0), (2.0, 2.0), (-1.0, 0.5)]):
        held_short = _drain_to_output(short, vector, stall=2 * index)
        held_long = _drain_to_output(long, vector, stall=2 * index)
        assert held_short.bits == held_long.bits, (
            f"vector={vector} stall={2 * index}: short=0x{held_short.bits:x} ({float(held_short)}) "
            f"long=0x{held_long.bits:x} ({float(held_long)})"
        )


# --------------------------------------------------------------------------------------------------------------------
# Shape 4: an overlap-spilling kernel -- the shared ``overlap_spill_kernel`` whose early branch condition lets the
# block terminator shrink, while a wide chain commits late and SPILLS past the shrunk terminator into BOTH arms. The
# spill's landing frame is computed from operator latency, so the same spill landing on the same logical value under
# two different staging depths (bit-identical across both polarities and at x == y) is precisely what is proved here.
# --------------------------------------------------------------------------------------------------------------------


def test_overlap_spill_timing_invariant() -> None:
    short, long = _pair(overlap_spill_kernel, "overlap_spill")
    rng = np.random.default_rng(0x09E2)
    samples = [(1.0, 2.0, 0.5), (2.0, 1.0, 0.5), (0.5, 0.5, 1.0), (-1.0, 3.0, 2.0), (3.0, -1.0, -2.0)]
    samples += [tuple(float(rng.uniform(-3.0, 3.0)) for _ in range(3)) for _ in range(60)]
    for x, y, z in samples:
        _assert_bits_equal(short, long, x, y, z)

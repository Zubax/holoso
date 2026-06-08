"""Unit tests for the pure-Python verification core."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from holoso import (
    FloatValue,
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FloatFormat,
    FMulILog2OperatorFamily,
    FMulOperator,
    OpConfig,
    UnsupportedConstruct,
)
from holoso._backend.numerical import generate as build_model
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import build
from holoso._mir import lower as lower_to_mir
from ._modelref import (
    bounded,
    default_tolerance,
    encode_inputs,
    evaluate_reference,
    log_uniform_positive,
    random_legal_bits,
    spd_matrix,
    unit_roundoff,
    within,
)

F32 = FloatFormat(8, 24)
FMT = FloatFormat(6, 18)
OPS = OpConfig(FAddOperator(FMT), FMulOperator(FMT), FDivOperator(FMT), FMulILog2OperatorFamily(FMT), FCmpOperator(FMT))


def _run(target):  # type: ignore[no-untyped-def]
    return lower_to_mir(optimize(lower(target)), OPS)


def _exact_int_comparison(a):  # type: ignore[no-untyped-def]
    # Regression (user): two compile-time integers must be compared exactly, not via a lossy float64 fold. As float64
    # both operands round to 9007199254740992.0 and the `==` would misfold true; as integers they are distinct.
    if 9007199254740993 == 9007199254740992:
        r = a
    else:
        r = a + 1.0
    return r


def test_model_exact_integer_comparison_is_not_folded_via_float() -> None:
    model = build_model(build(_run(_exact_int_comparison), "eic"))
    for a in (5.0, 0.0, -2.0):
        assert float(model(a)[0]) == _exact_int_comparison(a)  # a + 1.0 -- the `==` is false for distinct integers


def test_codec_known_binary32_values() -> None:
    assert F32.encode(1.0) == 0x3F800000
    assert F32.encode(2.0) == 0x40000000
    assert F32.encode(0.5) == 0x3F000000
    assert F32.encode(-1.0) == 0xBF800000
    assert F32.encode(0.0) == 0
    assert F32.decode(0x3F800000) == 1.0
    assert F32.decode(0) == 0.0


def test_codec_round_trip_within_unit_roundoff() -> None:
    rng = np.random.default_rng(1)
    for fmt in (F32, FMT):
        u = unit_roundoff(fmt)
        for _ in range(500):
            x = float(rng.uniform(-100.0, 100.0))
            y = fmt.decode(fmt.encode(x))
            assert abs(y - x) <= u * abs(x) + 1e-30


def test_codec_exact_powers_and_simple_fractions() -> None:
    for value in (3.0, 0.25, -7.5, 16.0, 0.125):
        assert FMT.decode(FMT.encode(value)) == value


def test_float_value_factories_and_fields() -> None:
    value = FloatValue.from_bits(F32, 0x3F800001)
    assert value.fmt == F32
    assert value.bits == 0x3F800001
    assert value.sign == 0
    assert value.exponent == 0x7F
    assert value.significand == 0x800001
    assert float(FloatValue.from_float(F32, 1.0)) == 1.0

    with pytest.raises(TypeError, match="from_bits"):
        FloatValue(F32, 1.0)
    with pytest.raises(TypeError, match="float"):
        FloatValue.from_float(F32, 1)
    with pytest.raises(TypeError, match="int"):
        FloatValue.from_bits(F32, True)
    with pytest.raises(ValueError, match="fit"):
        FloatValue.from_bits(F32, 1 << F32.width)


def test_is_legal_rejects_subnormal_and_negative_zero() -> None:
    # exp == 0 with nonzero fraction is subnormal; sign bit with zero magnitude is negative zero.
    assert not FMT.is_legal(0b1)  # subnormal
    neg_zero = 1 << (FMT.width - 1)
    assert not FMT.is_legal(neg_zero)
    assert FMT.is_legal(FMT.encode(1.0))


def test_reference_evaluates_and_flattens() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return [a + b, a * b]

    assert evaluate_reference(f, {"a": 2.0, "b": 3.0}) == [5.0, 6.0]


def test_model_matches_reference_small_kernels() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    inputs = {"a": 1.25, "b": -3.5}
    model = build_model(build(_run(f), "f"))
    got = model(*[inputs[name] for name in model.input_names])
    ref = evaluate_reference(f, inputs)
    rtol, atol = default_tolerance(FMT, model.lir.op_count, magnitude=max(abs(v) for v in inputs.values()))
    assert all(within(float(g), r, rtol, atol) for g, r in zip(got, ref))


def test_tuple_unpacking_matches_python_reference() -> None:
    # The reference runs the kernel as ordinary Python (which unpacks natively), so a bit-faithful hardware model must
    # route the swapped operands identically before the arithmetic.
    def f(a, b):  # type: ignore[no-untyped-def]
        x, y = b, a
        return [x - y, x * y]

    inputs = {"a": 1.25, "b": -3.5}
    model = build_model(build(_run(f), "f"))
    got = model(*[inputs[name] for name in model.input_names])
    ref = evaluate_reference(f, inputs)
    rtol, atol = default_tolerance(FMT, model.lir.op_count, magnitude=max(abs(v) for v in inputs.values()))
    assert all(within(float(g), r, rtol, atol) for g, r in zip(got, ref))


def test_for_counter_reassigned_to_runtime_clears_static_binding() -> None:
    # Regression: a static ``for`` counter later reassigned to a RUNTIME value must lose its compile-time-integer
    # binding, so a subsequent branch on that name is a real runtime branch -- not folded with the stale counter value.
    # With the defect, the loop-counter's static-int binding survived the reassignment and ``1.0 >= i`` was folded as
    # ``1.0 >= 0`` (the counter), silently taking the wrong arm and miscompiling the output for any ``i`` above 1.
    def f(a):  # type: ignore[no-untyped-def]
        for i in range(1):
            i = a  # reassign the loop variable to a runtime value inside the (single-trip) body
        if 1.0 >= i:
            r = 100.0
        else:
            r = 200.0
        return r

    model = build_model(build(_run(f), "f"))
    for a in (5.0, 0.5, -3.0, 2.0):  # a>1 must take the else arm (200); the defect always returns 100
        assert float(model(a)[0]) == float(f(a)), f"mismatch at a={a}"


def test_for_counter_reassigned_after_loop_clears_static_binding() -> None:
    # The same hazard when the counter (leaked after the loop) is reassigned to a runtime value past the loop: the
    # stale static-int binding must not fold a later branch. Without the fix, ``1.0 >= i`` folds with the counter's
    # final value (2) and the conditional state update is dropped on every call.
    def f(a):  # type: ignore[no-untyped-def]
        acc = 0.0
        for i in range(3):
            acc = acc + 1.0
        i = a
        if 1.0 >= i:
            acc = acc + 50.0
        return acc

    model = build_model(build(_run(f), "f"))
    for a in (5.0, 0.5, 2.0, -1.0):
        assert float(model(a)[0]) == float(f(a)), f"mismatch at a={a}"


def test_runtime_reassigned_for_counter_is_not_a_static_index() -> None:
    # A static ``for`` counter reassigned to a runtime value is no longer a compile-time integer, so using it as an
    # array index must be REJECTED (it is out-of-subset -- plain Python raises ``TypeError`` indexing with a float).
    # With the defect, the stale static-int binding let the index silently resolve to the counter value, miscompiling
    # an invalid kernel into a constant element selection.
    def f(a, b, c):  # type: ignore[no-untyped-def]
        vec = [a, b, c]
        for i in range(1):
            i = a  # i is now a runtime value, not a compile-time index
        return vec[i]

    with pytest.raises(UnsupportedConstruct):
        build(_run(f), "f")


def test_for_counter_reassign_keeps_scan_and_lowering_in_lockstep() -> None:
    # Regression: the persistent-state reachability scan must demote a runtime-reassigned ``for`` counter exactly as
    # lowering does. Here ``t`` (the counter) is reassigned to a runtime value, so ``if t >= ...`` is a real branch and
    # its else arm (a ``while`` writing ``self.s``) IS reachable. If the scan still folded the branch with the stale
    # counter (0), it would either drop ``self.s`` from the persistent-state set (a silent miscompile) or, once
    # lowering treats the branch as runtime, open a header phi for an attribute the scan never registered -- a
    # ``KeyError`` crash in ``_lower_while``. The state set and the emitted phis must agree.
    class K:
        def __init__(self) -> None:
            self.s = 4.0

        def step(self, a):  # type: ignore[no-untyped-def]
            for t in range(1):
                t = a  # reassign the counter to a runtime value -> the following branch is dynamic
            # Both sides are otherwise compile-time (the counter and a literal), so if the scan failed to demote the
            # reassigned counter it would fold ``0 < 1.0`` to True and never scan the else arm. With the counter
            # correctly demoted, this is a real runtime branch and the else arm's ``self.s`` write is reachable.
            if t < 1.0:
                pass
            else:
                c = 2.0
                while c > 0.0:
                    c = c - 1.0
                    self.s = 5.0  # written only on the else path's loop; must be persistent state
            return self.s

    hir = lower(K().step)
    assert "s" in {slot.name for slot in hir.state_slots}  # the loop-written attr must be registered as state

    model = build_model(build(_run(K().step), "k"))
    ref = K()
    for a in (0.5, 5.0, -7.0):  # t<1 (then arm, no write) for a<1; else arm runs the loop and writes s for a>=1
        assert float(model(a)[0]) == float(ref.step(a)), f"mismatch at a={a}"


def test_for_counter_reassigned_inside_while_is_demoted_after_the_loop() -> None:
    # Regression (differential fuzzer): a leaked ``for`` counter reassigned to a runtime value INSIDE a ``while`` body
    # must stay demoted after the loop. ``_lower_while`` restored the preheader static-int map verbatim on exit, which
    # resurrected the stale compile-time counter value (the body's ``_invalidate_static_int`` was undone). A later
    # comparison ``if i < 0.0`` was then folded against the stale counter (0) instead of the runtime value -- a SILENT
    # miscompile that took the wrong arm. The post-loop fold must follow the runtime value, matching plain Python.
    def kernel(a):  # type: ignore[no-untyped-def]
        for i in range(1):  # leaks i == 0 (a compile-time integer) into the enclosing scope
            pass
        w = 0.0
        while w < 1.0:
            i = a  # demote the counter to a runtime value INSIDE the while body
            w = w + 1.0
        r = 0.0
        if i < 0.0:  # must be a real runtime branch on the reassigned value, not a fold on the stale counter (0)
            r = 100.0
        else:
            r = 200.0
        return r

    # With the stale binding resurrected, ``i`` folds to 0 -> ``0 < 0.0`` is always False -> r is always 200.0.
    assert len(lower(kernel).blocks) > 1  # the comparison is a real branch, not folded away
    model = build_model(build(_run(kernel), "k"))
    for a in (-5.0, -0.5, 0.5, 7.0):
        assert float(model(a)[0]) == float(kernel(a)), f"mismatch at a={a}"  # 100.0 for a<0, else 200.0


def test_for_counter_reassigned_inside_while_rejects_later_static_use() -> None:
    # Companion to the above: once the counter is demoted by a while-body reassignment, a later static-only use of it
    # (a shift exponent) must be REJECTED as runtime, not silently folded to the stale compile-time counter value.
    def kernel(a):  # type: ignore[no-untyped-def]
        for i in range(2):
            pass
        w = 0.0
        while w < 1.0:
            i = a  # runtime reassignment inside the loop -> i is no longer a compile-time integer afterwards
            w = w + 1.0
        return a * 2.0**i  # a runtime exponent must be rejected, never folded against the stale counter

    with pytest.raises(UnsupportedConstruct, match="compile-time integer"):
        lower(kernel)


def test_model_uses_exact_ilog2_for_wide_supported_shift() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a * 16.0

    fmt = FloatFormat(3, 4)
    ops = OpConfig(
        FAddOperator(fmt), FMulOperator(fmt), FDivOperator(fmt), FMulILog2OperatorFamily(fmt), FCmpOperator(fmt)
    )
    model = build_model(build(lower_to_mir(optimize(lower(f)), ops), "f"))
    assert model(FloatValue.from_float(fmt, 0.5))[0] == FloatValue.from_float(fmt, 8.0)


def test_model_handles_unused_input_ports() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return b

    model = build_model(build(_run(f), "f"))
    assert [load.name for load in model.lir.float_inputs] == ["a", "b"]
    assert [load.dst.index for load in model.lir.float_inputs] == [0, 1]
    assert model.lir.float_regfile.nload == 2
    assert model(1.0, 2.0)[0] == FloatValue.from_float(FMT, 2.0)


def test_model_rejects_ambiguous_int_and_mismatched_float_value_format() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a

    model = build_model(build(_run(f), "f"))
    assert model(1.0)[0] == FloatValue.from_float(FMT, 1.0)

    with pytest.raises(TypeError, match="FloatValue or float"):
        model(1)
    with pytest.raises(ValueError, match="expected"):
        model(FloatValue.from_float(F32, 1.0))


def test_model_is_bit_exact_for_wide_zkf_multiply_regression() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a * b

    fmt = FloatFormat(8, 36)
    ops = OpConfig(
        FAddOperator(fmt), FMulOperator(fmt), FDivOperator(fmt), FMulILog2OperatorFamily(fmt), FCmpOperator(fmt)
    )
    mir = lower_to_mir(optimize(lower(f)), ops)
    model = build_model(build(mir, "f"))
    got = model(
        FloatValue.from_bits(fmt, 0x42BF30E6505),
        FloatValue.from_bits(fmt, 0xBD734F60F3A),
    )
    assert got[0].bits == 0xC0B5B6B31D9


def test_model_matches_reference_ekf1_stateless() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    rng = np.random.default_rng(12345)
    cov = spd_matrix(rng, 3, 0.5, 2.0)
    inputs = {
        "P00": float(cov[0, 0]),
        "P01": float(cov[0, 1]),
        "P02": float(cov[0, 2]),
        "P11": float(cov[1, 1]),
        "P12": float(cov[1, 2]),
        "P22": float(cov[2, 2]),
        "Q_R": log_uniform_positive(rng, 1e-3, 1e-1),
        "Q_g": log_uniform_positive(rng, 1e-3, 1e-1),
        "Q_i": log_uniform_positive(rng, 1e-3, 1e-1),
        "R_ct": log_uniform_positive(rng, 1e-1, 1.0),
        "R_shunt": log_uniform_positive(rng, 1e-1, 1.0),
        "dt": bounded(rng, 1e-3, 1e-2),
        "x_R": bounded(rng, -1.0, 1.0),
        "x_g": bounded(rng, -1.0, 1.0),
        "x_i": bounded(rng, -1.0, 1.0),
        "z_ct": bounded(rng, -1.0, 1.0),
        "z_shunt": bounded(rng, -1.0, 1.0),
    }
    model = build_model(build(_run(ekf1_stateless.update_x_P), "ekf1_stateless"))
    got = model(*[inputs[name] for name in model.input_names])
    ref = evaluate_reference(ekf1_stateless.update_x_P, inputs)
    assert len(ref) == 9 and all(np.isfinite(ref))
    rtol, atol = default_tolerance(FMT, model.lir.op_count, magnitude=max(abs(v) for v in inputs.values()))
    assert all(within(float(g), r, rtol, atol) for g, r in zip(got, ref))


def test_model_matches_reference_aggregates() -> None:
    def f(a, b, c):  # type: ignore[no-untyped-def]
        v = [a, b, c]
        head = v[0:2]
        return [v[2], *head]  # index, slice, and unpack -> [c, a, b]

    inputs = {"a": 1.0, "b": 2.0, "c": 3.0}
    model = build_model(build(_run(f), "agg"))
    got = model(*[inputs[name] for name in model.input_names])
    ref = evaluate_reference(f, inputs)  # these aggregate ops run identically in plain Python
    rtol, atol = default_tolerance(FMT, max(model.lir.op_count, 1), magnitude=3.0)
    assert all(within(float(g), r, rtol, atol) for g, r in zip(got, ref))


def test_model_matches_reference_ekf1_stateful() -> None:
    examples = str(Path(__file__).resolve().parents[1] / "examples")
    sys.path.insert(0, examples)
    import ekf1_stateful

    rng = np.random.default_rng(54321)
    cov = spd_matrix(rng, 3, 0.5, 2.0)
    p_urt = [float(cov[i, j]) for i, j in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))]
    x = [bounded(rng, -1.0, 1.0) for _ in range(3)]
    r_diag = [log_uniform_positive(rng, 1e-1, 1.0) for _ in range(2)]
    q_diag = [log_uniform_positive(rng, 1e-3, 1e-1) for _ in range(3)]
    dt = bounded(rng, 1e-3, 1e-2)
    step_inputs = {"dt": dt, "u_shunt": bounded(rng, -1.0, 1.0), "di_dt": bounded(rng, -1.0, 1.0)}

    def fresh():  # type: ignore[no-untyped-def]
        return ekf1_stateful.Ekf1(x=list(x), P_urt=list(p_urt), R_diag=list(r_diag), Q_diag=np.array(q_diag))

    model = build_model(build(_run(fresh().update), "ekf1_stateful"))
    got = model(*[step_inputs[name] for name in model.input_names])

    # update() is ordinary executable numpy, so the reference is just one native step from the same reset; the new
    # state in state-port order is x'(3) then P_urt'(6).
    reference = fresh()
    reference.update(**step_inputs)
    ref = [float(v) for v in (*reference.x, *reference.P_urt)]
    assert len(ref) == 9 and all(np.isfinite(ref))
    rtol, atol = default_tolerance(FMT, model.lir.op_count, magnitude=max(1.0, max(abs(v) for v in ref)))
    assert all(within(float(g), r, rtol, atol) for g, r in zip(got, ref))


def test_model_pickles_and_round_trips() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    model = build_model(build(_run(f), "f"))
    inputs = [1.25, -3.5]
    restored = pickle.loads(pickle.dumps(model))
    assert restored(*inputs) == model(*inputs)


def test_tolerance_predicate() -> None:
    assert within(1.0, 1.0, 0.0, 0.0)
    assert within(1.001, 1.0, 0.01, 0.0)
    assert not within(1.1, 1.0, 0.01, 0.0)
    assert within(float("inf"), float("inf"), 1.0, 1.0)
    assert not within(float("inf"), 1.0, 1.0, 1.0)


def test_default_tolerance_scales_with_format_and_size() -> None:
    coarse = default_tolerance(FMT, 100)[0]
    fine = default_tolerance(F32, 100)[0]
    assert coarse > fine  # 6/18 has a larger unit roundoff than 8/24
    assert default_tolerance(FMT, 200)[0] > default_tolerance(FMT, 10)[0]


def test_sampling_legal_and_spd() -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        bits = random_legal_bits(FMT, rng)
        assert FMT.is_legal(bits) and FMT.is_finite(bits)
    cov = spd_matrix(rng, 3)
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)
    encoded = encode_inputs(FMT, {"a": 1.0, "b": 2.0})
    assert set(encoded) == {"a", "b"} and encoded["a"] == FMT.encode(1.0)


class _Iir1Lpf:
    """A single-pole low-pass IIR with a boolean first-sample branch (mirrors examples/iir1_lpf.py)."""

    def __init__(self) -> None:
        self.alpha = 2**-16
        self.y = 0.0
        self._first = True

    def __call__(self, x):  # type: ignore[no-untyped-def]
        if self._first:
            self._first = False
            self.y = x
        else:
            self.y += self.alpha * (x - self.y)
        return self.y


def test_model_executes_first_sample_branch() -> None:
    # The model must follow the control-flow path: the first sample takes the then-arm (y = x exactly), and every
    # later sample takes the else-arm (the IIR update), staying bit-exact to the model's own re-evaluation.
    model = build_model(build(_run(_Iir1Lpf().__call__), "iir1_lpf"))
    reference = _Iir1Lpf()
    stream = [1.0, 2.0, 3.0, 0.5, -1.0, 8.0]
    rtol, atol = default_tolerance(FMT, model.lir.op_count, magnitude=8.0)
    for index, x in enumerate(stream):
        (got,) = model(x)
        if index == 0:
            assert float(got) == FMT.decode(FMT.encode(x))  # first sample is exactly x (then-arm)
        assert within(float(got), reference(x), rtol, atol)


def test_model_branch_reset_restarts_the_first_sample_arm() -> None:
    # reset() reloads the boolean state, so the first-sample arm fires again after a reset.
    model = build_model(build(_run(_Iir1Lpf().__call__), "iir1_lpf"))
    model(1.0)
    second = float(model(5.0)[0])
    assert second != 5.0  # the second sample took the IIR arm, not y = x
    model.reset()
    assert float(model(5.0)[0]) == FMT.decode(FMT.encode(5.0))  # first-sample arm again


class _PID:
    def __init__(self) -> None:
        self.kp = 0.5
        self.ki = 0.0625
        self.kd = 0.25
        self.limit = 4.0
        self.integral = 0.0
        self.prev_error = 0.0
        self._started = False  # boolean state: the first update has no derivative

    def __call__(self, setpoint, measurement):  # type: ignore[no-untyped-def]
        error = setpoint - measurement
        candidate = self.integral + self.ki * error
        if self._started:
            derivative = self.kd * (error - self.prev_error)
        else:
            derivative = 0.0
        self.prev_error = error
        self._started = True
        u = self.kp * error + candidate + derivative
        if u > self.limit:
            u = self.limit
        elif u < -self.limit:
            u = -self.limit
        else:
            self.integral = candidate
        return u


def test_model_pid_controller_all_arms_anti_windup_and_first_update() -> None:
    # A float comparison drives each saturation arm; the integrator freezes while saturated (anti-windup); the boolean
    # ``_started`` state suppresses the derivative on the first update (no prev_error yet) and enables it after.
    model = build_model(build(_run(_PID().__call__), "pid"))
    reference = _PID()
    ui = model.output_names.index("out_0")
    rtol, atol = default_tolerance(FMT, model.lir.op_count, magnitude=10.0)
    stream = [(10.0, 0.0), (10.0, 0.5), (0.0, 1.0), (0.5, 0.5), (-10.0, 0.0), (-10.0, -0.5), (0.0, 0.0)]
    for setpoint, measurement in stream:
        got = float(model(setpoint, measurement)[ui])
        assert within(got, reference(setpoint, measurement), rtol, atol)
    assert abs(reference.integral) < 0.1  # the integrator stayed bounded despite the saturating commands


def _remainder(x, y):  # type: ignore[no-untyped-def]
    # Mirrors examples/remainder.py: boolean quotient parity, a walrus-bound ``twice_r``, augmented assignment, and the
    # nested unit-place tie branch (``elif twice_r == ay: if quotient_is_odd:``) that the frontend folds to one branch.
    ax, ay = abs(x), abs(y)
    scaled = ay
    while scaled * 2 <= ax:
        scaled += scaled
    r = ax
    quotient_is_odd = False
    while scaled > ay:  # halve down to -- but not past -- the unit place (|y|*0.5 can clamp back in subnormal-free ZKF)
        if r >= scaled:
            r -= scaled
        scaled *= 0.5
    if r >= ay:
        r -= ay
        quotient_is_odd = True
    if (twice_r := r * 2) > ay:
        r -= ay
    elif twice_r == ay:
        if quotient_is_odd:
            r -= ay
    return -r if x < 0 else r


def _walrus(x):  # type: ignore[no-untyped-def]
    # ``(t := x*2)`` evaluates the subexpression once, binds ``t``, and yields it to the comparison; ``t`` then stays
    # visible to both arms (it is bound in the test, before the branch), as in Python.
    if (t := x * 2.0) > 4.0:
        r = t + 1.0
    else:
        r = t
    return r


def test_model_walrus_binds_once_and_stays_visible_after_the_test() -> None:
    model = build_model(build(_run(_walrus), "walrus"))
    for x in (3.0, 1.0, 2.0, -5.0, 0.0):  # >2 takes the then arm (reads t), else reads the same bound t
        assert float(model(x)[0]) == _walrus(x)


def _walrus_loop(x):  # type: ignore[no-untyped-def]
    # A walrus reassigning a pre-defined accumulator inside a loop body must be loop-carried (a header phi), so the
    # accumulation persists across iterations rather than resetting to the preheader value each trip.
    acc = 0.0
    i = 0.0
    while i < 4.0:
        y = (acc := acc + x)  # noqa: F841 -- the walrus side effect (rebinding acc) is the point
        i = i + 1.0
    return acc


def test_model_walrus_reassigned_loop_variable_is_loop_carried() -> None:
    model = build_model(build(_run(_walrus_loop), "walrus_loop"))
    for x in (2.5, 1.0, -3.0):  # the defect (walrus invisible to the loop scan) returns 0.0 instead of 4*x
        assert float(model(x)[0]) == _walrus_loop(x)


def test_model_remainder_iterative_reduction_is_exact_and_matches_ieee() -> None:
    # The data-dependent scaled-subtraction reduction is exact (every subtraction is Sterbenz-exact, no rounding), so
    # the model reproduces math.remainder bit-for-bit -- including the round-to-nearest-even ties (6/4 -> -2, 2/4 -> 2)
    # and negative exact multiples (which produce a -0.0, accepted by the example) -- for any normal-magnitude result
    # (these cases are; a subnormal-sized remainder would flush to +0 in subnormal-free ZKF). Regression: a divisor
    # equal to the smallest normal must still TERMINATE -- halving the unit place would clamp back to it and loop
    # forever, which the explicit unit-place handling avoids.
    import math

    model = build_model(build(_run(_remainder), "remainder"))
    ui = model.output_names.index("out_0")
    min_normal = 2.0 ** (1 - (2 ** (FMT.wexp - 1) - 1))
    cases = [(5.0, 3.0), (10.0, 3.0), (7.5, 2.0), (-7.5, 2.0), (13.0, 4.0), (6.0, 4.0), (2.0, 4.0), (0.0, 2.0)]
    cases += [(-3.0, 3.0), (-6.0, 3.0), (-9.0, 3.0)]  # negative exact multiples (the -0.0 the example accepts)
    cases += [(0.0, min_normal), (min_normal, min_normal), (3.0 * min_normal, 2.0 * min_normal)]
    for x, y in cases:
        assert float(model(x, y)[ui]) == math.remainder(x, y)


class _SchmittTrigger:
    def __init__(self) -> None:
        self.high = 1.0
        self.low = -1.0
        self.y = 0.0

    def __call__(self, x):  # type: ignore[no-untyped-def]
        if x > self.high:
            self.y = 1.0
        elif x < self.low:
            self.y = 0.0
        return self.y


def test_model_schmitt_trigger_hysteresis() -> None:
    # The output latches at the thresholds and holds across the deadband (a comparison branch with a one-arm phi).
    model = build_model(build(_run(_SchmittTrigger().__call__), "schmitt"))
    reference = _SchmittTrigger()
    for x in [0.0, 0.5, 1.5, 0.5, -0.5, -1.5, -0.5, 0.5, 2.0]:
        assert float(model(x)[0]) == reference(x)  # 0.0/1.0 are exact in ZKF


def test_compare_float_values_exact_for_wide_formats() -> None:
    # The model's comparison must be exact, not via a lossy float64 decode: two values differing only in the lowest
    # mantissa bit of a >53-bit mantissa must compare unequal (decode would collapse them).
    from holoso._value import compare_float_values

    fmt = FloatFormat(wexp=8, wman=60)
    bias = (1 << (fmt.wexp - 1)) - 1
    one = FloatValue.from_bits(fmt, bias << fmt.wman)
    one_plus_ulp = FloatValue.from_bits(fmt, (bias << fmt.wman) | 1)
    assert fmt.decode(one.bits) == fmt.decode(one_plus_ulp.bits)  # lossy: float64 cannot tell them apart
    assert compare_float_values(one, one_plus_ulp) == -1
    assert compare_float_values(one_plus_ulp, one) == 1
    assert compare_float_values(one, one) == 0
    # Signs, zero, and infinities form a total order (ZKF has no NaN).
    neg_one = one.apply_sign(negate=True, absolute=False)
    zero = FloatValue.from_float(fmt, 0.0)
    pos_inf = FloatValue.from_float(fmt, float("inf"))
    neg_inf = FloatValue.from_float(fmt, float("-inf"))
    ascending = [neg_inf, neg_one, zero, one, pos_inf]
    for lower_value, higher_value in zip(ascending, ascending[1:]):
        assert compare_float_values(lower_value, higher_value) == -1
        assert compare_float_values(higher_value, lower_value) == 1


def test_model_unrolled_for_loop_newton_reciprocal() -> None:
    # A fixed-count `for` unrolls to a flat datapath; Newton-Raphson converges to 1/x on the restricted domain.
    import math

    def reciprocal(x):  # type: ignore[no-untyped-def]
        y = 1.5 - 0.5 * x
        for _ in range(4):
            y = y * (2.0 - x * y)
        return y

    model = build_model(build(_run(reciprocal), "newton"))
    assert len(model.lir.blocks) == 1  # fully unrolled to a single straight-line block
    for x in [0.5, 0.75, 1.0, 1.3, 1.7, 2.0]:
        assert math.isclose(float(model(x)[0]), 1.0 / x, rel_tol=1e-5)


class _CordicSinCos:
    def __init__(self, iterations: int = 12) -> None:
        gain = 1.0
        for i in range(iterations):
            gain *= 1.0 / np.sqrt(1.0 + 2.0 ** (-2 * i))
        self.gain = float(gain)
        self.angles = tuple(float(np.arctan(2.0**-i)) for i in range(iterations))

    def __call__(self, theta):  # type: ignore[no-untyped-def]
        x, y, z = self.gain, 0.0, theta
        for i in range(12):
            if z >= 0.0:
                x_next, y_next, z = x - y * (2.0**-i), y + x * (2.0**-i), z - self.angles[i]
            else:
                x_next, y_next, z = x + y * (2.0**-i), y - x * (2.0**-i), z + self.angles[i]
            x, y = x_next, y_next
        return x, y


def test_model_unrolled_cordic_sin_cos() -> None:
    # An unrolled loop with a compile-time `2**-i` shift, a constant arctan table indexed by the counter, and a
    # per-iteration sign branch computes sin/cos.
    model = build_model(build(_run(_CordicSinCos().__call__), "cordic"))
    cos_index, sin_index = model.output_names.index("out_0"), model.output_names.index("out_1")
    for theta in [0.0, 0.3, 0.7, -0.5, 1.0, -1.0]:
        assert abs(float(model(theta)[cos_index]) - np.cos(theta)) < 1e-3
        assert abs(float(model(theta)[sin_index]) - np.sin(theta)) < 1e-3


class _LoopAccumulator:
    def __init__(self) -> None:
        self.acc = 0.0

    def __call__(self, x):  # type: ignore[no-untyped-def]
        for _ in range(3):
            self.acc = self.acc + x
        return self.acc


def test_model_attribute_written_only_in_loop_is_persistent_state() -> None:
    # An attribute assigned only inside a loop body must still become persistent state (regression: the write must not
    # be dropped). It accumulates within a call and carries across calls.
    model = build_model(build(_run(_LoopAccumulator().__call__), "accum"))
    assert [slot.name for slot in model.lir.float_state_slots] == ["acc"]
    reference = _LoopAccumulator()
    assert float(model(1.0)[0]) == reference(1.0)  # 3*1
    assert float(model(2.0)[0]) == reference(2.0)  # previous 3 + 3*2 = 9 (state carried)


class _LiveInClobberedByLiveOut:
    # Regression (Codex F1): a non-phi state live-out (here an input) must not be installed into the slot register in
    # the entry block, where the branch and arms still read the live-in.
    def __init__(self) -> None:
        self.y = 2.0

    def __call__(self, x):  # type: ignore[no-untyped-def]
        if self.y > 0.0:  # reads the live-in y
            z = self.y
        else:
            z = 0.0
        self.y = x  # live-out installed only at the boundary, not before the reads above
        return z


def test_model_state_liveout_does_not_clobber_live_in_branch() -> None:
    model = build_model(build(_run(_LiveInClobberedByLiveOut().__call__), "f1"))
    reference = _LiveInClobberedByLiveOut()
    for x in [5.0, -1.0, 3.0]:
        assert float(model(x)[0]) == reference(x)


class _LiveInReadAfterPhi:
    # Regression (Codex F2): a state phi must not be coalesced onto the slot register when the live-in is still read
    # afterwards (here returned), or the phi install corrupts the returned live-in.
    def __init__(self) -> None:
        self.y = 1.0

    def __call__(self, x):  # type: ignore[no-untyped-def]
        old = self.y
        if x > 0.0:
            self.y = x
        else:
            self.y = x + 10.0
        return old


def test_model_state_phi_does_not_clobber_returned_live_in() -> None:
    model = build_model(build(_run(_LiveInReadAfterPhi().__call__), "f2"))
    reference = _LiveInReadAfterPhi()
    for x in [2.0, 3.0, -4.0]:
        assert float(model(x)[0]) == reference(x)


class _SignedStateLiveOut:
    # Regression (Codex F3): a sign-conditioned state live-out must persist with the sign applied.
    def __init__(self) -> None:
        self.y = 0.0

    def __call__(self, x):  # type: ignore[no-untyped-def]
        if x > 0.0:
            t = x + 1.0
        else:
            t = self.y
        self.y = -t
        return self.y


def test_model_signed_state_liveout_persists_with_sign() -> None:
    model = build_model(build(_run(_SignedStateLiveOut().__call__), "f3"))
    reference = _SignedStateLiveOut()
    for x in [2.0, -1.0, -1.0, 4.0]:
        assert float(model(x)[0]) == reference(x)


def _neg_abs_phi(x):  # type: ignore[no-untyped-def]
    # A value whose sign is conditioned by the branch (-x on one arm, x on the other) merges through a phi -> -|x|.
    if x > 0.0:
        y = -x
    else:
        y = x
    return y


def test_model_sign_conditioned_phi_arm() -> None:
    # Regression (#16): the merge resolution carries a per-arm folded sign, so a sign-conditioned phi arm lowers and
    # evaluates correctly (it was previously rejected with "a sign-conditioned value merged by a phi").
    model = build_model(build(_run(_neg_abs_phi), "negabs"))
    for x in [3.0, -2.5, 0.0, 7.25, -10.0]:
        assert float(model(x)[0]) == (-x if x > 0.0 else x)


def _while_sum(x, n):  # type: ignore[no-untyped-def]
    # A real back-edge loop: add x to an accumulator a runtime number of times (n down to 0). Loop-carried locals only.
    acc = 0.0
    i = n
    while i > 0.0:
        acc = acc + x
        i = i - 1.0
    return acc


def test_model_while_loop_accumulates() -> None:
    # Regression (#14): a variable-count while loop follows the back-edge in the model and converges to x * n.
    model = build_model(build(_run(_while_sum), "whilesum"))
    for x, n in [(1.0, 3.0), (2.0, 0.0), (0.5, 5.0), (-1.0, 4.0)]:
        assert float(model(x, n)[0]) == pytest.approx(x * n)


class _WhileIntegrator:
    # A while loop that updates a persistent state attribute a runtime number of times: exercises the state scan's
    # while handling (the attribute must be classified as persistent state) and a loop-carried state phi.
    def __init__(self) -> None:
        self._total = 0.0

    def __call__(self, x, n):  # type: ignore[no-untyped-def]
        i = n
        while i > 0.0:
            self._total = self._total + x
            i = i - 1.0
        return self._total


def test_model_while_loop_carries_persistent_state() -> None:
    model = build_model(build(_run(_WhileIntegrator().__call__), "whileint"))
    reference = _WhileIntegrator()
    for x, n in [(1.0, 2.0), (3.0, 1.0), (0.5, 4.0), (2.0, 0.0)]:
        assert float(model(x, n)[0]) == pytest.approx(reference(x, n))


def _for_counter_inside_while(x):  # type: ignore[no-untyped-def]
    # Regression (Codex iter5): a `for` counter bound inside a `while` body is a loop-carried local; its value at the
    # body's end must flow through the while-header phi, not be dropped when the preheader environment is restored.
    j = 0.0
    i = 0.0
    while i < x:
        for j in range(2):
            pass
        i = i + 1.0
    return j


def test_model_for_counter_inside_while_is_loop_carried() -> None:
    model = build_model(build(_run(_for_counter_inside_while), "fciw"))
    for x in [0.0, 1.0, 2.0, 3.0]:
        assert float(model(x)[0]) == _for_counter_inside_while(x)


class _CounterGatedWhileState:
    # Regression (iter5): a leaked `for` counter reassigned in a `while` must be demoted from the static-int map for the
    # whole body, so an in-body branch on it is a real runtime branch -- both arms lowered, so the attribute written on
    # the otherwise-"folded-away" arm is correctly registered as persistent state and updated.
    def __init__(self) -> None:
        self.s1 = -1.0
        self._s2 = 2.0

    def step(self, a):  # type: ignore[no-untyped-def]
        for i in range(3):
            pass
        w = 2.0
        while w > 0.0:
            if 8.0 > i:
                self.s1 = a
            else:
                self._s2 = a
            i = a
            w = w - 1.0
        return self._s2


def _counter_dead_arm_in_while(x):  # type: ignore[no-untyped-def]
    # Regression (Codex iter7): a leaked `for` counter assigned only on a statically-dead path (`if False:`) inside a
    # `while` is NOT actually reassigned, so it stays a compile-time int -- a later static index must still resolve,
    # not be rejected (the demotion is fold-aware: only a counter reassigned on a reachable path is demoted).
    table = (10.0, 20.0, 30.0)
    for i in range(3):
        pass
    c = x
    while c > 0.0:
        if False:
            i = x  # noqa -- dead arm: i is not actually reassigned
        c = c - 1.0
    return table[i]


def test_model_counter_assigned_only_on_dead_path_stays_static() -> None:
    model = build_model(build(_run(_counter_dead_arm_in_while), "cdaiw"))
    for x in [0.0, 1.0, 3.0]:
        assert float(model(x)[0]) == _counter_dead_arm_in_while(x)  # table[2] == 30.0


def _zero_trip_inner_for(x):  # type: ignore[no-untyped-def]
    # Regression (Codex iter8): `for i in range(0)` runs zero times and never binds `i` (Python semantics), so it must
    # not be recorded as a loop-carried reassignment of the outer leaked counter -- the later static index still uses
    # the outer for's leaked value.
    table = (10.0, 20.0, 30.0)
    for i in range(3):
        pass
    while x > 0.0:
        for i in range(0):
            pass
        x = x - 1.0
    return table[i]


def test_model_zero_trip_inner_for_keeps_outer_counter_static() -> None:
    model = build_model(build(_run(_zero_trip_inner_for), "ztif"))
    for x in [0.0, 2.0, 5.0]:
        assert float(model(x)[0]) == _zero_trip_inner_for(x)  # table[2] == 30.0


def test_model_attr_written_under_counter_gated_branch_in_while() -> None:
    model = build_model(build(_run(_CounterGatedWhileState().step), "cgws"))
    assert "_s2" in {slot.name for slot in build(_run(_CounterGatedWhileState().step), "cgws").float_state_slots}
    reference = _CounterGatedWhileState()
    for a in [10.0, 9.0, 8.0, 0.0, -3.0, 12.0]:
        assert float(model(a)[0]) == reference.step(a)


class _SharedConstBranchCondition:
    # Regression (Codex F4): a constant branch condition shared by sibling branches (the interned `self.flag`) must be
    # materialized in every branching block that uses it, not only the first, or a path through the other reads a stale
    # boolean register.
    def __init__(self) -> None:
        self.flag = True

    def __call__(self, x):  # type: ignore[no-untyped-def]
        if x > 0.0:
            if self.flag:
                y = 1.0
            else:
                y = 2.0
        else:
            if self.flag:
                y = 3.0
            else:
                y = 4.0
        return y


def test_model_shared_constant_branch_condition() -> None:
    model = build_model(build(_run(_SharedConstBranchCondition().__call__), "f4"))
    reference = _SharedConstBranchCondition()
    for x in [1.0, -1.0, 2.0, -3.0]:
        assert float(model(x)[0]) == reference(x)


class _DeadLoopWrite:
    def __init__(self) -> None:
        self.y = 1.25

    def __call__(self, x):  # type: ignore[no-untyped-def]
        for _ in range(0):  # statically empty: the body never lowers
            self.y = x
        return self.y


class _DeadIfWrite:
    def __init__(self) -> None:
        self.y = 2.5

    def __call__(self, x):  # type: ignore[no-untyped-def]
        if False:  # statically never taken
            self.y = x
        return self.y


def test_model_statically_dead_attribute_write_is_not_state() -> None:
    # Regression (Codex F5): a write under a never-taken static branch or an empty loop must NOT be classified as
    # persistent state -- that changed the interface (a spurious state port) and crashed slot registration when the
    # dead-written attribute was not otherwise read. The attribute stays a compile-time constant.
    for kernel, constant in [(_DeadLoopWrite, 1.25), (_DeadIfWrite, 2.5)]:
        lir = build(_run(kernel().__call__), "dead")
        assert [slot.name for slot in lir.float_state_slots] == []  # no state slot; no crash building it
        model = build_model(lir)
        assert float(model(9.0)[0]) == constant  # unchanged across calls -- it never became state
        assert float(model(-3.0)[0]) == constant


class _CounterDependentEmptyInner:
    def __init__(self) -> None:
        self.y = 1.25

    def __call__(self, x):  # type: ignore[no-untyped-def]
        for i in range(1):  # the only outer trip has i == 0
            for _ in range(i):  # range(0) on that trip: the body never lowers
                self.y = x
        return self.y


class _CounterDependentLiveInner:
    def __init__(self) -> None:
        self.s = 0.0

    def __call__(self, x):  # type: ignore[no-untyped-def]
        for i in range(3):  # inner runs 0 + 1 + 2 = 3 times per call
            for _ in range(i):
                self.s = self.s + x
        return self.s


def test_model_counter_dependent_empty_inner_loop_is_not_state() -> None:
    # Regression (Codex F6): the attribute-write scan must mirror the unroll counter-by-counter, so a counter-dependent
    # inner range that is empty on every outer trip contributes no state (the scan ran before counter binding before
    # and over-approximated, crashing slot registration / adding a spurious port). A live nested loop still is state.
    dead = build(_run(_CounterDependentEmptyInner().__call__), "f6dead")
    assert [slot.name for slot in dead.float_state_slots] == []
    assert float(build_model(dead)(9.0)[0]) == 1.25  # builds without KeyError; y stays constant

    live = build(_run(_CounterDependentLiveInner().__call__), "f6live")
    assert [slot.name for slot in live.float_state_slots] == ["s"]
    model, reference = build_model(live), _CounterDependentLiveInner()
    for x in [1.0, 1.0, 2.0]:
        assert float(model(x)[0]) == reference(x)  # 3, 6, 12


class _ReturnInLiteralIfArm:
    def __init__(self) -> None:
        self.y = 2.0

    def __call__(self, x):  # type: ignore[no-untyped-def]
        if True:
            return x + self.y  # the only live path; reads y as a constant
        self.y = x  # unreachable: lowering stops at the return above
        return self.y


def test_model_return_in_literal_if_arm_ends_the_scan() -> None:
    # Regression (Codex/reviewer F7): the attribute-write scan must propagate a return reached in a taken literal-if
    # arm and stop, exactly as lowering does -- otherwise the dead post-if write is misclassified as state (spurious
    # port, or a slot-registration crash when the attribute is not otherwise read).
    lir = build(_run(_ReturnInLiteralIfArm().__call__), "f7")
    assert [slot.name for slot in lir.float_state_slots] == []
    assert float(build_model(lir)(5.0)[0]) == 7.0  # x + y = 5 + 2


class _CounterLeakAcrossArms:
    def __init__(self) -> None:
        self.y = 1.0

    def __call__(self, x):  # type: ignore[no-untyped-def]
        for i in range(1):  # leaves i == 0 before the branch
            pass
        if x > 0.0:
            for i in range(3):  # i == 2 here, but must NOT leak into the sibling arm
                pass
        else:
            for _ in range(i):  # i == 0 on this path: empty, so self.y is never written
                self.y = x
        return self.y


def test_model_loop_counter_does_not_leak_across_branch_arms_in_scan() -> None:
    # Regression (Codex F8): the attribute-write scan binds loop counters to mirror the unroll, so it must snapshot and
    # restore them per branch arm (and merge afterward) -- else the then-arm's counter leaks into the else-arm and a
    # statically-empty inner range there is mistaken for a live write.
    lir = build(_run(_CounterLeakAcrossArms().__call__), "f8")
    assert [slot.name for slot in lir.float_state_slots] == []
    assert float(build_model(lir)(1.0)[0]) == 1.0  # y stays the reset constant


class _DelayLine:
    def __init__(self) -> None:
        self._prev = 0.0
        self.c = 1.0

    def __call__(self, x):  # type: ignore[no-untyped-def]
        out = self._prev
        self._prev = self.c  # chained copy of c's live-in; c also self-accumulates below
        self.c = self.c + x
        return out


def test_model_chained_state_copy_delay_line() -> None:
    # Regression (Codex F9): a unit-delay idiom -- one slot copying another slot's live-in while that slot
    # self-accumulates -- must build (the single-block allocator must not coalesce the accumulator's live-out onto its
    # register, which would clobber the live-in before the chained copy reads it) and reproduce the delayed stream.
    model = build_model(build(_run(_DelayLine().__call__), "delay"))
    reference = _DelayLine()
    for x in [1.0, 1.0, 2.0, 3.0, -1.0]:
        assert float(model(x)[0]) == reference(x)  # out[n] == c[n-1]


def test_synthesis_result_reports_latency_metric() -> None:
    # The public latency metric is (min II, max II or None): a straight-line kernel is exact (min == max), a branching
    # kernel's max is unbounded for now (a branch can shortcut the PC), reported as None alongside the min lower bound.
    import holoso

    ops = OpConfig(
        FAddOperator(FMT), FMulOperator(FMT), FDivOperator(FMT), FMulILog2OperatorFamily(FMT), FCmpOperator(FMT)
    )

    def straight_line(a, b):  # type: ignore[no-untyped-def]
        return a * b + a

    flat = holoso.synthesize(straight_line, ops)
    flat_min, flat_max = flat.initiation_interval
    assert flat_min > 0 and flat_max == flat_min  # exact: min == max

    branching = holoso.synthesize(_PID().__call__, ops)
    branching_min, branching_max = branching.initiation_interval
    assert branching_min > 0 and branching_max is None  # inexact: unbounded max


def test_model_branch_state_is_picklable() -> None:
    # The boolean and float persistent state survive a pickle round-trip and keep advancing identically.
    model = build_model(build(_run(_Iir1Lpf().__call__), "iir1_lpf"))
    model(1.0)
    model(2.0)
    restored = pickle.loads(pickle.dumps(model))
    fresh = build_model(build(_run(_Iir1Lpf().__call__), "iir1_lpf"))
    for v in (1.0, 2.0):
        fresh(v)
    for v in (3.0, 4.0, 5.0):
        assert float(restored(v)[0]) == float(fresh(v)[0])


def test_model_boolean_connectives_and_chained_and_ternary_are_exact() -> None:
    # Connectives (and/or/not), a chained comparison, and ternaries all lower to combinational bool ops feeding
    # branch+phi merges; the model must reproduce the Python reference exactly across every arm.
    def kernel(x, lo, hi):  # type: ignore[no-untyped-def]
        deadband = 0.0 if lo < x < hi else x  # chained comparison + ternary
        gate = 1.0 if (x > lo and x < hi) else 0.0  # and-connective in a condition
        outside = 1.0 if (x < lo or x > hi) else 0.0  # or-connective
        inverted = -1.0 if not (x > lo) else 1.0  # not
        clamp = hi if x > hi else (lo if x < lo else x)  # nested ternary
        return (deadband, gate, outside, inverted, clamp)

    model = build_model(build(_run(kernel), "bool_kernel"))
    for x in (-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0):
        got = tuple(float(v) for v in model(x, 0.0, 1.0))
        ref = tuple(float(v) for v in evaluate_reference(kernel, {"x": x, "lo": 0.0, "hi": 1.0}))
        assert got == ref, f"x={x}: {got} != {ref}"


def test_model_bool_cast_matches_float_nonzero() -> None:
    # bool(x) is the ZKF exponent-nonzero test: true iff the value is nonzero *after* encoding into the format (a
    # magnitude too small to represent rounds to zero, like any ZKF value), including for +0.0 and -0.0.
    def kernel(x, y):  # type: ignore[no-untyped-def]
        return y if bool(x) else 0.0

    model = build_model(build(_run(kernel), "bool_cast"))
    for x in (0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 123.0, 2.0**-20):
        got = float(model(x, 7.0)[0])
        ref = 7.0 if float(FloatValue.from_float(FMT, x)) != 0.0 else 0.0
        assert got == ref, f"x={x}: {got} != {ref}"


def test_model_cross_domain_cast_chain_is_exact() -> None:
    # Regression: a branch-free float->bool->float->float chain (float(x>0)*k) builds via the CFG path even with a
    # single block (it has combinational ops, no branch); the model must take the same path and be bit-exact.
    def kernel(x, k):  # type: ignore[no-untyped-def]
        gate = float(x > 0.0) * k  # cross-domain chain
        cast = float(x < 0.0)  # branch-free bool->float
        return (gate, cast)

    model = build_model(build(_run(kernel), "cross_domain"))
    for x in (-2.0, -1.0, 0.0, 1.0, 2.0):
        got = tuple(float(v) for v in model(x, 5.0))
        ref = tuple(float(v) for v in evaluate_reference(kernel, {"x": x, "k": 5.0}))
        assert got == ref, f"x={x}: {got} != {ref}"


def test_model_bool_cast_of_underflowing_constant_is_false() -> None:
    # Regression (Codex): bool(c) of a compile-time constant is the ZKF exponent-nonzero test on the constant *encoded
    # into the format*, not a raw float64 ``c != 0.0``. In FMT(6,18) the tiny magnitude 2**-200 encodes to zero, so the
    # cast is False -- the HIR const-folder must not fold it to True.
    assert FMT.encode(2.0**-200) == 0  # the constant underflows to ZKF zero in this format

    def kernel(a):  # type: ignore[no-untyped-def]
        return a if bool(2.0**-200) else -a  # the gate is False -> the model returns -a

    model = build_model(build(_run(kernel), "tiny_bool"))
    for a in (1.0, -2.0, 3.5):
        assert float(model(a)[0]) == -a


def test_connective_branch_does_not_create_a_phantom_state_slot() -> None:
    # Regression (review): folding ``if u > 0.0 or True:`` to its live arm must keep the persistent-state scan
    # (``_scan_attr_writes``) and lowering in lockstep. Before the shared ``_static_condition`` predicate the scan
    # descended both arms (strict ``_static_bool`` does not fold ``X or True``) while lowering folded one, so the dead
    # else-arm's ``self.y`` became a state slot with no value and ``_register_state_slots`` crashed with KeyError.
    class K:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0

        def __call__(self, u):  # type: ignore[no-untyped-def]
            if u > 0.0 or True:
                self.x = self.x + u
            else:
                self.y = self.y + u  # unreachable: must not become persistent state
            return self.x

    hir = lower(K().__call__)
    assert [slot.name for slot in hir.state_slots] == ["x"]  # y is not a phantom slot
    assert len(hir.blocks) == 1  # the connective guard folded; no branch
    model = build_model(build(_run(K().__call__), "phantom_if"))
    assert float(model(2.0)[0]) == 2.0
    assert float(model(3.0)[0]) == 5.0  # accumulates 2 + 3, exact in this format


def test_connective_branch_in_a_loop_body_does_not_carry_a_phantom_attribute() -> None:
    # Same desync hazard via the loop-carried scan (``_loop_assigned``): a folded connective ``if`` inside a loop body
    # must not open a loop-header phi for an attribute the body never actually writes.
    class K:
        def __init__(self):
            self.acc = 0.0
            self.dead = 0.0

        def __call__(self, u):  # type: ignore[no-untyped-def]
            n = 0.0
            while n < 3.0:
                if u > 0.0 or True:
                    self.acc = self.acc + u
                else:
                    self.dead = self.dead + u  # unreachable
                n = n + 1.0
            return self.acc

    hir = lower(K().__call__)
    assert [slot.name for slot in hir.state_slots] == ["acc"]  # dead is not carried
    model = build_model(build(_run(K().__call__), "phantom_loop"))
    assert float(model(1.0)[0]) == 3.0  # three trips accumulate u

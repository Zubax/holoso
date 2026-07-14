"""
The differential correctness gate for the FIR front-end. The oracle is the Python float64 evaluation of the kernel
itself: the front-end lowered to MIR must reproduce it within the configured float format's round-off. Comparison
is on observable model I/O, never HIR structure.
"""

from collections.abc import Callable

import numpy as np
import pytest

from holoso._frontend._fir._analyze import AnalysisRejection
from holoso._frontend._fir._emit import EmissionRejection, lower_fir
from holoso._hir import optimize
from holoso._mir import MirInterpreter, lower as lower_to_mir
from holoso._type import FloatFormat
from holoso._value import FloatValue

from ._modelref import (
    branch_boundary_kernel,
    const_branch_kernel,
    default_ops,
    diamond_then_loop_kernel,
    phi_swap_computed_loop,
    phi_swap_loop,
)

_FMT = FloatFormat(8, 23)
_OPS = default_ops(_FMT)
_RTOL = 2.0 ** -(_FMT.wman - 4)  # a few ULP of headroom over the format's unit round-off


def _encode(value: float | bool) -> FloatValue | bool:
    return value if type(value) is bool else FloatValue.from_float(_FMT, float(value))


def _vectors(arity: int, seed: int, count: int = 24) -> list[tuple[float, ...]]:
    rng = np.random.default_rng(seed)
    vectors = [tuple(float(v) for v in rng.uniform(-4.0, 4.0, arity)) for _ in range(count)]
    vectors += [(0.0,) * arity, (1.0,) * arity, (-1.0,) * arity]  # boundary rows worth pinning explicitly
    return vectors


def _decode(value: float | bool) -> float | bool:
    # The hardware receives QUANTIZED inputs, so the Python oracle must too: evaluate the reference on the values the
    # kernel actually sees, not the raw float64s, or format round-off masquerades as a wiring bug near thresholds.
    return value if type(value) is bool else float(FloatValue.from_float(_FMT, float(value)))


def _new_hir(kernel: Callable[..., object]):  # type: ignore[no-untyped-def]
    return optimize(lower_fir(kernel))


def _assert_matches_python(kernel: Callable[..., object], vectors: list[tuple[float | bool, ...]]) -> None:
    # The oracle: the front-end's model output must reproduce the kernel's own float64 evaluation on the
    # QUANTIZED inputs.
    hir = _new_hir(kernel)
    new = MirInterpreter(lower_to_mir(hir, _OPS))
    port_count = len(hir.outputs)
    for vector in vectors:
        try:
            reference = kernel(*(_decode(v) for v in vector))
        except (ZeroDivisionError, ValueError):
            continue  # undefined at this input in Python; the hardware's defined error path is exercised elsewhere
        encoded = [_encode(v) for v in vector]
        new_out = new.run(*encoded)
        expected = reference if isinstance(reference, tuple) else (reference,)
        assert (
            len(new_out) == port_count == len(expected)
        ), f"{kernel.__name__} arity mismatch at {vector}: {len(new_out)} ports vs {len(expected)} returns"
        for produced, want in zip(new_out, expected, strict=True):
            assert float(produced) == pytest.approx(
                float(want), rel=_RTOL, abs=_RTOL
            ), f"{kernel.__name__} != python at {vector}: {float(produced)} vs {float(want)}"


def test_branchy_scalar_kernels_match_python() -> None:
    _assert_matches_python(branch_boundary_kernel, _vectors(3, 1))
    _assert_matches_python(const_branch_kernel, _vectors(2, 2))


def test_loop_kernels_match_python() -> None:
    _assert_matches_python(diamond_then_loop_kernel, _vectors(2, 3))


def test_loop_carried_swap_kernels_match_python() -> None:
    # The computed-recurrence family that motivated the stage-0 fix, now proven correct through the new front-end.
    _assert_matches_python(phi_swap_loop, [(x, n) for x in (-3.0, 0.5, 2.0, 5.0) for n in (1.0, 2.0, 3.0, 4.0)])
    _assert_matches_python(phi_swap_computed_loop, [(x, n) for x in (1.0, 2.0, -1.5) for n in (1.0, 2.0, 3.0, 4.0)])


def test_local_scalar_kernels_match_python() -> None:
    # `calls` exercises local-helper inlining; the Python oracle carries the proof.
    def affine(x: float, g: float) -> float:
        y = x * 2.0 + g
        if y > 1.0:
            y = y - 0.5
        return y * g

    def eager_bool(a: float, b: float) -> float:
        return (a if a > b else b) + 1.0

    def unrolled(x: float) -> float:
        acc = 0.0
        for k in (1.0, 2.0, 3.0):
            acc = acc + k * x
        return acc

    def helper(v: float, s: float) -> float:
        return v * s + 1.0

    def calls(x: float) -> float:
        return helper(x, 2.0) + helper(x, 3.0)

    kernels: list[tuple[Callable[..., object], int, int]] = [
        (affine, 2, 10),
        (eager_bool, 2, 11),
        (unrolled, 1, 12),
        (calls, 1, 13),
    ]
    for kernel, arity, seed in kernels:
        _assert_matches_python(kernel, _vectors(arity, seed))


def _public_slots(obj: object) -> set[str]:
    return {name for name in vars(obj) if not name.startswith("_")}


def _port_contract_violations(
    names: set[str],
    return_count: int,
    mutated: set[str],
    public_state: dict[str, float],
    returns0: list[float],
) -> list[str]:
    # The port contract the emitter must honor, checked against demands derived purely from Python behavior so the
    # check can never be satisfied by construction: (1) at least one port exists; (2) every slot the reference
    # actually mutates is genuine runtime state and MUST have a port -- a slot whose value never changes is
    # indistinguishable from a constant and may be folded away with no port (DeadFlag's gate/gain), so it is permitted
    # but not required; (3) no port beyond the returns and the public slots; (4) a returned leaf is absent only when
    # deduped onto a present public state port carrying the same value.
    problems: list[str] = []
    if not names:
        problems.append("no output ports at all")
    required_state = {f"state_{slot}" for slot in mutated}
    if not required_state <= names:
        problems.append(f"missing state ports for mutated slots: {required_state - names}")
    expected = {f"out_{i}" for i in range(return_count)} | {f"state_{slot}" for slot in public_state}
    if not names <= expected:
        problems.append(f"unexpected ports: {names - expected}")
    for i in range(return_count):
        deduped = any(f"state_{slot}" in names and returns0[i] == value for slot, value in public_state.items())
        if f"out_{i}" not in names and not deduped:
            problems.append(f"returned leaf out_{i} is absent and not deduped onto any state port")
    return problems


def _stateful_agree(make: Callable[[], object], transactions: list[tuple[float | bool, ...]]) -> None:
    # A stateful component holds a running reference instance; each transaction advances it and the front-end's
    # model output must track that Python state.
    reference = make()
    hir = optimize(lower_fir(make().step))  # type: ignore[attr-defined]
    new = MirInterpreter(lower_to_mir(hir, _OPS))
    names = [o.name for o in hir.outputs]
    assert len(names) == len(set(names)), f"duplicate output ports: {names}"  # a dedup miss shows up as a repeat
    sample: object = make()
    result = sample.step(*(_decode(t) for t in transactions[0]))  # type: ignore[attr-defined]
    return_count = len(result) if isinstance(result, tuple) else 1
    returns0 = [float(v) for v in (result if isinstance(result, tuple) else (result,))]
    sample_state = {slot: float(getattr(sample, slot)) for slot in _public_slots(sample)}
    # Discover which public slots the reference actually mutates over the stream, on a throwaway probe, so the demand
    # is derived from Python behavior rather than from the emitted port set (which would be circular).
    probe = make()
    initial = {slot: float(getattr(probe, slot)) for slot in _public_slots(probe)}
    mutated: set[str] = set()
    for vector in transactions:
        probe.step(*(_decode(v) for v in vector))  # type: ignore[attr-defined]
        mutated |= {slot for slot in _public_slots(probe) if float(getattr(probe, slot)) != initial[slot]}
    violations = _port_contract_violations(set(names), return_count, mutated, sample_state, returns0)
    assert not violations, "; ".join(violations)
    for vector in transactions:
        encoded = [_encode(v) for v in vector]
        new_by_name = dict(zip(names, new.run(*encoded), strict=True))
        want = reference.step(*(_decode(v) for v in vector))  # type: ignore[attr-defined]  # advance the running state
        returns = list(want) if isinstance(want, tuple) else [want]
        for name, produced in new_by_name.items():
            if name.startswith("state_"):
                target = float(getattr(reference, name.removeprefix("state_")))
            else:
                target = float(returns[int(name.removeprefix("out_"))])
            assert float(produced) == pytest.approx(
                target, rel=_RTOL, abs=_RTOL
            ), f"stateful port {name} != python at {vector}: {float(produced)} vs {target}"


def test_stateful_kernels_agree() -> None:
    class Integrator:
        def __init__(self) -> None:
            self.acc = 0.0

        def step(self, x: float) -> float:
            self.acc = self.acc + x
            return self.acc

    class Ema:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self, x: float) -> float:
            self.y = self.y * 0.9 + x * 0.1
            return self.y

    class PublicCounter:
        def __init__(self) -> None:
            self.count = 0.0

        def step(self, x: float) -> float:
            self.count = self.count + 1.0
            return x * self.count

    class DeadFlag:
        def __init__(self) -> None:
            self.gate = False
            self.gain = 4.0

        def step(self, x: float) -> float:
            if self.gate:
                self.gain = 0.0
            return self.gain * x

    rng = np.random.default_rng(20)
    stream: list[tuple[float | bool, ...]] = [(float(rng.uniform(-3.0, 3.0)),) for _ in range(16)]
    _stateful_agree(Integrator, stream)
    _stateful_agree(Ema, stream)
    _stateful_agree(PublicCounter, stream)
    _stateful_agree(DeadFlag, stream)


def test_intrinsic_calls_match_python() -> None:
    # abs is a registered Intrinsic(FloatAbs); a runtime-argument call lowers to that HIR operation. (A transcendental
    # like sqrt lowers correctly too -- the emitter produces FloatSqrt -- but needs a format-specific op table that is
    # orthogonal to front-end emission, so the intrinsic MECHANISM is pinned here with abs.)
    def with_abs(x: float, y: float) -> float:
        return abs(x - y) + 1.0

    _assert_matches_python(with_abs, _vectors(2, 30))


def test_loop_invariant_through_sealed_header_matches_python() -> None:
    # Codex-flagged: a loop-invariant value read AFTER a while-with-break flows through the sealed loop header; a
    # naive read recurses forever. The canonical-Braun cycle break makes it terminate and compute correctly.
    def kernel(x: float, n: float) -> float:
        g = x * 2.0
        acc = 0.0
        k = 0.0
        while k < n:
            acc = acc + 1.0
            if acc > 5.0:
                break
            k = k + 1.0
        return acc + g

    _assert_matches_python(kernel, [(x, n) for x in (-2.0, 0.5, 3.0) for n in (0.0, 2.0, 4.0, 8.0)])


def test_bool_parameter_kernel_matches_python() -> None:
    def kernel(flag: bool, x: float) -> float:
        return x + 1.0 if flag else x - 1.0

    new_hir = _new_hir(kernel)
    assert new_hir.input_names() == ["flag", "x"]  # a bool parameter must be a 1-bit port, not a float port
    for flag in (True, False):
        _assert_matches_python(kernel, [(flag, v) for v in (-2.0, 0.0, 3.5)])


def test_void_kernel_has_no_output_ports() -> None:
    def kernel(x: float) -> None:
        y = x + 1.0  # noqa: F841  # a pure computation with no return: no datapath output

    hir = _new_hir(kernel)
    assert hir.outputs == []


def test_new_frontend_matches_python_reference() -> None:
    # "Identical across front-ends" is only trustworthy if at least one is pinned to a correct oracle.
    def kernel(x: float, g: float) -> float:
        y = x * 2.0 + g
        if y > 1.0:
            y = y - 0.5
        return y * g

    interpreter = MirInterpreter(lower_to_mir(optimize(lower_fir(kernel)), _OPS))
    for x, g in [(0.5, 1.0), (-2.0, 3.0), (10.0, -1.5)]:
        out = interpreter.run(FloatValue.from_float(_FMT, x), FloatValue.from_float(_FMT, g))
        assert float(out[0]) == pytest.approx(kernel(x, g), rel=1e-5, abs=1e-5)


def test_aggregate_return_is_a_located_rejection() -> None:
    from holoso._frontend._fir._emit import EmissionRejection

    def kernel(x: float) -> tuple[float, float]:
        return x + 1.0, x - 1.0

    with pytest.raises(EmissionRejection, match="aggregate"):
        lower_fir(kernel)


def test_component_helper_method_matches_python() -> None:
    # Codex-flagged: self.helper(x) must dispatch, not be mistaken for state.
    class WithHelper:
        def __init__(self) -> None:
            self.gain = 2.0

        def helper(self, v: float) -> float:
            return v * self.gain

        def step(self, x: float) -> float:
            return self.helper(x) + 1.0

    _stateful_agree(WithHelper, [(v,) for v in (-2.0, 0.5, 3.0)])


def test_bool_equality_matches_python() -> None:
    def kernel(a: bool, b: bool) -> float:
        same = a == b
        return 1.0 if same else 0.0

    hir = _new_hir(kernel)
    assert hir.input_names() == ["a", "b"]
    for a in (True, False):
        for b in (True, False):
            _assert_matches_python(kernel, [(a, b)])


def test_aliased_bool_annotation_types_a_bool_port() -> None:
    import builtins

    def kernel(flag: builtins.bool, x: float) -> float:
        return x if flag else -x

    hir = _new_hir(kernel)
    from holoso._hir import BoolType

    flag_port = hir.nodes[hir.input_ids[0]]
    assert isinstance(flag_port.type, BoolType)


def test_inexact_integer_constant_promotes_and_rounds() -> None:
    from holoso._hir import FloatConst

    def kernel(x: float) -> float:
        big = 2**53 + 1  # not binary64-exact: the comparison promotes it and rounds onto 2**53 (accepted fastmath)
        return 1.0 if x == big else 0.0

    hir = lower_fir(kernel)
    assert float(2**53) in [n.value for n in hir.nodes.values() if isinstance(n, FloatConst)]


def test_unanchored_global_components_are_a_located_rejection() -> None:
    # Multiple stateful owners are now supported when they are MEMBERS of the synthesized component (see
    # test_hierarchical_components.py). Stateful components reached only through module globals from a plain function
    # have no member path from a root, so they are rejected as unanchored rather than silently slotted.
    from holoso._frontend._fir._emit import EmissionRejection

    class Filter:
        def __init__(self, reset: float) -> None:
            self.total = reset

    _a, _b = Filter(0.0), Filter(10.0)

    def kernel(x: float) -> float:
        _a.total = _a.total + x
        _b.total = _b.total + x
        return _a.total + _b.total

    with pytest.raises(EmissionRejection, match="unanchored reference"):
        lower_fir(kernel)


def test_lazy_init_state_port_order_matches_python() -> None:
    # Codex/Fable-flagged shape: a public slot first READ in a guard before another public slot is first STORED.
    # Emitting ports by first-touch would reorder them and (with return==state dedup) put a state port where the
    # harness expects the return. First-store ordering + name-based comparison both defend against it.
    class LazyEma:
        def __init__(self) -> None:
            self.initialized = False
            self.y = 0.0

        def step(self, x: float) -> float:
            if self.initialized:
                self.y = self.y * 0.9 + x * 0.1
            else:
                self.y = x
                self.initialized = True
            return self.y

    hir = _new_hir(LazyEma().step)
    state_ports = [o.name for o in hir.outputs if o.name.startswith("state_")]
    assert state_ports == ["state_y", "state_initialized"]  # first-STORE order (y stored before initialized), not
    # first-touch (initialized is READ first in the guard) -- matching the production port contract
    rng = np.random.default_rng(40)
    _stateful_agree(LazyEma, [(float(rng.uniform(-3.0, 3.0)),) for _ in range(12)])


def test_intrinsic_arity_mismatch_is_a_located_rejection() -> None:
    def kernel(a: float, b: float, c: float) -> float:
        return min(a, b, c)  # min lowers to a 2-ary operator; a 3-arg call must reject with a located message

    with pytest.raises(AnalysisRejection, match="expects 2 argument"):
        lower_fir(kernel)


@pytest.mark.parametrize("reset", [2**53 + 1, np.int64(2**53 + 1)], ids=["python_int", "np_int64"])
def test_inexact_integer_state_reset_promotes_and_rounds(reset: object) -> None:
    # An integer reset feeding a float accumulator promotes the slot to float; a reset binary64 cannot hold exactly
    # rounds onto 2**53 under fastmath instead of rejecting (both the Python-int and the np.int64 spellings).
    from holoso._hir import FloatConst, FloatType

    class Counter:
        def __init__(self) -> None:
            self.value: float = reset  # type: ignore[assignment]

        def step(self, x: float) -> float:
            self.value = self.value + x
            return self.value

    hir = lower_fir(Counter().step)
    (slot,) = hir.state_slots
    assert isinstance(hir.nodes[slot.live_out].type, FloatType)
    assert isinstance(slot.reset_value, FloatConst) and slot.reset_value.value == float(2**53)


def test_mutable_dataclass_component_lowers() -> None:
    import dataclasses

    @dataclasses.dataclass
    class Acc:  # a plain mutable (unhashable) dataclass component
        total: float = 0.0

        def step(self, x: float) -> float:
            self.total = self.total + x
            return self.total

    _stateful_agree(Acc, [(1.0,), (2.0,), (3.0,)])


def test_numpy_integer_constant_materializes() -> None:
    import numpy as np

    gain = np.int64(2)

    def kernel(x: float) -> float:
        return x * gain

    _assert_matches_python(kernel, _vectors(1, 50))


def test_mangled_bool_parameter_types_a_bool_port() -> None:
    from holoso._hir import BoolType

    class Gate:
        def __init__(self) -> None:
            self.g = 1.0

        def step(self, __enabled: bool, x: float) -> float:  # a class-private (mangled) bool parameter
            return x if __enabled else -x

    hir = _new_hir(Gate().step)
    assert isinstance(hir.nodes[hir.input_ids[0]].type, BoolType)


def test_nested_store_state_port_order_is_source_order() -> None:
    class TwoState:
        def __init__(self) -> None:
            self.first = 0.0
            self.second = 0.0

        def step(self, x: float) -> float:
            if x > 0.0:
                self.first = x  # stored in a nested branch (higher block id) but earlier in source
            self.second = x + 1.0
            return self.first + self.second

    hir = _new_hir(TwoState().step)
    state_ports = [o.name for o in hir.outputs if o.name.startswith("state_")]
    assert state_ports == ["state_first", "state_second"]  # source order, not block-id order


def test_static_comprehension_local_matches_python() -> None:
    # A fully-static comprehension stored in a local (the accumulator concatenates lists) must flow as a folded
    # value, not be rejected as an aggregate: its subscripts resolve to constants.
    def kernel(x: float) -> float:
        squares = [k * k for k in (1.0, 2.0, 3.0, 4.0)]
        return squares[0] + squares[3] * x

    _assert_matches_python(kernel, _vectors(1, 60))


def test_integer_power_matches_python() -> None:
    def kernel(x: float) -> float:
        return x**2 + x**3 - x**0 + x**-1  # compile-time integer exponents expand to multiply chains

    _assert_matches_python(kernel, [(v,) for v in (0.5, 2.0, -1.5, 3.0, 4.0)])


def test_nested_runtime_unpack_matches_python() -> None:
    def kernel(x: float, y: float) -> float:
        a, (b, c) = x, (y, x + y)  # a nested runtime aggregate exercises the typed sequence handles
        return a + 2.0 * b + 3.0 * c

    _assert_matches_python(kernel, _vectors(2, 70))


def test_port_contract_detects_dropped_state_and_zero_outputs() -> None:
    # Codex round-3 #4: the old completeness check intersected the demanded ports with the emitted ones, so a
    # silently dropped state port could never be caught. The oracle must FLAG a mutated slot that has no port and a
    # kernel that emits nothing, while still accepting a folded (never-mutated) public slot and a deduped return.
    assert _port_contract_violations({"out_0"}, 1, {"acc"}, {"acc": 1.0}, [5.0])  # mutated slot lacks its port
    assert _port_contract_violations(set(), 1, set(), {}, [5.0])  # a kernel that emits no ports at all
    assert _port_contract_violations({"out_0", "out_9"}, 1, set(), {}, [5.0])  # a stray port past the returns
    assert not _port_contract_violations({"state_gain"}, 0, set(), {"gate": 4.0, "gain": 4.0}, [])  # folded slots ok
    assert not _port_contract_violations({"state_acc"}, 1, {"acc"}, {"acc": 5.0}, [5.0])  # return deduped onto state
    assert not _port_contract_violations({"out_0", "state_count"}, 1, {"count"}, {"count": 3.0}, [7.0])  # well-formed


def test_namespace_attribute_store_is_a_located_rejection() -> None:
    # Codex round-3 #1: a class/module is a compile-time namespace; storing to it would let later reads (which
    # snapshot the live object) disagree with the store, a silent miscompile. Production rejects it; so must we.
    class Config:
        scale = 2.0

    def kernel(x: float) -> float:
        Config.scale = x  # honest mistake: rebinding a class attribute instead of an instance attribute
        return x * Config.scale

    with pytest.raises(AnalysisRejection, match="module or class attribute"):
        lower_fir(kernel)


def test_runtime_integer_in_float_datapath_promotes_to_float() -> None:
    # A runtime selection between two distinct int literals joins to a runtime int (a phi over two Known ints); feeding
    # that integer into a float operation promotes it via IntToFloat, exactly as Python promotes ``x + n`` int->float.
    from holoso._hir import FloatAdd, IntToFloat, Operation, optimize

    def kernel(flag: bool, x: float) -> float:
        n = 1 if flag else 0  # a runtime int (a phi over two distinct Known ints), not a foldable constant
        return x + n

    hir = optimize(lower_fir(kernel))
    ops = {type(n.operator).__name__ for n in hir.nodes.values() if isinstance(n, Operation)}
    assert IntToFloat.__name__ in ops and FloatAdd.__name__ in ops  # the integer edge is promoted, then added in float


def test_static_multidim_subscript_matches_python() -> None:
    # Codex round-3 #5: analysis accepts a static multi-dim index into a Known array, but emission asserted the index
    # was a plain int and crashed on the (1, 0) tuple. It must fold the concrete lookup, as production does.
    table = np.array([[1.0, 2.0], [3.0, 4.0]])

    def kernel(x: float) -> float:
        return x * table[1, 0] + table[np.int64(0), 1]  # type: ignore[no-any-return]  # tuple + np.int64 index

    _assert_matches_python(kernel, [(v,) for v in (0.5, 2.0, -1.5, 3.0)])


def test_mixed_bool_float_comparison_is_a_located_rejection() -> None:
    # Codex round-3 #6: comparing a bool against a float without a cast is ambiguous (Python coerces True to 1.0).
    # Emitting one side as a bool and the other as a float would compare mismatched domains; reject it explicitly.
    def kernel(flag: bool, x: float) -> float:
        hit = flag == x  # a bool compared directly against a float
        return 1.0 if hit else 0.0

    with pytest.raises(EmissionRejection, match="mixes a boolean and a non-boolean"):
        lower_fir(kernel)


def test_max_static_unroll_matches_python() -> None:
    # Codex round-3 #7: emission chases single-predecessor chains iteratively (a recursive walk once overflowed the
    # stack on a long chain). The unroll threshold caps a static loop at UNROLL_THRESHOLD trips, so this exercises the
    # deepest chain a static unroll can produce, right at the boundary.
    def kernel(x: float) -> float:
        acc = 0.0
        for _ in range(64):  # UNROLL_THRESHOLD: the largest unroll accepted
            acc = acc + x
        return acc

    _assert_matches_python(kernel, [(0.5,), (-1.0,), (2.0,)])


def _trivial_phis(hir) -> list[int]:  # type: ignore[no-untyped-def]
    from holoso._hir import Phi

    return [
        vid
        for vid, node in hir.nodes.items()
        if isinstance(node, Phi) and len({v for _, v in node.arms if v != vid}) == 1
    ]


def _empty_nonentry_blocks(hir) -> list[int]:  # type: ignore[no-untyped-def]
    return [b.id for b in hir.blocks if b.id != hir.entry and not b.phis and not b.operations]


def test_trivial_loop_phi_is_eliminated() -> None:
    # A loop-invariant carried through a data-dependent header leaves a phi(preheader: x, latch: self) that the
    # emitter's on-the-fly SSA cannot collapse in place; the shared trivial-phi pass must remove it (an extra live
    # value on the recurrence is both area and a scheduling slot). Regression guard on the metrics-recovery pass.
    def kernel(x: float, n: float) -> float:
        acc = 0.0
        i = n
        while i > 0.0:  # x is never reassigned -> its header phi is trivial
            acc = acc + x
            i = i - 1.0
        return acc

    assert not _trivial_phis(optimize(lower_fir(kernel))), "a trivial loop-invariant phi survived optimization"
    _assert_matches_python(kernel, [(2.0, 3.0), (0.5, 5.0), (-1.5, 4.0)])


def test_empty_trampoline_block_is_eliminated() -> None:
    # The structured single-exit emitter jumps straight-line bodies into an empty Ret block; the empty-block pass must
    # fold it into the predecessor so the schedule pays no per-block boundary drain for a block that does no work.
    def kernel(a: float, b: float) -> float:
        c = a + b
        d = c * 2.0
        return d - a

    assert not _empty_nonentry_blocks(optimize(lower_fir(kernel))), "an empty trampoline block survived optimization"
    _assert_matches_python(kernel, _vectors(2, 91))


def _has_op(hir, mnemonic: str) -> bool:  # type: ignore[no-untyped-def]
    from holoso._hir._ir import Operation

    return any(isinstance(n, Operation) and n.operator.mnemonic == mnemonic for n in hir.nodes.values())


def _has_branch(hir) -> bool:  # type: ignore[no-untyped-def]
    from holoso._hir._ir import Branch

    return any(isinstance(b.terminator, Branch) for b in hir.blocks)


def test_nested_guard_fuses_to_conjunction() -> None:
    # `if a: if b: <effect>` (no else) means `(a and b) ? effect : bypass`. Guarded-region if-conversion must fuse it
    # to one select over band(a, b), not the nested select(a, select(b, ...)) a bottom-up collapse leaves -- the extra
    # mux is a schedule regression (the remainder kernel's +1 cycle). Codex-recommended acceptance test.
    def kernel(a: bool, b: bool, x: float) -> float:
        r = x
        if a:
            if b:
                r = x + 1.0
        return r

    assert _has_op(optimize(lower_fir(kernel)), "band"), "the nested guard did not fuse to a conjunction"
    _assert_matches_python(kernel, [(p, q, v) for p in (True, False) for q in (True, False) for v in (2.0, -1.5)])


def test_guarded_division_is_not_speculated() -> None:
    # The fused guard condition may gate a non-speculatable effect (a division with an error sideband). Fusion forms
    # band(a, b) but the division must stay branch-gated, never hoisted to run on the bypass path -- otherwise a
    # div-by-zero fires the module error flag for a path never taken. The retained branch proves it is not speculated.
    def kernel(a: bool, b: bool, x: float, y: float) -> float:
        r = x
        if a:
            if b:
                r = x / y
        return r

    hir = optimize(lower_fir(kernel))
    assert _has_op(hir, "div") and _has_branch(hir), "the guarded division was speculated out of its branch"
    safe: list[tuple[float | bool, ...]] = [
        (True, True, 6.0, 2.0),
        (False, True, 5.0, 0.0),
        (True, False, 5.0, 0.0),
        (False, False, 3.0, 0.0),
    ]
    _assert_matches_python(kernel, safe)


def test_inner_walrus_guard_is_not_fused() -> None:
    # An assignment on the inner path makes some inner-false value differ from its outer-false peer, so the two
    # bypasses are NOT interchangeable and the region must NOT fuse. The value-equality guard must decline it.
    def kernel(a: bool, b: bool) -> float:
        y = True
        if a:
            if y := b:  # noqa: F841  -- the walrus rebinds y on the inner path
                pass
        return 1.0 if y else 0.0

    assert not _has_op(optimize(lower_fir(kernel)), "band"), "a region with an inner-path assignment was wrongly fused"
    _assert_matches_python(kernel, [(p, q) for p in (True, False) for q in (True, False)])


def test_non_returning_kernel_is_a_located_rejection() -> None:
    # Codex-flagged: a kernel that never reaches its return (an unconditional infinite loop) leaves the canonical
    # exit unreachable. Emission must refuse it cleanly, not crash reading a block that was never built.
    def kernel(x: float) -> float:
        while True:  # no break, no return
            x = x + 1.0

    with pytest.raises(EmissionRejection, match="never returns"):
        lower_fir(kernel)

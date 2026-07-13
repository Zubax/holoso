"""
The stage-5a correctness gate for the new (FIR) front-end. The PRIMARY oracle is the Python float64 evaluation of
the kernel itself: the new front-end lowered to MIR must reproduce it within the configured float format's
round-off. The old front-end is a LOW-CREDIBILITY baseline (it has known defects and rejects constructs the new one
supports), so old-vs-new agreement is only a SECONDARY cross-check, asserted exactly where the old front-end can
lower the kernel and skipped where it rejects. A divergence from the old front-end is therefore a signal to
investigate, not proof the new front-end is wrong -- the Python reference decides. Comparison is on observable model
I/O, never HIR structure.
"""

from collections.abc import Callable

import numpy as np
import pytest

from holoso._errors import UnsupportedConstruct
from holoso._frontend import lower as lower_frontend
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
    # PRIMARY oracle: the new front-end's model output must reproduce the kernel's own float64 evaluation on the
    # QUANTIZED inputs. Secondary: the low-credibility old front-end must agree EXACTLY where it can lower at all.
    hir = _new_hir(kernel)
    new = MirInterpreter(lower_to_mir(hir, _OPS))
    try:
        old: MirInterpreter | None = MirInterpreter(lower_to_mir(optimize(lower_frontend(kernel)), _OPS))
    except UnsupportedConstruct:
        old = None  # a genuine capability gap in the baseline; the Python oracle stands alone (any OTHER error raises)
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
        if old is not None:
            old_out = old.run(*encoded)
            assert new_out == old_out, f"{kernel.__name__} diverges from baseline at {vector}: {new_out} vs {old_out}"


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
    # `calls` exercises local-helper inlining, which the OLD front-end rejects (`unsupported call to 'helper'`) --
    # a capability gap in the baseline, not the new front-end; the Python oracle carries the proof here.
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
    # A stateful component holds a running reference instance; each transaction advances it and the new front-end's
    # model output must track that Python state. The baseline runs in parallel where it can, as a cross-check.
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
    old_hir = None
    try:
        old_hir = optimize(lower_frontend(make().step))  # type: ignore[attr-defined]
    except UnsupportedConstruct:
        old = None
    if old_hir is not None:
        old = MirInterpreter(lower_to_mir(old_hir, _OPS))
        old_names = [o.name for o in old_hir.outputs]
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
        if old_hir is not None:
            # Map BOTH frontends' outputs by their OWN port names before comparing, so a differing port order is not
            # mistaken for a value divergence (the whole point of a name-keyed cross-check).
            assert old is not None
            old_by_name = dict(zip(old_names, old.run(*encoded), strict=True))
            shared = set(names) & set(old_by_name)
            assert {k: new_by_name[k] for k in shared} == {
                k: old_by_name[k] for k in shared
            }, f"stateful diverges from baseline at {vector}"


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


def test_inexact_integer_constant_is_a_located_rejection() -> None:
    from holoso._frontend._fir._emit import EmissionRejection

    def kernel(x: float) -> float:
        big = 2**53 + 1  # not binary64-exact: materializing it would silently round and change comparisons
        return 1.0 if x == big else 0.0

    with pytest.raises(EmissionRejection, match="not exactly representable"):
        lower_fir(kernel)


def test_multiple_components_are_a_located_rejection() -> None:
    from holoso._frontend._fir._emit import EmissionRejection

    class Filter:
        def __init__(self, reset: float) -> None:
            self.total = reset

    _a, _b = Filter(0.0), Filter(10.0)

    def kernel(x: float) -> float:
        _a.total = _a.total + x
        _b.total = _b.total + x
        return _a.total + _b.total

    with pytest.raises(EmissionRejection, match="multiple stateful components"):
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


def test_inexact_integer_state_reset_is_a_located_rejection() -> None:
    from holoso._frontend._fir._emit import EmissionRejection

    class Counter:
        def __init__(self) -> None:
            self.value: float = 2**53 + 1  # a reset the float bank cannot hold exactly

        def step(self, x: float) -> float:
            self.value = self.value + x  # a store promotes value to a runtime slot; its reset must materialize
            return self.value

    with pytest.raises(EmissionRejection, match="not exactly representable"):
        lower_fir(Counter().step)


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


def test_runtime_integer_in_float_datapath_is_a_located_rejection() -> None:
    # Codex round-3 #2: a runtime selection between two distinct int literals joins to a runtime int, not a Known
    # value. The integer datapath is stage 8, so feeding that int into a float operation must be a located rejection
    # rather than a silent reinterpretation of the integer bits as a float.
    def kernel(flag: bool, x: float) -> float:
        n = 1 if flag else 0  # a runtime int (a phi over two distinct Known ints), not a foldable constant
        return x + n

    with pytest.raises(EmissionRejection, match="runtime integer value in the float datapath"):
        lower_fir(kernel)


def test_inexact_numpy_integer_state_reset_is_a_located_rejection() -> None:
    # Codex round-3 #3: a numpy integer reset value too wide for binary64 would silently round on the way in. The
    # exactness guard previously checked only Python int, letting np.int64 slip through; it must catch np.integer too.
    class Bad:
        acc: float

        def __init__(self) -> None:
            self.acc = np.int64(2**53 + 1)  # type: ignore[assignment]  # exact in int64, rounds in binary64

        def step(self, x: float) -> float:
            self.acc = self.acc + x
            return self.acc

    with pytest.raises(EmissionRejection, match="not exactly representable"):
        lower_fir(Bad().step)


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

    with pytest.raises(EmissionRejection, match="mixes a boolean and a float"):
        lower_fir(kernel)


def test_deep_static_unroll_does_not_overflow_the_stack() -> None:
    # Codex round-3 #7: emission chased single-predecessor chains and walked the CFG recursively, so a deep static
    # unroll (thousands of blocks, well past Python's recursion limit) overflowed the stack. It must be iterative.
    def kernel(x: float) -> float:
        acc = 0.0
        for _ in range(1500):  # a straight-line chain far deeper than sys.getrecursionlimit()
            acc = acc + x
        return acc

    _assert_matches_python(kernel, [(0.5,), (-1.0,), (2.0,)])

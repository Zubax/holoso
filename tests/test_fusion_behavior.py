"""
Public-API, black-box tests of what a settled branch leaves behind: a guard only a graph identity decides costs no
cycle once pruning takes it, and a phi that merges one value -- a loop's carried value its body leaves unchanged --
is that value, so what reads it simplifies as if it were written directly.

Each kernel is checked against CPython on vectors whose results the format holds exactly, and its cycle count against
a twin written without the guard. The fused and folded shapes are cosimulated too.
"""

from collections.abc import Callable

import pytest

import holoso
from holoso import FloatFormat

from ._cosim import run_cosim
from ._modelref import default_options, instantiated_modules
from .hdl.hdl_float_oracle import SIMULATORS

_FMT = FloatFormat(8, 24)


def _synthesize(kernel: Callable[..., object]) -> holoso.SynthesisResult:
    return holoso.synthesize(kernel, default_options(_FMT), name=f"k_{kernel.__name__}")


def _cycles(kernel: Callable[..., object], vectors: list[tuple[float | int, ...]]) -> list[int]:
    """Each vector's handshake-to-`out_valid` cycle count, its outputs checked against CPython on the way."""
    simulator = _synthesize(kernel).numerical_model.elaborate()
    counts: list[int] = []
    for vector in vectors:
        simulator.set_inputs(*vector)
        count = 1
        simulator.tick(in_valid=True, out_ready=False)
        while not simulator.out_valid:
            simulator.tick(in_valid=False, out_ready=False)
            count += 1
        (got,) = [float(value) for value in simulator.output_values]
        simulator.tick(in_valid=False, out_ready=True)
        assert got == kernel(*vector), vector
        counts.append(count)
    return counts


def guarded(x: float, y: float) -> float:
    z = x * y
    if x * 0.0 > 1.0:
        z = z + 1.0
    return z * y + x


def unguarded(x: float, y: float) -> float:
    z = x * y
    return z * y + x


def test_a_settled_guard_costs_no_cycle() -> None:
    assert _synthesize(guarded).initiation_interval == _synthesize(unguarded).initiation_interval
    assert _cycles(guarded, [(1.5, 2.0), (-3.0, 0.5)]) == _cycles(unguarded, [(1.5, 2.0), (-3.0, 0.5)])


def guarded_unrolled(x: float, y: float) -> float:
    acc = x
    for _ in range(100):
        if x * 0.0 > 1.0:
            acc = acc + 1.0
        acc = acc * y
    return acc


def unguarded_unrolled(x: float, y: float) -> float:
    acc = x
    for _ in range(100):
        acc = acc * y
    return acc


def test_a_settled_guard_costs_no_cycle_in_each_unrolled_trip() -> None:
    guarded_ii = _synthesize(guarded_unrolled).initiation_interval
    assert guarded_ii == _synthesize(unguarded_unrolled).initiation_interval
    assert _cycles(guarded_unrolled, [(1.5, 1.0), (-2.0, -1.0)]) == [guarded_ii[0]] * 2


def guarded_while(x: float, y: float, n: int) -> float:
    acc = x
    i = 0
    while i < n:
        acc = acc * y
        if x * 0.0 > 1.0:
            acc = acc + 1.0
        acc = acc + y
        i = i + 1
    return acc


def unguarded_while(x: float, y: float, n: int) -> float:
    acc = x
    i = 0
    while i < n:
        acc = acc * y
        acc = acc + y
        i = i + 1
    return acc


def test_a_settled_guard_between_dependent_work_costs_no_cycle_per_trip() -> None:
    # The latch block the guard's merge leaves behind carries the loop's value back to the header once fused.
    vectors: list[tuple[float | int, ...]] = [(1.5, 2.0, 0), (1.5, 2.0, 1), (0.25, -1.0, 4), (3.0, 0.5, 7)]
    assert _cycles(guarded_while, vectors) == _cycles(unguarded_while, vectors)


def unchanged_carry(x: float, n: int) -> float:
    acc = 2.0
    i = 0
    while i < n:
        acc *= 1.0
        i += 1
    return acc * x


def test_a_carried_value_the_body_leaves_unchanged_is_the_value_itself() -> None:
    # `acc` is 2.0 on every trip, so `acc * x` is a power-of-two scaling rather than a multiplication.
    assert "holoso_fmul" not in instantiated_modules(_synthesize(unchanged_carry))
    _cycles(unchanged_carry, [(1.5, 0), (-0.75, 3)])


def settled_flag(x: float, n: int) -> float:
    flag = False
    i = 0
    while i < n:
        if x * 0.0 > 1.0:
            flag = True
        i += 1
    if flag:
        y = x * 2.0
    else:
        y = x * 3.0
    return y


def unflagged(x: float, n: int) -> float:
    i = 0
    while i < n:
        i += 1
    return x * 3.0


def settled_flag_guarding_a_division(x: float, z: float, n: int) -> float:
    flag = False
    i = 0
    while i < n:
        if x * 0.0 > 1.0:
            flag = True
        i += 1
    if flag:
        y = x / z
    else:
        y = x * 3.0
    return y


def test_a_flag_its_loop_never_raises_settles_the_branch_it_guards() -> None:
    # The folded flag is itself the condition of a diamond small enough to convert, which must be pruned instead.
    vectors: list[tuple[float | int, ...]] = [(1.5, 0), (-0.5, 5)]
    assert _cycles(settled_flag, vectors) == _cycles(unflagged, vectors)
    assert "holoso_fdiv" not in instantiated_modules(_synthesize(settled_flag_guarding_a_division))
    _cycles(settled_flag_guarding_a_division, [(1.5, 2.0, 0), (-0.5, 4.0, 3)])


class UnchangedState:
    def __init__(self) -> None:
        self.s = 1.5

    def __call__(self, x: float, n: int) -> float:
        i = 0
        while i < n:
            self.s = self.s * 1.0
            i += 1
        self.s = self.s + x
        return self.s


def test_a_state_value_its_loop_leaves_unchanged_carries_across_transactions() -> None:
    simulator = holoso.synthesize(
        UnchangedState().__call__, default_options(_FMT), name="k_state"
    ).numerical_model.elaborate()
    reference = UnchangedState()
    for x, n in [(0.5, 0), (1.0, 3), (-2.0, 1), (0.25, 5)]:
        assert float(simulator.run(x, n)[0]) == reference(x, n)


def _settled_early_return(a: float, b: float) -> float:
    if a * 0.0 > 1.0:
        return b
    return a * b + 1.0


def inlined_settled_returns(x: float, y: float) -> float:
    return _settled_early_return(x, y) + _settled_early_return(y, x)


def inlined_twin(x: float, y: float) -> float:
    return (x * y + 1.0) + (y * x + 1.0)


def test_settled_early_returns_cost_no_cycle_in_inlined_helpers() -> None:
    # Each helper's surviving return site reaches its frame's exit block by a lone jump, as the kernel's reaches its own.
    assert _cycles(inlined_settled_returns, [(1.5, 2.0), (-3.0, 0.5)]) == _cycles(
        inlined_twin, [(1.5, 2.0), (-3.0, 0.5)]
    )


class GuardedState:
    def __init__(self) -> None:
        self._s = 1.0

    def __call__(self, x: float) -> float:
        if x * 0.0 > 1.0:
            self._s = 5.0
            return x
        self._s = self._s + x
        return self._s * 2.0


class UnguardedState:
    def __init__(self) -> None:
        self._s = 1.0

    def __call__(self, x: float) -> float:
        self._s = self._s + x
        return self._s * 2.0


def test_a_settled_early_return_leaves_the_state_update_to_the_surviving_site() -> None:
    vectors: list[tuple[float | int, ...]] = [(0.5,), (1.0,), (-2.0,)]
    assert _cycles(GuardedState().__call__, vectors) == _cycles(UnguardedState().__call__, vectors)


@pytest.mark.cosim
@pytest.mark.parametrize(
    ("kernel", "vectors"),
    [
        (guarded_while, [{"x": 1.5, "y": 2.0, "n": 0}, {"x": 0.25, "y": -1.0, "n": 4}, {"x": 3.0, "y": 0.5, "n": 7}]),
        (unchanged_carry, [{"x": 1.5, "n": 0}, {"x": -0.75, "n": 3}]),
        (settled_flag, [{"x": 1.5, "n": 0}, {"x": -0.5, "n": 5}]),
    ],
    ids=lambda case: case.__name__ if callable(case) else "",
)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_fused_and_folded(sim: str, kernel: Callable[..., object], vectors: list[dict[str, float | int]]) -> None:
    run_cosim(sim, _synthesize(kernel), vectors)


def guarded_arm(a: float, b: float) -> float:
    if a > b:
        y = a + b
    else:
        y = a - b
        if a * 0.0 < 1.0:
            y = y * 3.0
    return y


def unguarded_arm(a: float, b: float) -> float:
    if a > b:
        y = a + b
    else:
        y = (a - b) * 3.0
    return y


def test_a_settled_guard_leaves_its_diamond_convertible() -> None:
    # The guard's chain inside the arm is one block again, so the diamond collapses into a select.
    assert _synthesize(guarded_arm).initiation_interval == _synthesize(unguarded_arm).initiation_interval
    assert _cycles(guarded_arm, [(1.5, 0.5), (0.5, 1.5)]) == _cycles(unguarded_arm, [(1.5, 0.5), (0.5, 1.5)])


def guarded_repeat(x: float, y: float) -> float:
    z = x / y
    if x * 0.0 < 1.0:
        z = z + x / y
    return z


def unguarded_repeat(x: float, y: float) -> float:
    z = x / y
    z = z + x / y
    return z


def test_a_settled_guard_leaves_a_repeated_expression_shared() -> None:
    # Identical expressions are one value only within a block, so the division behind the guard is shared once fused.
    assert _cycles(guarded_repeat, [(1.5, 0.5), (3.0, 4.0)]) == _cycles(unguarded_repeat, [(1.5, 0.5), (3.0, 4.0)])

"""
Public-API, black-box tests of arm-block threading: a branch arm that does nothing but install its merge's phi arms is
threaded out, its predecessor installing them instead, when that shortens the arm's path and lengthens none.

Each path's cycle count is the handshake-to-`out_valid` span, and its values are checked against CPython. Threading
shortens a threaded arm's path by exactly one arm frame (4 cycles) and leaves every other path alone; each shortened
count is annotated with its unthreaded value. A kernel the rule turns down realizes its unthreaded counts. The threaded
ones are cosimulated too.
"""

from collections.abc import Callable
import dataclasses

import pytest

import holoso
from holoso import FloatFormat

from ._cosim import run_cosim
from ._modelref import default_options
from .hdl.hdl_float_oracle import SIMULATORS

_FMT = FloatFormat(6, 18)


def _paths(
    kernel: Callable[..., object], vectors: list[tuple[float | int | bool, ...]], ifconv_max_ops: int | None = None
) -> list[int]:
    """Each vector's realized cycle count, its outputs checked against CPython on the way."""
    options = default_options(_FMT)
    if ifconv_max_ops is not None:
        options = dataclasses.replace(options, ifconv_max_ops=ifconv_max_ops)
    simulator = holoso.synthesize(kernel, options, name=kernel.__name__).numerical_model.elaborate()
    counts: list[int] = []
    for vector in vectors:
        simulator.set_inputs(*vector)
        count = 1
        simulator.tick(in_valid=True, out_ready=False)
        while not simulator.out_valid:
            simulator.tick(in_valid=False, out_ready=False)
            count += 1
        got = [float(value) for value in simulator.output_values]
        simulator.tick(in_valid=False, out_ready=True)
        want = kernel(*vector)
        assert got == [float(value) for value in (want if isinstance(want, tuple) else (want,))], vector
        counts.append(count)
    return counts


def constant_arm(x: float, y: float) -> float:
    if x > 1.0:
        r = 1.0
    else:
        r = x / y
    return r + y


def boolean_arm(x: float, y: float) -> tuple[bool, float]:
    if x > 1.0:
        f = True
        r = x
    else:
        f = x < -1.0
        r = x / y
    return f, r


def into_a_loop(x: float, n: int) -> float:
    y = x
    if x > 0.0:
        i = 0
        while i < n:
            y = y * 0.5
            i = i + 1
    return y


def inside_a_loop(x: float, n: int) -> float:
    acc = 0.0
    i = 0
    while i < n:
        if acc > x:
            t = 1.0
        else:
            t = acc / (x + 3.0)
        acc = acc + t
        i = i + 1
    return acc


def overlapping(x: float, y: float, c: bool) -> float:
    if c:
        r = 0.0
    else:
        r = x / y
    return r + x


def three_way(x: float, y: float, b: bool) -> float:
    if x > 1.0:
        z = 1.0
    elif b:
        z = x
    else:
        z = x / y
    return z * y


def latch(n: float, k: float) -> float:
    x = n
    while x > 1.0:
        if x > k:
            x = x / 2.0
        else:
            x = 0.5
    return x


def test_a_constant_arm_costs_its_path_nothing() -> None:
    assert _paths(constant_arm, [(2.0, 4.0), (0.5, 4.0)]) == [12, 28]  # 16 and 28 without the threading


def test_a_boolean_arm_costs_its_path_nothing() -> None:
    assert _paths(boolean_arm, [(2.0, 4.0), (0.5, 4.0)]) == [4, 20]  # 8 and 20


def test_an_arm_threads_straight_into_a_loop_header() -> None:
    # The branch then targets the loop header, a shape the front end never emits.
    assert _paths(into_a_loop, [(2.0, 3), (-2.0, 3)]) == [46, 4]  # 50 and 4


def test_an_arm_inside_a_loop_saves_its_frame_on_every_trip() -> None:
    assert _paths(inside_a_loop, [(-1.0, 4), (2.0, 4)]) == [85, 169]  # 101 and 169: four trips through the arm


def test_an_arm_whose_predecessor_overlaps_keeps_its_frame() -> None:
    # The predecessor's results land past its terminator in both arms; installing there would make it drain.
    assert _paths(overlapping, [(2.0, 4.0, True), (2.0, 4.0, False)]) == [13, 25]


def test_an_arm_whose_threading_would_cost_another_path_keeps_its_frame() -> None:
    # Threading the constant arm leaves its predecessor's frame alone, but the merge phi then occupies its register
    # across that frame's boundary, where `x` is still live toward the other arms: the `z = x` arm could no longer
    # coalesce onto the phi, and would take a frame of its own.
    assert _paths(three_way, [(2.0, 4.0, True), (0.5, 4.0, True), (0.5, 4.0, False)]) == [13, 10, 26]


def test_a_latch_arm_is_never_threaded() -> None:
    # The constant arm jumps back to the loop header, whose phi the other arm still reads (`x / 2.0`): installing the
    # constant in their predecessor would overwrite it first.
    assert _paths(latch, [(10.0, 3.0), (10.0, 20.0), (0.5, 3.0)], ifconv_max_ops=0) == [50, 20, 6]


@pytest.mark.cosim
@pytest.mark.parametrize("kernel", [constant_arm, boolean_arm, into_a_loop, inside_a_loop], ids=lambda k: k.__name__)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_threaded_arms(sim: str, kernel: Callable[..., object]) -> None:
    run_cosim(sim, holoso.synthesize(kernel, default_options(_FMT), name=f"threaded_{kernel.__name__}"))

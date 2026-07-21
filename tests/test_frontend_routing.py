"""
Frontend tests: cell routing -- that the RIGHT source cell reaches each result cell.

Every kernel here is swap-sensitive by construction: distinct values per cell and a non-commutative readout, so a
routing error that exchanges two cells of the same type changes the answer rather than producing a well-formed
wrong-but-equal result. That property is the whole point of the module and is easy to destroy accidentally -- a
readout changed to a plain sum, or test data made symmetric, silently stops testing anything.

What this module is NOT, measured rather than assumed. Four routing mutants (a perturbed transpose route, a
rotated repeat, a swap inside each repeated unit, a rotated aligned copy) were each run against this module and
against the pre-existing suites: every one that this module caught, the example-driven matrix and aggregate
tests caught too. So these tests close no hole that was actually open at the whole-suite level, and claiming
otherwise would repeat the mistake this campaign keeps making. What they add is LOCALIZATION and a named
invariant: a rotated repeat surfaces here as "a repeated aggregate does not route each copy" rather than as a
failure in the IMU frame-transform example, and M2 gets a per-construct net it can be checked against
construct by construct.

One route is inherently untestable and worth knowing before trying: repetition order. `seq * n` yields n
identical copies, so permuting whole repetitions maps identical content onto identical content. The rotated-
repeat mutant is invisible to the kernel below for exactly that reason, and is observable elsewhere only
because a mixed Known/residual sequence writes some cells and skips others, which makes the rotation stop
being a pure permutation. A swap WITHIN the repeated unit is observable, and is what the kernel below pins.
"""

from collections.abc import Callable

import numpy as np
import pytest
from jaxtyping import Float64

import holoso
from holoso import FloatFormat

from ._modelref import default_ops

_FMT = FloatFormat(8, 23)


def _run(fn: Callable[..., object], *args: float) -> list[float]:
    sim = holoso.synthesize(fn, default_ops(_FMT), name="routing").numerical_model.elaborate()
    return [float(v) for v in sim.run(*args)]


def _assert_matches_python(fn: Callable[..., object], *args: float) -> None:
    # The Python call doubles as proof the kernel is genuinely runnable Python, so a construct Holoso accepts but
    # Python does not fails here rather than passing as a spurious positive.
    want = np.asarray(fn(*args), dtype=np.float64).flatten().tolist()
    assert _run(fn, *args) == pytest.approx(want), fn.__name__


def test_a_repeated_aggregate_routes_each_copy() -> None:
    # The suite's only repetition test repeats a SCALAR (`[x] * 3`), so all three cells hold one value and no
    # cell-level error inside the repeated unit can show. Repeating a two-cell aggregate makes that half visible:
    # the six result cells alternate between two distinct sources. It does not make repetition ORDER visible --
    # see the module docstring; that is not a gap but an identity.
    def kernel(x: float, y: float) -> float:
        pair = [x, y]
        six = pair * 3
        return six[0] + six[1] * 10.0 + six[2] * 100.0 + six[3] * 1000.0 + six[4] * 1e4 + six[5] * 1e5

    _assert_matches_python(kernel, 2.0, 3.0)
    # Pinned explicitly as well as differentially: the weights make each cell's contribution a distinct decimal
    # digit, so the expected value reads as the routing itself -- source cells x,y,x,y,x,y.
    assert _run(kernel, 2.0, 3.0) == [323232.0]


def test_a_reversed_repetition_finds_its_sequence_operand() -> None:
    # `3 * seq` is accepted as well as `seq * 3`, so the sequence is NOT always the first operand. Consult X6a
    # rejected a routing schema that addressed cells by operand INDEX partly on this case: with a one-cell
    # sequence a bounds check still accepts the wrong operand, so the error would have been silent. The compiler
    # is correct here today; this pins that, because the schema that would have broken it looked plausible.
    def kernel(x: float, y: float) -> float:
        six = 3 * [x, y]
        return six[0] + six[1] * 10.0 + six[2] * 100.0 + six[3] * 1000.0 + six[4] * 1e4 + six[5] * 1e5

    def scalar_sequence(x: float, y: float) -> float:
        # The one-cell case the bounds check cannot distinguish: picking the wrong operand yields the OTHER
        # input, which is well-formed and wrong. `y` is present only to make that confusion observable.
        return (3 * [x])[0] + (3 * [x])[1] * 10.0 + (3 * [x])[2] * 100.0 + y * 1000.0

    _assert_matches_python(kernel, 2.0, 3.0)
    assert _run(kernel, 2.0, 3.0) == [323232.0]
    _assert_matches_python(scalar_sequence, 2.0, 7.0)
    assert _run(scalar_sequence, 2.0, 7.0) == [7222.0]


def test_a_built_tuple_reversed_by_index_routes_the_reversal() -> None:
    # The suite's dedicated build-and-index test spells this exact reversal and asserts only that two ports exist,
    # so dropping the reversal entirely would pass it. Here the reversal is observable.
    def kernel(a: float, b: float) -> tuple[float, float]:
        z = a, b
        return z[1], z[0]

    _assert_matches_python(kernel, 4.0, 7.0)
    assert _run(kernel, 4.0, 7.0) == [7.0, 4.0]


def test_a_comprehension_built_transpose_routes_its_axes() -> None:
    # A hand-written transpose through a nested comprehension. The existing test builds precisely this and checks
    # only `out_{i}_{j}` port names, which a transposed-the-wrong-way result satisfies just as well.
    def kernel(a: float, b: float, c: float, d: float, e: float, f: float) -> Float64[np.ndarray, "3 2"]:
        m = [[a, b, c], [d, e, f]]
        return np.array([[m[i][j] for i in range(2)] for j in range(3)])

    _assert_matches_python(kernel, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert _run(kernel, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]


def test_a_transpose_of_a_nonsymmetric_matrix_routes_its_cells() -> None:
    # The bundled transpose example is exercised through data that is half degenerate: one of its two rotation
    # cases yields a SYMMETRIC matrix, for which transpose is the identity and the route is untested. This pins
    # the route on data that cannot be symmetric.
    def kernel(a: float, b: float, c: float, d: float, e: float, g: float) -> Float64[np.ndarray, "3 2"]:
        return np.array([[a, b, c], [d, e, g]]).T

    _assert_matches_python(kernel, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert _run(kernel, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]


def test_a_concatenation_routes_both_operands_in_order() -> None:
    def kernel(a: float, b: float, c: float) -> float:
        joined = (a, b) + (c,)
        return joined[0] + joined[1] * 10.0 + joined[2] * 100.0

    _assert_matches_python(kernel, 1.0, 2.0, 3.0)
    assert _run(kernel, 1.0, 2.0, 3.0) == [321.0]


def test_a_slice_window_routes_within_the_window() -> None:
    # A commutative readout over a slice catches a wrong WINDOW but not a swap inside it; the weights here catch
    # both, and the reversed slice pins the direction.
    def kernel(a: float, b: float, c: float, d: float) -> float:
        items = [a, b, c, d]
        mid = items[1:3]
        backwards = items[::-1]
        return mid[0] + mid[1] * 10.0 + backwards[0] * 100.0 + backwards[3] * 1000.0

    _assert_matches_python(kernel, 1.0, 2.0, 3.0, 4.0)
    assert _run(kernel, 1.0, 2.0, 3.0, 4.0) == [1432.0]


def test_a_reflavored_aggregate_keeps_its_cell_order() -> None:
    def kernel(a: float, b: float) -> float:
        items = list((a, b))
        pair = tuple([b, a])
        return items[0] + items[1] * 10.0 + pair[0] * 100.0 + pair[1] * 1000.0

    _assert_matches_python(kernel, 2.0, 5.0)
    assert _run(kernel, 2.0, 5.0) == [2552.0]


def test_a_state_rotation_routes_across_transactions() -> None:
    # A permuting store into component state. The two existing tests of this shape assert on a SET of slot names,
    # which no permutation can disturb; persistence across transactions is where the routing actually shows.
    class Rotate:
        def __init__(self) -> None:
            self.a = 1.0
            self.b = 2.0
            self.c = 3.0

        def step(self, x: float) -> float:
            self.a, self.b, self.c = self.c, self.a, self.b + x
            return self.a + self.b * 10.0 + self.c * 100.0

    reference = Rotate()
    sim = holoso.synthesize(Rotate().step, default_ops(_FMT), name="rotate").numerical_model.elaborate()
    for step in range(4):
        drive = float(step)
        assert [float(v) for v in sim.run(drive)][0] == pytest.approx(reference.step(drive))


def test_a_record_projection_routes_its_fields() -> None:
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class Gains:
        gain: float
        offset: float
        trim: float = 4.0

    def kernel(x: float) -> float:
        built = Gains(x, 2.0)
        return built.gain + built.offset * 10.0 + built.trim * 100.0

    _assert_matches_python(kernel, 1.0)
    # The default-filled field is the `:= const` half of the routing record, and its window is the one a
    # misplaced default would land in.
    assert _run(kernel, 1.0) == [421.0]

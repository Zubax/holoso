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

    def scalar_sequence(x: float) -> float:
        # The one-cell case: here the other operand is the literal 3, not another input, so a schema that indexed
        # operands positionally would reach a non-aggregate rather than a wrong value. Kept because it is the
        # shape whose cell count cannot distinguish the two operands by width.
        return (3 * [x])[0] + (3 * [x])[1] * 10.0 + (3 * [x])[2] * 100.0

    _assert_matches_python(kernel, 2.0, 3.0)
    assert _run(kernel, 2.0, 3.0) == [323232.0]
    _assert_matches_python(scalar_sequence, 2.0)
    assert _run(scalar_sequence, 2.0) == [222.0]


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


def test_a_component_aggregate_read_routes_from_its_state_cells() -> None:
    # The component-attribute arm reads `StateLeaf` cells, NOT cells of the op's operands -- which is one of the
    # reasons a routing schema cannot address a source by operand index (consult X6a). Distinct initial values
    # and a weighted readout make the state-cell mapping observable, and the rotation makes it observable again
    # on the next transaction from a different starting permutation.
    class Held:
        def __init__(self) -> None:
            self.v = [1.0, 2.0, 3.0]

        def step(self, x: float) -> float:
            got = self.v
            self.v = [got[2], got[0], got[1] + x]
            return got[0] + got[1] * 10.0 + got[2] * 100.0

    reference = Held()
    sim = holoso.synthesize(Held().step, default_ops(_FMT), name="held").numerical_model.elaborate()
    for step in range(4):
        drive = float(step)
        assert [float(v) for v in sim.run(drive)][0] == pytest.approx(reference.step(drive))


def test_a_zero_cell_source_routes_no_cells() -> None:
    # An empty aggregate has a LEGITIMATE zero-cell route. It is why "not a route" and "a route with zero rows"
    # must stay distinct in the M2 plan: collapsing them would rebuild the absence-versus-intent ambiguity the
    # step exists to remove.
    def kernel(x: float, y: float) -> float:
        empty = ()
        joined = empty + (x, y)
        return joined[0] + joined[1] * 10.0

    _assert_matches_python(kernel, 3.0, 4.0)
    assert _run(kernel, 3.0, 4.0) == [43.0]


def test_a_nonvalue_leaf_routes_to_no_cell() -> None:
    # A `Reference` leaf deliberately has no datapath cell, so a total per-leaf plan needs a third disposition
    # beyond "a source cell" and "a constant". Carrying one beside a real value proves the cell ordinals of the
    # surviving leaves are not disturbed by the leaf that produces nothing.
    def kernel(x: float, y: float) -> float:
        carried = (x, None, y)
        return carried[0] + carried[2] * 10.0

    _assert_matches_python(kernel, 2.0, 5.0)
    assert _run(kernel, 2.0, 5.0) == [52.0]


def test_a_known_condition_selection_routes_the_chosen_aggregate() -> None:
    # A `PySelect` with a compile-time-known condition re-chooses its source during EMISSION today, which makes
    # it a routing re-derivation the schema's first two revisions both omitted from the site set (consult X6a
    # round 2). Two equal-width sources make the choice unobservable by width alone.
    def kernel(a: float, b: float, c: float, d: float) -> float:
        left = (a, b)
        right = (c, d)
        p = True and left
        q = right or left
        return p[0] + 10.0 * p[1] + 100.0 * q[0] + 1000.0 * q[1]

    _assert_matches_python(kernel, 1.0, 2.0, 3.0, 4.0)
    assert _run(kernel, 1.0, 2.0, 3.0, 4.0) == [4321.0]


def test_a_record_built_from_reordered_keywords_routes_by_field() -> None:
    # Construction mixes positional and keyword sources with no numbering between them, which is one of the
    # reasons a source cannot be addressed by operand index. Here the keywords are supplied OUT of field order,
    # so a route that followed argument order rather than field identity produces a well-formed wrong answer.
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class Fields:
        a: float
        b: float
        c: float

    def kernel(x: float, y: float, z: float) -> float:
        p = Fields(x, c=z, b=y)
        return p.a + 10.0 * p.b + 100.0 * p.c

    _assert_matches_python(kernel, 1.0, 2.0, 3.0)
    assert _run(kernel, 1.0, 2.0, 3.0) == [321.0]


def test_a_zero_cell_conversion_stays_a_conversion() -> None:
    # A conversion whose route has zero rows is still a conversion. Classification must therefore not be
    # inferred from whether a route exists -- the reason `_conversion_calls` leaves the routing type entirely.
    def kernel(x: float) -> float:
        empty = tuple([x][:0])
        return x + float(len(empty))

    _assert_matches_python(kernel, 5.0)
    assert _run(kernel, 5.0) == [5.0]


def test_a_write_only_aggregate_state_registers_every_slot() -> None:
    # The aggregate `PyStoreAttr` walk carries state-slot registration as a side effect of routing, so a cutover
    # that moved the route without the registration would silently drop ports. A kernel returning None makes the
    # state slots the ONLY observable, which is what pins the side effect.
    class WriteOnly:
        def __init__(self) -> None:
            self.v = [0.0, 0.0]

        def step(self, x: float, y: float) -> None:
            self.v = [x, y]

    built = holoso.synthesize(WriteOnly().step, default_ops(_FMT), name="write_only")
    assert [port.name for port in built.numerical_model.outputs] == ["state_v_0", "state_v_1"]
    assert [float(v) for v in built.numerical_model.elaborate().run(3.0, 7.0)] == [3.0, 7.0]


def test_a_known_condition_selection_routes_both_polarities() -> None:
    # The AND/OR selection inverts with its mode: AND takes the RIGHT operand when the condition is true, OR
    # takes the LEFT. Exercising only true conditions lets a hardcoded "AND->right, OR->left" pass, so a false
    # condition is needed too. A false AND is unobservable and will STAY unobservable: a falsy aggregate is an
    # empty one, so the route has zero cells, and a zero-width route encodes no source to compare -- arm
    # identity there is not merely untested but semantically absent.
    def kernel(a: float, b: float, c: float, d: float) -> float:
        left = (a, b)
        right = (c, d)
        q = () or right
        r = left or right
        return q[0] + 10.0 * q[1] + 100.0 * r[0] + 1000.0 * r[1]

    _assert_matches_python(kernel, 1.0, 2.0, 3.0, 4.0)
    assert _run(kernel, 1.0, 2.0, 3.0, 4.0) == [2143.0]


def test_a_known_integer_stored_into_a_float_slot_is_promoted() -> None:
    # A compile-time-known integer reaching a slot whose reset fixes it as a float. This pins the TARGET-SIDE
    # normalization -- the cell's value arrives already conformed to the slot's kind -- and NOT the runtime
    # integer transfer, which needs a residual source and is covered in the schema suite.
    class Promote:
        def __init__(self) -> None:
            self.v = 0.0

        def step(self, x: float) -> float:
            self.v = 3
            return x + self.v

    built = holoso.synthesize(Promote().step, default_ops(_FMT), name="promote")
    assert [float(v) for v in built.numerical_model.elaborate().run(2.0)] == [5.0, 3.0]


def test_a_reflavor_preserves_a_boolean_cell_as_boolean() -> None:
    # A conversion carrying a runtime boolean beside a float, which routes with IDENTITY: the cell keeps its own
    # kind. This is NOT the boolean promotion -- an earlier version of this comment claimed the promotion was
    # unreachable, on the strength of two probes that missed the route entirely (see the promotion test below).
    def kernel(x: float, f: bool) -> tuple[float, bool]:
        items = list((x, f))
        # mypy unifies the list's element type to float; Holoso tracks the two cells' kinds separately, which
        # is the property under test, so the annotation states the truth and the checker gets the exception.
        return items[0], items[1]  # type: ignore[return-value]

    built = holoso.synthesize(kernel, default_ops(_FMT), name="reflavor_bool")
    assert [port.name for port in built.numerical_model.outputs] == ["out_0", "out_1"]
    assert [float(v) for v in built.numerical_model.elaborate().run(2.0, True)] == [2.0, 1.0]


def test_an_explicit_float_dtype_promotes_a_boolean_cell() -> None:
    # The routing-path boolean promotion, which an earlier probe of mine wrongly concluded was unreachable. The
    # route is the EXPLICIT dtype: an array factory forced to float turns a residual boolean source into a
    # residual float destination, and emission inserts the promotion. Reaching it needs `dtype=float`; the
    # implicit mixed-kind literal is refused and the plain re-flavor routes with identity, which is why probing
    # only those two produced a false negative.
    def kernel(x: float, f: bool) -> Float64[np.ndarray, "2"]:
        return np.array([f, x], dtype=float)

    _assert_matches_python(kernel, 2.0, True)
    assert _run(kernel, 2.0, True) == [1.0, 2.0]
    assert _run(kernel, 2.0, False) == [0.0, 2.0]


def test_an_active_construction_carries_a_nondatapath_default() -> None:
    # An ADMITTED default that is not datapath-capable. It is a separate emission branch from both the datapath
    # constant and the fully static construction, and the schema first specified it wrongly as a constant cell:
    # admission covers strings, ranges, slices and records, while emission materializes only numeric and
    # boolean Knowns. Here the construction is ACTIVE -- its other field is residual -- so the site emits, and
    # the string field must still contribute no cell without disturbing the field that does.
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class WithText:
        gain: float
        label: str = "unset"

    def kernel(x: float) -> float:
        built = WithText(x)
        return built.gain * 2.0

    _assert_matches_python(kernel, 3.0)
    assert _run(kernel, 3.0) == [6.0]


def test_a_nondatapath_scalar_state_store_stays_a_located_rejection() -> None:
    # A non-datapath scalar state store must stay a located public rejection. NOTE WHAT THIS DOES NOT PIN: the
    # rejection it actually observes comes from ANALYSIS ("values of irreconcilable kinds merge here"), not
    # from emission materialization, so this does not exercise the emitter path the M2 cutover removes. The
    # storage-conformance coverage proper lives in the schema suite. Kept because the located-rejection
    # property is worth holding across the cutover, but its name promises more than it delivers.
    class Stores:
        def __init__(self) -> None:
            self.v = 0.0

        def step(self, x: float) -> float:
            # Deliberately ill-typed: the kernel under test is one the compiler must REFUSE, and the refusal is
            # the assertion. mypy is told so rather than the kernel being weakened into something type-clean.
            self.v = "text"  # type: ignore[assignment]
            return x

    with pytest.raises(holoso.UnsupportedConstruct) as raised:
        holoso.synthesize(Stores().step, default_ops(_FMT), name="nondatapath_store")
    assert "Stores.step" in str(raised.value)  # located: the diagnostic names the kernel and its line
    assert ":" in str(raised.value)

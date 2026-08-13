"""
The native integer format and type: the arithmetic surface, the plumbing from ``Options`` down into the emitted RTL,
and the port conditioning an integer type admits. The format sizes the shared wide register bank (the ``WINT``
localparam), pinned here from the public entry point down to the Verilog backend; the operators the lowering selects
for it are covered by ``test_int_operators`` and ``test_int_selection``.
"""

import dataclasses
from collections.abc import Callable
import re

import pytest

import holoso
from holoso import FloatFormat, IntFormat, Options
from holoso._mir import MirOperation, MirPhi
from holoso._operators import BoolInversion, FloatSignControl, IntIdentity, SelectOperator, identity_conditioner
from holoso._type import BoolType, IntType
from holoso._value import IntValue

from ._modelref import default_options

FMT = FloatFormat(6, 18)


@pytest.mark.parametrize("width", (2, 3, 8, 33, 64))
def test_range_round_trip_and_saturation(width: int) -> None:
    fmt = IntFormat(width)
    assert (fmt.min, fmt.max) == (-(2 ** (width - 1)), 2 ** (width - 1) - 1)
    assert fmt.fits(fmt.min) and fmt.fits(fmt.max)
    assert not fmt.fits(fmt.min - 1) and not fmt.fits(fmt.max + 1)

    for value in (fmt.min, fmt.min + 1, -1, 0, 1, fmt.max - 1, fmt.max):
        assert fmt.decode(fmt.encode(value)) == value
        assert fmt.saturate(value) == value

    assert fmt.encode(0) == 0
    assert fmt.encode(-1) == (1 << width) - 1
    assert fmt.encode(fmt.min) == 1 << (width - 1)  # the sign bit alone
    assert fmt.encode(fmt.max) == (1 << (width - 1)) - 1

    assert fmt.saturate(fmt.min - 1) == fmt.min
    assert fmt.saturate(fmt.max + 1) == fmt.max
    assert fmt.saturate(10**30) == fmt.max
    assert fmt.saturate(-(10**30)) == fmt.min

    assert str(fmt) == f"int{width}"


def test_decode_walks_the_whole_bit_space() -> None:
    fmt = IntFormat(4)
    assert [fmt.decode(bits) for bits in range(16)] == [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1]


@pytest.mark.parametrize("width", (2, 17, 33))
def test_int_type_lands_in_the_wide_bank(width: int) -> None:
    ty = IntType(IntFormat(width))
    assert ty.width == width
    assert ty.is_wide, "integers share the wide data register bank with floats; IntFormat forbids width < 2"
    assert str(ty) == f"int{width}"


@pytest.mark.parametrize("width", (2, 17, 33))
def test_select_carries_an_integer_scalar_type_through_its_signature(width: int) -> None:
    # The mux is type-polymorphic across the scalar families, at every integer width rather than the machine's own.
    ty = IntType(IntFormat(width))
    fmt = ty.fmt
    signature = SelectOperator(ty).signature
    assert signature.operand_types == (BoolType(), ty, ty)
    assert signature.result_types == (ty,)

    arms = (IntValue.from_int(fmt, fmt.max), IntValue.from_int(fmt, fmt.min))
    assert SelectOperator(ty).evaluate(True, *arms) == (arms[0],)
    assert SelectOperator(ty).evaluate(False, *arms) == (arms[1],)


@pytest.mark.parametrize("width", (2, 17, 33))
def test_integer_ports_condition_with_the_identity_and_nothing_else(width: int) -> None:
    # An integer port folds nothing into a sideband: two's-complement negation is not free in fabric the way
    # holoso_fsgnop is. So the wide bank has no single conditioner algebra.
    ty = IntType(IntFormat(width))
    assert identity_conditioner(ty) == IntIdentity()
    assert IntIdentity().is_identity
    assert IntIdentity().decorate("r3") == "r3"

    operation = MirOperation(
        SelectOperator(ty), [0, 1, 2], [BoolInversion(), IntIdentity(), IntIdentity()], 0, IntIdentity(), ()
    )
    assert operation.scalar_type == ty

    # The phi path matters separately: it is what the wide-bank allocator narrows when lowering a merge into
    # per-predecessor install copies.
    assert MirPhi(ty, ((0, 1, IntIdentity()), (2, 3, IntIdentity()))).scalar_type == ty


def test_float_and_bool_ports_keep_their_own_conditioners() -> None:
    assert identity_conditioner(holoso.FloatType(FMT)) == FloatSignControl()
    assert identity_conditioner(BoolType()) == BoolInversion()
    assert FloatSignControl().is_identity and BoolInversion().is_identity
    assert not FloatSignControl(negate=True).is_identity
    assert not FloatSignControl(absolute=True).is_identity
    assert not BoolInversion(invert=True).is_identity


def _add(a: float, b: float) -> float:
    return a + b


def _increment(n: int) -> int:
    return n + 1


def _both(a: float, n: int) -> tuple[float, int]:
    return a + 1.0, n + 1


def _flags(a: bool, b: bool) -> tuple[bool, bool]:
    return a and b, a or b


def _word_of(kernel: Callable[..., object], options: Options, name: str) -> tuple[IntFormat, str]:
    result = holoso.synthesize(kernel, options, name=name)
    return result.int_format, result.verilog_output.verilog


@pytest.mark.parametrize("wint_min,width", ((2, 24), (16, 24), (33, 33), (44, 44)))
def test_a_float_carrying_kernel_takes_the_wider_of_the_two(wint_min: int, width: int) -> None:
    options = dataclasses.replace(default_options(FMT), wint_min=wint_min)
    assert _word_of(_add, options, "WintFloat")[0] == IntFormat(width)
    assert _word_of(_both, options, "WintMixed")[0] == IntFormat(width)


@pytest.mark.parametrize("wint_min", (2, 16, 17, 33))
def test_a_kernel_carrying_no_float_answers_to_the_floor_alone(wint_min: int) -> None:
    """The float format sizes nothing where the kernel instantiates none of it, however wide it is configured."""
    for ffmt in (FMT, FloatFormat(8, 36)):
        options = dataclasses.replace(default_options(ffmt), wint_min=wint_min)
        assert _word_of(_increment, options, "WintInt")[0] == IntFormat(wint_min), ffmt
        assert _word_of(_flags, options, "WintBool")[0] == IntFormat(wint_min), ffmt


def test_the_machine_word_sizes_the_wide_register_bank_in_the_rtl() -> None:
    # The end-to-end pin: driven through the public entry point, so a build that dropped the derivation anywhere
    # between here and the Verilog backend -- or substituted a plausible-but-wrong width -- fails right here.
    options = dataclasses.replace(default_options(FMT), wint_min=33)
    for kernel, name, width in ((_add, "WintProbe", 33), (_increment, "WintProbeInt", 33)):
        word, verilog = _word_of(kernel, options, name)
        (wint,) = re.findall(r"^localparam\s+WINT\s*=\s*(\d+);", verilog, re.MULTILINE)
        (wreg,) = re.findall(r"^localparam\s+WREG\s*=\s*(\d+);", verilog, re.MULTILINE)
        assert word == IntFormat(width) and int(wint) == width and int(wreg) == width
        assert re.search(r"reg\s+\[WREG-1:0\]\s+regs\b", verilog)
    assert "WFLT      = WEXP + WMAN;" in _word_of(_add, options, "WintProbe")[1]


def test_a_boolean_only_kernel_allocates_no_wide_register() -> None:
    """Its word is inert -- nothing wide is stored -- which is why the floor may answer for it unexamined."""
    verilog = _word_of(_flags, default_options(FMT), "BoolOnlyBank")[1]
    assert not re.search(r"reg\s+\[WREG-1:0\]\s+regs\b", verilog)


def _shift_past_the_narrow_word(x: int) -> int:
    return x << 20  # survives a 24-bit word and truncates to zero under a 16-bit one


def _float_arm_behind_a_shift(x: int, y: int) -> int:
    """Oscillates: the word that erases the float arm is the one whose own graph then keeps it."""
    q = (x << 24) // (y << 16)
    if q == 0:
        return 1
    return int(float(x) / float(y))


def _zero_div_behind_a_shift(a: int) -> int:
    """Refused at a 24-bit word, accepted at 16: the narrower word erases the arm the wider one convicts."""
    z = a - a
    if (a << 20) != 0:
        return 7 // z
    return 0


class _OversizedReset:
    def __init__(self) -> None:
        self.acc = 100_000

    def step(self, x: int) -> int:
        self.acc = self.acc + x
        return self.acc


def test_the_second_derivation_reshapes_the_graph_under_the_word_it_settled_on() -> None:
    """
    The word is an INPUT to the graph that answers it, so settling on a narrower one obliges a re-derivation. A
    machine sized in one pass would emit a shifter its own word cannot express.
    """
    options = dataclasses.replace(default_options(FMT), wint_min=16)  # the count lies inside [16, 24)
    result = holoso.synthesize(_shift_past_the_narrow_word, options, name="ShiftFoldsLate")
    assert result.int_format == IntFormat(16)
    assert "holoso_ishl" not in result.verilog_output.verilog, "the count is past the settled word, so it folded"
    (out,) = result.numerical_model.elaborate().run(3)
    assert isinstance(out, IntValue) and int(out) == 0


def test_a_narrowing_that_does_not_settle_keeps_the_widest_word() -> None:
    """No fixpoint need exist, so the loop is bounded and falls back rather than failing on a legal kernel."""
    operator = dataclasses.replace(
        default_options(FMT).operator, ffromint=holoso.FFromIntOptions(), ftoint=holoso.FToIntOptions()
    )
    options = dataclasses.replace(default_options(FMT), operator=operator, wint_min=16)
    result = holoso.synthesize(_float_arm_behind_a_shift, options, name="Oscillates")
    assert result.int_format == IntFormat(24), "the widest word every family fits stands where narrowing will not"


def _loop_behind_a_shift(x: int) -> int:
    y = 0
    while (x << 20) == 0:  # a real loop at the wide word; at the narrow one the header decides and never exits
        y = y + 1
    return y


def _scale_past_a_tiny_word(n: int) -> int:
    return n * 4  # the absorbed count is itself a machine integer, which a two-bit word cannot hold


def test_a_narrowing_that_leaves_nothing_buildable_keeps_the_widest_word() -> None:
    """A refusal inside the speculative narrow derivation is a failed narrowing, not a kernel that cannot be built."""
    options = dataclasses.replace(default_options(FMT), wint_min=16)
    result = holoso.synthesize(_loop_behind_a_shift, options, name="LoopBehindShift")
    assert result.int_format == IntFormat(24)
    (out,) = result.numerical_model.elaborate().run(1)
    assert isinstance(out, IntValue) and int(out) == 0


def test_an_absorbed_shift_count_fits_the_word_that_carries_it() -> None:
    """Only a two-bit word cannot represent its own width, and every count from width-1 up rails identically."""
    options = dataclasses.replace(default_options(FMT), wint_min=2)
    result = holoso.synthesize(_scale_past_a_tiny_word, options, name="TinyWordScale")
    assert result.int_format == IntFormat(2)
    sim = result.numerical_model.elaborate()
    for n, expect in ((0, 0), (1, 1), (-1, -2), (-2, -2)):  # int2 holds -2..1, so every nonzero product rails
        (out,) = sim.run(n)
        assert isinstance(out, IntValue) and int(out) == expect, n


def test_judgement_waits_for_the_graph_the_machine_actually_builds() -> None:
    """Refusal convicts what survives, and what survives depends on the word, so it may not run on a draft graph."""
    options = dataclasses.replace(default_options(FMT), wint_min=16)
    result = holoso.synthesize(_zero_div_behind_a_shift, options, name="ZeroDivErased")
    assert result.int_format == IntFormat(16)
    (out,) = result.numerical_model.elaborate().run(3)
    assert isinstance(out, IntValue) and int(out) == 0


def test_a_state_reset_past_the_floor_is_refused_by_its_own_gate() -> None:
    """A slot's reset is no HIR node, so it is the one oversized value the literal gate cannot catch."""
    options = dataclasses.replace(default_options(FloatFormat(8, 36)), wint_min=16)
    with pytest.raises(holoso.UnsupportedConstruct, match=r"slot 'acc' reset 100000 does not fit int16"):
        holoso.synthesize(_OversizedReset().step, options, name="ResetTooWide")
    widened = dataclasses.replace(options, wint_min=24)
    assert holoso.synthesize(_OversizedReset().step, widened, name="ResetFits").int_format == IntFormat(24)


def test_a_literal_past_the_floor_is_refused_where_a_wide_float_used_to_carry_it() -> None:
    """The float format no longer lends its width to an integer kernel, so the kernel must ask for what it needs."""

    def big(n: int) -> int:
        return n + 100_000

    options = dataclasses.replace(default_options(FloatFormat(8, 36)), wint_min=16)
    with pytest.raises(holoso.UnsupportedConstruct, match="does not fit int16; raise wint_min"):
        holoso.synthesize(big, options, name="TooNarrow")
    widened = dataclasses.replace(options, wint_min=24)
    assert holoso.synthesize(big, widened, name="WideEnough").int_format == IntFormat(24)

"""
The native integer format and type: the arithmetic surface, the plumbing from ``Options`` down into the emitted RTL,
and the port conditioning an integer type admits. No lowering selects an integer operator yet, so the format's only
hardware footprint is the ``WINT`` localparam -- which is nevertheless enough to pin the whole chain from the public
entry point through MIR and LIR to the Verilog backend.
"""

import dataclasses
import re

import pytest

import holoso
from holoso import FloatFormat, IntFormat, OperatorOptions, Options
from holoso._eel import lower as lower_frontend
from holoso._mir import MirOperation, MirPhi, lower as lower_to_mir
from holoso._operators import BoolInversion, FloatSignControl, IntIdentity, SelectOperator, identity_conditioner
from holoso._type import BoolType, IntType
from holoso._value import IntValue

from ._modelref import build_lir, build_ops, default_options

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


def test_default_options_carry_the_documented_int_format() -> None:
    # The floor is low enough that a float-only build at a practical format pays nothing for the integer half.
    assert Options(OperatorOptions()).ifmt == IntFormat(24)
    assert Options(OperatorOptions(), ffmt=FloatFormat(4, 8)).ifmt == IntFormat(16)


@pytest.mark.parametrize("wint_min,width", ((2, 24), (16, 24), (33, 33), (44, 44)))
def test_the_int_format_is_never_narrower_than_the_float(wint_min: int, width: int) -> None:
    options = dataclasses.replace(default_options(FMT), wint_min=wint_min)
    assert options.ifmt == IntFormat(width)
    mir = lower_to_mir(lower_frontend(_add).hir, build_ops(options), options.ffmt, options.ifmt, options.ifconv_max_ops)
    assert mir.int_format == options.ifmt
    assert build_lir(mir, "int_format_probe").int_format == options.ifmt


@pytest.mark.parametrize("wint_min,width", ((2, 24), (16, 24), (33, 33), (44, 44)))
def test_the_int_format_sizes_the_wide_register_bank_in_the_rtl(wint_min: int, width: int) -> None:
    # The end-to-end pin: driven through the public entry point, so a build that dropped the derivation anywhere
    # between here and the Verilog backend -- or substituted a plausible-but-wrong width -- fails right here.
    options = dataclasses.replace(default_options(FMT), wint_min=wint_min)
    verilog = holoso.synthesize(_add, options, name="WintProbe").verilog_output.verilog
    (wint,) = re.findall(r"^localparam\s+WINT\s*=\s*(\d+);", verilog, re.MULTILINE)
    (wreg,) = re.findall(r"^localparam\s+WREG\s*=\s*(\d+);", verilog, re.MULTILINE)
    assert int(wint) == width and int(wreg) == width
    assert "WFLT      = WEXP + WMAN;" in verilog, "the float keeps its own width inside the wider register"
    assert re.search(r"reg\s+\[WREG-1:0\]\s+regs\b", verilog)

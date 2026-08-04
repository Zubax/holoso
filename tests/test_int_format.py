"""
The native integer format: its arithmetic surface, and its plumbing from ``Options`` down into the emitted RTL. No
operator reads the format yet, so its only hardware footprint is the ``WINT`` localparam -- which is nevertheless
enough to pin the whole chain from the public entry point through MIR and LIR to the Verilog backend.
"""

import dataclasses
import re

import pytest

import holoso
from holoso import FloatFormat, IntFormat, OperatorOptions, Options
from holoso._eel import lower as lower_frontend
from holoso._hir import optimize
from holoso._mir import lower as lower_to_mir
from holoso._operators import SelectOperator
from holoso._type import BoolType, IntType

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
    # Signature only, and deliberately so: ``evaluate``'s operand contract is ``FloatValue | bool`` and there is no
    # ``IntValue`` yet, while ``_reject_integers`` refuses every integer HIR value before lowering begins, so an
    # integer select can be neither evaluated nor lowered into MIR. This pins the generalization, not working muxes.
    ty = IntType(IntFormat(width))
    signature = SelectOperator(ty).signature
    assert signature.operand_types == (BoolType(), ty, ty)
    assert signature.result_types == (ty,)


def _add(a: float, b: float) -> float:
    return a + b


def test_default_options_carry_the_documented_int_format() -> None:
    assert Options(OperatorOptions()).ifmt == IntFormat(33)


def test_configured_int_format_reaches_the_scheduled_machine() -> None:
    ifmt = IntFormat(17)
    options = dataclasses.replace(default_options(FMT), ifmt=ifmt)
    mir = lower_to_mir(optimize(lower_frontend(_add).hir), build_ops(options), options.ffmt, options.ifmt)
    assert mir.int_format == ifmt
    assert build_lir(mir, "int_format_probe").int_format == ifmt


@pytest.mark.parametrize("width", (2, 17, 33, 44))
def test_configured_int_format_surfaces_as_wint_in_the_rtl(width: int) -> None:
    # The end-to-end pin: driven through the public entry point, so a build that dropped ``Options.ifmt`` anywhere
    # between here and the Verilog backend -- or substituted a plausible-but-wrong width -- fails right here.
    options = dataclasses.replace(default_options(FMT), ifmt=IntFormat(width))
    verilog = holoso.synthesize(_add, options, name="WintProbe").verilog_output.verilog
    (emitted,) = re.findall(r"^localparam\s+WINT\s*=\s*(\d+);", verilog, re.MULTILINE)
    assert int(emitted) == width
    assert "WFLT      = WEXP + WMAN;" in verilog, "the float width stays independent of the integer width"

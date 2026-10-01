"""
The inline integer primitives in fabric: each one's own `verilog_expr` spliced into a combinational harness, scored
against an independent Python bit formula per primitive family.

The generated transport -- the emitter's operand nets, conditioners and register writes -- is owned elsewhere:
test_cosim_int drives inline primitives end-to-end through generated machines (the inline xor in `sat_mix`, the
inline shift/mask in `pow2_strength`) and test_int_selection pins which primitives MIR selects.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import Timer
from cocotb_tools.runner import get_runner

from holoso._operators import (
    BoolToIntPrimitive,
    InlinePrimitive,
    IntBwAndPrimitive,
    IntBwNotPrimitive,
    IntBwOrPrimitive,
    IntBwXorPrimitive,
    IntShiftConstPrimitive,
    IntToBoolPrimitive,
)
from holoso._type import BoolType, IntFormat, IntType

from .hdl_float_oracle import REPO_ROOT, SIMULATORS, build_args, get_seed

EXHAUSTIVE_WIDTH_LIMIT = 6
"""At and below this width every operand pair is driven, so the sweep proves the primitives there, not samples them."""

TOPLEVEL = "holoso_int_inline_tb"


# Every width up to the exhaustive limit, then the production formats. Widths where 4 divides `width - 1` matter to
# the shift, whose largest amounts land on a group boundary.
_WIDTHS = (*range(2, EXHAUSTIVE_WIDTH_LIMIT + 1), 9, 24, 33, 44)


def _shift_counts(fmt: IntFormat) -> list[int]:
    """Every count the primitive accepts, which at the production widths is too many to instantiate one output each."""
    legal = [count for count in range(1 - fmt.width, fmt.width) if count != 0]
    if fmt.width <= EXHAUSTIVE_WIDTH_LIMIT:
        return legal
    corners = {1, 2, 3, fmt.width - 2, fmt.width - 1}
    return sorted(count for count in legal if abs(count) in corners)


@dataclass(frozen=True, slots=True)
class _Case:
    """
    One primitive over one folded-conditioner assignment of its operands, owning its literal operand nets and result
    family: the boolean port takes the harness net `c`, the integer ones `a` then `b`. An integer port folds
    nothing, but a boolean port folds an inversion that the emitter splices as `~net`, so a primitive taking one
    must serve both spellings.
    """

    primitive: InlinePrimitive
    invert: bool
    operands: tuple[str, ...]
    wide_result: bool

    @property
    def nets(self) -> list[str]:
        return [f"~{net}" if self.invert and net == "c" else net for net in self.operands]


def _cases(fmt: IntFormat) -> list[_Case]:
    table: list[tuple[InlinePrimitive, tuple[str, ...], bool]] = [
        (IntBwAndPrimitive(fmt), ("a", "b"), True),
        (IntBwOrPrimitive(fmt), ("a", "b"), True),
        (IntBwXorPrimitive(fmt), ("a", "b"), True),
        (IntBwNotPrimitive(fmt), ("a",), True),
        (IntToBoolPrimitive(fmt), ("a",), False),
        (BoolToIntPrimitive(fmt), ("c",), True),
        *((IntShiftConstPrimitive(fmt, count), ("a",), True) for count in _shift_counts(fmt)),
    ]
    cases: list[_Case] = []
    for primitive, operands, wide_result in table:
        # The production signature is checked against the case's own literal shape, never consulted for it: a
        # signature defect must fail here, not be coerced through the numeric compare downstream.
        expected_operands = tuple(BoolType() if net == "c" else IntType(fmt) for net in operands)
        assert primitive.signature.operand_types == expected_operands, primitive.mnemonic
        expected_result = (IntType(fmt),) if wide_result else (BoolType(),)
        assert primitive.signature.result_types == expected_result, primitive.mnemonic
        polarities = (False, True) if "c" in operands else (False,)
        cases.extend(_Case(primitive, invert, operands, wide_result) for invert in polarities)
    return cases


def _port(index: int, case: _Case) -> str:
    return f"y{index}_{case.primitive.mnemonic}"  # the index disambiguates the several shift counts and both polarities


def _nested_port(index: int, case: _Case) -> str:
    return f"n{index}_{case.primitive.mnemonic}"


def _harness(width: int, cases: Sequence[_Case]) -> str:
    ports = [
        f"    input  wire [{width - 1}:0] a,",
        f"    input  wire [{width - 1}:0] b,",
        "    input  wire  c,",
        "    input  wire  sel,",
    ]
    assigns = []
    for index, case in enumerate(cases):
        declaration = f"[{width - 1}:0] " if case.wide_result else " "
        expression = case.primitive.verilog_expr(*case.nets)
        ports.append(f"    output wire {declaration}{_port(index, case)},")
        ports.append(f"    output wire {declaration}{_nested_port(index, case)},")
        assigns.append(f"    assign {_port(index, case)} = {expression};")
        # Substituted beside an UNSIGNED sibling, which is what nesting will do: the ternary is then unsigned, and an
        # expression that leaks its signedness to the context degrades to a logical shift.
        sibling = "c" if not case.wide_result else "a"
        assigns.append(f"    assign {_nested_port(index, case)} = sel ? {expression} : {sibling};")
    ports[-1] = ports[-1].rstrip(",")
    return "\n".join(
        [
            "`default_nettype none",
            "",
            f"module {TOPLEVEL} (",
            *ports,
            ");",
            *assigns,
            "endmodule",
            "",
            "`default_nettype wire",
            "",
        ]
    )


def _expected_bits(primitive: InlinePrimitive, width: int, a: int, b: int, c: bool) -> int:
    """
    The independent oracle: a plain Python bit formula per primitive family over the signed operands `a`/`b` and
    the boolean `c`, never the primitive's own `evaluate`.
    """
    mask = (1 << width) - 1
    if isinstance(primitive, IntBwAndPrimitive):
        return (a & b) & mask
    if isinstance(primitive, IntBwOrPrimitive):
        return (a | b) & mask
    if isinstance(primitive, IntBwXorPrimitive):
        return (a ^ b) & mask
    if isinstance(primitive, IntBwNotPrimitive):
        return ~a & mask
    if isinstance(primitive, IntToBoolPrimitive):
        return int(a != 0)
    if isinstance(primitive, BoolToIntPrimitive):
        return int(c)
    assert isinstance(primitive, IntShiftConstPrimitive)
    return ((a << primitive.shamt) if primitive.shamt > 0 else (a >> -primitive.shamt)) & mask


def _operand_values(fmt: IntFormat, rng: np.random.Generator) -> list[int]:
    if fmt.width <= EXHAUSTIVE_WIDTH_LIMIT:
        return list(range(fmt.min, fmt.max + 1))
    values = {fmt.min, fmt.min + 1, -2, -1, 0, 1, 2, fmt.max - 1, fmt.max}
    for shift in range(fmt.width - 1):
        values.update((1 << shift, -(1 << shift), (1 << shift) - 1))
    values.update(int(rng.integers(fmt.min, fmt.max + 1)) for _ in range(8))
    return sorted(value for value in values if fmt.fits(value))


@cocotb.test()
async def holoso_int_inline_cocotb(dut: Any) -> None:
    fmt = IntFormat(int(os.environ["HOLOSO_INT_INLINE_WINT"]))
    cases = _cases(fmt)
    values = _operand_values(fmt, np.random.default_rng(get_seed()))
    mask = (1 << fmt.width) - 1

    for a in values:
        for b in values:
            for c in (False, True):
                dut.a.value = a & mask
                dut.b.value = b & mask
                dut.c.value = int(c)
                dut.sel.value = 1
                await Timer(1, unit="ns")
                for index, case in enumerate(cases):
                    want = _expected_bits(case.primitive, fmt.width, a, b, c ^ case.invert)
                    for port in (_port(index, case), _nested_port(index, case)):
                        actual = int(getattr(dut, port).value)
                        assert actual == want, f"{port} {fmt} a={a} b={b} c={c}: got {actual:#x}, want {want:#x}"


@pytest.mark.parametrize("width", _WIDTHS, ids=lambda width: f"w{width}")
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_int_inline(sim: str, width: int) -> None:
    # Named per simulator as well as per width: the two simulator rows of one width are separate xdist items, and a
    # shared path would let one truncate the file the other is building from.
    generated = REPO_ROOT / "build" / "cocotb" / "int_inline" / f"{TOPLEVEL}_{sim}_w{width}.v"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(_harness(width, _cases(IntFormat(width))), encoding="utf-8")
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"int_inline_w{width}"
    runner.build(
        sources=[generated],  # the expressions call no support function, so the harness stands alone
        hdl_toplevel=TOPLEVEL,
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel=TOPLEVEL,
        test_module="tests.hdl.test_int_inline",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={"HOLOSO_INT_INLINE_WINT": str(width)},
        results_xml=str(build_dir / "results.xml"),
    )

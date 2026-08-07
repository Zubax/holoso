"""
The inline integer operators in fabric: each one's own ``verilog_expr`` spliced into a combinational harness, scored
against its own ``evaluate`` over the whole register, which an integer fills.

What this does NOT cover is the generated transport -- the emitter's operand nets, conditioners and register writes --
which no test can reach until MIR selects an integer operator.
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
    BoolToIntOperator,
    InlineHardwareOperator,
    IntBwAndOperator,
    IntBwNotOperator,
    IntBwOrOperator,
    IntBwXorOperator,
    IntShiftConstOperator,
    IntToBoolOperator,
)
from holoso._type import BoolType, IntFormat
from holoso._value import IntValue, ScalarValue

from .hdl_float_oracle import REPO_ROOT, SIMULATORS, build_args, get_seed

EXHAUSTIVE_WIDTH_LIMIT = 6
"""At and below this width every operand pair is driven, so the sweep proves the operators there, not samples them."""

TOPLEVEL = "holoso_int_inline_tb"


# Every width up to the exhaustive limit, then the production formats. Widths where 4 divides ``width - 1`` matter to
# the shift, whose largest amounts land on a group boundary.
_WIDTHS = (*range(2, EXHAUSTIVE_WIDTH_LIMIT + 1), 9, 24, 33, 44)


def _shift_counts(fmt: IntFormat) -> list[int]:
    """Every count the operator accepts, which at the production widths is too many to instantiate one output each."""
    legal = [count for count in range(1 - fmt.width, fmt.width) if count != 0]
    if fmt.width <= EXHAUSTIVE_WIDTH_LIMIT:
        return legal
    corners = {1, 2, 3, fmt.width - 2, fmt.width - 1}
    return sorted(count for count in legal if abs(count) in corners)


def _operators(fmt: IntFormat) -> list[InlineHardwareOperator]:
    return [
        IntBwAndOperator(fmt),
        IntBwOrOperator(fmt),
        IntBwXorOperator(fmt),
        IntBwNotOperator(fmt),
        IntToBoolOperator(fmt),
        BoolToIntOperator(fmt),
        *(IntShiftConstOperator(fmt, count) for count in _shift_counts(fmt)),
    ]


def _operand_slots(operator: InlineHardwareOperator) -> list[str]:
    """The boolean port takes the harness net ``c``, the integer ones ``a`` then ``b``."""
    wide = iter(("a", "b"))
    return ["c" if isinstance(ty, BoolType) else next(wide) for ty in operator.signature.operand_types]


@dataclass(frozen=True, slots=True)
class _Case:
    """
    One operator over one folded-conditioner assignment of its operands. An integer port folds nothing, but a boolean
    port folds an inversion that the emitter splices as ``~net``, so an operator taking one must serve both spellings.
    """

    operator: InlineHardwareOperator
    invert: bool

    @property
    def nets(self) -> list[str]:
        return [f"~{slot}" if self.invert and slot == "c" else slot for slot in _operand_slots(self.operator)]


def _cases(fmt: IntFormat) -> list[_Case]:
    cases: list[_Case] = []
    for operator in _operators(fmt):
        takes_bool = any(isinstance(ty, BoolType) for ty in operator.signature.operand_types)
        cases.extend(_Case(operator, invert) for invert in ((False, True) if takes_bool else (False,)))
    return cases


def _port(index: int, case: _Case) -> str:
    return f"y{index}_{case.operator.mnemonic}"  # the index disambiguates the several shift counts and both polarities


def _harness(width: int, cases: Sequence[_Case]) -> str:
    ports = [
        f"    input  wire [{width - 1}:0] a,",
        f"    input  wire [{width - 1}:0] b,",
        "    input  wire  c,",
    ]
    assigns = []
    for index, case in enumerate(cases):
        (result,) = case.operator.signature.result_types
        declaration = f"[{width - 1}:0] " if result.is_wide else " "
        ports.append(f"    output wire {declaration}{_port(index, case)},")
        assigns.append(f"    assign {_port(index, case)} = {case.operator.verilog_expr(*case.nets)};")
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

    for a in values:
        for b in values:
            for c in (False, True):
                dut.a.value = fmt.encode(a)
                dut.b.value = fmt.encode(b)
                dut.c.value = int(c)
                await Timer(1, unit="ns")
                for index, case in enumerate(cases):
                    payload: dict[str, ScalarValue] = {
                        "a": IntValue.from_int(fmt, a),
                        "b": IntValue.from_int(fmt, b),
                        "c": c ^ case.invert,
                    }
                    (expected,) = case.operator.evaluate(*(payload[slot] for slot in _operand_slots(case.operator)))
                    assert isinstance(expected, IntValue | bool)
                    want = int(expected) if isinstance(expected, bool) else expected.bits
                    actual = int(getattr(dut, _port(index, case)).value)
                    assert actual == want, f"{case!r} {fmt} a={a} b={b} c={c}: got {actual:#x}, want {want:#x}"


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

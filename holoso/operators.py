"""Operator kinds, their latency model (mirroring ``holoso_support.v``), and sign-op encoding.

This module is the single source of truth shared by the passes (latency annotation), the scheduler (issue priority),
and the Verilog backend (instantiation), so the latency formulas cannot drift from the HDL wrappers.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import assert_never

from .format import FloatFormat


class Sgnop(enum.IntFlag):
    """Folded sign manipulation applied to an operator operand or output.

    A 2-bit field: bit 0 = negate, bit 1 = absolute value (so ``ABS | NEG`` means ``-|x|``). The integer values match
    ``HOLOSO_FSGNOP_*`` in ``holoso_support.vh`` (NONE=0, NEG=1, ABS=2, ABS|NEG=3), which is why ``IntFlag`` is used:
    ``int(op)`` yields the Verilog encoding while membership tests (``Sgnop.ABS in op``) express the bit semantics.
    """

    NONE = 0
    NEG = 1
    ABS = 2

    def decorate(self, text: str) -> str:
        """Wrap a value's name to show this sign-op: NEG -> ``-x``, ABS -> ``|x|``, ABS|NEG -> ``-|x|``."""
        if Sgnop.ABS in self:
            text = f"|{text}|"
        if Sgnop.NEG in self:
            text = f"-{text}"
        return text


class OpKind(enum.Enum):
    FADD = "fadd"
    FMUL = "fmul"
    FDIV = "fdiv"
    FMUL_ILOG2 = "fmul_ilog2_const"


MODULE_NAMES: dict[OpKind, str] = {
    OpKind.FADD: "holoso_fadd",
    OpKind.FMUL: "holoso_fmul",
    OpKind.FDIV: "holoso_fdiv",
    OpKind.FMUL_ILOG2: "holoso_fmul_ilog2_const",
}


@dataclass(frozen=True, slots=True)
class StageConfig:
    """Optional pipeline-stage knobs per operator (default off). They affect latency and instantiation parameters."""

    fadd_decode: int = 0
    fadd_align: int = 0
    fmul_product: int = 0
    fdiv_input: int = 0
    fmul_ilog2_decode: int = 0


DEFAULT_STAGES = StageConfig()


def arity(kind: OpKind) -> int:
    """Number of operand inputs (``fmul_ilog2_const`` is unary; the rest are binary)."""
    return 1 if kind is OpKind.FMUL_ILOG2 else 2


def has_div0(kind: OpKind) -> bool:
    return kind is OpKind.FDIV


def latency_of(kind: OpKind, fmt: FloatFormat, stages: StageConfig = DEFAULT_STAGES) -> int:
    """Replicate the ``LATENCY`` localparam of each ``holoso_support.v`` wrapper (used for scheduling only)."""
    match kind:
        case OpKind.FADD:
            return 6 + stages.fadd_decode + stages.fadd_align
        case OpKind.FMUL:
            return 3 + stages.fmul_product
        case OpKind.FMUL_ILOG2:
            return 1 + stages.fmul_ilog2_decode
        case OpKind.FDIV:
            w = fmt.wman
            return 4 + ((w + 2 + ((w + 2) % 2)) // 2) + stages.fdiv_input
        case _ as unreachable:
            assert_never(unreachable)

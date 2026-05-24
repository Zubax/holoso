"""Operator kinds, their latency model (mirroring ``holoso_support.v``), and sign-op encoding.

This module is the single source of truth for operator latency, shared by the passes (latency annotation), the
scheduler (exact issue/commit timing), and the Verilog backend (instantiation). The latency formulas MUST match the
HDL wrappers cycle-for-cycle: the static schedule commits each result on ``issue + latency`` without watching
``out_valid``, so any drift is a correctness bug, not merely a bad estimate.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, fields
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
class ResourceKey:
    """A distinct physical operator module that ops may share.

    Two ops share a key iff they elaborate to identical hardware: same ``kind`` and same elaboration-time
    ``params``. ``params`` are the parameters baked into the module at elaboration -- today only
    ``fmul_ilog2_const``'s exponent ``K`` -- and is empty for ``fadd``/``fmul``/``fdiv``. The scheduler pools and
    caps operator instances by this key, so ops sharing a key can time-share one instance.
    """

    kind: OpKind
    params: tuple[int, ...] = ()

    @staticmethod
    def of(kind: OpKind, k: int | None) -> "ResourceKey":
        """Derive the key from a kind and its optional exponent (``k`` is set iff ``kind`` is ``FMUL_ILOG2``)."""
        return ResourceKey(kind, () if k is None else (k,))


@dataclass(frozen=True, slots=True)
class StageConfig:
    """Optional pipeline-stage knobs per operator (default off). They affect latency and instantiation parameters.

    Each knob is 0 or 1: the HDL wrappers gate every stage as ``(STAGE_X ? 1 : 0)``, so a value above 1 would lengthen
    ``latency_of`` without changing the RTL -- silently desyncing the static schedule from the hardware.
    """

    fadd_decode: int = 0
    fadd_align: int = 0
    fmul_product: int = 0
    fdiv_input: int = 0
    fmul_ilog2_decode: int = 0

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if value not in (0, 1):
                raise ValueError(f"StageConfig.{f.name} must be 0 or 1 (RTL gates each stage ?:1:0); got {value!r}")


DEFAULT_STAGES = StageConfig()

# kind -> ((HDL wrapper param, StageConfig field), ...): the optional pipeline stages and what drives each. The single
# source for both the latency contribution (latency_of) and the instantiation parameters (stage_params), so they cannot
# drift; adding a stage to an operator is a one-line edit here.
_STAGE_KNOBS: dict["OpKind", tuple[tuple[str, str], ...]] = {
    OpKind.FADD: (("STAGE_DECODE", "fadd_decode"), ("STAGE_ALIGN", "fadd_align")),
    OpKind.FMUL: (("STAGE_PRODUCT", "fmul_product"),),
    OpKind.FDIV: (("STAGE_INPUT", "fdiv_input"),),
    OpKind.FMUL_ILOG2: (("STAGE_DECODE", "fmul_ilog2_decode"),),
}


def arity(kind: OpKind) -> int:
    """Number of operand inputs (``fmul_ilog2_const`` is unary; the rest are binary)."""
    return 1 if kind is OpKind.FMUL_ILOG2 else 2


def has_div0(kind: OpKind) -> bool:
    return kind is OpKind.FDIV


def latency_of(kind: OpKind, fmt: FloatFormat, stages: StageConfig = DEFAULT_STAGES) -> int:
    """The exact ``LATENCY`` of each ``holoso_support.v`` wrapper, in clocks -- load-bearing, not a hint.

    The schedule commits each result on ``issue + latency`` and the backend trusts that timing without watching
    ``out_valid``, so this must replicate the RTL wrapper's ``LATENCY`` localparam exactly.
    """
    extra = sum(int(getattr(stages, field)) for _, field in _STAGE_KNOBS[kind])
    match kind:
        case OpKind.FADD:
            return 6 + extra
        case OpKind.FMUL:
            return 3 + extra
        case OpKind.FMUL_ILOG2:
            return 1 + extra
        case OpKind.FDIV:
            w = fmt.wman
            return 4 + ((w + 2 + ((w + 2) % 2)) // 2) + extra
        case _ as unreachable:
            assert_never(unreachable)


def stage_params(kind: OpKind, stages: StageConfig) -> dict[str, int]:
    """The ``STAGE_*`` instantiation parameters for an operator of ``kind``, from the same table as ``latency_of``."""
    return {param: int(getattr(stages, field)) for param, field in _STAGE_KNOBS[kind]}

from dataclasses import dataclass
from typing import Any

import zkf
from amaranth.hdl import Const, Module, Signal, Value

from ..._type import IntFormat, IntType, ScalarType
from ..._value import IntValue


@dataclass(frozen=True, kw_only=True, eq=False)
class IntBlock(zkf.Block):
    """
    An operator on signed `wint`-bit two's complement words. `stage_input` and `stage_output` count the register stages
    at the input and the output ports; zero leaves that side combinational.
    """

    wint: int
    stage_input: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.wint >= 2, "the machine word comes from an IntFormat, which refuses less"
        if self.stage_input < 0:
            raise ValueError(f"stage_input={self.stage_input} is negative")
        if self.stage_output < 0:
            raise ValueError(f"stage_output={self.stage_output} is negative")

    def _word(self, value: int) -> IntValue:
        return IntValue.from_int(IntFormat(self.wint), value)


class Datapath:
    """
    The register stages of an operator, built one group of parallel signals at a time, so that an immediate or a flag
    cannot drift apart from its payload and `depth` counts what was built rather than what was intended.
    """

    def __init__(self, m: Module) -> None:
        self._m = m
        self.depth = 0

    def register(self, label: str, stages: int, **values: Any) -> list[Value]:
        """
        Every value delayed by `stages` registers without reset, in argument order. A constant, such as an immediate the
        block is specialized to, is the same at every stage and takes no register.
        """
        assert stages >= 0 and values
        group = {name: Value.cast(value) for name, value in values.items()}
        for k in range(stages):
            for name, value in group.items():
                if isinstance(value, Const):
                    continue
                q = Signal(value.shape(), name=f"{name}_{label}_q{k}", reset_less=True)
                self._m.d.sync += q.eq(value)
                group[name] = q
        self.depth += stages
        return list(group.values())


def int_ports(fmt: IntFormat, *names: str) -> dict[str, ScalarType]:
    return {name: IntType(fmt) for name in names}

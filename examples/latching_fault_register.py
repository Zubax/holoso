#!/usr/bin/env python3
"""
A latching fault register: the sticky status/alarm bits behind a hardware safety interlock. Each fault input, once
asserted, stays latched until the synchronous reset clears it, so a transient overcurrent/overvoltage/overtemperature
event that has already disappeared remains visible to the supervisor that polls the register.
A combinational ``any_fault`` summary ORs the latched channels for a single trip line.
"""

from pathlib import Path

import holoso


class LatchingFaultRegister:
    def __init__(self) -> None:
        self._overcurrent: bool = False
        self._overvoltage: bool = False
        self._overtemp: bool = False

    def __call__(self, overcurrent: bool, overvoltage: bool, overtemp: bool, /) -> tuple[bool, bool, bool, bool]:
        # Each latch is one `or` of its own prior state and its input, so the new-state producer reads only resident
        # values (carried state and fresh input) and is eligible for the entry block's very first microcode word.
        # That is precisely what the scheduler's accept-dwell guard holds off cycle 0: while the sequencer waits for
        # `in_valid` it re-fires `ucode[0]` every idle cycle, so a state-bearing write there would be re-driven with
        # stale inputs. This kernel is the exemplar of that guard -- its three latches are floored to cycle 1,
        # one step later than an unguarded combinational write would land.
        self._overcurrent = self._overcurrent or overcurrent
        self._overvoltage = self._overvoltage or overvoltage
        self._overtemp = self._overtemp or overtemp
        any_fault = self._overcurrent or self._overvoltage or self._overtemp
        return any_fault, self._overcurrent, self._overvoltage, self._overtemp


def main() -> None:
    float_format = holoso.FloatFormat(wexp=8, wman=36)
    ops = holoso.OpConfig(
        holoso.FAddOperator(float_format),
        holoso.FMulOperator(float_format),
        holoso.FDivOperator(float_format),
        holoso.FMulILog2OperatorFamily(float_format),
        holoso.FCmpOperator(float_format),
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(LatchingFaultRegister().__call__, ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

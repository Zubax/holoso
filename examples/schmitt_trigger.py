#!/usr/bin/env python3
"""A Schmitt trigger: a comparator with hysteresis."""

from pathlib import Path

import holoso


class SchmittTrigger:
    """Between the two thresholds the output holds its previous value (the hysteresis deadband)."""

    def __init__(self, *, high: float = 1.0, low: float = -1.0, initial: bool = False) -> None:
        self.high: float = high
        self.low: float = low
        self.y: bool = initial  # persistent output state held in the deadband

    def __call__(self, x: float, /) -> bool:
        if x > self.high:
            self.y = True
        elif x < self.low:
            self.y = False
        return self.y  # Duplicates the public member `y`, so this output port will be elided.


def main() -> None:
    float_format = holoso.FloatFormat(wexp=8, wman=36)
    options = holoso.Options(
        operator=holoso.OperatorOptions(
            fadd=holoso.FAddOptions(stage_input=0, stage_core=holoso.FAddOptions.CoreStages.NONE, stage_output=0),
            fmul=holoso.FMulOptions(stage_input=0, stage_core_mul=0, stage_output=0),
            fdivsqrt=holoso.FDivsqrtOptions(
                stage_input=0, stage_core=holoso.FDivsqrtOptions.CoreStages.NONE, stage_output=0, fast=True
            ),
            fmul_ilog2=holoso.FMulILog2Options(
                stage_input=0, stage_core=holoso.FMulILog2Options.CoreStages.NONE, stage_output=0
            ),
            fcmp=holoso.FCmpOptions(stage_input=0, stage_output=0),
        ),
        ffmt=float_format,
        wmultiplier=18,
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(SchmittTrigger().__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

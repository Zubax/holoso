#!/usr/bin/env python3
"""A single-pole low-pass IIR filter."""

from pathlib import Path
import holoso


class IIR1LPF:
    """
    A single-pole low-pass IIR filter. Difference equation: y[n] = y[n-1] + alpha * (x[n] - y[n-1]).
    """

    def __init__(self, *, ALPHA: float = 2**-16):
        self.alpha: float = ALPHA
        self.y: float = 0.0
        self._first: bool = True

    def __call__(self, x: float) -> float:
        if self._first:
            self._first = False
            self.y = x
        else:
            self.y += self.alpha * (x - self.y)
        return self.y


def main() -> None:
    float_format = holoso.FloatFormat(wexp=8, wman=36)
    options = holoso.Options(
        operator=holoso.OperatorOptions(
            fadd=holoso.FAddOptions(
                stage_input=0,
                stage_core=holoso.FAddOptions.CoreStages.DECODE
                | holoso.FAddOptions.CoreStages.OPERANDS
                | holoso.FAddOptions.CoreStages.ALIGNED
                | holoso.FAddOptions.CoreStages.SUM
                | holoso.FAddOptions.CoreStages.NORMALIZE
                | holoso.FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
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
    result = holoso.synthesize(IIR1LPF().__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
A single-pole high-pass IIR filter based on LF baseline subtraction.
"""

from pathlib import Path
from iir1_lpf import IIR1LPF
import holoso


class IIR1HPF:
    """
     A single-pole high-pass IIR filter. Difference equation:

        m[n] = m[n-1] + alpha * (x[n] - m[n-1])
        y[n] = x[n] - m[n]

    The LPF state `m` is the estimated low-frequency/DC bias.
    """

    def __init__(self, *, ALPHA: float = 2**-16):
        self.lpf = IIR1LPF(ALPHA=ALPHA)

    def step(self, x: float) -> float:
        x = float(x)  # No-op in Holoso (ignored), while it may be useful in numerical Python.
        bias = self.lpf(x)
        return x - bias


def main() -> None:
    float_format = holoso.FloatFormat(wexp=8, wman=36)
    options = holoso.Options(
        operator=holoso.OperatorOptions(
            fadd=holoso.FAddOptions(
                stage_input=0,
                stage_core=holoso.FAddOptions.CoreStages.DECODE
                | holoso.FAddOptions.CoreStages.OPERANDS
                | holoso.FAddOptions.CoreStages.ALIGNED
                | holoso.FAddOptions.CoreStages.NORMALIZE
                | holoso.FAddOptions.CoreStages.NORMALIZED,
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
    result = holoso.synthesize(IIR1HPF().step, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A 4-tap finite impulse response filter on a numpy delay line."""

from pathlib import Path

import numpy as np

import holoso


class Fir4:
    """
    A 4-tap FIR: `y[n] = sum(taps[i] * x[n-3+i])`.
    The taps fold to constants; the default ones are a unit-gain smoothing kernel.
    The delay line is shifted by rebuilding it from its own tail plus the new sample -- the idiom compiled into
    a register-to-register move in hardware.
    """

    def __init__(self, taps: tuple[float, float, float, float] = (0.1, 0.3, 0.4, 0.2)) -> None:
        self._taps = np.array(taps)
        self._line = np.zeros(4)

    def __call__(self, x: float) -> float:
        self._line = np.array((*self._line[1:], x))
        acc = 0.0
        for i in range(4):
            acc = acc + self._taps[i] * self._line[i]
        return acc


def main() -> None:
    float_format = holoso.FloatFormat(wexp=8, wman=36)
    options = holoso.Options(
        operator=holoso.OperatorOptions(
            fadd=holoso.FAddOptions(
                stage_input=0,
                stage_core=holoso.FAddOptions.CoreStages.OPERANDS
                | holoso.FAddOptions.CoreStages.ALIGNED
                | holoso.FAddOptions.CoreStages.SUM
                | holoso.FAddOptions.CoreStages.NORMALIZE
                | holoso.FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=holoso.FMulOptions(stage_input=0, stage_core_mul=3, stage_output=1),
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
    result = holoso.synthesize(Fir4().__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

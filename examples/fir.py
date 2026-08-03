#!/usr/bin/env python3
"""
A 4-tap finite impulse response filter on a numpy delay line.
"""

from pathlib import Path

import numpy as np

import holoso


class Fir4:
    """
    A 4-tap FIR: ``y[n] = sum(taps[i] * x[n-3+i])``. The taps are frozen at construction and fold to constants; the
    default ones are a unit-gain smoothing kernel and none is a power of two, so each costs a real multiply rather
    than a shift. The delay line is private persistent state, shifted by rebuilding it from its own tail plus the new
    sample -- the idiom that keeps the array unaliased, so the shift is a register-to-register move in hardware.
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
        holoso.OperatorOptions(
            fadd=holoso.FAddOptions(),
            fmul=holoso.FMulOptions(),
            fdiv=holoso.FDivOptions(),
            fmul_ilog2=holoso.FMulILog2Options(),
            fcmp=holoso.FCmpOptions(),
        ),
        ffmt=float_format,
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(Fir4().__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

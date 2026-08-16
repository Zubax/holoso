#!/usr/bin/env python3
"""A second-order IIR section in the transposed direct form II."""

from pathlib import Path

import holoso


class Biquad:
    """
    A biquad in transposed direct form II: `y = b0*x + s1`, then `s1 <- b1*x - a1*y + s2` and `s2 <- b2*x - a2*y`.
    The two accumulators are the whole state, each new value depends only on the current sample and the previous
    accumulators, and no feedback path is longer than one addition.
    """

    def __init__(self, b: tuple[float, float, float] = (0.2, 0.4, 0.2), a: tuple[float, float] = (-0.5, 0.25)) -> None:
        self._b0, self._b1, self._b2 = b
        self._a1, self._a2 = a
        self.s1 = 0.0
        self.s2 = 0.0

    def __call__(self, x: float) -> float:
        y = self._b0 * x + self.s1
        self.s1 = self._b1 * x - self._a1 * y + self.s2
        self.s2 = self._b2 * x - self._a2 * y
        return y


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
    result = holoso.synthesize(Biquad().__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

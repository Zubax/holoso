#!/usr/bin/env python3
"""
A single-pole low-pass IIR filter.
"""

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
    ops = holoso.OpConfig(
        holoso.FAddOperator(float_format),
        holoso.FMulOperator(float_format),
        holoso.FDivOperator(float_format),
        holoso.FMulILog2OperatorFamily(float_format),
        holoso.FCmpOperator(float_format),
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(IIR1LPF().__call__, ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

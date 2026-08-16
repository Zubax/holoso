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
    result = holoso.synthesize(IIR1HPF().step, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Tiny multiply-add kernel, a leftover from the very early days of Holoso.
The const mults strength-reduce to fmul_ilog2 (K = -2); `a` and `b` are each read twice, the products are single-use.
"""

from pathlib import Path
import holoso


def madd(a: float, b: float, c: float) -> float:
    """`c` is an unused argument, kept as given to exercise dead-input handling."""
    return (a - b) * 0.25 + a * b * 8


def main() -> None:
    float_format = holoso.FloatFormat(wexp=6, wman=18)
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
    result = holoso.synthesize(madd, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

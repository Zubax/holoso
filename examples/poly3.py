#!/usr/bin/env python3
"""
A simple polynomial demo kernel that survived from the very early days of Holoso.
Illustrates how the optional FMA operator changes the synthesis when enabled.
"""

import dataclasses
from math import fma
from pathlib import Path
import holoso


def poly3(x: float, c0: float, c1: float, c2: float, c3: float) -> float:
    """Degree-3 polynomial evaluated in Horner form: ((c3 * x + c2) * x + c1) * x + c0."""
    return fma(fma(fma(c3, x, c2), x, c1), x, c0)


def main() -> None:
    float_format = holoso.FloatFormat(wexp=6, wman=18)
    operators = holoso.OperatorOptions(
        fadd=holoso.FAddOptions(),
        fmul=holoso.FMulOptions(),
        fdivsqrt=holoso.FDivsqrtOptions(),
        fmul_ilog2=holoso.FMulILog2Options(),
        fcmp=holoso.FCmpOptions(),
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    # The same kernel with and without the fused multiply-add: a Horner step rounded once, or a product and a sum.
    for label, ffma in (("poly3", None), ("poly3_fma", holoso.FFmaOptions())):
        options = holoso.Options(dataclasses.replace(operators, ffma=ffma), ffmt=float_format)
        result = holoso.synthesize(poly3, options, name=label)
        print(f"{label}: II {result.initiation_interval}")
        for filename, path in result.write(out_dir).items():
            print(f"    {filename}: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
A simple polynomial demo kernel that survived from the very early days of Holoso.
Illustrates how the optional FMA operator changes the synthesis when enabled.
"""

from math import fma
from pathlib import Path
import holoso


def poly3(x: float, c0: float, c1: float, c2: float, c3: float) -> float:
    """Degree-3 polynomial evaluated in Horner form: ((c3 * x + c2) * x + c1) * x + c0."""
    return fma(fma(fma(c3, x, c2), x, c1), x, c0)


def main() -> None:
    float_format = holoso.FloatFormat(wexp=6, wman=18)
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    # The same kernel with and without the fused multiply-add: a Horner step rounded once, or a product and a sum.
    for label, options in (
        (
            "poly3",
            holoso.Options(
                operator=holoso.OperatorOptions(
                    fadd=holoso.FAddOptions(
                        stage_input=0,
                        stage_core=holoso.FAddOptions.CoreStages.OPERANDS
                        | holoso.FAddOptions.CoreStages.ALIGNED
                        | holoso.FAddOptions.CoreStages.NORMALIZE
                        | holoso.FAddOptions.CoreStages.PACK,
                        stage_output=0,
                    ),
                    fmul=holoso.FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
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
            ),
        ),
        (
            "poly3_fma",
            holoso.Options(
                operator=holoso.OperatorOptions(
                    fadd=holoso.FAddOptions(
                        stage_input=0, stage_core=holoso.FAddOptions.CoreStages.NONE, stage_output=0
                    ),
                    fmul=holoso.FMulOptions(stage_input=0, stage_core_mul=0, stage_output=0),
                    fdivsqrt=holoso.FDivsqrtOptions(
                        stage_input=0, stage_core=holoso.FDivsqrtOptions.CoreStages.NONE, stage_output=0, fast=True
                    ),
                    fmul_ilog2=holoso.FMulILog2Options(
                        stage_input=0, stage_core=holoso.FMulILog2Options.CoreStages.NONE, stage_output=0
                    ),
                    fcmp=holoso.FCmpOptions(stage_input=0, stage_output=0),
                    ffma=holoso.FFmaOptions(
                        stage_input=0,
                        stage_core_mul=1,
                        stage_core_add=holoso.FAddOptions.CoreStages.DECODE
                        | holoso.FAddOptions.CoreStages.OPERANDS
                        | holoso.FAddOptions.CoreStages.ALIGNED
                        | holoso.FAddOptions.CoreStages.NORMALIZE
                        | holoso.FAddOptions.CoreStages.PACK,
                        stage_output=0,
                    ),
                ),
                ffmt=float_format,
                wmultiplier=18,
            ),
        ),
    ):
        result = holoso.synthesize(poly3, options, name=label)
        print(f"{label}: II {result.initiation_interval}")
        for filename, path in result.write(out_dir).items():
            print(f"    {filename}: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cartesian <-> polar conversion as two 2-vector kernels.
Contrast with `cordic_sincos.py`, which computes sin/cos the hard way in software.
"""

import math
from pathlib import Path

import numpy as np
from jaxtyping import Float64

import holoso


def to_polar(v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
    x, y = v[0], v[1]
    return np.array([math.hypot(y, x), math.atan2(y, x)])


def from_polar(v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
    magnitude, angle = v[0], v[1]
    return np.array([magnitude * math.cos(angle), magnitude * math.sin(angle)])


def main() -> None:
    base = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    narrow = holoso.FloatFormat(wexp=6, wman=18)
    wide = holoso.FloatFormat(wexp=8, wman=36)
    # Each kernel takes its own configuration at each format: the stages that close its timing.
    configs = [
        (
            holoso.Options(
                operator=holoso.OperatorOptions(
                    fadd=holoso.FAddOptions(
                        stage_input=0, stage_core=holoso.FAddOptions.CoreStages.NONE, stage_output=0
                    ),
                    fmul=holoso.FMulOptions(stage_input=0, stage_core_mul=1, stage_output=1),
                    fdivsqrt=holoso.FDivsqrtOptions(
                        stage_input=0, stage_core=holoso.FDivsqrtOptions.CoreStages.NONE, stage_output=0, fast=True
                    ),
                    fmul_ilog2=holoso.FMulILog2Options(
                        stage_input=0, stage_core=holoso.FMulILog2Options.CoreStages.NONE, stage_output=0
                    ),
                    fcmp=holoso.FCmpOptions(stage_input=0, stage_output=0),
                    fcordic=holoso.FCordicOptions(
                        stage_input=0,
                        stage_core_mul=3,
                        stage_core=holoso.FCordicOptions.CoreStages.TAKEN
                        | holoso.FCordicOptions.CoreStages.ORDERED
                        | holoso.FCordicOptions.CoreStages.ANGLE
                        | holoso.FCordicOptions.CoreStages.NORMALIZE
                        | holoso.FCordicOptions.CoreStages.PACK,
                        stage_output=0,
                        fast=True,
                    ),
                ),
                ffmt=narrow,
                wmultiplier=18,
            ),
            holoso.Options(
                operator=holoso.OperatorOptions(
                    fadd=holoso.FAddOptions(
                        stage_input=0, stage_core=holoso.FAddOptions.CoreStages.NONE, stage_output=0
                    ),
                    fmul=holoso.FMulOptions(stage_input=0, stage_core_mul=1, stage_output=0),
                    fdivsqrt=holoso.FDivsqrtOptions(
                        stage_input=0, stage_core=holoso.FDivsqrtOptions.CoreStages.NONE, stage_output=0, fast=True
                    ),
                    fmul_ilog2=holoso.FMulILog2Options(
                        stage_input=0, stage_core=holoso.FMulILog2Options.CoreStages.NONE, stage_output=0
                    ),
                    fcmp=holoso.FCmpOptions(stage_input=0, stage_output=0),
                    fcordic=holoso.FCordicOptions(
                        stage_input=0,
                        stage_core_mul=3,
                        stage_core=holoso.FCordicOptions.CoreStages.REDUCED
                        | holoso.FCordicOptions.CoreStages.FOLDED
                        | holoso.FCordicOptions.CoreStages.NORMALIZE
                        | holoso.FCordicOptions.CoreStages.PACK,
                        stage_output=0,
                        fast=True,
                    ),
                ),
                ffmt=narrow,
                wmultiplier=18,
            ),
        ),
        (
            holoso.Options(
                operator=holoso.OperatorOptions(
                    fadd=holoso.FAddOptions(
                        stage_input=0, stage_core=holoso.FAddOptions.CoreStages.NONE, stage_output=0
                    ),
                    fmul=holoso.FMulOptions(stage_input=0, stage_core_mul=3, stage_output=1),
                    fdivsqrt=holoso.FDivsqrtOptions(
                        stage_input=0, stage_core=holoso.FDivsqrtOptions.CoreStages.NONE, stage_output=0, fast=True
                    ),
                    fmul_ilog2=holoso.FMulILog2Options(
                        stage_input=0, stage_core=holoso.FMulILog2Options.CoreStages.NONE, stage_output=0
                    ),
                    fcmp=holoso.FCmpOptions(stage_input=0, stage_output=0),
                    fcordic=holoso.FCordicOptions(
                        stage_input=0,
                        stage_core_mul=3,
                        stage_core=holoso.FCordicOptions.CoreStages.TAKEN
                        | holoso.FCordicOptions.CoreStages.ORDERED
                        | holoso.FCordicOptions.CoreStages.ANGLE
                        | holoso.FCordicOptions.CoreStages.NORMALIZE
                        | holoso.FCordicOptions.CoreStages.PACK,
                        stage_output=0,
                        fast=True,
                    ),
                ),
                ffmt=wide,
                wmultiplier=18,
            ),
            holoso.Options(
                operator=holoso.OperatorOptions(
                    fadd=holoso.FAddOptions(
                        stage_input=0, stage_core=holoso.FAddOptions.CoreStages.NONE, stage_output=0
                    ),
                    fmul=holoso.FMulOptions(stage_input=0, stage_core_mul=3, stage_output=1),
                    fdivsqrt=holoso.FDivsqrtOptions(
                        stage_input=0, stage_core=holoso.FDivsqrtOptions.CoreStages.NONE, stage_output=0, fast=True
                    ),
                    fmul_ilog2=holoso.FMulILog2Options(
                        stage_input=0, stage_core=holoso.FMulILog2Options.CoreStages.NONE, stage_output=0
                    ),
                    fcmp=holoso.FCmpOptions(stage_input=0, stage_output=0),
                    fcordic=holoso.FCordicOptions(
                        stage_input=0,
                        stage_core_mul=3,
                        stage_core=holoso.FCordicOptions.CoreStages.REDUCED
                        | holoso.FCordicOptions.CoreStages.FOLDED
                        | holoso.FCordicOptions.CoreStages.NORMALIZE_INPUT
                        | holoso.FCordicOptions.CoreStages.NORMALIZE
                        | holoso.FCordicOptions.CoreStages.PACK,
                        stage_output=0,
                        fast=True,
                    ),
                ),
                ffmt=wide,
                wmultiplier=18,
            ),
        ),
    ]
    for to_options, from_options in configs:
        label = f"e{to_options.ffmt.wexp}m{to_options.ffmt.wman}"
        models = {}
        for fn, options in ((to_polar, to_options), (from_polar, from_options)):
            result = holoso.synthesize(fn, options)
            models[fn.__name__] = result.numerical_model.elaborate()
            for filename, path in result.write(base / label / fn.__name__).items():
                print(f"{label}/{fn.__name__}/{filename}: {path}")
        for x, y in [(3.0, 4.0), (-1.0, 2.0), (0.5, -0.5)]:
            r, theta = (float(v) for v in models["to_polar"].run(x, y))
            xr, yr = (float(v) for v in models["from_polar"].run(r, theta))
            print(f"{label}: ({x:+.2f}, {y:+.2f}) -> (r={r:.4f}, theta={theta:+.4f}) -> ({xr:+.4f}, {yr:+.4f})")


if __name__ == "__main__":
    main()

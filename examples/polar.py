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
    # Each scalar width wants its own operator pipelining to close timings; the wide one also halves the CORDIC's
    # iterations per cycle. The CORDIC's product is wider than a hardware multiplier's 18-bit input at either width,
    # so it is split to that width.
    configs = [
        holoso.Options(
            holoso.OperatorOptions(
                fadd=holoso.FAddOptions(),
                fmul=holoso.FMulOptions(),
                fdivsqrt=holoso.FDivsqrtOptions(),
                fmul_ilog2=holoso.FMulILog2Options(),
                fcmp=holoso.FCmpOptions(),
                fcordic=holoso.FCordicOptions(stage_product=2, stage_normalize=2, stage_pack=1),
            ),
            ffmt=holoso.FloatFormat(wexp=6, wman=18),
            wmultiplier=18,
        ),
        holoso.Options(
            holoso.OperatorOptions(
                fadd=holoso.FAddOptions(),
                fmul=holoso.FMulOptions(stage_product=2, stage_pack=1),
                fdivsqrt=holoso.FDivsqrtOptions(),
                fmul_ilog2=holoso.FMulILog2Options(),
                fcmp=holoso.FCmpOptions(),
                fcordic=holoso.FCordicOptions(unroll100=50, stage_product=2, stage_normalize=2, stage_pack=2),
            ),
            ffmt=holoso.FloatFormat(wexp=8, wman=36),
            wmultiplier=18,
        ),
    ]
    for options in configs:
        label = f"e{options.ffmt.wexp}m{options.ffmt.wman}"
        models = {}
        for fn in (to_polar, from_polar):
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

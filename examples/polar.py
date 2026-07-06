#!/usr/bin/env python3
"""
Cartesian <-> polar conversion as two 2-vector kernels.
Contrast with ``cordic_sincos.py``, which computes sin/cos the hard way in software.
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
    for fmt in (holoso.FloatFormat(wexp=6, wman=18), holoso.FloatFormat(wexp=8, wman=36)):
        label = f"e{fmt.wexp}m{fmt.wman}"
        ops = holoso.OpConfig(
            holoso.FAddOperator(fmt),
            holoso.FMulOperator(fmt),
            holoso.FDivOperator(fmt),
            holoso.FMulILog2OperatorFamily(fmt),
            holoso.FCmpOperator(fmt),
            fsincos=holoso.FSincosOperator(fmt),
            fatan2=holoso.FAtan2Operator(fmt),
        )
        models = {}
        for fn in (to_polar, from_polar):
            result = holoso.synthesize(fn, ops=ops)
            models[fn.__name__] = result.numerical_model.elaborate()
            for filename, path in result.write(base / label / fn.__name__).items():
                print(f"{label}/{fn.__name__}/{filename}: {path}")
        for x, y in [(3.0, 4.0), (-1.0, 2.0), (0.5, -0.5)]:
            r, theta = (float(v) for v in models["to_polar"].run(x, y))
            xr, yr = (float(v) for v in models["from_polar"].run(r, theta))
            print(f"{label}: ({x:+.2f}, {y:+.2f}) -> (r={r:.4f}, theta={theta:+.4f}) -> ({xr:+.4f}, {yr:+.4f})")


if __name__ == "__main__":
    main()

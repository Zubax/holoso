#!/usr/bin/env python3
"""
Rotation-mode CORDIC computing cos(theta), sin(theta) with no multiplier on the rotation: each iteration is an
add/subtract and a power-of-two scale `2**-i` (an exact shift), with a per-iteration sign decision driving the
direction of micro-rotation. The fixed arctan table and the aggregate gain fold at compile time; the loop unrolls.

Note that this is illustrative; Holoso supports native trigonometry in hardware.
"""

import math
from pathlib import Path
import holoso


class CordicSinCos:
    def __init__(self, *, iterations: int = 12) -> None:
        self.iterations: int = iterations
        gain = 1.0
        for i in range(iterations):
            gain *= 1.0 / math.sqrt(1.0 + 2.0 ** (-2 * i))
        self.gain: float = gain  # the CORDIC scale factor, folded in as the seed for x
        self.angles: tuple[float, ...] = tuple(math.atan(2.0**-i) for i in range(iterations))

    def __call__(self, theta: float, /) -> tuple[float, float]:
        x, y, z = self.gain, 0.0, theta  # theta is the residual angle, driven toward zero
        for i in range(self.iterations):
            if z >= 0.0:
                x_next = x - y * (2.0**-i)
                y_next = y + x * (2.0**-i)
                z -= self.angles[i]
            else:
                x_next = x + y * (2.0**-i)
                y_next = y - x * (2.0**-i)
                z += self.angles[i]
            x, y = x_next, y_next
        return x, y


def main() -> None:
    float_format = holoso.FloatFormat(wexp=8, wman=36)
    options = holoso.Options(
        operator=holoso.OperatorOptions(
            fadd=holoso.FAddOptions(
                stage_input=0,
                stage_core=holoso.FAddOptions.CoreStages.DECODE
                | holoso.FAddOptions.CoreStages.ALIGN
                | holoso.FAddOptions.CoreStages.ALIGNED
                | holoso.FAddOptions.CoreStages.NORMALIZE
                | holoso.FAddOptions.CoreStages.PACK,
                stage_output=0,
            ),
            fmul=holoso.FMulOptions(stage_input=0, stage_core_mul=0, stage_output=0),
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
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(CordicSinCos(iterations=12).__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

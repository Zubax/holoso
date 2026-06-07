#!/usr/bin/env python3
"""
A PI controller with output saturation and integral anti-windup (conditional integration).

The controller computes ``u = kp*e + (integral + ki*e)`` and saturates it to +-LIMIT. The integrator only accumulates
when the output is within range (the ``else`` arm), so it cannot wind up while saturated -- the state is bounded by
construction. This is a data-dependent three-way branch (saturate high / saturate low / integrate) over persistent
float state, on the path toward the finite-set current controller.
"""

from pathlib import Path

import holoso


class SaturatingPI:
    def __init__(self, *, KP: float = 0.5, KI: float = 0.0625, LIMIT: float = 4.0) -> None:
        self.kp: float = KP
        self.ki: float = KI
        self.limit: float = LIMIT
        self.integral: float = 0.0  # persistent state; frozen while the output is saturated (anti-windup)

    def __call__(self, setpoint: float, measurement: float, /) -> float:
        error = setpoint - measurement
        candidate = self.integral + self.ki * error  # tentative integrator update
        u = self.kp * error + candidate
        if u > self.limit:
            u = self.limit
        elif u < -self.limit:
            u = -self.limit
        else:
            self.integral = candidate  # integrate only when not saturated
        return u


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
    result = holoso.synthesize(SaturatingPI().__call__, ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

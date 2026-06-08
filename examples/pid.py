#!/usr/bin/env python3
"""
A PID controller with output saturation, integral anti-windup (conditional integration), and a derivative channel.
The controller computes ``u = kp*e + (integral + ki*e) + kd*(e - e_prev)`` and saturates it to +-limit.
The very first update is treated specially to avoid the initial spike.
"""

from pathlib import Path

import holoso


class PID:
    def __init__(self, *, kp: float = 0.5, ki: float = 0.0625, kd: float = 0.25, limit: float = 4.0) -> None:
        self.kp: float = kp
        self.ki: float = ki
        self.kd: float = kd
        self.limit: float = limit
        self.integral: float = 0.0  # persistent state; frozen while the output is saturated (anti-windup)
        self.prev_error: float = 0.0  # persistent state; previous error for the derivative term
        self._started: bool = False  # boolean persistent state; False only on the first update (no derivative yet)

    def __call__(self, setpoint: float, measurement: float, /) -> float:
        error = setpoint - measurement
        candidate = self.integral + self.ki * error  # tentative integrator update
        if self._started:
            derivative = self.kd * (error - self.prev_error)  # kd * d(error)/dt
        else:
            derivative = 0.0  # first update: no previous error, so emit no derivative kick
        self.prev_error = error
        self._started = True
        u = self.kp * error + candidate + derivative
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
    result = holoso.synthesize(PID().__call__, ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

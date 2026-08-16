#!/usr/bin/env python3
"""
One explicit-Euler step of rigid-body rotational dynamics (Euler's equations): from the body-frame inertia tensor,
angular rate, and applied torque, produce the propagated rate and the angular momentum.

The inertia is a runtime input -- it changes with payload, fuel, or configuration --
so the kernel inverts it on every transaction with np.linalg.inv.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Float64

import holoso


def update(
    inertia: Float64[np.ndarray, "3 3"],  # body-frame inertia tensor (symmetric positive-definite)
    omega: Float64[np.ndarray, "3"],  # body angular rate
    tau: Float64[np.ndarray, "3"],  # applied torque
    dt: float,
) -> tuple[Float64[np.ndarray, "3"], Float64[np.ndarray, "3"]]:
    L = inertia @ omega  # angular momentum
    gyro = np.array(  # the gyroscopic torque omega × L
        [
            omega[1] * L[2] - omega[2] * L[1],
            omega[2] * L[0] - omega[0] * L[2],
            omega[0] * L[1] - omega[1] * L[0],
        ]
    )
    omega_dot = np.linalg.inv(inertia) @ (tau - gyro)
    return omega + omega_dot * dt, L


def main() -> None:
    inertia = np.array(
        [
            [2.0, 0.1, 0.0],
            [0.1, 3.0, -0.2],
            [0.0, -0.2, 4.0],
        ]
    )
    args = (inertia, np.array([0.5, -0.3, 0.8]), np.array([0.1, 0.0, -0.2]), 0.005)
    narrow = holoso.FloatFormat(wexp=6, wman=18)
    wide = holoso.FloatFormat(wexp=8, wman=36)
    configs = [
        holoso.Options(
            holoso.OperatorOptions(
                fadd=holoso.FAddOptions(),
                fmul=holoso.FMulOptions(),
                fdiv=holoso.FDivOptions(),
                fcmp=holoso.FCmpOptions(),
            ),
            ffmt=narrow,
        ),
        holoso.Options(
            holoso.OperatorOptions(
                fadd=holoso.FAddOptions(stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1),
                fmul=holoso.FMulOptions(stage_input=1, stage_product=1, stage_pack=1),
                fdiv=holoso.FDivOptions(stage_input=1, stage_pack=1, stage_output=1),
                fcmp=holoso.FCmpOptions(),
            ),
            ffmt=wide,
        ),
    ]
    base = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    flat_inputs = [float(x) for x in np.concatenate([np.asarray(a, dtype=np.float64).flatten() for a in args])]
    for options in configs:
        label = f"e{options.ffmt.wexp}m{options.ffmt.wman}"
        result = holoso.synthesize(update, options, name="rigid_body_rates")
        outputs = [float(v) for v in result.numerical_model.elaborate().run(*flat_inputs)]
        print(f"{label}: omega'/L = {outputs}")
        for filename, path in result.write(base / label).items():
            print(f"{label}/{filename}: {path}")


if __name__ == "__main__":
    main()

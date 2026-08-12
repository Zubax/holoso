#!/usr/bin/env python3
"""
Rigid-body frame transform for a strapdown IMU: place a body-frame point in the world frame and resolve a specific-force
measurement into world-frame linear acceleration, using only matrix products.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Float64

import holoso

GRAVITY = np.array([0.0, 0.0, 9.80665])  # world-frame gravity, subtracted from specific force to get linear accel
# Fixed sensor-to-body mounting rotation (here a 90-degree yaw); applies to every measurement.
MOUNT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def transform(
    R: Float64[np.ndarray, "3 3"],  # body-to-world rotation
    t: Float64[np.ndarray, "3"],  # world-frame position of the body origin
    a_meas: Float64[np.ndarray, "3"],  # accelerometer specific force, sensor frame
    p_sensor: Float64[np.ndarray, "3"],  # a point, sensor frame
) -> tuple[Float64[np.ndarray, "3"], Float64[np.ndarray, "3"], Float64[np.ndarray, "3"]]:
    R_ws = R @ MOUNT  # world-from-sensor rotation
    p_world = R_ws @ p_sensor + t
    a_world = R_ws @ a_meas - GRAVITY
    p_recovered = R_ws.T @ (p_world - t)  # world-to-sensor is the transpose, R_ws being orthonormal
    return p_world, a_world, p_recovered


def main() -> None:
    # A right angle about z, so the transform stays exact and easy to read in the emitted report.
    yaw90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    args = (yaw90, np.array([1.0, 2.0, 3.0]), np.array([0.1, -0.2, 9.9]), np.array([2.0, 0.0, -1.0]))
    narrow = holoso.FloatFormat(wexp=6, wman=18)
    wide = holoso.FloatFormat(wexp=8, wman=36)
    # Each format twice: the plain fmul+fadd expansion of the matrix products, and the ffma-contracted datapath where
    # every dot-product multiply-accumulate fuses into one rounding (a*b+c). The staged variants show representative
    # pipeline depths -- this script only elaborates and writes RTL/reports, so the knobs are illustrative rather than
    # timing-closed; the wide FMA multiplicand exceeds one DSP tile, hence its STAGE_PRODUCT split.
    configs = [
        holoso.Options(
            holoso.OperatorOptions(
                fadd=holoso.FAddOptions(),
                fmul=holoso.FMulOptions(),
                fdiv=holoso.FDivOptions(),
                fmul_ilog2=holoso.FMulILog2Options(),
                fcmp=holoso.FCmpOptions(),
            ),
            ffmt=narrow,
        ),
        holoso.Options(
            holoso.OperatorOptions(
                fadd=holoso.FAddOptions(stage_input=1, stage_decode=1, stage_pack=1),
                fmul=holoso.FMulOptions(stage_product=1),
                fdiv=holoso.FDivOptions(),
                fmul_ilog2=holoso.FMulILog2Options(),
                fcmp=holoso.FCmpOptions(),
                ffma=holoso.FFmaOptions(stage_product=1, stage_decode=1, stage_normalize=1, stage_pack=1),
            ),
            ffmt=narrow,
        ),
        holoso.Options(
            holoso.OperatorOptions(
                fadd=holoso.FAddOptions(stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1),
                fmul=holoso.FMulOptions(stage_input=1, stage_product=1, stage_pack=1),
                fdiv=holoso.FDivOptions(stage_input=1, stage_pack=1, stage_output=1),
                fmul_ilog2=holoso.FMulILog2Options(),
                fcmp=holoso.FCmpOptions(),
            ),
            ffmt=wide,
        ),
        holoso.Options(
            holoso.OperatorOptions(
                fadd=holoso.FAddOptions(stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1),
                fmul=holoso.FMulOptions(stage_input=1, stage_product=1, stage_pack=1),
                fdiv=holoso.FDivOptions(stage_input=1, stage_pack=1, stage_output=1),
                fmul_ilog2=holoso.FMulILog2Options(),
                fcmp=holoso.FCmpOptions(),
                ffma=holoso.FFmaOptions(
                    stage_input=1, stage_product=2, stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1
                ),
            ),
            ffmt=wide,
        ),
    ]
    base = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    flat_inputs = [float(x) for x in np.concatenate([a.flatten() for a in args])]
    for options in configs:
        label = f"e{options.ffmt.wexp}m{options.ffmt.wman}" + ("_fma" if options.operator.ffma is not None else "")
        result = holoso.synthesize(transform, options)
        world = [float(v) for v in result.numerical_model.elaborate().run(*flat_inputs)]
        print(f"{label}: p_world/a_world/p_recovered = {world}")
        for filename, path in result.write(base / label).items():
            print(f"{label}/{filename}: {path}")


if __name__ == "__main__":
    main()

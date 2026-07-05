#!/usr/bin/env python3
"""
Rigid-body frame transform for a strapdown IMU: place a body-frame point in the world frame and resolve a specific-force
measurement into world-frame linear acceleration, using only matrix products (no inversion, no trigonometry).
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
        holoso.OpConfig(
            holoso.FAddOperator(narrow),
            holoso.FMulOperator(narrow),
            holoso.FDivOperator(narrow),
            holoso.FMulILog2OperatorFamily(narrow),
            holoso.FCmpOperator(narrow),
        ),
        holoso.OpConfig(
            holoso.FAddOperator(narrow, stage_input=1, stage_decode=1, stage_pack=1),
            holoso.FMulOperator(narrow, stage_product=1),
            holoso.FDivOperator(narrow),
            holoso.FMulILog2OperatorFamily(narrow),
            holoso.FCmpOperator(narrow),
            ffma=holoso.FFmaOperator(narrow, stage_product=1, stage_decode=1, stage_normalize=1, stage_pack=1),
        ),
        holoso.OpConfig(
            holoso.FAddOperator(wide, stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1),
            holoso.FMulOperator(wide, stage_input=1, stage_product=1, stage_pack=1),
            holoso.FDivOperator(wide, stage_input=1, stage_pack=1, stage_output=1),
            holoso.FMulILog2OperatorFamily(wide),
            holoso.FCmpOperator(wide),
        ),
        holoso.OpConfig(
            holoso.FAddOperator(wide, stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1),
            holoso.FMulOperator(wide, stage_input=1, stage_product=1, stage_pack=1),
            holoso.FDivOperator(wide, stage_input=1, stage_pack=1, stage_output=1),
            holoso.FMulILog2OperatorFamily(wide),
            holoso.FCmpOperator(wide),
            ffma=holoso.FFmaOperator(
                wide, stage_input=1, stage_product=2, stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1
            ),
        ),
    ]
    base = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    flat_inputs = [float(x) for x in np.concatenate([a.flatten() for a in args])]
    for ops in configs:
        label = f"e{ops.float_format.wexp}m{ops.float_format.wman}" + ("_fma" if ops.ffma is not None else "")
        result = holoso.synthesize(transform, ops=ops)
        world = [float(v) for v in result.numerical_model.elaborate().run(*flat_inputs)]
        print(f"{label}: p_world/a_world/p_recovered = {world}")
        for filename, path in result.write(base / label).items():
            print(f"{label}/{filename}: {path}")


if __name__ == "__main__":
    main()

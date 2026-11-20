#!/usr/bin/env python3
"""
Kepler's equation solver: given mean anomaly M and eccentricity e, find the eccentric anomaly E satisfying
M = E - e*sin(E), by Newton-Raphson from E0 = M.
Convergence is quadratic on the well-conditioned domain e <= 0.9; loops until the Newton update falls below tolerance.
"""

import math
from pathlib import Path

import holoso


def eccentric_anomaly(mean_anomaly: float, eccentricity: float) -> float:
    E = mean_anomaly
    delta = 1.0  # seed above tolerance so the convergence test runs at least once
    while abs(delta) > 2.0**-12:
        delta = (E - eccentricity * math.sin(E) - mean_anomaly) / (1.0 - eccentricity * math.cos(E))
        E = E - delta
    return E


def main() -> None:
    fmt = holoso.FloatFormat(wexp=8, wman=36)
    options = holoso.Options(
        operator=holoso.OperatorOptions(
            fadd=holoso.FAddOptions(
                stage_input=0,
                stage_core=holoso.FAddOptions.CoreStages.OPERANDS
                | holoso.FAddOptions.CoreStages.ALIGNED
                | holoso.FAddOptions.CoreStages.SUM
                | holoso.FAddOptions.CoreStages.NORMALIZE
                | holoso.FAddOptions.CoreStages.NORMALIZED,
                stage_output=0,
            ),
            fmul=holoso.FMulOptions(stage_input=0, stage_core_mul=3, stage_output=1),
            fdivsqrt=holoso.FDivsqrtOptions(
                stage_input=0, stage_core=holoso.FDivsqrtOptions.CoreStages.NONE, stage_output=1, fast=True
            ),
            fmul_ilog2=holoso.FMulILog2Options(
                stage_input=0, stage_core=holoso.FMulILog2Options.CoreStages.NONE, stage_output=0
            ),
            fcmp=holoso.FCmpOptions(stage_input=0, stage_output=0),
            fcordic=holoso.FCordicOptions(
                stage_input=0,
                stage_core_mul=4,
                stage_core=holoso.FCordicOptions.CoreStages.REDUCED
                | holoso.FCordicOptions.CoreStages.FOLDED
                | holoso.FCordicOptions.CoreStages.CORRECTIONS
                | holoso.FCordicOptions.CoreStages.NORMALIZE
                | holoso.FCordicOptions.CoreStages.PACK,
                stage_output=0,
                fast=True,
            ),
        ),
        ffmt=fmt,
        wmultiplier=18,
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(eccentric_anomaly, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")
    model = result.numerical_model.elaborate()
    for m, e in [(0.8, 0.3), (2.5, 0.6), (-1.0, 0.9), (0.1, 0.0)]:
        anomaly = float(model.run(m, e)[0])
        print(f"M={m:+.2f} e={e:.2f} -> E={anomaly:+.6f}  (residual {anomaly - e * math.sin(anomaly) - m:+.2e})")


if __name__ == "__main__":
    main()

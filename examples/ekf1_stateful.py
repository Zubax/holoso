#!/usr/bin/env python3
"""
Stateful wrapper around the pure ``ekf1_stateless.update_x_P`` kernel.

This is the same extended-Kalman-filter covariance update as ``ekf1_stateless``, but packaged as a resettable filter
module that owns its state. The persisted ``x`` and ``P_urt`` are public, so they become observable
``state_x_*`` / ``state_P_urt_*`` ports; ``R_diag`` and ``Q_diag`` are read-only configuration folded into the design at
construction time. ``update`` is ordinary executable numpy -- holoso lowers the very source that runs natively.
"""

import dataclasses
from pathlib import Path

import numpy as np
from jaxtyping import Float64

import holoso
from ekf1_stateless import update_x_P  # inlined into the wrapper at synthesis time


@dataclasses.dataclass
class Ekf1:
    """
    An Extended Kalman Filter with state carried between updates.
    The initial state is set when the object is constructed and passed to synthesize() as the reset value.
    In this example, update() returns nothing -- the synthesizer will identify public state variables and make them
    RTL output ports.

    An aggregate field may be a Python list, whose length holoso reads from the reset value, or a numpy array whose
    shape is stated explicitly with jaxtyping; ``Q_diag`` exercises the latter and the rest the former.
    """

    x: list[float]  # state vector [x_R, x_g, x_i]
    P_urt: list[float]  # covariance upper-right triangle [P00, P01, P02, P11, P12, P22]
    R_diag: list[float]  # measurement-noise diagonal [R_ct, R_shunt]
    Q_diag: Float64[np.ndarray, "3"]  # process-noise diagonal [Q_R, Q_g, Q_i]

    def update(self, *, dt: float, u_shunt: float, di_dt: float) -> None:
        z = [u_shunt, di_dt]
        q = self.Q_diag * dt  # process noise integrated over the step; each diagonal entry scales with dt
        x_p = np.asarray(update_x_P(*self.P_urt, *q, *self.R_diag, dt, *self.x, *z)).flatten()  # type: ignore
        self.x = list(x_p[0:3])
        self.P_urt = list(x_p[3:9])


def main() -> None:
    # The values stored in the object at the time of synthesize() become its reset values.
    filt = Ekf1(
        x=[0.1e-3, 0.0, 0.0],
        P_urt=[1e3, 0.0, 0.0, 1e6, 0.0, 1e-3],
        R_diag=[1e3, 1e-6],
        Q_diag=np.array([1e-3, 1e9, 1e-9]),
    )
    # The kernel is float-format-agnostic, but each scalar width wants its own operator pipelining, so the OpConfig is
    # built per float format. The narrow 24-bit default (e6/m18) closes single-cycle, so its operators take no extra
    # stages; the wide 44-bit datapath (e8/m36) needs deeper operator pipelines to close timing.
    narrow = holoso.FloatFormat(wexp=6, wman=18)
    wide = holoso.FloatFormat(wexp=8, wman=36)
    configs = [
        holoso.OpConfig(
            holoso.FAddOperator(narrow),
            holoso.FMulOperator(narrow),
            holoso.FDivOperator(narrow),
            holoso.FMulILog2OperatorFamily(narrow),
        ),
        holoso.OpConfig(
            holoso.FAddOperator(wide, stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1),
            holoso.FMulOperator(wide, stage_input=1, stage_product=1, stage_pack=1),
            holoso.FDivOperator(wide, stage_input=1, stage_pack=1, stage_output=1),
            holoso.FMulILog2OperatorFamily(wide),
        ),
    ]
    base = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    for ops in configs:
        label = f"e{ops.float_format.wexp}m{ops.float_format.wman}"
        result = holoso.synthesize(filt.update, ops=ops)
        for filename, path in result.write(base / label).items():
            print(f"{label}/{filename}: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""A controller of a VSI inverter operating in current control mode."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast
import numpy as np
from jaxtyping import Float64
import holoso

type Vec2 = Float64[np.ndarray, "2"]
type Vec3 = Float64[np.ndarray, "3"]


@dataclass(frozen=True)
class Kinematics:
    pos: float
    vel: float
    accel: float


@dataclass(frozen=True)
class CurrentControllerDecision:
    switch_ac: tuple[bool, bool, bool]
    switch_balance: Vec3


def _dq0_to_ac(dq0: np.ndarray, theta: float) -> np.ndarray:
    dq0 = dq0.reshape((2, 1))
    d, q = dq0[0, 0], dq0[1, 0]
    ct, st = np.cos(theta), np.sin(theta)
    alpha = d * ct - q * st  # inverse Park
    beta = d * st + q * ct
    a = alpha  # inverse Clarke
    b = -0.5 * alpha + (np.sqrt(3.0) / 2.0) * beta
    c = -0.5 * alpha - (np.sqrt(3.0) / 2.0) * beta
    return np.array([[a], [b], [c]])


class FiniteSetCurrentController:
    # Constants that are folded at synthesis time (not registers)
    _BALANCE_WEIGHT = 4
    _CURRENT_DERIVATIVE_DAMPING_DT = 4e-6

    def __init__(self) -> None:
        self._n_phases = 3  # This one is read-only so constant-folded at synthesis time, not a state.
        # The following defines actual states; each vector is spilled into separate registers.
        self._switch_balance = np.zeros(self._n_phases, dtype=float)  # Shape deduced statically!
        (
            self._active_switch_candidates,
            self._active_switch_vectors,
            self._active_drive_threshold,
        ) = self._make_active_switch_candidates()  # Evaluated at synthesis time since everything is known statically

    def __call__(
        self, kin: Kinematics, i_ac: Vec3, di_ac_dt: Vec3, u_dc: float, i_dq_ref: Vec2, /
    ) -> CurrentControllerDecision:
        i_ac_ref = _dq0_to_ac(i_dq_ref, kin.pos)
        switch_ac = self._select_switch(i_ac_ref, i_ac, di_ac_dt, u_dc)
        self._switch_balance = self._zero_mean(self._switch_balance + self._balance_step(switch_ac))
        return CurrentControllerDecision(
            switch_ac=switch_ac,
            switch_balance=np.array(self._switch_balance),
        )

    def _select_switch(
        self, i_ac_ref: np.ndarray, i_ac: np.ndarray, di_ac_dt: np.ndarray, u_dc: float, /
    ) -> tuple[bool, bool, bool]:
        error = self._zero_mean(
            i_ac_ref.reshape(self._n_phases)  # "reshape()" is a no-op
            - (i_ac.reshape(self._n_phases) + self._CURRENT_DERIVATIVE_DAMPING_DT * di_ac_dt.reshape(self._n_phases))
        )
        # Expanding the finite-state score leaves only the strongest phase of this vector.
        phase_drive = (u_dc * error) - ((4.0 * self._BALANCE_WEIGHT) * self._switch_balance)
        active_drive = np.array([float(phase_drive @ vector) for vector in self._active_switch_vectors])
        best_drive = float(np.max(active_drive))
        if best_drive <= self._active_drive_threshold:
            return False, False, False
        tolerance = best_drive - 1e-9 * max(abs(best_drive), 1.0)
        chosen = self._active_switch_candidates[0]
        for k, candidate in enumerate(self._active_switch_candidates):
            if float(active_drive[k]) >= tolerance:
                chosen = candidate
                break
        return chosen

    def _make_active_switch_candidates(self) -> tuple[
        tuple[tuple[bool, bool, bool], ...],
        tuple[np.ndarray, ...],
        float,
    ]:
        candidates: list[tuple[bool, ...]] = []
        vectors: list[np.ndarray] = []
        for state in range(2**self._n_phases):
            switch_ac = tuple(bool((state >> phase) & 1) for phase in range(self._n_phases))
            vector = self._zero_mean(np.array(switch_ac, dtype=float))
            if not np.allclose(vector, 0.0, rtol=0.0, atol=0.0):
                candidates.append(switch_ac)
                vectors.append(vector)
        if not candidates:
            raise ValueError("No active switch candidates")  # Fails synthesis (because evaluated statically)
        norm_squares = np.array([float(vector @ vector) for vector in vectors])
        if not np.allclose(norm_squares, norm_squares[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"Active vectors must have equal norms: {norm_squares}")
        return (
            cast(tuple[tuple[bool, bool, bool], ...], tuple(candidates)),
            tuple(vectors),
            4.0 * self._BALANCE_WEIGHT * float(norm_squares[0]),
        )

    def _balance_step(self, switch_ac: tuple[bool, bool, bool], /) -> np.ndarray:
        return 2.0 * self._zero_mean(np.array(switch_ac, dtype=float))

    @staticmethod
    def _zero_mean(x: np.ndarray, /) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x - float(np.mean(x))


def main() -> None:
    float_format = holoso.FloatFormat(wexp=8, wman=36)
    options = holoso.Options(
        holoso.OperatorOptions(
            fadd=holoso.FAddOptions(),
            fmul=holoso.FMulOptions(),
            fdiv=holoso.FDivOptions(),
            fmul_ilog2=holoso.FMulILog2Options(),
            fcmp=holoso.FCmpOptions(),
            fsort=holoso.FSortOptions(),
            fsincos=holoso.FSincosOptions(),
        ),
        ffmt=float_format,
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(FiniteSetCurrentController().__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

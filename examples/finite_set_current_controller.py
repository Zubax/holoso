#!/usr/bin/env python3
"""
A complex example of a larger-scale control system for a VSI inverter operating in current control mode.
TODO FIXME: Currently unsupported.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Kinematics:
    pos: float
    vel: float
    accel: float


@dataclass(frozen=True)
class CurrentControllerDecision:
    switch_ac: tuple[bool, bool, bool]
    switch_balance: np.ndarray  # TODO: needs some kind of static size annotation, so this one won't work as-is


def _dq0_to_ac(dq0: np.ndarray, theta: float) -> np.ndarray:  # Inlined at place of invocation; shape annotation needed
    dq0 = dq0.reshape((2, 1))
    d, q = dq0[0, 0], dq0[1, 0]
    ct, st = np.cos(theta), np.sin(theta)  # Assume holoso_sincos is available.
    # Inverse Park: dq0 -> alpha-beta-zero
    alpha = d * ct - q * st
    beta = d * st + q * ct
    # Inverse Clarke: alpha-beta-zero -> abc
    a = alpha
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
        self,
        kin: Kinematics,
        i_ac: np.ndarray,  # these also need static shape annotations instead of np.ndarray
        di_ac_dt: np.ndarray,
        u_dc: float,
        i_dq_ref: np.ndarray,
        /,
    ) -> CurrentControllerDecision:
        i_ac_ref = _dq0_to_ac(i_dq_ref, kin.pos)
        switch_ac = self._select_switch(i_ac_ref, i_ac, di_ac_dt, u_dc)
        self._switch_balance = self._zero_mean(self._switch_balance + self._balance_step(switch_ac))
        return CurrentControllerDecision(
            switch_ac=switch_ac,
            switch_balance=self._switch_balance.reshape((self._n_phases, 1)),
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
        best_drive = float(np.max(active_drive))  # e.g., max() by sequential application of holoso_sort, or similar
        active = int(np.flatnonzero(active_drive >= best_drive - (1e-12 * max(abs(best_drive), 1.0)))[0])
        if best_drive <= self._active_drive_threshold:
            return False, False, False
        # We know the size statically so we can treat it as separate output registers:
        return self._active_switch_candidates[active]

    def _make_active_switch_candidates(self) -> tuple[  # This one doesn't make it to the final Verilog at all
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
        return tuple(candidates), tuple(vectors), 4.0 * self._BALANCE_WEIGHT * float(norm_squares[0])

    def _balance_step(self, switch_ac: tuple[bool, bool, bool], /) -> np.ndarray:
        return 2.0 * self._zero_mean(np.array(switch_ac, dtype=float))

    @staticmethod
    def _zero_mean(x: np.ndarray, /) -> np.ndarray:  # As always, we need static shape annotation here
        x = np.asarray(x, dtype=float)
        return x - float(np.mean(x))

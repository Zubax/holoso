#!/usr/bin/env python3
"""
Two-axis alpha-beta-frame PMSM flux observer -- a sensorless rotor angle estimator.
The `foc.py` example composes this observer into a complete sensorless field-oriented controller.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float64

import holoso

type Vec2 = Float64[np.ndarray, "2"]


@dataclass(frozen=True)
class MotorParams:
    R: float
    """Phase resistance [ohm]"""

    L_d: float
    """Direct-axis phase inductance [henry]"""

    flux_linkage: float
    """Rotor flux linkage magnitude [weber]"""


class FluxObserver:
    """
    The classical PMSM stationary-frame flux observer, derived from `u = R*i + dpsi/dt` where `psi` is the stator
    flux linkage; subtracting the stator contribution `L_d*i` leaves the rotor flux vector, whose argument is the
    electrical rotor angle:

        flux += (u - R*i) * dt - L_d * (i - i_last)

    Popularized in the drone ESC space by Shane Colton's 2014 sensorless FOC writeup
    (https://scolton-www.s3.amazonaws.com/motordrive/sensorless_gen1_Rev1.pdf),
    implemented in MESC (David Molony, aka mxlemming), and later brought into VESC (Benjamin Vedder) et al.

    The pure integrator drifts under offset errors, so each flux component is hard-clamped to the rotor flux
    linkage magnitude after the update, MESC-style.
    """

    def __init__(self) -> None:
        self.flux: Vec2 = np.zeros(2)  # Rotor flux linkage estimate [weber]
        self.i_last: Vec2 = np.zeros(2)  # Previous alpha-beta frame current [ampere]
        self._aligned = False

    @property
    def _theta_e(self) -> float:
        return float(np.atan2(self.flux[1], self.flux[0]))  # [-pi, +pi]

    def _integrate(self, params: MotorParams, dt: float, u_ab: Vec2, i_ab: Vec2) -> Vec2:
        integral: Vec2 = self.flux + (u_ab - params.R * i_ab) * dt - params.L_d * (i_ab - self.i_last)
        return integral

    def tick(self, params: MotorParams, dt: float, u_alpha_beta: Vec2, i_alpha_beta: Vec2) -> float:
        """
        Update the flux estimate with the alpha-beta frame voltage and current vectors sampled over the period dt.
        Returns the electrical rotor angle estimate in radians in [-pi, +pi].
        """
        if self._aligned:
            flux = self._integrate(params, dt, u_alpha_beta, i_alpha_beta)
        else:
            flux = np.array([params.flux_linkage, 0.0])
        self._aligned = True
        self.flux = np.clip(flux, -params.flux_linkage, params.flux_linkage)
        self.i_last = np.array(i_alpha_beta)
        return self._theta_e


def main() -> None:
    options = holoso.Options(
        holoso.OperatorOptions(
            fadd=holoso.FAddOptions(),
            fmul=holoso.FMulOptions(),
            fsort=holoso.FSortOptions(),
            fatan2=holoso.FAtan2Options(),
        ),
        ffmt=holoso.FloatFormat(wexp=8, wman=24),
    )
    # The demo integrates the exact back-EMF of an unloaded rotor spinning up from 50 Hz.
    params = MotorParams(R=0.05, L_d=2e-5, flux_linkage=0.005)
    observer = FluxObserver()
    result = holoso.synthesize(observer.tick, options)
    model = result.numerical_model.elaborate()
    dt = 1e-4
    model.run(params.R, params.L_d, params.flux_linkage, dt, 0.0, 0.0, 0.0, 0.0)
    theta = 0.0
    for k in range(8):
        f_Hz = 50.0 + 10.0 * k
        theta_next = theta + 2.0 * math.pi * f_Hz * dt
        psi_now = params.flux_linkage * np.array([math.cos(theta), math.sin(theta)])
        psi_next = params.flux_linkage * np.array([math.cos(theta_next), math.sin(theta_next)])
        u_ab = (psi_next - psi_now) / dt
        theta_est = float(
            model.run(params.R, params.L_d, params.flux_linkage, dt, float(u_ab[0]), float(u_ab[1]), 0.0, 0.0)[0]
        )
        print(f"theta true={theta_next:+.5f} estimated={theta_est:+.5f}")
        theta = theta_next
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

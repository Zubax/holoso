#!/usr/bin/env python3
"""Two-axis alpha-beta-frame PMSM flux observer -- a sensorless rotor-angle estimator."""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from jaxtyping import Float64

import holoso

type Vec2 = Float64[np.ndarray, "2"]


@dataclass
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

    R: float
    """Phase resistance [ohm]"""

    L_d: float
    """Direct-axis phase inductance [henry]"""

    flux_linkage: float
    """Rotor flux linkage magnitude [weber]; the clamp that arrests integrator drift"""

    flux: Vec2 = field(default_factory=lambda: np.zeros(2))
    """Rotor flux linkage estimate [weber]"""

    i_last: Vec2 = field(default_factory=lambda: np.zeros(2))
    """Previous alpha-beta frame current [ampere]"""

    @property
    def _theta_e(self) -> float:
        return float(np.atan2(self.flux[1], self.flux[0]))  # [-pi, +pi]

    def _integrate(self, dt: float, u_ab: Vec2, i_ab: Vec2) -> Vec2:
        return self.flux + (u_ab - self.R * i_ab) * dt - self.L_d * (i_ab - self.i_last)  # type: ignore[no-any-return]

    def tick(self, dt: float, u_ab: Vec2, i_ab: Vec2) -> float:
        """
        Update the flux estimate with the alpha-beta frame voltage and current vectors sampled over the period dt.
        Returns the electrical rotor angle estimate in radians in [-pi, +pi].
        """
        flux = self._integrate(dt, u_ab, i_ab)
        self.flux = np.minimum(np.maximum(flux, -self.flux_linkage), self.flux_linkage)
        self.i_last = np.array(i_ab)
        return self._theta_e


def main() -> None:
    narrow = holoso.FloatFormat(wexp=6, wman=18)
    wide = holoso.FloatFormat(wexp=8, wman=36)
    configs = [
        holoso.Options(
            holoso.OperatorOptions(
                fadd=holoso.FAddOptions(),
                fmul=holoso.FMulOptions(),
                fsort=holoso.FSortOptions(),
                fatan2=holoso.FAtan2Options(),
            ),
            ffmt=narrow,
        ),
        holoso.Options(
            holoso.OperatorOptions(
                fadd=holoso.FAddOptions(stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1),
                fmul=holoso.FMulOptions(stage_input=1, stage_product=1, stage_pack=1),
                fsort=holoso.FSortOptions(),
                fatan2=holoso.FAtan2Options(),
            ),
            ffmt=wide,
        ),
    ]
    # The reset flux is aligned with the true rotor flux, so the demo integrates the exact back-EMF of an unloaded
    # spinning rotor and the estimate tracks the true angle from the first transaction.
    flux_linkage_Wb = 0.005
    f_Hz = 50.0
    dt = 1e-4
    base = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    for options in configs:
        label = f"e{options.ffmt.wexp}m{options.ffmt.wman}"
        observer = FluxObserver(R=0.05, L_d=2e-5, flux_linkage=flux_linkage_Wb, flux=np.array([flux_linkage_Wb, 0.0]))
        result = holoso.synthesize(observer.tick, options)
        model = result.numerical_model.elaborate()
        for k in range(8):
            theta_now = 2.0 * math.pi * f_Hz * k * dt
            theta_true = 2.0 * math.pi * f_Hz * (k + 1) * dt
            psi_now = flux_linkage_Wb * np.array([math.cos(theta_now), math.sin(theta_now)])
            psi_next = flux_linkage_Wb * np.array([math.cos(theta_true), math.sin(theta_true)])
            u_ab = (psi_next - psi_now) / dt
            theta_est = float(model.run(dt, float(u_ab[0]), float(u_ab[1]), 0.0, 0.0)[0])
            print(f"{label}: theta true={theta_true:+.5f} estimated={theta_est:+.5f}")
        for filename, path in result.write(base / label).items():
            print(f"{label}/{filename}: {path}")


if __name__ == "__main__":
    main()

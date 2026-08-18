#!/usr/bin/env python3
"""
Sensorless field-oriented control of a PMSM: one call is one PWM cycle, from phase current samples to phase duty cycles.
The chain is the stationary-frame flux observer from `flux_observer.py`, a delay-compensated Park rotation (one sincos),
decoupled dq-axis PI current control with vector voltage limiting and conditional-integration anti-windup,
inverse Park, and midpoint-clamp space-vector modulation.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from jaxtyping import Float64

import holoso
from flux_observer import FluxObserver, MotorParams as ObserverParams

type Vec2 = Float64[np.ndarray, "2"]
type Vec3 = Float64[np.ndarray, "3"]

_ONE_OVER_SQRT3 = 1.0 / math.sqrt(3.0)
_SQRT3_OVER_2 = math.sqrt(3.0) / 2.0
_CLARKE = np.array(
    [
        [1.0, 0.0],
        [_ONE_OVER_SQRT3, 2.0 * _ONE_OVER_SQRT3],
    ]
)
_INVERSE_CLARKE = np.array(
    [
        [1.0, 0.0],
        [-0.5, _SQRT3_OVER_2],
        [-0.5, -_SQRT3_OVER_2],
    ]
)


@dataclass(frozen=True)
class MotorParams:
    R: float
    """Phase resistance [ohm]"""

    L_dq: Vec2
    """Direct- and quadrature-axis phase inductances [henry]; equal on a surface-magnet machine"""

    flux_linkage: float
    """Rotor flux linkage magnitude [weber]"""

    speed_filter_gain: float
    """Per-sample first-order filter gain of the angle-difference speed estimate [dimensionless]"""


class FocController:
    """
    The two-shunt convention: phase currents a and b are measured, c is implied by Kirchhoff.
    The amplitude-invariant Clarke/Park convention is used throughout.
    The voltage command is limited to the inscribed circle of the hexagon, v_dc/sqrt(3),
    so the duty cycles never leave [0, 1] and the commanded stator voltage always equals the applied one;
    the PI integrators freeze while the limiter is engaged.
    """

    def __init__(
        self,
        *,
        bandwidth: float = 2.0 * math.pi * 1500.0,  # Current loop bandwidth [radian/second]
        angle_advance: float = 1.5,  # Park angle advance in sample periods: 0.5 for ZOH avg + 1 for loop delay
    ) -> None:
        self.bandwidth = bandwidth
        self.angle_advance = angle_advance
        self.observer = FluxObserver()
        self.integral_dq: Vec2 = np.zeros(2)  # dq-axis PI integrator [volt]
        self.u_alpha_beta: Vec2 = np.zeros(2)  # Previous cycle stator voltage cmd, applied over the last period [volt]
        self.theta_prev: float = 0.0  # Observer angle on the previous cycle [radian]
        self.omega: float = 0.0  # Filtered electrical angular speed estimate [radian/second]

    def tick(
        self,
        params: MotorParams,
        dt: float,  # [second] PWM period elapsed since the previous call
        i_ab: Vec2,  # [ampere] phase-a and phase-b current samples
        i_dq_ref: Vec2,  # [ampere] direct- and quadrature-axis current setpoints
        v_dc: float,  # [volt] DC bus voltage sample
    ) -> tuple[Vec3, Vec2]:
        """
        Runs one PWM cycle and returns the three phase duty cycles in [0, 1] and the measured dq-frame currents.
        The estimated rotor angle and speed are observable through the persisted `theta_prev` and `omega` state.
        """
        i_alpha_beta = _CLARKE @ i_ab
        observer_params = ObserverParams(R=params.R, L_d=params.L_dq[0], flux_linkage=params.flux_linkage)
        theta = self.observer.tick(observer_params, dt, self.u_alpha_beta, i_alpha_beta)

        delta = theta - self.theta_prev
        if delta > math.pi:
            delta -= math.tau
        elif delta < -math.pi:
            delta += math.tau
        self.theta_prev = theta
        self.omega = self.omega + params.speed_filter_gain * (delta / dt - self.omega)

        advanced = theta + self.omega * (self.angle_advance * dt)
        s = math.sin(advanced)
        c = math.cos(advanced)
        i_dq = np.array([[c, s], [-s, c]]) @ i_alpha_beta

        e_dq = i_dq_ref - i_dq
        candidate_dq = self.integral_dq + (params.R * self.bandwidth) * e_dq * dt
        decouple = self.omega * np.array([-params.L_dq[1] * i_dq[1], params.L_dq[0] * i_dq[0] + params.flux_linkage])
        u_dq = (self.bandwidth * params.L_dq) * e_dq + candidate_dq + decouple

        v_max = v_dc * _ONE_OVER_SQRT3
        norm = float(np.linalg.norm(u_dq))
        if norm > v_max:
            u_dq = u_dq * (v_max / norm)
        else:
            self.integral_dq = np.array(candidate_dq)

        self.u_alpha_beta = np.array([[c, -s], [s, c]]) @ u_dq

        v_phase = _INVERSE_CLARKE @ self.u_alpha_beta
        v_common = 0.5 * (float(np.max(v_phase)) + float(np.min(v_phase)))
        duty = np.clip((v_phase - v_common) * (1.0 / v_dc) + 0.5, 0.0, 1.0)
        return duty, i_dq


CURRENT_NOISE_A = 0.03
"""Deterministic pseudo-noise amplitude on the measured phase currents [ampere], standing in for ADC noise."""


@dataclass
class PmsmSim:
    """
    A surface-magnet PMSM with a viscous-plus-Coulomb load, integrated by explicit Euler with substeps much shorter
    than the electrical time constant. Not synthesized: this is the plant the demo closes the loop around,
    and the physics truth the controller is judged against.
    """

    R: float
    L: float
    flux_linkage: float
    pole_pairs: int
    inertia: float
    load_coulomb: float
    load_viscous: float
    theta_e: float = 0.0
    omega_e: float = 0.0
    i_ab: Vec2 = field(default_factory=lambda: np.zeros(2))

    def step(self, u_ab: Vec2, dt: float, substeps: int = 20) -> None:
        h = dt / substeps
        for _ in range(substeps):
            bemf = self.flux_linkage * self.omega_e * np.array([-math.sin(self.theta_e), math.cos(self.theta_e)])
            di = (u_ab - self.R * self.i_ab - bemf) / self.L
            i_q = math.cos(self.theta_e) * self.i_ab[1] - math.sin(self.theta_e) * self.i_ab[0]
            torque = 1.5 * self.pole_pairs * self.flux_linkage * i_q
            omega_m = self.omega_e / self.pole_pairs
            load = self.load_coulomb * math.copysign(1.0, omega_m) + self.load_viscous * omega_m
            self.i_ab = self.i_ab + di * h
            self.omega_e = self.omega_e + self.pole_pairs * (torque - load) / self.inertia * h
            self.theta_e = math.remainder(self.theta_e + self.omega_e * h, math.tau)

    def phase_currents(self) -> tuple[float, float]:
        return float(self.i_ab[0]), float(-0.5 * self.i_ab[0] + _SQRT3_OVER_2 * self.i_ab[1])

    def measured_currents(self, k: int) -> tuple[float, float]:
        i_a, i_b = self.phase_currents()
        return i_a + CURRENT_NOISE_A * math.sin(1.7 * k + 0.3), i_b + CURRENT_NOISE_A * math.sin(2.3 * k + 1.1)


def scenario(k: int) -> tuple[float, float, float]:
    """
    The demo mission profile: settle at low torque, track a dithered dq setpoint below the voltage ceiling, then
    accelerate hard into the ceiling so the vector limiter engages and stays engaged. Returns (i_d_ref [ampere],
    i_q_ref [ampere], v_dc [volt]); the bus carries a slow ripple so the modulator divides by a moving voltage.
    """
    if k < 96:
        i_d_ref, i_q_ref = 0.0, 1.5
    elif k < 240:
        i_d_ref = -1.5
        i_q_ref = 2.0 + 0.9 * math.sin(2.0 * math.pi * (k - 96) / 48.0)
    else:
        i_d_ref, i_q_ref = 0.0, 8.0
    return i_d_ref, i_q_ref, 24.0 + 0.8 * math.sin(2.0 * math.pi * k / 97.0)


def main() -> None:
    dt = 2e-5
    controller = FocController()
    params = MotorParams(R=0.05, L_dq=np.array([2e-5, 2e-5]), flux_linkage=0.005, speed_filter_gain=0.05)
    plant = PmsmSim(
        R=0.05,
        L=2e-5,
        flux_linkage=0.005,
        pole_pairs=7,
        inertia=1e-6,
        load_coulomb=1e-3,
        load_viscous=3e-4,
        omega_e=2.0 * math.pi * 50.0,
    )
    for k in range(512):
        i_d_ref, i_q_ref, v_dc = scenario(k)
        i_a, i_b = plant.measured_currents(k)
        duty, i_dq = controller.tick(params, dt, np.array([i_a, i_b]), np.array([i_d_ref, i_q_ref]), v_dc)
        plant.step(np.array(controller.u_alpha_beta), dt)
        if k % 64 == 0:
            print(
                f"k={k:3d} f_e={plant.omega_e / math.tau:6.1f} Hz "
                f"i_dq=({i_dq[0]:+6.2f}, {i_dq[1]:+6.2f}) A duty=({duty[0]:.3f}, {duty[1]:.3f}, {duty[2]:.3f}) "
                f"angle err={math.remainder(controller.theta_prev - plant.theta_e, math.tau):+.4f} rad"
            )

    options = holoso.Options(
        holoso.OperatorOptions(
            fadd=holoso.FAddOptions(),
            fmul=holoso.FMulOptions(),
            fdiv=holoso.FDivOptions(),
            fsqrt=holoso.FSqrtOptions(),
            fsincos=holoso.FSincosOptions(),
            fatan2=holoso.FAtan2Options(),
            fcmp=holoso.FCmpOptions(),
            fsort=holoso.FSortOptions(),
            fmul_ilog2=holoso.FMulILog2Options(),
        ),
        ffmt=holoso.FloatFormat(wexp=8, wman=24),
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(FocController().tick, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

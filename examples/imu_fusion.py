#!/usr/bin/env python3
"""
Strapdown IMU sensor fusion: a Mahony-type complementary filter with lever-arm compensation for an IMU mounted away
from the center of mass. Each transaction takes one raw 6-axis sample with its calibration matrices plus the sensor
temperature and the time step, and produces the world-frame linear acceleration of the center of mass together with
a measurement-validity flag; the carried state is the attitude quaternion and the in-run gyro bias estimate.

The example is somewhat simplified compared to actual production systems to keep it readable.
Obviously, Holoso would have no problem expressing an actual production-grade estimator with bells and whistles,
but the margins here are too narrow to contain it.

Processing chain:

- raw rate/specific-force samples pass through factory calibration matrices (mounting rotation composed with
  scale-factor and misalignment correction), which arrive as runtime inputs so one core serves recalibration
  on the fly -- the lever arm stays a construction constant to show the opposite binding-time choice;

- the gyro bias model is the sum of a temperature polynomial (factory calibration) and the in-run estimate
  (fed by the tilt error, clamped to a plausibility bound);

- angular acceleration is recovered by backward difference so the accelerometer sample can be compensated for
  the lever arm: f_cm = f - w' x r - w x (w x r);

- tilt correction runs only when the compensated magnitude is close to one g (dynamic maneuvers and freefall
  are rejected), and a sticky flag latches if the gyro ever nears its full-scale range;

- the attitude coarse-aligns from the first accepted accelerometer sample (accelerometer leveling with zero
  initial yaw), as a production INS does before fine alignment takes over.
"""

from pathlib import Path

import numpy as np
from jaxtyping import Float64

import holoso

type Vec3 = Float64[np.ndarray, "3"]
type Vec4 = Float64[np.ndarray, "4"]
type Mat3x3 = Float64[np.ndarray, "3 3"]

GRAVITY = np.array([0.0, 0.0, 9.80665])  # [m/s^2] world frame is z-up: at rest the specific force is +1 g along +z
ACCEL_BAND = (8.8, 10.8)  # [m/s^2] tilt correction accepted only within this acceleration magnitude band


class ImuFusion:
    """
    The persisted public `attitude` (body-to-world unit quaternion [w, x, y, z]) and `bias` (in-run gyro bias)
    become observable ports, and the sticky `gyro_clip` reports a ranged-out gyro until reset.
    The attitude starts at identity and coarse-aligns from the first accepted accelerometer sample.
    """

    def __init__(
        self,
        *,
        lever_arm: Vec3 | None = None,  # [meter] IMU position relative to the center of mass
        kp: float = 2.0,  # [rad/s] per unit tilt error; proportional tilt gain
        ki: float = 0.1,  # [rad/s^2] per unit tilt error; integral tilt gain
        bias_limit: float = 0.05,  # [rad/s] bound on the in-run estimate, the temperature model carries the rest
        gyro_limit: float = 30.0,  # [rad/s] nearing full scale means the attitude can no longer be trusted
        temp_model: Mat3x3 | None = None,  # [rad/s] per [1, T, T^2] with T in degrees Celsius, one row per axis
    ) -> None:
        self.lever_arm = lever_arm if lever_arm is not None else np.array([0.05, -0.02, 0.03])
        self.kp = kp
        self.ki = ki
        self.bias_limit = bias_limit
        self.gyro_limit = gyro_limit
        self.temp_model = (
            temp_model
            if temp_model is not None
            else np.array(
                [
                    [1e-3, 2e-5, -3e-7],
                    [-2e-3, -1e-5, 2e-7],
                    [5e-4, 3e-5, 1e-7],
                ]
            )
        )
        self.attitude = np.array([1.0, 0.0, 0.0, 0.0])
        self.bias = np.zeros(3)  # [rad/s]
        self.gyro_clip = False
        self._w_prev = np.zeros(3)  # [rad/s]
        self._started = False
        self._aligned = False

    def update(
        self,
        # Sensor samples.
        gyro: Vec3,  # [rad/s] raw rate gyroscope sample
        accel: Vec3,  # [m/s^2] raw accelerometer specific force
        # Per-instance calibration data.
        gyro_cal: Mat3x3,  # sensor-to-body rotation composed with the gyro scale/misalignment correction
        accel_cal: Mat3x3,  # likewise for the accelerometer
        # Auxiliaries.
        temperature: float,  # [degree Celsius] sensor temperature
        dt: float,  # [second] time since the previous sample
    ) -> tuple[Vec3, bool]:
        """
        Updates the IMU state estimate based on the angular rate and linear acceleration samples.
        Returns the world-frame linear acceleration of the center of mass [m/s^2] and the sample validity.
        """
        # Latch the sticky clip flag while the raw rate is near full scale.
        self.gyro_clip = self.gyro_clip or np.linalg.norm(gyro, np.inf) > self.gyro_limit

        # Calibrate the rate and subtract the temperature-modeled plus the estimated bias.
        t_powers = np.array([1.0, temperature, temperature**2])
        w = gyro_cal @ gyro - (self.temp_model @ t_powers + self.bias)

        # Recover the angular acceleration by backward difference; the first sample has no history, so it is zero.
        started = self._started
        if started:
            w_dot = (w - self._w_prev) / dt
        else:
            w_dot = np.zeros(3)
        self._w_prev = w
        self._started = True

        # Calibrate the specific force and strip the lever-arm terms to relocate it to the center of mass.
        f = accel_cal @ accel
        f_cm = f - np.cross(w_dot, self.lever_arm) - np.cross(w, np.cross(w, self.lever_arm))

        # Accept the sample for tilt correction only when its magnitude is close to one g.
        f_norm = np.linalg.norm(f_cm)
        valid = ACCEL_BAND[0] < f_norm < ACCEL_BAND[1]
        if valid:
            a_n = f_cm * (1.0 / f_norm)
            if self._aligned:
                # The tilt error against the attitude's own gravity direction feeds the clamped bias integrator
                # and the proportional rate correction.
                qw, qx, qy, qz = self.attitude[0], self.attitude[1], self.attitude[2], self.attitude[3]
                gravity_body = np.array(
                    [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)]
                )
                e = np.cross(a_n, gravity_body)
                raw_bias = self.bias - self.ki * e * dt
                self.bias = np.clip(raw_bias, -self.bias_limit, self.bias_limit)
                w = w + self.kp * e
            elif a_n[2] > -0.99:
                # Coarse-align on the first accepted sample: the shortest arc taking the measured gravity direction
                # onto world up. Yaw is unobservable so it starts at zero. The arc degenerates near the antipode,
                # where the alignment defers to a later accepted sample rather than committing to a singular one.
                qa = np.array([1.0 + a_n[2], a_n[1], -a_n[0], 0.0])
                self.attitude = qa * (1.0 / np.linalg.norm(qa))
                self._aligned = True

        # Propagate the attitude by one explicit-Euler step of q' = Omega(w)/2 @ q and renormalize.
        rate = np.array(
            [
                [0.0, -w[0], -w[1], -w[2]],
                [w[0], 0.0, w[2], -w[1]],
                [w[1], -w[2], 0.0, w[0]],
                [w[2], w[1], -w[0], 0.0],
            ]
        )
        q = self.attitude + (rate @ self.attitude) * (0.5 * dt)
        self.attitude = q * (1.0 / np.linalg.norm(q))

        # Resolve the compensated acceleration into the world frame and remove gravity.
        qw, qx, qy, qz = self.attitude[0], self.attitude[1], self.attitude[2], self.attitude[3]
        rotation = np.array(
            [
                [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
                [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
                [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
            ]
        )
        return rotation @ f_cm - GRAVITY, valid


def main() -> None:
    fusion = ImuFusion()
    narrow = holoso.FloatFormat(wexp=6, wman=18)
    wide = holoso.FloatFormat(wexp=8, wman=36)
    configs = [
        holoso.Options(
            holoso.OperatorOptions(
                fadd=holoso.FAddOptions(),
                fmul=holoso.FMulOptions(),
                fdiv=holoso.FDivOptions(),
                fmul_ilog2=holoso.FMulILog2Options(),
                fcmp=holoso.FCmpOptions(),
                fexp2=holoso.FExp2Options(),
                flog2=holoso.FLog2Options(),
                fsort=holoso.FSortOptions(),
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
                fexp2=holoso.FExp2Options(stage_reduce=1, stage_product=2),
                flog2=holoso.FLog2Options(stage_product=2, stage_product_final=2, stage_normalize=1, stage_pack=1),
                fsort=holoso.FSortOptions(),
            ),
            ffmt=wide,
        ),
    ]
    base = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    # The wide config's stage knobs are illustrative rather than timing-closed (this script only elaborates and
    # writes RTL/reports); the measured e6m18 closure knobs live in the test suite's synthesis matrix.
    # A static vehicle pitched 10 degrees, with raw samples generated through the inverse of the sensor model the
    # filter assumes (the demonstration calibration below and the temperature bias at 25 degrees Celsius). From
    # the identity reset the estimated attitude must converge toward the true tilt.
    gyro_cal = np.array([[0.002, -1.002, 0.001], [0.998, 0.001, -0.002], [0.001, 0.002, 1.001]])
    accel_cal = np.array([[-0.001, -0.999, 0.002], [1.003, -0.002, 0.001], [-0.002, 0.001, 0.997]])
    pitch = np.deg2rad(10.0)
    temperature = 25.0
    f_body = np.array([-np.sin(pitch), 0.0, np.cos(pitch)]) * 9.80665
    f_sensor = np.linalg.inv(accel_cal) @ f_body
    g_sensor = np.linalg.inv(gyro_cal) @ (fusion.temp_model @ np.array([1.0, temperature, temperature**2]))
    q_true = [float(np.cos(pitch / 2.0)), 0.0, float(np.sin(pitch / 2.0)), 0.0]
    row = [float(v) for v in (*g_sensor, *f_sensor, *gyro_cal.flatten(), *accel_cal.flatten(), temperature, 0.01)]
    for options in configs:
        label = f"e{options.ffmt.wexp}m{options.ffmt.wman}"
        result = holoso.synthesize(fusion.update, options, name="imu_fusion")
        model = result.numerical_model.elaborate()
        for _ in range(400):
            outputs = dict(zip((port.name for port in model.outputs), model.run(*row)))
        q_est = [float(outputs[f"state_attitude_{k}"]) for k in range(4)]
        print(f"{label}: attitude after 400 static steps = {q_est}, true tilt = {q_true}")
        for filename, path in result.write(base / label).items():
            print(f"{label}/{filename}: {path}")


if __name__ == "__main__":
    main()

"""Extended Kalman filter covariance + state update for a current/resistance estimator.

A dense straight-line expression DAG (17 scalar inputs, a 9-element flattened matrix out, one reciprocal) produced by
offline symbolic differentiation -- a realistic stress test for scheduling and operator sharing.
"""


def update_x_P(P00, P01, P02, P11, P12, P22, Q_R, Q_g, Q_i, R_ct, R_shunt, dt, x_R, x_g, x_i, z_ct, z_shunt):
    """
    All inputs are floating-point scalars -> Verilog ``input wire [WFULL-1:0]`` ports, where WFULL = WEXP + WMAN.
    The return is a sequence (here a flattened matrix in row-major order); each element becomes a separate output port.
    """
    x0 = R_ct * R_shunt
    x1 = x_R**2
    x2 = P01 + P11 * dt
    x3 = P00 + P01 * dt + Q_i + dt * x2
    x4 = x1 * x3
    x5 = P22 + Q_R
    x6 = dt * x_g + x_i
    x7 = x6**2
    x8 = x5 * x7
    x9 = P02 + P12 * dt
    x10 = x6 * x_R
    x11 = 2 * x10
    x12 = x11 * x9
    x13 = P11 + Q_g
    x14 = R_shunt * x13
    x15 = P12**2
    x16 = x15 * x7
    x17 = x2**2
    x18 = x1 * x17
    x19 = x11 * x2
    x20 = -P12 * x19 + x12 * x13 + x13 * x4 + x13 * x8 + x14 - x16 - x18
    x21 = R_ct * x12 + R_ct * x4 + R_ct * x8 + x0 + x20
    x22 = 1 / x21
    x23 = x10 - z_shunt
    x24 = P12 * x6 + x2 * x_R
    x25 = R_ct + x13
    x26 = x3 * x_R + x6 * x9
    x27 = x_g - z_ct
    x28 = R_shunt + x12 + x4 + x8
    x29 = x5 * x6 + x9 * x_R
    x30 = x9**2
    x31 = x30 * x7
    x32 = R_shunt * x2
    x33 = P12 * x9
    x34 = x2 * x9
    x35 = x10 * x3
    x36 = R_ct * x22
    x37 = x10 * x30
    x38 = x10 * x5
    x39 = x35 * x5
    x40 = x1 * x30
    return [
        [x22 * (x21 * x6 + x23 * (x2 * x24 - x25 * x26) - x27 * (x2 * x28 - x24 * x26))],
        [x22 * (-R_ct * x23 * x24 + x21 * x_g - x27 * (x13 * x28 - x24**2))],
        [x22 * (x21 * x_R + x23 * (P12 * x24 - x25 * x29) - x27 * (P12 * x28 - x24 * x29))],
        [
            x22
            * (
                2 * P12 * x2 * x7 * x9
                + R_ct * R_shunt * x3
                + R_ct * x3 * x5 * x7
                - R_ct * x31
                + R_shunt * x13 * x3
                - R_shunt * x17
                + x13 * x3 * x5 * x7
                - x13 * x31
                - x16 * x3
                - x17 * x8
            )
        ],
        [x36 * (-P12 * x35 + x10 * x34 + x2 * x8 + x32 - x33 * x7)],
        [
            x22
            * (
                -P12 * x32
                + R_ct * x37
                - R_ct * x39
                + x0 * x9
                + x13 * x37
                - x13 * x39
                + x14 * x9
                + x15 * x35
                + x17 * x38
                - x19 * x33
            )
        ],
        [x20 * x36],
        [x36 * (P12 * R_shunt + P12 * x4 - x1 * x34 + x10 * x33 - x2 * x38)],
        [
            x22
            * (
                2 * P12 * x1 * x2 * x9
                + R_ct * R_shunt * x5
                + R_ct * x1 * x3 * x5
                - R_ct * x40
                + R_shunt * x13 * x5
                - R_shunt * x15
                + x1 * x13 * x3 * x5
                - x13 * x40
                - x15 * x4
                - x18 * x5
            )
        ],
    ]

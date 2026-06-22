#!/usr/bin/env python3
"""
A sample-by-sample integrator with an optional DC leak term.
"""

from pathlib import Path
import holoso


class TrapezoidalLeakyStreamingIntegrator:
    """
    Sample-by-sample trapezoidal (Tustin, AM2) integrator with an optional DC leak term.
    The leak term, unless zero, avoids long-term integrator bias in the presence of transient input biasing.
    Notation used below:

        x[n]  - input derivative sample at discrete time n.
        y[n]  - integrator output at discrete time n.
        T     - sample period, i.e. ``dt``.
        k     - leak coefficient.
        r     - leak pole, ``r = 1 - k``.
        z^-1  - one-sample delay.
        omega - normalized angular frequency, ``omega = 2*pi*f/f_s``.

    The returned output obeys the recurrence:

        y[n] = r*y[n-1] + T*a[n]

    where ``a[n]`` is the addend, and the leak function forms the standard IIR filter with a single pole at r.

        a[n] = (x[n] + x[n-1]) / 2
        H(z) = Y(z)/X(z) = T*(1 + z^-1) / (2*(1 - r*z^-1))

    The exact steady-state amplitude gain is:

        A_out / D = T*cos(omega/2) / sqrt(1 - 2*r*cos(omega) + r^2),  for 0 <= omega <= pi.

    With the leak disabled, this reduces to:

        A_out / D = T*cos(omega/2) / (2*sin(omega/2)),  for 0 < omega < pi.

    The DC gain is T/k when the leak is enabled. The numerator has a zero at Nyquist, so the trapezoidal
    response attenuates exactly to zero when omega=pi.
    """

    def __init__(self, *, k: float = 2**-22) -> None:
        self.k: float = k
        self.y: float = 0.0
        self._x_prev: float = 0.0

    def __call__(self, x: float, dt: float, /) -> float:
        self.y = (1.0 - self.k) * self.y + (x + self._x_prev) * 0.5 * dt
        self._x_prev = x
        return self.y  # Duplicates the public member `y`, so this output port will be elided.


def main() -> None:
    float_format = holoso.FloatFormat(wexp=8, wman=36)
    ops = holoso.OpConfig(
        holoso.FAddOperator(float_format),
        holoso.FMulOperator(float_format),
        holoso.FDivOperator(float_format),
        holoso.FMulILog2OperatorFamily(float_format),
        holoso.FCmpOperator(float_format),
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem

    # Construct the instance with the desired constructor arguments, then pass the bound method to synthesize: its
    # __self__ snapshot seeds the reset state, and __func__ is the analyzed method.
    integrator = TrapezoidalLeakyStreamingIntegrator(k=2**-22)
    result = holoso.synthesize(integrator.__call__, ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

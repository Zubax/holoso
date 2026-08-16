#!/usr/bin/env python3
"""
A quadrature (I/Q) oscillator: a direct-digital-synthesis (DDS) sinewave generator whose phase lives in an integer
accumulator, as in `nco.py`, while its outputs are the float pair I = cos(phase), Q = sin(phase).
"""

import math
from pathlib import Path
from nco import Nco
import holoso


class IqOscillator:
    """
    Holding the phase as a fixed-point turn fraction keeps it exact; a float sheds the low bits as the angle grows
    and drifts. The finest frequency step is one unit per tick, 1/(dt*2**wphase): 233 nHz at a 1 ms tick.

    The conversion of frequency*dt into accumulator units saturates at the integer rail rather than at the phase
    modulus, so the aliasing is faithful only while |frequency*dt| stays inside the native word.
    """

    def __init__(self, *, wphase: int = 32) -> None:
        self._lsb_per_turn = float(1 << wphase)
        self._radians_per_lsb = math.tau / self._lsb_per_turn
        self.nco = Nco(wphase=wphase, wout=wphase)

    def tick(self, frequency: float, dt: float, phase_offset: float, /) -> tuple[float, float]:
        assert frequency * dt < self._lsb_per_turn
        # A signed increment is what a sampled oscillator wants: negative runs backwards, past-Nyquist folds.
        increment = round(frequency * dt * self._lsb_per_turn)
        # In turns, and added on the way out: the phase control word of a classic NCO.
        offset = round(phase_offset * self._lsb_per_turn)
        angle = float(self.nco.tick(increment, offset)) * self._radians_per_lsb
        return math.cos(angle), math.sin(angle)  # one sincos firing computes both, so Q rides along free


def main() -> None:
    options = holoso.Options(
        holoso.OperatorOptions(
            fmul=holoso.FMulOptions(),
            fmul_ilog2=holoso.FMulILog2Options(),  # scaling a turn fraction by a power of two is an exponent add
            fsincos=holoso.FSincosOptions(),
            ffromint=holoso.FFromIntOptions(),
            ftoint=holoso.FToIntOptions(),
        ),
        ffmt=holoso.FloatFormat(wexp=8, wman=36),
        wint_min=34,  # a carry bit above the phase, then a sign bit
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(IqOscillator().tick, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

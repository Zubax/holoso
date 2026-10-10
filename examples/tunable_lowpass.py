#!/usr/bin/env python3
"""
A first-order low-pass filter (an exponential moving average) in fixed point whose time constant and output gain are
powers of two chosen at run time, which makes it tunable without a multiplier: both scalings are shifts by a count
that arrives with the sample. It is the integer counterpart of `iir1_lpf.py` with alpha = 2**-k.
"""

from pathlib import Path

import holoso


class TunableLowpass:
    def __init__(self) -> None:
        self.acc: int = 0

    def __call__(self, x: int, k: int, g: int, /) -> int:
        self.acc += (x - self.acc) >> k  # right shift floors, which leaves a deadband
        return self.acc << g


def main() -> None:
    # The scaled output needs 30 bits and the difference 25, so a 32-bit word holds every value of the domain.
    options = holoso.Options(holoso.OperatorOptions(), wint_min=32)
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(TunableLowpass().__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

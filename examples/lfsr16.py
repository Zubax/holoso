#!/usr/bin/env python3
"""
A 16-bit Galois linear-feedback shift register: the cheap pseudorandom bit sequence (PRBS) source behind bit-error-rate
patterns, scramblers, dither, and memory BIST sweeps.
"""

from pathlib import Path

import holoso


class Lfsr16:
    def __init__(self, *, seed: int = 0xACE1, taps: int = 0xB400) -> None:
        """
        Default taps: x**16 + x**14 + x**13 + x**11 + 1, primitive over GF(2),
        hence the maximal-length 65535-state cycle.
        """
        assert 0 < seed <= 0xFFFF, "zero is the lock-up state, mapping to itself forever"
        self.register = seed
        self.taps = taps

    def __call__(self, advance: bool, /) -> bool:
        if advance:  # a clock enable: the generator pauses without losing phase
            if (self.register & 1) == 1:
                self.register = (self.register >> 1) ^ self.taps
            else:
                self.register = self.register >> 1
        return (self.register & 1) == 1  # the bit the next advance shifts out, so it holds steady while gated


def main() -> None:
    # The taps and the register are 16 unsigned bits, which a signed word must carry one bit above.
    options = holoso.Options(holoso.OperatorOptions(), wint_min=17)
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(Lfsr16().__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

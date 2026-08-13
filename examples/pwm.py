#!/usr/bin/env python3
"""An ordinary center-aligned pulse width modulator."""

from pathlib import Path

import holoso


class Pwm:
    """
    The PWM counter walks up to top and back down, so the output pulse is symmetric about the counter's extremum rather
    than pinned to a period edge -- which is why multiphase drives prefer it, the phase legs switching around a common
    centre instead of together.

    The triangle costs half the carrier frequency of a sawtooth: a period is 2*top ticks, a tick being one accepted
    transaction rather than one clock edge. Duty d carries 2*d-1 of them high, with 0 always off and anything above
    top always on, so the steps are the odd counts and the two rails.
    """

    def __init__(self, top: int = 100) -> None:
        self._top = top
        self.counter: int = 0
        self._down: bool = False

    def tick(self, duty: int, /) -> bool:
        out = self.counter < duty  # sampled before the update, so the first tick after reset is already correct
        # Count-and-reverse rather than a modulo: the direction flag turns the comparator into the whole wrap logic,
        # and each one-armed if rewrites a value written this transaction, so both if-convert into selects.
        if self._down:
            self.counter = self.counter - 1
            if self.counter <= 0:
                self._down = False
        else:
            self.counter = self.counter + 1
            if self.counter >= self._top:
                self._down = True
        return out


def main() -> None:
    # The counter walks up to top and the duty compares against it, so a word that holds top, plus the sign bit.
    options = holoso.Options(holoso.OperatorOptions(), wint_min=8)
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(Pwm().tick, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

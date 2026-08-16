#!/usr/bin/env python3
"""A numerically controlled oscillator (NCO): the core of direct digital synthesis (DDS)."""

from pathlib import Path

import holoso


class Nco:
    """
    The output frequency is f_tick*increment/2**wphase and the tuning step is f_tick/2**wphase,
    where a tick is one accepted transaction rather than one clock edge.
    Half scale is Nyquist; above it an increment aliases to its complement running backwards.

    The output is the top `wout` bits of the phase: a sawtooth, or at the default width the square wave that is the
    coarsest phase-to-amplitude conversion there is. See `iq_oscillator.py` for the sine pair.
    """

    def __init__(self, *, wphase: int = 32, wout: int = 1) -> None:
        assert 0 < wout <= wphase
        self._mask = (1 << wphase) - 1
        self._shift = wphase - wout
        self.phase: int = 0

    def tick(self, increment: int, phase_offset: int, /) -> int:
        # Masking each input keeps every sum inside the word, which would otherwise saturate before its own mask.
        increment &= self._mask
        phase_offset &= self._mask
        self.phase = (self.phase + increment) & self._mask  # the mask makes the accumulator modular
        return ((self.phase + phase_offset) & self._mask) >> self._shift  # the offset shifts the output, not the phase


def main() -> None:
    # A masked sum exists before its mask: a carry bit above the phase, then a sign bit.
    options = holoso.Options(holoso.OperatorOptions(), wint_min=34)
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(Nco().tick, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

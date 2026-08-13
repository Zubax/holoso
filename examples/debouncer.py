#!/usr/bin/env python3
"""
A counting switch debouncer -- the standard filter for a mechanical contact, whose surfaces ring for a few
milliseconds after every open and close and hand the sampler a burst of spurious edges. The cost is latency:
at a sample period Ts any disturbance shorter than `samples` consecutive disagreeing samples is erased,
and a genuine edge is delayed by (samples-1)*Ts.

This is hysteresis in time, where `schmitt_trigger.py` is hysteresis in amplitude: the Schmitt trigger separates its
two switching thresholds so that noise narrower than the deadband cannot cross back, while here both directions share
one threshold and are separated by the dwell requirement instead.
"""

from pathlib import Path

import holoso


class Debouncer:
    def __init__(self, *, samples: int = 4, initial: bool = False) -> None:
        self._samples: int = samples
        self._count: int = 0  # consecutive samples disagreeing with the reported level
        self.level: bool = initial

    def __call__(self, raw: bool, /) -> bool:
        # Both if diamonds if-convert here, resulting in a branchless kernel with speculative execution & selection.
        if raw == self.level:
            self._count = 0
        else:
            self._count += 1
            if self._count >= self._samples:
                self.level = raw
                self._count = 0
        return self.level  # Duplicates the public member `level`, so this output port will be elided.


def main() -> None:
    # The dwell count reaches samples, plus the sign bit; wint_min=4 --> max count 15.
    options = holoso.Options(holoso.OperatorOptions(), wint_min=4)
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(Debouncer().__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

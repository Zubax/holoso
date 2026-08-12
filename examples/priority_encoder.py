#!/usr/bin/env python3
"""
A fixed-priority encoder: the front end of an arbiter, turning a bus of simultaneous requests into the index of the
one to serve, or -1 when idle. The lowest set bit wins, as interrupt vectors, DMA channels and bus masters are all
numbered in descending urgency. Bits at or above `width` are not on the bus and are ignored.
"""

from pathlib import Path

import holoso


class PriorityEncoder:
    def __init__(self, *, width: int = 8) -> None:
        self._width: int = width

    def __call__(self, request: int, /) -> int:
        index = -1
        found = False  # the kill line masking every requester below the winner
        # A `break` would unroll the same but leave a data-dependent exit in every copy, swinging the latency with
        # the request pattern; the flag keeps every arm if-convertible.
        for i in range(self._width):
            if not found and (request >> i) & 1 == 1:
                index = i
                found = True
        return index


def main() -> None:
    options = holoso.Options(holoso.OperatorOptions())  # bit tests and an index: no float operator is built
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(PriorityEncoder().__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

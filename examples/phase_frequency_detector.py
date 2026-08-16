#!/usr/bin/env python3
"""A sampled phase/frequency detector."""

from pathlib import Path

import holoso


class PhaseFrequencyDetector:
    """
    A reference edge that arrives before feedback asserts `up`; a feedback edge that arrives first asserts `down`.
    Once both sides have been seen, both pending latches and outputs clear, matching the reset action of a classic PFD.
    """

    def __init__(self) -> None:
        self._ref_pending: bool = False
        self._fb_pending: bool = False
        self.up: bool = False
        self.down: bool = False

    def __call__(self, ref_edge: bool, fb_edge: bool, clear: bool, /) -> tuple[bool, bool]:
        if clear:  # Once we added support for multiple methods, this 'clear' branch would be a separate method.
            self._ref_pending = False
            self._fb_pending = False
            self.up = False
            self.down = False
        else:
            if ref_edge:
                self._ref_pending = True
            if fb_edge:
                self._fb_pending = True
            if self._ref_pending and self._fb_pending:
                self._ref_pending = False
                self._fb_pending = False
                self.up = False
                self.down = False
            else:
                self.up = self._ref_pending
                self.down = self._fb_pending
        return self.up, self.down


def main() -> None:
    options = holoso.Options(holoso.OperatorOptions())  # purely boolean: no float operator, no float format
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(PhaseFrequencyDetector().__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
A Schmitt trigger: a comparator with hysteresis. The output latches high once the input rises above ``HIGH`` and low
once it falls below ``LOW``; between the two thresholds it holds its previous value (the hysteresis deadband).
The two threshold tests are data-dependent branches that leave the state untouched in the deadband.
TODO FIXME: boolean ports are not yet supported.
"""

from pathlib import Path

import holoso


class SchmittTrigger:
    def __init__(self, *, high: float = 1.0, low: float = -1.0, initial: bool = False) -> None:
        self.high: float = high
        self.low: float = low
        self.y: bool = initial  # persistent output state held in the deadband

    def __call__(self, x: float, /) -> bool:
        if x > self.high:
            self.y = True
        elif x < self.low:
            self.y = False
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
    result = holoso.synthesize(SchmittTrigger().__call__, ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
A stateless signal-window conditioner exercising various boolean/float expression forms:

- clamped: x clamped into [lo, hi] with a nested conditional (ternary) expression;

- inside: whether x lies strictly inside the open window, from a chained comparison fed through a bool->float cast;

- outside: the boundary/outside region, from an or-connective in a conditional expression;

- live: a "live" sample -- nonzero and inside -- from a float->bool cast and an and-connective, cast back to float;

- gated: the input passed through only inside the window -- a cross-domain chain
  (comparison -> bool -> float cast -> float multiply) that the bool->float result must feed on time.

The conditional expressions lower to branch + phi, so the kernel is a small CFG; the comparisons, connectives, and
casts are combinational ops within it. No branch arm drives an output directly -- every output is a phi merge or a
combinational cast result.
"""

from pathlib import Path

import holoso


def signal_window(x: float, lo: float, hi: float) -> tuple[float, float, float, float, float]:
    clamped = hi if x > hi else (lo if x < lo else x)
    inside = float(lo < x < hi)
    outside = 1.0 if (x <= lo or x >= hi) else 0.0
    live = float(bool(x) and lo < x < hi)
    gated = float(lo < x < hi) * x
    return clamped, inside, outside, live, gated


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
    result = holoso.synthesize(signal_window, ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

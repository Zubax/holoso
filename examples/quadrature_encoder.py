#!/usr/bin/env python3
"""
A quadrature encoder transition filter. The input pair ``(a, b)`` is expected to move through a Gray-code sequence;
single-bit transitions emit a one-cycle ``step`` pulse with ``forward`` indicating the direction, while simultaneous
changes are flagged as invalid. The previous sampled input pair is persistent state.
"""

from pathlib import Path
import holoso


class QuadratureEncoder:
    def __init__(self, *, initial_a: bool = False, initial_b: bool = False) -> None:
        self._a: bool = initial_a
        self._b: bool = initial_b

    def __call__(self, a: bool, b: bool, /) -> tuple[bool, bool, bool]:
        changed_a = a != self._a  # an input edge: the sample differs from the previous one (lowers to an exclusive-or)
        changed_b = b != self._b
        step = False
        forward = True
        fault = False
        if changed_a and changed_b:
            fault = True
            self._a = a
            self._b = b
        elif changed_a or changed_b:
            step = True
            forward = a == self._b  # direction: forward iff the changed line now matches the other channel (xnor)
            self._a = a
            self._b = b
        return step, forward, fault


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
    result = holoso.synthesize(QuadratureEncoder().__call__, ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

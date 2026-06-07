#!/usr/bin/env python3

from pathlib import Path
import holoso


class TrapezoidalLeakyStreamingIntegrator:
    def __init__(self, *, k: float = 2**-22) -> None:
        self.k: float = k
        self.y: float = 0.0
        self._x_prev: float = 0.0

    def __call__(self, x: float, /) -> float:
        self.y = (1.0 - self.k) * self.y + 0.5 * (x + self._x_prev)
        self._x_prev = x
        return self.y


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

    # Construct the instance with the desired constructor arguments, then pass the bound method to synthesize: its
    # __self__ snapshot seeds the reset state, and __func__ is the analyzed method.
    integrator = TrapezoidalLeakyStreamingIntegrator(k=2**-22)
    result = holoso.synthesize(integrator.__call__, ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

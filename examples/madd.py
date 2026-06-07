#!/usr/bin/env python3

from pathlib import Path
import holoso


def madd(a, b, c):
    """
    Tiny multiply-add kernel.
    The const mults strength-reduce to fmul_ilog2 (K = -2); ``a`` and ``b`` are each read twice (small reuse),
    the products are single-use. ``c`` is an unused argument, kept as given to exercise dead-input handling.
    """
    return (a - b) * 0.25 + a * b * 8


def main() -> None:
    float_format = holoso.FloatFormat(wexp=6, wman=18)
    ops = holoso.OpConfig(
        holoso.FAddOperator(float_format),
        holoso.FMulOperator(float_format),
        holoso.FDivOperator(float_format),
        holoso.FMulILog2OperatorFamily(float_format),
        holoso.FCmpOperator(float_format),
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(madd, ops=ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from pathlib import Path
import holoso


def poly3(x, c0, c1, c2, c3):
    """
    Degree-3 polynomial evaluated in Horner form: ((c3 * x + c2) * x + c1) * x + c0.
    A pure multiply-add dependency chain (alternating fmul and fadd) in which every intermediate is single-use,
    making it the forwarding-ideal counterpoint to the reuse-heavy ekf1_stateless kernel.
    """
    return ((c3 * x + c2) * x + c1) * x + c0


def main() -> None:
    float_format = holoso.FloatFormat(wexp=6, wman=18)
    ops = holoso.OpConfig(
        holoso.FAddOperator(float_format),
        holoso.FMulOperator(float_format),
        holoso.FDivOperator(float_format),
        holoso.FMulILog2OperatorFamily(float_format),
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(poly3, ops=ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

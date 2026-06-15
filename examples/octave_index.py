#!/usr/bin/env python3
"""
The order of magnitude of a number in octaves: how many halvings (or, for a magnitude below unity, doublings) bring
|x| into the unit interval (0.5, 1]. This is ceil(|log2(|x|)|) computed with only compares, a reciprocal,
and a halving loop -- a pure-arithmetic exponent estimator for designs without bit-level access to the float field
(auto-ranging front-ends, gain staging, coarse logarithms).

A magnitude below one is first inverted, so a single halving loop serves both ranges. That reciprocal 1/|x| is the
one non-speculatable operation, so the magnitude-selecting diamond stays a real branch; its merge -- which only carries
the selected magnitude into the loop -- is an empty pass-through that merge threading eliminates, folding both arms
straight into the loop header. The trip count is the octave distance, hence a data-dependent number of cycles per input.
"""

from pathlib import Path
import holoso


def octave_index(x: float) -> float:
    magnitude = abs(x)
    if magnitude >= 1.0:
        scaled = magnitude
    else:
        scaled = 1.0 / magnitude  # the lone non-speculatable op: keeps the diamond a real branch
    # FIXME: ``octaves`` is an integer count carried as a float only because integer operands are not yet supported;
    # TODO: make it an int once the frontend grows typed int operands/constants.
    octaves = 0.0
    while scaled > 1.0:
        scaled = scaled * 0.5
        octaves = octaves + 1.0
    return octaves


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
    result = holoso.synthesize(octave_index, ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

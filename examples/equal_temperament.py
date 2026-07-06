#!/usr/bin/env python3
"""
Twelve-tone equal temperament -- the transcendental counterpart to octave_index.py (which does log2 the hard way). It
maps a MIDI note to its frequency (exp2, reached through ``2 ** x``) and recovers the note back from that frequency
(log2), so the two operators are exercised as inverses. log2's argument is the exp2 output, always positive.
"""

import math
from pathlib import Path

import holoso

A4_Hz = 440.0
A4_NOTE = 69.0
SEMITONES_PER_OCTAVE = 12.0


def equal_temperament(note: float) -> tuple[float, float]:
    hertz = A4_Hz * 2 ** ((note - A4_NOTE) / SEMITONES_PER_OCTAVE)
    recovered = A4_NOTE + SEMITONES_PER_OCTAVE * math.log2(hertz / A4_Hz)
    return hertz, recovered


def main() -> None:
    float_format = holoso.FloatFormat(wexp=8, wman=36)
    ops = holoso.OpConfig(
        holoso.FAddOperator(float_format),
        holoso.FMulOperator(float_format),
        holoso.FDivOperator(float_format),
        holoso.FMulILog2OperatorFamily(float_format),
        holoso.FCmpOperator(float_format),
        fexp2=holoso.FExp2Operator(float_format),
        flog2=holoso.FLog2Operator(float_format),
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(equal_temperament, ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
A 3-out-of-5 (3oo5) majority voter with latched channel diagnostics -- the modular-redundancy pattern from
safety-critical control. The voted output is the majority of five redundant boolean inputs; while ``enabled``, each
channel that disagrees with the voted value (an exclusive-or against it) latches a sticky per-channel fault that holds
until the synchronous reset clears it. Five-way redundancy of this flavour flew on the Space Shuttle, whose five
general-purpose computers ran a four-way cross-voted primary avionics set alongside a dissimilarly-developed backup;
this kernel is the idealised flat 3oo5 majority rather than that exact 4-plus-1 arrangement.

The diagnostic update is deliberately gated behind ``enabled`` rather than folded into the always-on vote, so it stays
a real conditional branch: the five distinct ``channel ^ voted`` disagreements and five sticky-fault latches keep the
arm's work irreducible, so it survives even aggressive boolean simplification rather than collapsing into the vote.
"""

from pathlib import Path

import holoso


class MajorityVoter:
    def __init__(self) -> None:
        self._fault_a: bool = False
        self._fault_b: bool = False
        self._fault_c: bool = False
        self._fault_d: bool = False
        self._fault_e: bool = False

    @staticmethod
    def _majority(a: bool, b: bool, c: bool, d: bool, e: bool) -> bool:
        """True when at least three of the five redundant channels agree -- the 3-of-5 voted value."""
        return (
            (a and b and c)
            or (a and b and d)
            or (a and b and e)
            or (a and c and d)
            or (a and c and e)
            or (a and d and e)
            or (b and c and d)
            or (b and c and e)
            or (b and d and e)
            or (c and d and e)
        )

    def __call__(self, enabled: bool, a: bool, b: bool, c: bool, d: bool, e: bool, /) -> tuple[bool, ...]:
        voted = self._majority(a, b, c, d, e)
        if enabled:
            # a channel is faulty when it disagrees with the voted majority -- exactly an exclusive-or; each fault is
            # sticky, accumulating across transactions until reset.
            self._fault_a = self._fault_a or (a ^ voted)
            self._fault_b = self._fault_b or (b ^ voted)
            self._fault_c = self._fault_c or (c ^ voted)
            self._fault_d = self._fault_d or (d ^ voted)
            self._fault_e = self._fault_e or (e ^ voted)
        return voted, self._fault_a, self._fault_b, self._fault_c, self._fault_d, self._fault_e


def main() -> None:
    # This kernel is purely boolean and emits no float arithmetic, so the float format is immaterial; the default wide
    # format is used only because OpConfig requires a complete operator set. This may be improved in the future.
    float_format = holoso.FloatFormat(wexp=8, wman=36)
    ops = holoso.OpConfig(
        holoso.FAddOperator(float_format),
        holoso.FMulOperator(float_format),
        holoso.FDivOperator(float_format),
        holoso.FMulILog2OperatorFamily(float_format),
        holoso.FCmpOperator(float_format),
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(MajorityVoter().__call__, ops)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

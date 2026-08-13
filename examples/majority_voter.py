#!/usr/bin/env python3
"""
A 3-out-of-5 (3oo5) majority voter with latched channel diagnostics -- the modular-redundancy pattern from
safety-critical control. Five-way redundancy of this flavour flew on the Space Shuttle, whose five general-purpose
computers ran a four-way cross-voted primary avionics set alongside a dissimilarly-developed backup;
this kernel is the idealised flat 3oo5 majority rather than that exact 4-plus-1 arrangement.

The diagnostic update is deliberately gated behind `enabled` rather than folded into the always-on vote, so it stays
a real conditional branch: the five distinct `channel ^ voted` disagreements and five sticky-fault latches keep the
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
        packed = int(a) | (int(b) << 1) | (int(c) << 2) | (int(d) << 3) | (int(e) << 4)
        return packed.bit_count() >= 3  # Synthesized into popcount and compare

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
    # The five channels pack into as many bits, which a signed word carries one above.
    options = holoso.Options(holoso.OperatorOptions(), wint_min=6)
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(MajorityVoter().__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

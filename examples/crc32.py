#!/usr/bin/env python3
"""
The CRC-32 algorithm compatible with various standard polynomials (IEEE 802.3, gzip, PNG, zlib, iSCSI, SCTP, NVMe, ...).
"""

from pathlib import Path

import holoso

POLY_IEEE8023 = 0x04C11DB7
POLY_CRC32C = 0x1EDC6F41


class Crc32:
    def __init__(self, polynomial: int, initial: int = 0xFFFF_FFFF, width: int = 32) -> None:
        # The constructor is not synthesized here, it is evaluated in Python.
        self._mask = (1 << width) - 1
        assert 0 <= polynomial <= self._mask
        assert 0 <= initial <= self._mask
        self._polynomial = int(f"{polynomial:0{width}b}"[::-1], 2)  # reflect
        self.register = initial

    def __call__(self, byte: int, /) -> int:
        acc = self.register ^ (byte & 0xFF)
        # The small constant-trip loop will be unrolled, see Options.
        # The conditional branches are small so they are if-converted into speculative execution with result selection.
        # The whole byte is thus consumed in a single transaction without loops or branches, very efficient.
        for _ in range(8):
            if (acc & 1) == 1:
                acc = (acc >> 1) ^ self._polynomial
            else:
                acc = acc >> 1
        self.register = acc
        return acc ^ self._mask


def main() -> None:
    # The register holds 32 unsigned bits, which a signed word must carry one bit above.
    options = holoso.Options(holoso.OperatorOptions(), wint_min=33)
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(Crc32(POLY_IEEE8023).__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

from dataclasses import dataclass

EXHAUSTIVE_MAX_WIDTH = 9
"""
At and below this width the integer HDL tests drive every operand combination, so they prove the operator at those
widths instead of sampling it. The widest exhaustive case is a few hundred thousand vectors, which is seconds.
"""

TEST_WIDTHS = (2, 3, 4, 5, 6, 7, 8, 9, 24, 33, 44)
"""
Every width up to the exhaustive limit, then the two production formats and the default one. Widths where 4 divides
``width - 1`` matter to the shifter: its magnitude ends exactly on a group boundary there, so the largest shift
amounts select a group that lies wholly in the padding.
"""


def signed(bits: int, width: int) -> int:
    sign = 1 << (width - 1)
    return bits - (1 << width) if bits & sign else bits


@dataclass(frozen=True, slots=True)
class ShiftResult:
    shft: int
    prod: int
    saturated: bool


def ishift(a_bits: int, b_bits: int, width: int) -> ShiftResult:
    """
    A left shift is a multiplication by a power of two, so it can overflow: `shft` lets the high bits fall off the
    word while `prod` clamps to the representable range. Right shifts cannot overflow and the two agree there.
    """
    assert width >= 2
    mask = (1 << width) - 1
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    assert 0 <= a_bits <= mask and 0 <= b_bits <= mask
    b = signed(b_bits, width)
    a = signed(a_bits, width)
    if b < 0:
        y = (a >> min(-b, width)) & mask
        return ShiftResult(y, y, False)
    shift = min(b, width)  # any larger amount is indistinguishable from width inside a width-bit word
    exact = a << shift
    clamped = min(max(exact, minimum), maximum)
    return ShiftResult((a_bits << shift) & mask, clamped & mask, clamped != exact)

EXHAUSTIVE_MAX_WIDTH = 9
"""
At and below this width the integer HDL tests drive every operand combination, so they prove the operator at those
widths instead of sampling it. The widest exhaustive case is a few hundred thousand vectors, which is seconds.
"""

TEST_WIDTHS = (2, 3, 4, 5, 6, 7, 8, 9, 24, 33, 44)
"""
Every width up to the exhaustive limit, then the two production formats and the default one. Widths where 4 divides
``width - 1`` matter to the shifter, whose group decomposition runs its last index one past the array there.
"""


def signed(bits: int, width: int) -> int:
    sign = 1 << (width - 1)
    return bits - (1 << width) if bits & sign else bits


def ashift(a_bits: int, b_bits: int, width: int, saturating: bool) -> tuple[int, int]:
    """
    Returns the result bits and the saturation flag. A left shift is a multiplication by a power of two, so under
    `saturating` it clamps to the representable range instead of letting the high bits fall off the word.
    """
    mask = (1 << width) - 1
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    b = signed(b_bits, width)
    a = signed(a_bits, width)
    if b < 0:
        if b <= -width:
            return mask if a < 0 else 0, 0
        return (a >> -b) & mask, 0
    if saturating:
        if a == 0:
            return 0, 0
        if b >= width:  # the shift amount is unbounded, so a nonzero operand overflows without evaluating the shift
            return (maximum if a > 0 else minimum) & mask, 1
        exact = a << b
        clamped = min(max(exact, minimum), maximum)
        return clamped & mask, int(clamped != exact)
    if b >= width:
        return 0, 0
    return (a_bits << b) & mask, 0

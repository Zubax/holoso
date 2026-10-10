EXHAUSTIVE_MAX_WIDTH = 9
"""
At and below this width the integer HDL tests drive every operand combination, so they prove the operator at those
widths instead of sampling it. The widest exhaustive case is a few hundred thousand vectors, which is seconds.
"""

TEST_WIDTHS = (2, 3, 4, 5, 6, 7, 8, 9, 24, 33, 44)
"""
Every width up to the exhaustive limit, then the two production formats and the default one. The exhaustive range
holds both kinds of width the shifter tells apart: a power of two, whose count field spans exactly the word, and the
others, where a count past the word still fits the field.
"""


def signed(bits: int, width: int) -> int:
    sign = 1 << (width - 1)
    return bits - (1 << width) if bits & sign else bits


def ishl(a_bits: int, b_bits: int, width: int) -> int:
    """
    What `holoso_ishft` answers with `right` low: left for a positive count, right for a negative one. The shift is
    raw, so a left shift lets the high bits fall off the word.
    """
    assert width >= 2
    mask = (1 << width) - 1
    assert 0 <= a_bits <= mask and 0 <= b_bits <= mask
    a = signed(a_bits, width)
    b = signed(b_bits, width)
    shift = min(abs(b), width)  # any larger amount is indistinguishable from width inside a width-bit word
    return ((a_bits << shift) if b >= 0 else (a >> shift)) & mask


def ishr(a_bits: int, b_bits: int, width: int) -> int:
    """The mirror of `ishl`, `right` high: right for a positive count, left for a negative one, raw either way."""
    assert width >= 2
    mask = (1 << width) - 1
    assert 0 <= a_bits <= mask and 0 <= b_bits <= mask
    a = signed(a_bits, width)
    b = signed(b_bits, width)
    shift = min(abs(b), width)  # any larger amount is indistinguishable from width inside a width-bit word
    return ((a >> shift) if b >= 0 else (a_bits << shift)) & mask


def expected_iadds_add(a_bits: int, b_bits: int, width: int) -> dict[str, int]:
    """
    What `holoso_iadds` answers when it adds, keyed by the output ports that mode drives and its saturation sideband.
    Each mode of the adder has its own reference, the module's name selecting none of them.
    """
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    exact = signed(a_bits, width) + signed(b_bits, width)
    clamped = min(max(exact, minimum), maximum)
    return {"y": clamped & ((1 << width) - 1), "saturated": int(clamped != exact)}


def expected_iadds_sub(a_bits: int, b_bits: int, width: int) -> dict[str, int]:
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    exact = signed(a_bits, width) - signed(b_bits, width)
    clamped = min(max(exact, minimum), maximum)
    return {"y": clamped & ((1 << width) - 1), "saturated": int(clamped != exact)}


def expected_iadds_cmp(a_bits: int, b_bits: int, width: int) -> dict[str, int]:
    a = signed(a_bits, width)
    b = signed(b_bits, width)
    return {"a_gt_b": int(a > b), "a_eq_b": int(a == b), "a_lt_b": int(a < b)}


def expected_simple(module: str, a_bits: int, width: int) -> dict[str, int]:
    """
    What one of the single-operand modules taking neither a mode nor a parameter beyond the width answers, keyed by
    its own output port names. This, the shift directions and the adder's modes above and the two configurable
    siblings below are the reference the HDL benches score every module against, so they are also what the Python
    operator model must reproduce.
    """
    a = signed(a_bits, width)
    if module == "holoso_ipopcnt":
        return {"y": abs(a).bit_count()}
    assert module == "holoso_iabss"
    clamped = min(abs(a), (1 << (width - 1)) - 1)
    return {"y": clamped, "saturated": int(clamped != abs(a))}


def expected_imuls(a_bits: int, b_bits: int, width: int) -> dict[str, int]:
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    exact = signed(a_bits, width) * signed(b_bits, width)
    clamped = min(max(exact, minimum), maximum)
    return {"y": clamped & ((1 << width) - 1), "saturated": int(clamped != exact)}


def expected_idivs(num_bits: int, den_bits: int, width: int, quotient_floor: bool) -> dict[str, int]:
    mask = (1 << width) - 1
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    num = signed(num_bits, width)
    den = signed(den_bits, width)
    if den == 0:
        quotient = minimum if num < 0 else maximum
        remainder = num
        saturated = 1
        div0 = 1
    elif num == minimum and den == -1:
        quotient = maximum
        remainder = 0
        saturated = 1
        div0 = 0
    else:
        if quotient_floor:
            quotient = num // den
        else:
            quotient_magnitude = abs(num) // abs(den)
            quotient = -quotient_magnitude if (num < 0) != (den < 0) else quotient_magnitude
        remainder = num - den * quotient
        saturated = 0
        div0 = 0
        assert num == den * quotient + remainder
        assert abs(remainder) < abs(den)
        if remainder:
            assert (remainder < 0) == ((den if quotient_floor else num) < 0)
    return {
        "quo": quotient & mask,
        "rem": remainder & mask,
        "saturated": saturated,
        "div0": div0,
    }

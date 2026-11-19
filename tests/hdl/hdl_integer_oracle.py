"""
A fixed-width reference for the integer operators, written on raw bit patterns independently of `IntValue`: the
operators' `evaluate` is scored against it, and `tests/test_rtl.py` scores their RTL against `evaluate`. Each answer is
keyed by the name of the output port that carries it.
"""


def signed(bits: int, width: int) -> int:
    sign = 1 << (width - 1)
    return bits - (1 << width) if bits & sign else bits


def _saturated(exact: int, width: int) -> int:
    clamped = min(max(exact, -(1 << (width - 1))), (1 << (width - 1)) - 1)
    return clamped & ((1 << width) - 1)


def ishl(a_bits: int, b_bits: int, width: int) -> int:
    """
    What the shifter answers shifting left: left for a positive count, right for a negative one. The shift is raw, so
    a left shift lets the high bits fall off the word.
    """
    assert width >= 2
    mask = (1 << width) - 1
    assert 0 <= a_bits <= mask and 0 <= b_bits <= mask
    a = signed(a_bits, width)
    b = signed(b_bits, width)
    shift = min(abs(b), width)  # any larger amount is indistinguishable from width inside a width-bit word
    return ((a_bits << shift) if b >= 0 else (a >> shift)) & mask


def ishr(a_bits: int, b_bits: int, width: int) -> int:
    """The mirror of `ishl`: right for a positive count, left for a negative one, raw either way."""
    assert width >= 2
    mask = (1 << width) - 1
    assert 0 <= a_bits <= mask and 0 <= b_bits <= mask
    a = signed(a_bits, width)
    b = signed(b_bits, width)
    shift = min(abs(b), width)  # any larger amount is indistinguishable from width inside a width-bit word
    return ((a >> shift) if b >= 0 else (a_bits << shift)) & mask


def expected_iadds_add(a_bits: int, b_bits: int, width: int) -> dict[str, int]:
    """What the adder answers when it adds. Each operation of the adder has its own reference."""
    return {"y": _saturated(signed(a_bits, width) + signed(b_bits, width), width)}


def expected_iadds_sub(a_bits: int, b_bits: int, width: int) -> dict[str, int]:
    return {"y": _saturated(signed(a_bits, width) - signed(b_bits, width), width)}


def expected_iadds_cmp(a_bits: int, b_bits: int, width: int) -> dict[str, int]:
    a = signed(a_bits, width)
    b = signed(b_bits, width)
    return {"a_gt_b": int(a > b), "a_eq_b": int(a == b), "a_lt_b": int(a < b)}


def expected_unary(operator: str, a_bits: int, width: int) -> dict[str, int]:
    """What one of the single-operand operators, named as the operator is, answers."""
    a = signed(a_bits, width)
    if operator == "ipopcnt":
        return {"y": abs(a).bit_count()}
    assert operator == "iabss"
    return {"y": min(abs(a), (1 << (width - 1)) - 1)}


def expected_imuls(a_bits: int, b_bits: int, width: int) -> dict[str, int]:
    return {"y": _saturated(signed(a_bits, width) * signed(b_bits, width), width)}


def expected_idivs(num_bits: int, den_bits: int, width: int) -> dict[str, int]:
    """
    The floor division and its remainder. A zero divisor answers the rail of the numerator's sign and keeps the
    numerator as the remainder; MIN // -1 saturates.
    """
    mask = (1 << width) - 1
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    num = signed(num_bits, width)
    den = signed(den_bits, width)
    if den == 0:
        quotient = minimum if num < 0 else maximum
        remainder = num
    elif num == minimum and den == -1:
        quotient = maximum
        remainder = 0
    else:
        quotient = num // den
        remainder = num - den * quotient
        assert num == den * quotient + remainder
        assert abs(remainder) < abs(den)
        assert not remainder or (remainder < 0) == (den < 0)
    return {"quo": quotient & mask, "rem": remainder & mask}

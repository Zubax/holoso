def signed(bits: int, width: int) -> int:
    sign = 1 << (width - 1)
    return bits - (1 << width) if bits & sign else bits


def ashift(a_bits: int, b_bits: int, width: int) -> int:
    mask = (1 << width) - 1
    b = signed(b_bits, width)
    if b >= 0:
        if b >= width:
            return 0
        return (a_bits << b) & mask
    a = signed(a_bits, width)
    if b <= -width:
        return mask if a < 0 else 0
    return (a >> -b) & mask

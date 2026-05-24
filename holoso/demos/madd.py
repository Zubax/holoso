"""Subtract, scale by a power of two, then fused multiply-add."""


def madd(a, b, c):
    # The 0.25 scaling is a power of two, so it costs no multiplier -- only an exponent adjustment.
    return (a - b) * 0.25 + a * b

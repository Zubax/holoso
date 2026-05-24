"""Cubic polynomial evaluated in Horner form: three fused multiply-adds, no powers."""


def poly3(x, c0, c1, c2, c3):
    return ((c3 * x + c2) * x + c1) * x + c0

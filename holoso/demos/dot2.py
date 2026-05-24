"""Dot product of two 2-vectors: the simplest multiply-add kernel."""


def dot2(a, b, c, d):
    ab = a * b
    cd = c * d
    return ab + cd

"""
B1 competing-error precedence: two schema-violating stores on sibling arms with different messages; the
surfaced one is pinned to the then-arm store by CFG preorder, independent of any set iteration order.
"""


def kernel(c: bool, x: float) -> float:
    if c:
        a = 1
        a = x
    else:
        b = True
        b = x
    return x

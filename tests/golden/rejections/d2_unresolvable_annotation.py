"""D2: a PEP-649 lazy annotation that fails to resolve (a typo) is a located rejection, not a NameError."""


def kernel(x: flaot) -> float:
    return x

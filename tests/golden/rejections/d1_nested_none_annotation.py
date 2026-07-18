"""D1: None nested inside a return annotation is rejected; None is only the whole-return spelling."""


def kernel(a: float) -> tuple[float, None]:
    return a, None

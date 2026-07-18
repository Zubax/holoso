"""Pre-existing shape: a library-stub shape mismatch (matmul) is attributed to the user call site."""

import numpy as np
from jaxtyping import Float64


def kernel(a: Float64[np.ndarray, "2 3"], x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
    return a @ x

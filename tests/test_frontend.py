"""Unit tests for the Python-to-HIR frontend."""

import numpy as np
import pytest

from holoso import UnsupportedConstruct
from holoso._frontend import lower


def test_nested_function_definition_is_rejected() -> None:
    # A nested function or class definition inside a kernel is unsupported -- even a dead one after a return. The
    # original scope-shadowing concern (the nested ``np`` leaking to the outer scope) cannot arise, because the nested
    # def is rejected outright at build time before any name resolution.
    def kernel(x: float) -> float:
        y = np.asarray([x])
        return y[0]  # type: ignore[no-any-return]

        def helper() -> int:  # noqa -- dead nested scope; its ``np`` is not the outer's
            np = 1
            return np

    with pytest.raises(UnsupportedConstruct, match="nested function"):
        lower(kernel)

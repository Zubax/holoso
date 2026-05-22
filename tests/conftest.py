"""Make `tests/` importable as a flat directory.

This lets bare-name imports like `from hdl_float_oracle import ...` work the same way in two contexts: pytest, which
would normally resolve them through the `tests` package, and the cocotb subprocess, which puts its `test_dir` on
sys.path directly. The existing `tests` package (used by `from tests import kernels`) still works because the
package itself is untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

"""Make `tests/` and `tests/hdl/` importable as flat directories.

This lets bare-name imports like `from hdl_float_oracle import ...` work the same way in two contexts: pytest (where
the root-level `test_cosim.py`/`test_backend.py` import the shared oracle that now lives in `tests/hdl/`), and the
cocotb subprocess (which puts only its `test_dir` -- `tests/hdl/` -- on sys.path). Adding both keeps the flat
bare-name import style consistent in either context.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE / "hdl"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

"""
Tool discovery for the synthesis-evaluation harness.
"""

import functools
import os
import shutil
from pathlib import Path

# Roots searched recursively, in order, after PATH. Deliberately broad; over-searching is cheap because results
# are cached and a missing tool just returns None.
_SEARCH_ROOTS = (Path("/opt"), Path("/usr"), Path("/home"))


def _walk_for(name: str) -> Path | None:
    for root in _SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for dirpath, _dirnames, filenames in os.walk(root, onerror=lambda _e: None, followlinks=False):
            if name in filenames:
                candidate = Path(dirpath) / name
                if os.access(candidate, os.X_OK):
                    return candidate
    return None


@functools.lru_cache(maxsize=None)
def find_tool(name: str) -> Path | None:
    """Locate an executable by name on PATH first, then by searching predefined locations on the filesystem."""
    found = shutil.which(name)
    if found:
        return Path(found)
    return _walk_for(name)


def require_tool(name: str) -> Path:
    """Like :func:`find_tool` but raise ``FileNotFoundError`` when the tool is absent."""
    path = find_tool(name)
    if path is None:
        raise FileNotFoundError(f"required executable {name!r} was not found on PATH, /opt, /usr, or /home")
    return path

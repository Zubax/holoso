"""
Shared flattening of a function's return into ordered, named output ports.

Both the code generator (which walks the return AST) and the verification reference (which walks the runtime return
value) must agree on output-port order and naming. The naming convention lives here, expressed over a *path* of keys
(integer sequence indices or string dataclass field names), so the two walkers cannot drift apart.

Convention (matches the examples in ``DESIGN.md`` / ``DESIGN.draft.md``):

- a bare scalar return is output ``0``           -> ``out_0``
- a sequence yields positional outputs           -> ``out_0``, ``out_1``, ...
- a dataclass yields its fields by name          -> ``out_<field>``
- nesting concatenates                           -> ``out_0_0`` (matrices), ``out_0_foo_bar`` (nested dataclasses)
"""

import dataclasses
from typing import Any

PathKey = int | str
Path = tuple[PathKey, ...]


def port_name(path: Path) -> str:
    """Map a leaf path to its output-port name, e.g. ``(0,)`` -> ``out_0``; ``(0, "foo", "bar")`` -> ``out_0_foo_bar``."""
    return "out" + "".join(f"_{key}" for key in path)


def flatten_value(root: object) -> list[tuple[Path, Any]]:
    """Flatten a runtime return value into ``(path, leaf)`` pairs in row-major / declaration order."""
    leaves: list[tuple[Path, Any]] = []

    def walk(node: object, path: Path) -> None:
        if isinstance(node, (list, tuple)) and not isinstance(node, str):
            for index, item in enumerate(node):
                walk(item, (*path, index))
        elif dataclasses.is_dataclass(node) and not isinstance(node, type):
            for field in dataclasses.fields(node):
                walk(getattr(node, field.name), (*path, field.name))
        else:
            leaves.append((path, node))

    if (isinstance(root, (list, tuple)) and not isinstance(root, str)) or (
        dataclasses.is_dataclass(root) and not isinstance(root, type)
    ):
        walk(root, ())
    else:
        leaves.append(((0,), root))
    return leaves


def output_names(root: object) -> list[str]:
    """The ordered output-port names for a runtime return value."""
    return [port_name(path) for path, _ in flatten_value(root)]

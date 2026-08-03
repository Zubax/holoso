"""Port naming shared by HIR emission and the test harnesses."""

import itertools


def port_name(path: tuple[int | str, ...]) -> str:
    """Map a returned leaf path to its output-port name, e.g. ``(0, "x")`` -> ``out_0_x``."""
    return "out" + "".join(f"_{key}" for key in path)


def slot_name(path: tuple[str | int, ...]) -> str:
    """The flattened slot-register name of a state leaf path, e.g. ``("x", 0, 1)`` -> ``x_0_1``."""
    assert path and isinstance(path[0], str)
    return "_".join(str(key) for key in path)


def state_port_name(path: tuple[str | int, ...]) -> str:
    """The observability port of a public state leaf, e.g. ``("x", 0)`` -> ``state_x_0``."""
    return "state_" + slot_name(path)


def public_slot(path: tuple[str | int, ...]) -> bool:
    """A public attribute gets a ``state_<attr>`` port; an underscore-prefixed one stays register-only."""
    assert path and isinstance(path[0], str)
    return not path[0].startswith("_")


def indexed_names(base: str, shape: tuple[int, ...]) -> list[str]:
    """
    The row-major per-element names of a shaped base name: ``()`` -> ``[base]``, ``(2,)`` -> ``[base_0, base_1]``,
    ``(2, 2)`` -> ``[base_0_0, base_0_1, base_1_0, base_1_1]``. The one naming convention shared by decomposed
    state slots and decomposed array-parameter input ports.
    """
    return [base + "".join(f"_{i}" for i in index) for index in itertools.product(*(range(dim) for dim in shape))]

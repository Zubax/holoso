"""
Port naming shared by HIR emission and the test harnesses.
``state_port_name`` joins when persistent state (M8) lands.
"""

import itertools


def port_name(path: tuple[int | str, ...]) -> str:
    """Map a returned leaf path to its output-port name, e.g. ``(0, "x")`` -> ``out_0_x``."""
    return "out" + "".join(f"_{key}" for key in path)


def indexed_names(base: str, shape: tuple[int, ...]) -> list[str]:
    """
    The row-major per-element names of a shaped base name: ``()`` -> ``[base]``, ``(2,)`` -> ``[base_0, base_1]``,
    ``(2, 2)`` -> ``[base_0_0, base_0_1, base_1_0, base_1_1]``. The one naming convention shared by decomposed
    state slots and decomposed array-parameter input ports.
    """
    return [base + "".join(f"_{i}" for i in index) for index in itertools.product(*(range(dim) for dim in shape))]

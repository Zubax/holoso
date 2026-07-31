"""
Port naming shared by HIR emission and the test harnesses.
``state_port_name`` and ``indexed_names`` join when persistent state (M8) and aggregates (M5) land.
"""


def port_name(path: tuple[int | str, ...]) -> str:
    """Map a returned leaf path to its output-port name, e.g. ``(0, "x")`` -> ``out_0_x``."""
    return "out" + "".join(f"_{key}" for key in path)

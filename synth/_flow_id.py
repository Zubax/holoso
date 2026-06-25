from enum import StrEnum


class FlowId(StrEnum):
    """
    The supported synthesis flows; each value is the flow id used on the command line and recorded in reports. Each
    names its target device so the id stays unambiguous as more devices appear. Kept in this dependency-free leaf
    module so every layer (the report dataclasses, the flow modules, the CLI, the test matrix) shares one source.
    """

    YOSYS_ECP5 = "yosys-ecp5"
    DIAMOND_ECP5 = "diamond-ecp5"
    VIVADO_ARTIX7 = "vivado-artix7"

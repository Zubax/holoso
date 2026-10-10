from enum import StrEnum


class FlowId(StrEnum):
    YOSYS_ECP5 = "yosys-ecp5"
    DIAMOND_ECP5 = "diamond-ecp5"
    VIVADO_ARTIX7 = "vivado-artix7"


class DeviceClass(StrEnum):
    """Each flow maps a class to a part of its own family; the larger one is for a design that outgrows the default."""

    DEFAULT = "default"
    LARGE = "large"

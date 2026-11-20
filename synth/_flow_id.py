from enum import StrEnum


class FlowId(StrEnum):
    YOSYS_ECP5 = "yosys-ecp5"
    DIAMOND_ECP5 = "diamond-ecp5"
    VIVADO_ARTIX7 = "vivado-artix7"


class DeviceClass(StrEnum):
    """Each flow maps a class to a part of its own family; the larger one is for a design that outgrows the default."""

    DEFAULT = "default"
    LARGE = "large"


class SpeedGrade(StrEnum):
    """
    A family's speed grades from its slowest, which each flow names in its own part numbering: ECP5 grades 6, 7 and 8;
    Artix-7 grades -1, -2 and -3. A wide datapath whose limiting path no register splits may close on a faster part.
    """

    SLOWEST = "slowest"
    MIDDLE = "middle"
    FASTEST = "fastest"

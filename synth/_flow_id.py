from enum import StrEnum


class FlowId(StrEnum):
    YOSYS_ECP5 = "yosys-ecp5"
    DIAMOND_ECP5 = "diamond-ecp5"
    VIVADO_ARTIX7 = "vivado-artix7"

from typing import assert_never

from .._flow_id import DeviceClass as DeviceClass
from .._flow_id import FlowId as FlowId
from .._flow_id import SpeedGrade as SpeedGrade
from ._flow import Flow as Flow
from .diamond import DiamondEcp5Device, DiamondEcp5Flow
from .vivado import VivadoArtix7Flow, XilinxPart
from .yosys import Ecp5Device, YosysEcp5Flow


def make_flow(
    flow_id: FlowId,
    target_frequency_MHz: float,
    device_class: DeviceClass = DeviceClass.DEFAULT,
    speed_grade: SpeedGrade = SpeedGrade.SLOWEST,
) -> Flow:
    match flow_id:
        case FlowId.YOSYS_ECP5:
            return YosysEcp5Flow(Ecp5Device.of(device_class, speed_grade), target_frequency_MHz)
        case FlowId.DIAMOND_ECP5:
            return DiamondEcp5Flow(DiamondEcp5Device.of(device_class, speed_grade), target_frequency_MHz)
        case FlowId.VIVADO_ARTIX7:
            return VivadoArtix7Flow(XilinxPart.of(device_class, speed_grade), target_frequency_MHz)
        case _:
            assert_never(flow_id)

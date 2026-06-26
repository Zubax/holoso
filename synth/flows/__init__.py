from collections.abc import Callable

from .._flow_id import FlowId as FlowId
from ._flow import Flow as Flow
from .diamond import DiamondEcp5Flow
from .vivado import VivadoArtix7Flow
from .yosys import YosysEcp5Flow

_CONSTRUCTORS: dict[FlowId, Callable[[float], Flow]] = {
    FlowId.YOSYS_ECP5: lambda freq: YosysEcp5Flow(target_frequency_MHz=freq),
    FlowId.DIAMOND_ECP5: lambda freq: DiamondEcp5Flow(target_frequency_MHz=freq),
    FlowId.VIVADO_ARTIX7: lambda freq: VivadoArtix7Flow(target_frequency_MHz=freq),
}
assert _CONSTRUCTORS.keys() == set(FlowId), "every FlowId needs a constructor"


def make_flow(flow_id: FlowId, target_frequency_MHz: float) -> Flow:
    return _CONSTRUCTORS[flow_id](target_frequency_MHz)

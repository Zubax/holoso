"""Fixed-point and integer-enumeration boundaries the shipped example does not reach, checked against CPython."""

import dataclasses
import enum
import sys
from pathlib import Path

import holoso

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from fixed_point_pi import Current, Fault, Fix  # noqa: E402


class Mode(enum.IntEnum):
    IDLE = 0
    RUN = 3


@dataclasses.dataclass(frozen=True)
class Sample:
    current: Current
    mode: Mode


class Monitor:
    def __init__(self) -> None:
        self.offset = Current(-5)

    def __call__(self, sample: Sample, mask: Fault, /) -> tuple[int, Fault]:
        level = int(sample.current) - int(self.offset) if sample.mode == Mode.RUN else 0
        return level, mask | Fault.OVERCURRENT


def test_enumerations_and_records_cross_the_boundary_as_integers() -> None:
    result = holoso.synthesize(Monitor().__call__, holoso.Options(holoso.OperatorOptions()), name="monitor")
    assert [p.name for p in result.input_ports] == ["in_sample_current_word", "in_sample_mode", "in_mask"]
    sim = result.numerical_model.elaborate()
    for word, mode, mask in ((100, Mode.RUN, Fault(0)), (-2048, Mode.RUN, Fault.SATURATED), (7, Mode.IDLE, Fault(0))):
        got = [v for v in sim.run(word, int(mode), int(mask)) if isinstance(v, holoso.IntValue)]
        assert [int(v) for v in got] == list(Monitor()(Sample(Current(word), mode), mask))


def _doubled(value: Fix) -> int:
    return 2 * int(value)


def test_a_base_class_annotation_admits_a_subclass_record() -> None:
    def kernel(current: Current) -> int:
        return _doubled(current)

    result = holoso.synthesize(kernel, holoso.Options(holoso.OperatorOptions()), name="doubled")
    sim = result.numerical_model.elaborate()
    for word in (0, 5, -2048):
        got = sim.run(word)[0]
        assert isinstance(got, holoso.IntValue) and int(got) == kernel(Current(word))

#!/usr/bin/env python3
"""
A PI current regulator computed entirely in fixed point.
The word formats are ordinary user classes; Holoso ships nothing for them to import.
"""

import dataclasses
import enum
from pathlib import Path
from typing import ClassVar, Self

import holoso


@dataclasses.dataclass(frozen=True)
class Fix:
    """A signed word of `width` bits whose least significant bit weighs `2**-frac`."""

    word: int

    frac: ClassVar[int]
    lo: ClassVar[int]
    hi: ClassVar[int]

    def __init_subclass__(cls, *, width: int, frac: int) -> None:
        super().__init_subclass__()
        cls.frac, cls.lo, cls.hi = frac, -(1 << (width - 1)), (1 << (width - 1)) - 1

    @classmethod
    def saturated(cls, word: int, /) -> Self:
        return cls(min(max(word, cls.lo), cls.hi))

    @classmethod
    def encode(cls, value: float, /) -> Self:
        return cls.saturated(round(value * 2**cls.frac))

    @classmethod
    def product(cls, a: Fix, b: Fix, /) -> int:
        return (a.word * b.word) >> (a.frac + b.frac - cls.frac)

    @classmethod
    def rescaled(cls, x: Fix, /) -> int:
        return x.word >> (x.frac - cls.frac)

    def __int__(self) -> int:
        return self.word


class Current(Fix, width=12, frac=4): ...  # [A]


class CurrentError(Fix, width=13, frac=4): ...  # [A] the difference of two currents, held exactly


class Voltage(Fix, width=12, frac=4): ...  # [V]


class Integral(Fix, width=16, frac=8): ...  # [V] Voltage's range at a finer LSB, so a small error still integrates


class Gain(Fix, width=8, frac=4): ...  # [V/A]


class Fault(enum.IntFlag):
    SATURATED = 1 << 0
    OVERCURRENT = 1 << 1


@dataclasses.dataclass(frozen=True)
class Command:
    voltage: Voltage
    fault: Fault


class CurrentRegulator:
    """The integrator holds while the command is on a rail, so a demand the bridge cannot meet does not wind it up."""

    def __init__(self, *, ki: Gain = Gain.encode(0.125), overcurrent: Current = Current.encode(120.0)) -> None:
        self.ki = ki  # per tick
        self.overcurrent = overcurrent
        self.integral = 0  # in Integral words

    def __call__(self, reference: Current, measurement: Current, kp: Gain, /) -> Command:
        error = CurrentError(int(reference) - int(measurement))
        integral = Integral.saturated(self.integral + Integral.product(self.ki, error))
        demand = Voltage.product(kp, error) + Voltage.rescaled(integral)
        voltage = Voltage.saturated(demand)
        saturated = int(voltage) != demand
        if not saturated:
            self.integral = int(integral)
        fault = Fault.SATURATED if saturated else Fault(0)
        if abs(int(measurement)) >= int(self.overcurrent):
            fault |= Fault.OVERCURRENT
        return Command(voltage, fault)


def main() -> None:
    # The widest intermediate is the proportional product, at most 4095 * 128 in magnitude: 19 bits plus the sign.
    options = holoso.Options(holoso.OperatorOptions(), wint_min=20)
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(CurrentRegulator().__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

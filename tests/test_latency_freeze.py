"""
Chained-copy schedule freeze and behavioral check. The example schedule rows that used to live here moved into
the golden corpus (``tests/_golden_cases.py`` + ``tests/test_golden.py``), whose ABI manifests pin the exact
(min II, last PC) per catalogued case; this file keeps only the purpose-built kernels no committed example
exercises.

The chained-copy kernel shape -- a state assignment ``self.a = self.b`` where both sides are slots, so one
slot's live-out reads another slot's live-in (the register allocator's ``tapped_by_other`` path) -- is pinned
in both banks by two purpose-built kernels (a float delay line and a boolean shift register), and a behavioral
check confirms the chained copy still captures each old value before it is overwritten, which a value-blind
schedule freeze cannot.
"""

import pytest

import holoso
from holoso import FloatFormat
from holoso._frontend import lower as lower_frontend
from holoso._hir import optimize
from holoso._lir import FloatStateSlot, build
from holoso._lir._ir import BoolStateSlot
from holoso._mir import lower as lower_to_mir

from ._modelref import default_ops

_FMT_WIDE = FloatFormat(8, 36)


class _Delay3:
    def __init__(self) -> None:
        self.x0 = 0.0
        self.x1 = 0.0
        self.x2 = 0.0

    def __call__(self, x: float) -> float:
        out = self.x2
        self.x2 = self.x1
        self.x1 = self.x0
        self.x0 = x
        return out


class _BoolShift3:
    def __init__(self) -> None:
        self.b0 = False
        self.b1 = False
        self.b2 = False

    def __call__(self, b: bool) -> bool:
        out = self.b2
        self.b2 = self.b1
        self.b1 = self.b0
        self.b0 = b
        return out


# Chained-copy kernels and their frozen (min II, last PC). _Delay3 exercises the wide bank's tapped_by_other path,
# _BoolShift3 the boolean bank's, both at the wide e8m36 datapath the example matrix uses.
_CHAINED_COPY: list[tuple[str, type[_Delay3] | type[_BoolShift3], tuple[int, int]]] = [
    ("delay3", _Delay3, (3, 3)),
    ("bool_shift3", _BoolShift3, (3, 3)),
]


@pytest.mark.parametrize("name,kernel_cls,frozen", _CHAINED_COPY)
def test_chained_copy_schedule_is_frozen(
    name: str, kernel_cls: type[_Delay3] | type[_BoolShift3], frozen: tuple[int, int]
) -> None:
    lir = build(
        lower_to_mir(optimize(lower_frontend(kernel_cls().__call__)), default_ops(_FMT_WIDE)), name, fetch_stages=3
    )
    slots: list[FloatStateSlot | BoolStateSlot] = [*lir.float_state_slots, *lir.bool_state_slots]
    assert all(
        slot.needs_copy for slot in slots
    ), f"{name}: a chained-copy slot unexpectedly coalesced; the tapped_by_other path is no longer exercised"
    got = (lir.min_initiation_interval, lir.last_pc)
    assert got == frozen, (
        f"{name}: scheduling efficiency changed -- (min II, last PC) {got} differs from the frozen {frozen}. "
        f"If this is a deliberate schedule improvement, update the frozen value."
    )


def test_chained_copy_captures_old_values() -> None:
    """
    The chained copy must sample each slot's OLD value before the bundle overwrites it, so a 3-tap line delays its
    input by exactly three transactions. A value-blind schedule freeze cannot see a read-after-write miscompile here.
    """
    fmt = _FMT_WIDE
    fmodel = holoso.synthesize(_Delay3().__call__, default_ops(fmt), name="delay3").numerical_model.elaborate()
    fref = _Delay3()
    for raw in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
        x = fmt.decode(fmt.encode(raw))  # quantize so the model and the reference see the identical operand
        assert float(fmodel.run(x)[0]) == fref(x)

    bmodel = holoso.synthesize(_BoolShift3().__call__, default_ops(fmt), name="bool_shift3").numerical_model.elaborate()
    bref = _BoolShift3()
    for b in (True, False, True, True, False, False, True, False):
        assert bool(bmodel.run(b)[0]) == bref(b)

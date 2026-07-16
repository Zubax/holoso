"""
Golden schedule-length freeze for a representative cross-section of example kernels -- a committed regression
guard on scheduling efficiency, standing in for the deferred makespan/II optimization work.

The differential fuzzer and the example-reference suite compare output VALUES only; they are structurally blind
to cycle count, so a schedule that still computes the right result but takes longer -- a wasted cycle, a lost
cross-block overlap, an over-pipelining congestion regression -- passes them silently. This test pins each
kernel's (min initiation interval, last microcode PC). The min II is the throughput of the shortest path; the
last PC is the out_valid boundary PC -- the end of the static schedule across every block (a zero-based PC, not a
word count) -- so the full schedule is pinned even for a data-dependent branch/loop kernel whose public max II is
reported as None (its loop body would otherwise be unguarded). Both are fixed by the scheduler before register-
allocation annealing, so they are independent of ``HOLOSO_REGALLOC_EFFORT``. A deliberate schedule change is
expected to update the frozen value in the same commit.

It also folds in the chained-copy kernel shape -- a state assignment ``self.a = self.b`` where both sides are
slots, so one slot's live-out reads another slot's live-in (the register allocator's ``tapped_by_other`` path). No
committed example exercises it. Two purpose-built kernels (a float delay line and a boolean shift register) pin it in
both banks, and a behavioral check confirms the chained copy still captures each old value before it is overwritten --
which the value-blind schedule freeze cannot.
"""

import pytest

import holoso
from holoso import FloatFormat
from holoso._frontend import lower as lower_frontend
from holoso._hir import optimize
from holoso._lir import FloatStateSlot, build
from holoso._lir._ir import BoolStateSlot
from holoso._mir import lower as lower_to_mir

from ._examples import SPECS
from ._modelref import default_ops

# Kernel name -> frozen (min initiation interval, last microcode PC). last_pc is the out_valid boundary PC -- the
# end of the static schedule across all blocks -- so it pins the full schedule even for data-dependent (branch/loop)
# kernels. A representative cross-section of shapes: straight-line and deep arithmetic, clamp/select, stateful
# filters, branchy logic, data-dependent loops, and a large kernel.
# FIR_PARITY_PENDING: ekf1_stateless (ndarray returns) is dropped from the freeze until its stage lands. The
# tuple-returning rows (signal_window, majority_voter, uart_rx/uart_tx, cordic_sincos) are restored at their exact
# pre-cutover values: with jump-chain fusion the new front-end schedules them identically to the old. iir1_hpf
# improved when fusion let its guard diamond if-convert (24 -> 20).
_FROZEN_SCHEDULE: dict[str, tuple[int, int]] = {
    "madd": (14, 14),
    "signal_window": (9, 9),
    "poly3": (23, 23),
    "iir1_lpf": (15, 15),
    "iir1_hpf": (20, 20),
    "schmitt_trigger": (6, 6),
    "majority_voter": (14, 19),
    # The loop body's tail copy (y <- y_next) sources y_next, which is NOT the block's last work (delta = y_next - y
    # is), so the install fits at the work makespan instead of one past it -- shaving a cycle off every iteration.
    "recip_newton": (15, 32),
    "remainder": (36, 53),
    "cordic_sincos": (104, 104),
    # Frozen at first activation: these rows join the table the day their features land (aggregate state at
    # stage 9a-3, array ports at 9b), so they are the new front-end's own baselines, not legacy carry-overs.
    "ekf1_stateful": (125, 125),
    "polar_to": (63, 63),
    "polar_from": (38, 38),
    # Branchy kernels whose phi-arm installs source block-entry-resident values (boolean/float live-out constants, or
    # an input/state read) on the normal path -- the inline-class timing (no source-sample edge, no +1 step) lands each
    # within the work makespan rather than at the copy-pipeline boundary, shrinking every downstream block base.
    "uart_rx": (6, 120),
    # uart_tx additionally has an empty overlapping branch block (the idle "not busy" arm) whose only act is to test a
    # resident input condition; a non-entry branch may redirect at its own base PC, so its terminator drains nothing.
    "uart_tx": (7, 103),
    "octave_index": (14, 38),
}

_SPEC_BY_NAME = {spec.name: spec for spec in SPECS}


@pytest.mark.parametrize("name", sorted(_FROZEN_SCHEDULE))
def test_schedule_length_is_frozen(name: str) -> None:
    spec = _SPEC_BY_NAME[name]
    lir = build(
        lower_to_mir(optimize(lower_frontend(spec.make_kernel())), default_ops(spec.formats[0])), name, fetch_stages=3
    )
    got = (lir.min_initiation_interval, lir.last_pc)
    assert got == _FROZEN_SCHEDULE[name], (
        f"{name}: scheduling efficiency changed -- (min II, last PC) {got} differs from the frozen "
        f"{_FROZEN_SCHEDULE[name]}. If this is a deliberate schedule improvement, update the frozen value."
    )


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
# _BoolShift3 the boolean bank's. _FMT is the wide e8m36 datapath the example matrix uses.
_FMT = FloatFormat(8, 36)
_CHAINED_COPY: list[tuple[str, type[_Delay3] | type[_BoolShift3], tuple[int, int]]] = [
    ("delay3", _Delay3, (3, 3)),
    ("bool_shift3", _BoolShift3, (3, 3)),
]


@pytest.mark.parametrize("name,kernel_cls,frozen", _CHAINED_COPY)
def test_chained_copy_schedule_is_frozen(
    name: str, kernel_cls: type[_Delay3] | type[_BoolShift3], frozen: tuple[int, int]
) -> None:
    lir = build(lower_to_mir(optimize(lower_frontend(kernel_cls().__call__)), default_ops(_FMT)), name, fetch_stages=3)
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
    fmt = _FMT
    fmodel = holoso.synthesize(_Delay3().__call__, default_ops(fmt), name="delay3").numerical_model.elaborate()
    fref = _Delay3()
    for raw in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
        x = fmt.decode(fmt.encode(raw))  # quantize so the model and the reference see the identical operand
        assert float(fmodel.run(x)[0]) == fref(x)

    bmodel = holoso.synthesize(_BoolShift3().__call__, default_ops(fmt), name="bool_shift3").numerical_model.elaborate()
    bref = _BoolShift3()
    for b in (True, False, True, True, False, False, True, False):
        assert bool(bmodel.run(b)[0]) == bref(b)

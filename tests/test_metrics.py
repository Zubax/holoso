"""
Steering/area non-regression gate for the LIR build.

Every currently-synthesizing example is built to LIR and measured on the figures that bound the synthesized fabric:
the wide and boolean register counts, the register-file steering -- the arms of every operand port's read mux and of
every register's write select, counted off the same source lists the Verilog codebooks are numbered from
(`read_sources_per_port`, `write_sources_per_register`) plus the handshake-gated arms the emitter adds beside them,
so a frozen figure is the emitted mux on every kernel, straight-line or not -- the allocator's own score
`steering + register_price * registers`, the widest read port and the widest wide write select, and the statically
known latency figures. Here "straight-line" means the pure-float flat path: single block, no boolean fabric (an
if-converted kernel can be single-block without being straight-line in this sense). The baseline is frozen at a
fixed allocator tuning, passed explicitly so no environment speed-up leaks in. Value numbering is seed-independent
(`tests/test_determinism.py` proves byte-identical Verilog across `PYTHONHASHSEED` values), so these figures hold in
any process without pinning the hash seed.

The contract: `score` may never regress, and nothing else may regress either except through a deliberate
register-for-arm trade at the configured price, which re-freezes `nreg` (or `bnreg`) and `steering` together. The
control-flow rows encode the convergence win (cross-block reuse and coalescing collapsed the former fresh-per-value
register explosion), so any backslide toward it fails the same gate.
"""

import sys
from collections.abc import Callable
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pytest

from holoso import FloatFormat, FMulOptions, FSortOptions, OperatorOptions
from holoso._eel import lower
from holoso._lir import (
    BoolRegRef,
    Lir,
    MoveWriteSource,
    RegRef,
    read_arms,
    read_sources_per_port,
    write_arms,
    write_events,
)
from holoso._mir import MirOptions
from holoso._mir import lower as lower_to_mir
from ._modelref import (
    build_lir,
    default_mir,
    default_options,
    mir_options,
    DEFAULT_UNROLL_MAX_TRIPS,
    FROZEN_TUNING,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import madd  # noqa: E402
import poly3  # noqa: E402
from cordic_sincos import CordicSinCos  # noqa: E402
from ekf1_stateful import Ekf1  # noqa: E402
from ekf1_stateless import update_x_P  # noqa: E402
from iir1_hpf import IIR1HPF  # noqa: E402
from iir1_lpf import IIR1LPF  # noqa: E402
from imu_fusion import ImuFusion  # noqa: E402
from latching_fault_register import LatchingFaultRegister  # noqa: E402
from majority_voter import MajorityVoter  # noqa: E402
from octave_index import octave_index  # noqa: E402
from phase_frequency_detector import PhaseFrequencyDetector  # noqa: E402
from pid import PID  # noqa: E402
from quadrature_encoder import QuadratureEncoder  # noqa: E402
from recip_newton import NewtonReciprocal  # noqa: E402
from remainder import remainder  # noqa: E402
from schmitt_trigger import SchmittTrigger  # noqa: E402
from signal_window import signal_window  # noqa: E402
from trapezoidal_leaky_streaming_integrator import TrapezoidalLeakyStreamingIntegrator  # noqa: E402
from biquad import Biquad  # noqa: E402
from finite_set_current_controller import FiniteSetCurrentController  # noqa: E402
from fir import Fir4  # noqa: E402
from foc import FocController  # noqa: E402

_FMT = FloatFormat(8, 36)

# Kernels the shared default operator set cannot lower take their spec-style adjustment here.
_EXTRA_OPERATORS: dict[str, Callable[[OperatorOptions], OperatorOptions]] = {
    "ekf1_stateless_fmul2": lambda ops: dataclasses.replace(ops, fmul=FMulOptions(instances=2)),
    "imu_fusion": lambda ops: dataclasses.replace(ops, fsort=FSortOptions()),
    "finite_set_current_controller": lambda ops: dataclasses.replace(ops, fsort=FSortOptions()),
    "foc": lambda ops: dataclasses.replace(ops, fsort=FSortOptions()),
}

_EXAMPLES: dict[str, Callable[[], Callable[..., object]]] = {
    "madd": lambda: madd.madd,
    "poly3": lambda: poly3.poly3,
    "signal_window": lambda: signal_window,
    "iir1_hpf": lambda: IIR1HPF().step,
    "iir1_lpf": lambda: IIR1LPF().__call__,
    "pid": lambda: PID().__call__,
    "schmitt_trigger": lambda: SchmittTrigger().__call__,
    "quadrature_encoder": lambda: QuadratureEncoder().__call__,
    "phase_frequency_detector": lambda: PhaseFrequencyDetector().__call__,
    "latching_fault_register": lambda: LatchingFaultRegister().__call__,
    "majority_voter": lambda: MajorityVoter().__call__,
    "recip_newton": lambda: NewtonReciprocal().__call__,
    "remainder": lambda: remainder,
    "octave_index": lambda: octave_index,
    "cordic_sincos": lambda: CordicSinCos().__call__,
    "integrator": lambda: TrapezoidalLeakyStreamingIntegrator(k=2**-22).__call__,
    "imu_fusion": lambda: ImuFusion().update,
    "fir": lambda: Fir4().__call__,
    "biquad": lambda: Biquad().__call__,
    "ekf1_stateless": lambda: update_x_P,
    "ekf1_stateless_fmul2": lambda: update_x_P,
    "finite_set_current_controller": lambda: FiniteSetCurrentController().__call__,
    "ekf1_stateful": lambda: Ekf1().update,
    "foc": lambda: FocController().tick,
}


@dataclass(frozen=True, slots=True)
class Metrics:
    """
    The non-regression figures sampled off a built Lir.

    `steering` is the total register-file mux fan-in: over every operand port, its read sources beyond the first
    (registers and constants alike, a constant being an arm of the same mux), plus over every register, its write
    sources beyond the first -- the structurally distinct opcode sources of its write select (a pooled lane, an
    inline result, a move; one arm however many steps select it) and the handshake-gated arms beside them (an input
    load, a boundary state install). Counting the moves matters: phi-arm coalescing trades pc-gated copies for shared
    pooled writeback lanes, so a copy-blind proxy would mis-report a coalescing win as a regression. `score` is the
    allocator's objective form at the frozen register price (its own count leaves the one-bit arms and the pinned
    registers out); `max_read_port` and `max_write_select` are the widest read
    mux and the widest wide-bank write select, the localized view a total cannot give. `copies` is the total phi-arm
    install count (wide copies plus boolean writes), the direct measure of how many phi arms still install by copy
    rather than coalescing onto the merged register.

    `last_pc` is the static ROM length (Lir.initiation_interval) -- the total stage count: blocks tile the
    ROM, so a per-block drain regression in ANY block inflates it, the primary "excessive stages" guard.
    `max_block_span` is the largest per-block terminator offset (`max term_offset`), localizing a per-block drain
    regression to one block (unlike `last_pc`, it does not move with the number of blocks).
    """

    straight_line: bool
    nreg: int
    bnreg: int
    steering: int
    score: float
    max_read_port: int
    max_write_select: int
    copies: int
    min_ii: int
    last_pc: int
    max_block_span: int


def _mir_for(name: str) -> MirOptions:
    """The shared default, adjusted per kernel where the default operator set cannot lower it (min/max needs fsort)."""
    if name in _EXTRA_OPERATORS:
        options = default_options(_FMT)
        options = dataclasses.replace(options, operator=_EXTRA_OPERATORS[name](options.operator))
        return mir_options(options)
    return default_mir(_FMT)


def _build(kernel: Callable[..., object], name: str, options: MirOptions) -> Lir:
    return build_lir(lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, options), name, FROZEN_TUNING)


def _measure(name: str) -> Metrics:
    lir = _build(_EXAMPLES[name](), name, _mir_for(name))
    straight = (
        len(lir.blocks) == 1
        and not lir.bool_state_slots
        and not any(b.inline_ops or b.wide_copies or b.bool_writes for b in lir.blocks)
        and lir.bool_regfile.nreg == 0
    )
    reads, writes = read_arms(lir), write_arms(lir)
    steering = sum(max(0, n - 1) for n in reads.values()) + sum(max(0, n - 1) for n in writes.values())
    copies = sum(len(block.wide_copies) + len(block.bool_writes) for block in lir.blocks)
    nreg, bnreg = lir.regfile.nreg, lir.bool_regfile.nreg
    return Metrics(
        straight_line=straight,
        nreg=nreg,
        bnreg=bnreg,
        steering=steering,
        score=steering + FROZEN_TUNING.register_price * (nreg + bnreg),
        max_read_port=max(reads.values(), default=0),
        max_write_select=max((n for dst, n in writes.items() if isinstance(dst, RegRef)), default=0),
        copies=copies,
        min_ii=lir.min_initiation_interval,
        last_pc=lir.initiation_interval,
        max_block_span=max(block.term_offset for block in lir.blocks),
    )


# Each row is an upper bound (a build must be <= every field), frozen on the current build. What the figures reflect,
# per mechanism:
#
# - Registers and steering reflect the unified cross-block allocator: liveness-bounded reuse with coalesced state
#   slots, per-(instance, port) lane accounting of both banks' write selects, the comparator's read ports steered like
#   any other operand muxes, and commutative orientation (the comparator swaps with its gt/lt tap exchange), all
#   decided by one annealer minimizing `steering + 2 * registers`. steering is the emitted mux arm count: a constant
#   operand is a read arm (cordic's 23 constant arms are its angle table), and a write source repeated on several
#   steps is one arm. Rows where nreg and steering moved in opposite directions are the annealer's trades at that
#   price: the large kernels spent registers to remove arms (ekf1_stateless 41 -> 44 registers for 100 -> 91 arms,
#   ekf1_stateful 39 -> 42 for 99 -> 79), the phi-dense and small kernels spent arms to remove registers (imu_fusion
#   51 -> 41 registers for 151 -> 160 arms, remainder 9 -> 5 for 13 -> 17, pid 9 -> 6 for 15 -> 17); every score
#   fell or held. max_write_select records how wide those merges made the widest wide-bank write select, the
#   localized view the total hides (foc reaches 9). Write keys are the emitted opcodes (an inline result over the same
#   registers is one arm, two arms moving one operand are one), so the widest port and select of foc and imu_fusion
#   are those of the allocations that exact count chose.
# - copies and the phi-dense figures reflect phi-arm coalescing: a phi whose register-backed, identity-conditioner arms
#   do not interfere with it shares their register (the install-free oracle decides), so the install copy vanishes.
#   recip_newton keeps its one loop-carried copy (its phi overlaps the back-edge arm), proving the oracle refuses
#   unsound merges.
# - in-place state commit extends slot-live-out coalescing to both banks and to conditional (phi/select) updates: a
#   slot live-out (an operator result, an inline select from an if-converted update, or a phi whose "unchanged" arm is
#   the slot live-in) is written directly into its slot register read-first, eliding the boundary copy-back and the
#   scratch register. A validate-and-retry loop demotes any slot whose in-place commit the colorer finds unsound (a
#   live-in feeding another phi, or a dominator-arm clobber) back to a copy-back.
# - min_ii reflects uniform dependency edges (both banks read latch-free), diamond if-conversion (small pure branch
#   diamonds become muxes -- an fselect, or a bselect reduced to and/or/not for a boolean/mixed merge),
#   NOT-folding (a semantic NOT is a free consumer-side inversion), and cross-block software pipelining. Bool-phi
#   if-conversion runs both arms unconditionally, so it can RAISE min_ii (the shortest static path) while LOWERING
#   realized per-transaction latency -- the true goal, guarded by test_cycle_model. A converted diamond keeps both
#   arms' values simultaneously live, so it can need an extra register -- the cost of the mux. Coalescing never
#   changes behavior, but the surviving copy set feeds the drain/push classification, so min_ii can move with it.
# - bnreg reflects exact per-consumer boolean read steps and phi coalescing: a condition consumed mid-block frees
#   its register for a later value in the same block, and a boolean phi merging onto its arms drops its own register.
# - last_pc and max_block_span are the stage-count guards. They reflect the per-block drain tightener: the
#   coalesced-install fixpoint -- a phi-arm predecessor whose every arm coalesces installs nothing, so its
#   +1 install drain is dropped. The drained boundary is the latest value LANDING per op: every op -- inline (a
#   select or a bool->float cast) or pooled -- lands at the same uniform per-op landing, and an install with a
#   SETTLED source (`install_source_commit`) fires at the work makespan and drops its +1 install drain. last_pc tiles
#   every block's span (a per-block drain regression anywhere inflates it); max_block_span localizes it to one
#   block. These timing rules move the schedule-length guards but not nreg/bnreg/steering/copies; signal_window
#   carries a deliberately-loosened steering arm (one freed boolean register traded for one write-select mux) --
#   refrozen rather than chased, since the rules are global and correctness-neutral. Settling a commutative operator's
#   constant operand on one side re-colors what it interns, at an unchanged schedule, and is refrozen the same way.
# - pid uses a variable sample interval: the derivative path contains a real divide, and the first-sample and saturation
#   guards keep the kernel multi-block with one residual copy. The larger PID row is therefore a property of the example
#   itself, not a scheduler regression to chase.
# fmt: off
_BASELINE: dict[str, Metrics] = {
    "madd": Metrics(
        True, nreg=3, bnreg=0, steering=5, score=11.0, max_read_port=2, max_write_select=4,
        copies=0, min_ii=14, last_pc=14, max_block_span=14,
    ),
    "poly3": Metrics(
        True, nreg=5, bnreg=0, steering=4, score=14.0, max_read_port=3, max_write_select=3,
        copies=0, min_ii=23, last_pc=23, max_block_span=23,
    ),
    "signal_window": Metrics(
        False, nreg=4, bnreg=5, steering=8, score=26.0, max_read_port=2, max_write_select=2,
        copies=0, min_ii=9, last_pc=9, max_block_span=9,
    ),
    "iir1_hpf": Metrics(
        False, nreg=3, bnreg=1, steering=2, score=10.0, max_read_port=2, max_write_select=2,
        copies=0, min_ii=20, last_pc=20, max_block_span=20,
    ),
    "iir1_lpf": Metrics(
        False, nreg=3, bnreg=1, steering=2, score=10.0, max_read_port=2, max_write_select=2,
        copies=0, min_ii=15, last_pc=15, max_block_span=15,
    ),
    "pid": Metrics(
        False, nreg=6, bnreg=2, steering=17, score=33.0, max_read_port=4, max_write_select=5,
        copies=1, min_ii=36, last_pc=68, max_block_span=31,
    ),
    "schmitt_trigger": Metrics(
        False, nreg=1, bnreg=2, steering=2, score=8.0, max_read_port=1, max_write_select=1,
        copies=0, min_ii=6, last_pc=6, max_block_span=6,
    ),
    "quadrature_encoder": Metrics(
        False, nreg=0, bnreg=7, steering=7, score=21.0, max_read_port=0, max_write_select=0,
        copies=0, min_ii=6, last_pc=6, max_block_span=6,
    ),
    "phase_frequency_detector": Metrics(
        False, nreg=0, bnreg=5, steering=5, score=15.0, max_read_port=0, max_write_select=0,
        copies=0, min_ii=6, last_pc=6, max_block_span=6,
    ),
    "latching_fault_register": Metrics(
        False, nreg=0, bnreg=6, steering=2, score=14.0, max_read_port=0, max_write_select=0,
        copies=0, min_ii=5, last_pc=5, max_block_span=5,
    ),
    "majority_voter": Metrics(
        False, nreg=5, bnreg=11, steering=15, score=47.0, max_read_port=1, max_write_select=4,
        copies=0, min_ii=15, last_pc=20, max_block_span=11,
    ),
    # recip_newton's loop opens with a statically-true convergence test, so the partial evaluator peels the first
    # trip: one more live value across the loop entry (nreg, steering) and one body's worth of extra microcode, in
    # exchange for a shorter realized transaction (test_cycle_model).
    "recip_newton": Metrics(
        False, nreg=4, bnreg=1, steering=8, score=18.0, max_read_port=3, max_write_select=3,
        copies=1, min_ii=29, last_pc=46, max_block_span=23,
    ),
    "remainder": Metrics(
        False, nreg=5, bnreg=4, steering=16, score=34.0, max_read_port=3, max_write_select=3,
        copies=2, min_ii=37, last_pc=51, max_block_span=17,
    ),
    "octave_index": Metrics(
        False, nreg=3, bnreg=1, steering=5, score=13.0, max_read_port=2, max_write_select=3,
        copies=3, min_ii=14, last_pc=45, max_block_span=24,
    ),
    "cordic_sincos": Metrics(
        False, nreg=5, bnreg=1, steering=29, score=41.0, max_read_port=14, max_write_select=4,
        copies=0, min_ii=104, last_pc=104, max_block_span=104,
    ),
    "integrator": Metrics(
        True, nreg=4, bnreg=0, steering=5, score=13.0, max_read_port=2, max_write_select=2,
        copies=0, min_ii=16, last_pc=16, max_block_span=16,
    ),
    # The capability-probe controller: records, reductions, reshape, dtype conversions, and a branchy scan in one
    # kernel, so it gates the whole new-frontend surface against fabric regressions. The pairwise extrema over the
    # six active drives hold two partial maxima live where a fold held one, at no cost in latency: the dot products
    # bound that block, not the max.
    "finite_set_current_controller": Metrics(
        False, nreg=16, bnreg=4, steering=69, score=109.0, max_read_port=8, max_write_select=7,
        copies=12, min_ii=160, last_pc=204, max_block_span=108,
    ),
    # The heaviest matrix-library user (matmul, cross, norm, elementwise clamp) composed with real control flow,
    # so it is the gate that would catch a linear-algebra stub expanding into more hardware than it replaced. Its
    # three Euclidean norms carry an exponent extraction per leg and the scalings around it, which this row prices.
    "imu_fusion": Metrics(
        False, nreg=42, bnreg=5, steering=158, score=252.0, max_read_port=26, max_write_select=8,
        copies=14, min_ii=270, last_pc=464, max_block_span=139,
    ),
    # The two graduated filter examples: both straight-line, so every figure is one block's.
    "fir": Metrics(
        True, nreg=8, bnreg=0, steering=8, score=24.0, max_read_port=4, max_write_select=1,
        copies=0, min_ii=20, last_pc=20, max_block_span=20,
    ),
    "biquad": Metrics(
        True, nreg=5, bnreg=0, steering=6, score=16.0, max_read_port=3, max_write_select=2,
        copies=0, min_ii=21, last_pc=21, max_block_span=21,
    ),
    # The two largest kernels carry the highest register pressure: the uniform landing keeps min_ii/last_pc tight,
    # so a result resides a cycle longer, and the allocator spends a few more registers still to cut their read
    # muxes. The baselines are non-regression ceilings (`<=`) pinned tight to the converged build, so a later
    # improvement may sit below its bound until the next re-freeze.
    "ekf1_stateless": Metrics(
        True, nreg=44, bnreg=0, steering=91, score=179.0, max_read_port=26, max_write_select=3,
        copies=0, min_ii=125, last_pc=125, max_block_span=125,
    ),
    # The EKF with a second multiplier, the knob this allocator was built for: the co-issued products shorten the
    # transaction by 38 stages and the allocator binds them across the two instances (the scheduler's first-free binding
    # measured 140 arms here, the annealer 112). A monotonicity guard for the binding, not its proof -- the directed
    # kernels in test_regalloc.py are that.
    "ekf1_stateless_fmul2": Metrics(
        True, nreg=50, bnreg=0, steering=112, score=212.0, max_read_port=21, max_write_select=3,
        copies=0, min_ii=87, last_pc=87, max_block_span=87,
    ),
    "ekf1_stateful": Metrics(
        True, nreg=42, bnreg=0, steering=79, score=163.0, max_read_port=30, max_write_select=3,
        copies=0, min_ii=125, last_pc=125, max_block_span=125,
    ),
    # A deep composition: a nested component instance (the flux observer) whose state joins the controller's own,
    # every transcendental the library offers, and two data-dependent branches -- so it gates cross-component slot
    # allocation against the register and steering blowup that inlining a component can cause.
    "foc": Metrics(
        False, nreg=28, bnreg=3, steering=89, score=151.0, max_read_port=14, max_write_select=9,
        copies=4, min_ii=298, last_pc=349, max_block_span=231,
    ),
}
# fmt: on


@pytest.mark.parametrize("name", list(_EXAMPLES))
def test_metrics_do_not_regress(name: str) -> None:
    base = _BASELINE[name]
    got = _measure(name)
    assert got.straight_line == base.straight_line, f"{name}: control-flow classification changed"
    for field in (
        "nreg",
        "bnreg",
        "steering",
        "score",
        "max_read_port",
        "max_write_select",
        "copies",
        "min_ii",
        "last_pc",
        "max_block_span",
    ):
        assert getattr(got, field) <= getattr(
            base, field
        ), f"{name}: {field} regressed {getattr(base, field)} -> {getattr(got, field)}"


def test_build_is_deterministic() -> None:
    """The allocator's annealing is `seed=0`; two builds of the same kernel must agree, so the baseline is stable."""
    first = _measure("ekf1_stateless")
    second = _measure("ekf1_stateless")
    assert first == second


def test_constant_operands_are_read_arms() -> None:
    # A constant is an arm of the operand mux like any register. Three products of one input by three distinct
    # non-power-of-two constants (separate outputs, so the linear-combination fold cannot merge them; a power-of-two
    # scale would never reach the pool) put three constants on the multiplier's two ports, at least two of them
    # sharing a port whichever way the firings orient -- arms a register-only count reports as zero.
    def kernel(x: float) -> tuple[float, float, float]:
        return x * 1.3, x * 2.7, x * 3.9

    lir = _build(kernel, "three_scales", default_mir(_FMT))
    assert len(lir.wide_consts) == 3
    books = read_sources_per_port(lir)
    registers_only = sum(max(0, sum(isinstance(source, RegRef) for source in book) - 1) for book in books.values())
    assert sum(max(0, n - 1) for n in read_arms(lir).values()) - registers_only >= 2


def test_a_move_repeated_on_several_steps_is_one_write_arm() -> None:
    # The loop counter's initial value installs into its phi register from both arms of the real diamond ahead of the
    # loop (the divide keeps it a branch): two steps, one structurally identical move, one arm of the write select --
    # the emitter selects it by one opcode on both steps, so a per-step count overstates the mux.
    def kernel(x: float) -> int:
        if x >= 1.0:
            scaled = x
        else:
            scaled = 1.0 / x
        octaves = 0
        while scaled > 1.0:
            scaled = scaled * 0.5
            octaves = octaves + 1
        return octaves

    lir = _build(kernel, "counter_seed", default_mir(_FMT))
    moves: dict[RegRef | BoolRegRef, list[MoveWriteSource]] = {}
    for event in write_events(lir):
        if isinstance(event.source, MoveWriteSource):
            moves.setdefault(event.dst, []).append(event.source)
    repeated = [dst for dst, sources in moves.items() if len(sources) > len(set(sources))]
    assert repeated, "the premise needs one move source installed on several steps"
    arms = write_arms(lir)
    for dst in repeated:
        sources = [event.source for event in write_events(lir) if event.dst == dst]
        assert arms[dst] == len(set(sources)) < len(sources)

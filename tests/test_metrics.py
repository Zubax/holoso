"""
Steering/area non-regression gate for the LIR build.

Every currently-synthesizing example (all except ``iir1_hpf`` and ``finite_set_current_controller``, which the
frontend cannot yet lower) is built to LIR and measured on the metrics that bound the synthesized fabric (here
"straight-line" means the pure-float flat path: single block, no boolean fabric -- an if-converted kernel can be
single-block without being straight-line in this sense): the wide and
boolean register counts, the per-port read-mux fan-in and per-register write-select fan-in (the steering cost that
dominates the LUTs), and the statically-known latency lower bound. The baseline below was re-frozen on the converged
build with the bank-independent read/landing model, at the default register-allocation effort. Value numbering is
seed-independent (``tests/test_determinism.py`` proves byte-identical Verilog across ``PYTHONHASHSEED`` values), so
these figures hold in any process without pinning the hash seed.

The contract: NO example may regress past its frozen baseline on any metric. A single-block (straight-line) kernel is
expected to hit equality -- the unified allocator subsumes the former straight-line path. The control-flow rows
themselves encode the convergence win (cross-block reuse and coalescing collapsed the former fresh-per-value register
explosion), so any backslide toward it fails the same gate.

These read/write fan-in figures are exact steering proxies only for single-block kernels (the read-set/write-set
properties union across the flat op stream, which conflates mutually-exclusive blocks on a CFG); for control-flow
kernels the register counts are the directly-comparable figures, with the fan-in kept as a same-kernel before/after
monotonicity guard.
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from holoso import FloatFormat
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import Lir, build
from holoso._mir import lower as lower_to_mir
from ._modelref import default_ops

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import madd  # noqa: E402
import poly3  # noqa: E402
from cordic_sincos import CordicSinCos  # noqa: E402
from ekf1_stateful import Ekf1  # noqa: E402
from ekf1_stateless import update_x_P  # noqa: E402
from iir1_lpf import IIR1LPF  # noqa: E402
import imu_frame_transform  # noqa: E402
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

_FMT = FloatFormat(8, 36)

_EXAMPLES: dict[str, Callable[[], Callable[..., object]]] = {
    "madd": lambda: madd.madd,
    "poly3": lambda: poly3.poly3,
    "signal_window": lambda: signal_window,
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
    "imu_frame_transform": lambda: imu_frame_transform.transform,
    "ekf1_stateless": lambda: update_x_P,
    "ekf1_stateful": lambda: Ekf1().update,
}


@dataclass(frozen=True, slots=True)
class Metrics:
    """
    The non-regression figures sampled off a built :class:`Lir`.

    ``steering`` is the total sparse-regfile mux fan-in -- read-mux fan-in plus the upper-bound write-select fan-in
    (:attr:`Lir.write_select_fanin`, which counts every write-chain driver the backend synthesizes: pooled lanes,
    inline casts, phi-arm copies, and slot installs). Counting the copies matters here: phi-arm coalescing trades
    pc-gated copies for shared pooled writeback lanes, so a copy-blind proxy would mis-report a coalescing win as a
    regression. ``copies`` is the total phi-arm install count (float copies plus boolean writes), the direct measure
    of how many phi arms still install by copy rather than coalescing onto the merged register.

    ``last_pc`` is the static ROM length (:attr:`Lir.initiation_interval`) -- the total stage count: blocks tile the
    ROM, so a per-block drain regression in ANY block inflates it, the primary "excessive stages" guard.
    ``max_block_span`` is the largest per-block terminator offset (``max term_offset``), localizing a per-block drain
    regression to one block (unlike ``last_pc``, it does not move with the number of blocks).
    """

    straight_line: bool
    nreg: int
    bnreg: int
    steering: int
    copies: int
    min_ii: int
    last_pc: int
    max_block_span: int


@pytest.fixture(autouse=True)
def _pinned_regalloc_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Pin every register-allocation tuning knob to its shipped default so the frozen baselines are reproducible
    regardless of the developer's environment (``HOLOSO_REGALLOC_EFFORT`` speed-ups, write-cap/price experiments).
    The knobs are env-read-once at import, so the module attributes are patched to the named defaults; changing a
    default deliberately re-freezes the baselines.
    """
    import holoso._lir._regalloc as regalloc

    # Pinned to the knobs' default values (the getenv fallbacks in _regalloc), restated here as literals: the
    # baselines are frozen against the defaults, so an env override must not leak into this gate.
    monkeypatch.setattr(regalloc, "_REFINE_MAXITER", 5000)
    monkeypatch.setattr(regalloc, "_REG_REUSE_WRITE_CAP", 2)
    monkeypatch.setattr(regalloc, "_REG_PRICE", 2.0)


def _measure(name: str) -> Metrics:
    lir: Lir = build(lower_to_mir(optimize(lower(_EXAMPLES[name]())), default_ops(_FMT)), name, fetch_stages=3)
    straight = (
        len(lir.blocks) == 1
        and not lir.bool_state_slots
        and not any(b.inline_ops or b.copies or b.bool_writes for b in lir.blocks)
        and lir.bool_regfile.nreg == 0
    )
    read_fanin = sum(max(0, len(regs) - 1) for regs in lir.read_set_per_port.values())
    copies = sum(len(block.copies) + len(block.bool_writes) for block in lir.blocks)
    return Metrics(
        straight_line=straight,
        nreg=lir.regfile.nreg,
        bnreg=lir.bool_regfile.nreg,
        steering=read_fanin + lir.write_select_fanin,
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
#   any other operand muxes, and commutative orientation (the comparator swaps with its gt/lt tap exchange). steering
#   counts the upper-bound write-select fan-in (the emitter's full per-register write chain incl. phi-arm copies), not
#   a pooled-lane-only proxy. Cordic's read-mux steering sits a couple of arms above the best coloring in its
#   near-equal-cost band -- the disclosed price of deterministic value numbering; deeper annealing is optional.
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
#   diamonds become muxes -- a float select, or a bool_select reduced to and/or/not for a boolean/mixed merge),
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
#   select or a bool->float cast) or pooled -- lands at the same uniform per-op landing, and an install reading a
#   block-entry-RESIDENT source (``value_resident_at_entry``) fires at the
#   combinational step and drops its +1 install drain. last_pc tiles every block's span (a per-block drain regression
#   anywhere inflates it); max_block_span localizes it to one block. These timing rules move the schedule-length
#   guards but not nreg/bnreg/steering/copies; signal_window carries a deliberately-loosened steering arm (one freed
#   boolean register traded for one write-select mux) -- refrozen rather than chased, since the rules are global and
#   correctness-neutral.
# - pid uses a variable sample interval: the derivative path contains a real divide, and the first-sample and saturation
#   guards keep the kernel multi-block with one residual copy. The larger PID row is therefore a property of the example
#   itself, not a scheduler regression to chase.
BASELINE: dict[str, Metrics] = {
    "madd": Metrics(True, nreg=4, bnreg=0, steering=3, copies=0, min_ii=15, last_pc=15, max_block_span=15),
    "poly3": Metrics(True, nreg=5, bnreg=0, steering=5, copies=0, min_ii=24, last_pc=24, max_block_span=24),
    "signal_window": Metrics(False, nreg=4, bnreg=5, steering=8, copies=0, min_ii=10, last_pc=10, max_block_span=10),
    "iir1_lpf": Metrics(False, nreg=3, bnreg=2, steering=2, copies=0, min_ii=16, last_pc=16, max_block_span=16),
    "pid": Metrics(False, nreg=10, bnreg=2, steering=13, copies=1, min_ii=38, last_pc=71, max_block_span=32),
    "schmitt_trigger": Metrics(False, nreg=1, bnreg=2, steering=2, copies=0, min_ii=7, last_pc=7, max_block_span=7),
    "quadrature_encoder": Metrics(False, nreg=1, bnreg=7, steering=7, copies=0, min_ii=6, last_pc=6, max_block_span=6),
    "phase_frequency_detector": Metrics(
        False, nreg=0, bnreg=5, steering=5, copies=0, min_ii=6, last_pc=6, max_block_span=6
    ),
    "latching_fault_register": Metrics(
        False, nreg=1, bnreg=6, steering=2, copies=0, min_ii=6, last_pc=6, max_block_span=6
    ),
    "majority_voter": Metrics(False, nreg=1, bnreg=21, steering=20, copies=0, min_ii=14, last_pc=19, max_block_span=12),
    "recip_newton": Metrics(False, nreg=4, bnreg=1, steering=4, copies=2, min_ii=15, last_pc=32, max_block_span=16),
    "remainder": Metrics(False, nreg=8, bnreg=4, steering=12, copies=2, min_ii=39, last_pc=58, max_block_span=17),
    "octave_index": Metrics(False, nreg=3, bnreg=1, steering=6, copies=3, min_ii=16, last_pc=51, max_block_span=25),
    "cordic_sincos": Metrics(
        False, nreg=7, bnreg=1, steering=53, copies=0, min_ii=105, last_pc=105, max_block_span=105
    ),
    "integrator": Metrics(True, nreg=5, bnreg=0, steering=4, copies=0, min_ii=17, last_pc=17, max_block_span=17),
    # The only example whose datapath is built entirely from the matrix product and transpose, so it is the gate that
    # would catch a linear-algebra library stub expanding into more hardware than the operators it replaced.
    "imu_frame_transform": Metrics(
        True, nreg=20, bnreg=0, steering=35, copies=0, min_ii=42, last_pc=42, max_block_span=42
    ),
    # The two largest kernels carry slightly higher register pressure as a deliberate latency-for-area point: the
    # uniform landing keeps min_ii/last_pc tight, so a result resides a cycle longer, raising register
    # pressure (nreg, and ekf1_stateless's steering with it). The baselines are non-regression ceilings (``<=``) pinned
    # tight to the converged build, so a later improvement may sit below its bound until the next re-freeze.
    "ekf1_stateless": Metrics(
        True, nreg=41, bnreg=0, steering=100, copies=0, min_ii=126, last_pc=126, max_block_span=126
    ),
    "ekf1_stateful": Metrics(
        True, nreg=40, bnreg=0, steering=89, copies=0, min_ii=128, last_pc=128, max_block_span=128
    ),
}


@pytest.mark.parametrize("name", list(_EXAMPLES))
def test_metrics_do_not_regress(name: str) -> None:
    base = BASELINE[name]
    got = _measure(name)
    assert got.straight_line == base.straight_line, f"{name}: control-flow classification changed"
    for field in ("nreg", "bnreg", "steering", "copies", "min_ii", "last_pc", "max_block_span"):
        assert getattr(got, field) <= getattr(
            base, field
        ), f"{name}: {field} regressed {getattr(base, field)} -> {getattr(got, field)}"


def test_build_is_deterministic() -> None:
    """The allocator's annealing is ``seed=0``; two builds of the same kernel must agree, so the baseline is stable."""
    first = _measure("ekf1_stateless")
    second = _measure("ekf1_stateless")
    assert first == second

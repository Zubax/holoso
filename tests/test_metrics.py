"""
Steering/area non-regression gate for the LIR build.

Every currently-synthesizing example (all except ``iir1_hpf`` and ``finite_set_current_controller``, which the
frontend cannot yet lower) is built to LIR and measured on the metrics that bound the synthesized fabric (here
"straight-line" means the pure-float flat path: single block, no boolean fabric -- an if-converted kernel can be
single-block without being straight-line in this sense): the wide and
boolean register counts, the per-port read-mux fan-in and per-register write-select fan-in (the steering cost that
dominates the LUTs), and the statically-known latency lower bound. The baseline below was re-frozen on the converged
build with the per-bank timing model, at the default register-allocation effort. Value numbering is
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
    "recip_newton": lambda: NewtonReciprocal().__call__,
    "remainder": lambda: remainder,
    "octave_index": lambda: octave_index,
    "cordic_sincos": lambda: CordicSinCos().__call__,
    "integrator": lambda: TrapezoidalLeakyStreamingIntegrator(k=2**-22).__call__,
    "ekf1_stateless": lambda: update_x_P,
    "ekf1_stateful": lambda: Ekf1().update,
}


@dataclass(frozen=True, slots=True)
class Metrics:
    """
    The non-regression figures sampled off a built :class:`Lir`.

    ``steering`` is the total sparse-regfile mux fan-in -- read-mux fan-in plus the GROUND-TRUTH write-select fan-in
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
    lir: Lir = build(lower_to_mir(optimize(lower(_EXAMPLES[name]())), default_ops(_FMT)), name)
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


# Each row is an upper bound (a build must be <= every field), frozen on the current build. What explains the
# figures, per mechanism rather than per re-freeze:
#
# - Registers and steering reflect the unified cross-block allocator: liveness-bounded reuse with coalesced state
#   slots, the per-(instance, port) lane accounting of BOTH banks' write selects, the comparator's read ports
#   counted (and steered) like any other operand muxes, and commutative orientation (the comparator swaps with its
#   gt/lt tap exchange). steering now counts the GROUND-TRUTH write-select fan-in (the emitter's full per-register
#   write chain incl. phi-arm copies), so the figures are larger than the former pooled-lane-only proxy but reflect
#   real hardware. Cordic's read-mux steering sits a couple of arms above the best coloring observed in its
#   near-equal-cost band -- the disclosed price of deterministic value numbering; deeper annealing is optional.
# - copies and the phi-dense steering/registers reflect phi-arm coalescing (M5): a phi whose register-backed,
#   identity-conditioner arms do not interfere with it (the install-free oracle decides) shares their register, so
#   the install copy vanishes. iir1_lpf's two slot-feeding copies, remainder's diamond copies, and the boolean
#   phi-arm writes of schmitt/quadrature/pfd coalesce away, dropping copies AND registers (recip_newton 5->4 wide,
#   quadrature 13->8 bool) AND the true write-select fan-in (remainder 10->7); recip_newton keeps its one
#   loop-carried copy (its phi overlaps the back-edge arm), proving the oracle refuses unsound merges.
# - min_ii reflects the bank-true dependency edges (latch-free boolean bank), diamond if-conversion (small pure
#   branch diamonds become muxes -- a float select or, for a boolean/mixed merge, a bool_select reduced to and/or/not:
#   signal_window/pid/cordic and now schmitt/pfd/iir1_lpf collapse to a single block; quadrature collapses partway
#   bounded by the per-arm op budget; remainder converts its float AND bool diamonds, keeping only its while-loop
#   branches), NOT-folding (a semantic NOT is a free consumer-side inversion), and cross-block software pipelining
#   (a branch block whose successors are all single-predecessor shrinks its terminator below the drained boundary).
#   Bool-phi if-conversion runs both arms unconditionally, so it can RAISE min_ii (the shortest static path) while
#   LOWERING realized per-transaction latency -- the true goal, guarded by test_cycle_model's realized-latency test:
#   iir1_lpf's rare cheap first-iteration path disappears (min_ii 15->21) but its steady-state latency drops 30->20;
#   schmitt 27->7, pfd 48->8, quadrature 32->21, remainder every path down. Coalescing is layout-neutral, so it does
#   not move min_ii. The register/steering side of the same trade: both arms' values are simultaneously live, so a
#   converted diamond can need an extra register (remainder nreg 6->8, schmitt bnreg 2->3) -- the cost of the mux.
# - bnreg reflects exact per-consumer boolean read steps and phi coalescing: a condition consumed mid-block frees
#   its register for a later value in the same block, and a boolean phi merging onto its arms drops its own register.
# - last_pc and max_block_span are the stage-count guards (the "excessive number of stages" gate). They reflect the two
#   per-block drain tighteners: (1) the bank-aware drained boundary -- a drained block carrying only boolean values at
#   its boundary AND no tail install drains one fetch step earlier than a wide one (a pc-gated install lands at the wide
#   boundary regardless of bank, so an install-bearing block keeps the wide drain): quadrature last_pc 28->27 (only its
#   install-free Ret tightens; its bool-install blocks stay wide), while schmitt min_ii 8->7 and pfd 9->8 tighten on
#   their install-free boolean-output single blocks; and (2) the coalesced-install fixpoint -- a phi-arm predecessor
#   whose every arm coalesces installs nothing, so its +1 install drain is dropped (remainder last_pc 73->71 on its two
#   fully-coalesced diamond arms). last_pc tiles every block's span, so a per-block drain regression anywhere inflates
#   it; max_block_span localizes it to one block.
BASELINE: dict[str, Metrics] = {
    "madd": Metrics(True, nreg=4, bnreg=0, steering=3, copies=0, min_ii=20, last_pc=20, max_block_span=20),
    "poly3": Metrics(True, nreg=5, bnreg=0, steering=5, copies=0, min_ii=35, last_pc=35, max_block_span=35),
    "signal_window": Metrics(False, nreg=4, bnreg=6, steering=7, copies=0, min_ii=13, last_pc=13, max_block_span=13),
    "iir1_lpf": Metrics(False, nreg=3, bnreg=2, steering=3, copies=0, min_ii=21, last_pc=21, max_block_span=21),
    "pid": Metrics(False, nreg=9, bnreg=3, steering=10, copies=0, min_ii=40, last_pc=40, max_block_span=40),
    "schmitt_trigger": Metrics(False, nreg=1, bnreg=3, steering=2, copies=0, min_ii=7, last_pc=7, max_block_span=7),
    "quadrature_encoder": Metrics(
        False, nreg=1, bnreg=8, steering=14, copies=4, min_ii=17, last_pc=27, max_block_span=9
    ),
    "phase_frequency_detector": Metrics(
        False, nreg=1, bnreg=8, steering=6, copies=0, min_ii=8, last_pc=8, max_block_span=8
    ),
    "recip_newton": Metrics(False, nreg=4, bnreg=1, steering=4, copies=2, min_ii=25, last_pc=51, max_block_span=25),
    "remainder": Metrics(False, nreg=8, bnreg=4, steering=12, copies=2, min_ii=50, last_pc=71, max_block_span=22),
    "octave_index": Metrics(False, nreg=3, bnreg=1, steering=6, copies=3, min_ii=22, last_pc=60, max_block_span=27),
    "cordic_sincos": Metrics(
        False, nreg=9, bnreg=3, steering=54, copies=0, min_ii=150, last_pc=150, max_block_span=150
    ),
    "integrator": Metrics(True, nreg=5, bnreg=0, steering=4, copies=0, min_ii=24, last_pc=24, max_block_span=24),
    "ekf1_stateless": Metrics(
        True, nreg=39, bnreg=0, steering=95, copies=0, min_ii=129, last_pc=129, max_block_span=129
    ),
    "ekf1_stateful": Metrics(
        True, nreg=38, bnreg=0, steering=89, copies=0, min_ii=132, last_pc=132, max_block_span=132
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

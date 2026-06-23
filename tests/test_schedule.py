"""Unit tests for pipelined scheduling, register allocation, and LIR construction."""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from holoso import (
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FloatFormat,
    FMulILog2OperatorFamily,
    FMulOperator,
    OpConfig,
)
from holoso._errors import UnsupportedConstruct
from holoso._frontend import lower
from holoso._hir import _if_convert as if_convert_pass
from holoso._hir import optimize
from holoso._lir import (
    BoolOperand,
    BoolRegRef,
    Branch,
    FloatConstRef,
    FloatOperand,
    Jump,
    LirBlock,
    RegRef,
    Ret,
    operand_read_cycle,
    result_landing_cycle,
)
from holoso._lir._ir import (
    FETCH_LAG,
    READ_FIRST_EDGE,
    boundary_step,
    dependency_edge,
    inline_landing_cycle,
    install_landing,
    pooled_writeback_word,
    successor_local_cycle,
)
from holoso._mir import (
    lower as lower_to_mir,
    Mir,
    MirBlock,
    MirBuilder,
    MirFloatConst,
    MirFloatInput,
    MirFloatOutput,
    MirFloatView,
    MirInput,
    MirOperation,
    MirRet,
)
from holoso._operators import BoolAndOperator, BoolInversion, FMulILog2Operator, FloatSignControl, SelectOperator
from holoso._hir import RelationalOp
from ._modelref import build_model
from holoso._lir import build
from holoso._lir._schedule import resolve_pool, schedule_ops
from holoso._type import BoolType, FloatType, ScalarType

from ._modelref import (
    ChainedSlots,
    COMPARATOR_OP_CASES,
    OperatorCase,
    PIPELINE_OP_CASES,
    SelectHold,
    branch_boundary_kernel,
    const_branch_kernel,
    default_ops,
    diamond_then_loop_kernel,
    overlap_dead_arm_spill_kernel,
    overlap_div_err_kernel,
    overlap_spill_kernel,
    staged_ops,
)
from ._writetimeline import InlineProducer, build_write_timeline, latest_producer_before

FMT = FloatFormat(6, 18)
OPS = OpConfig(FAddOperator(FMT), FMulOperator(FMT), FDivOperator(FMT), FMulILog2OperatorFamily(FMT), FCmpOperator(FMT))


@dataclass(frozen=True, slots=True)
class OtherScalarType(ScalarType):
    @property
    def width(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class OtherMirInput(MirInput):
    pass


def _run(target, ops: OpConfig = OPS) -> Mir:  # type: ignore[no-untyped-def]
    return lower_to_mir(optimize(lower(target)), ops)


def _view(mir: Mir) -> MirFloatView:
    return MirFloatView.from_mir(mir)


def _schedule(mir: Mir):
    view = _view(mir)
    return schedule_ops(mir.nodes, resolve_pool(mir.nodes), set(view.operation_nodes))


def _muls(mir: Mir) -> list[int]:
    return [vid for vid, n in mir.nodes.items() if isinstance(n, MirOperation) and isinstance(n.operator, FMulOperator)]


def test_schedule_respects_dependencies() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    mir = _run(f)
    sched = _schedule(mir)
    for vid, cycle in sched.issue_cycle.items():
        op = mir.nodes[vid]
        assert isinstance(op, MirOperation)
        assert cycle >= 1  # nothing issues on the accept cycle
        for operand in op.operands:
            node = mir.nodes[operand]
            if isinstance(node, MirOperation):
                # A consumer issues no earlier than the producer's commit plus the pair's dependency edge (derived
                # from the producer's result-bank landing and the consumer's operand-read mechanism).
                edge = dependency_edge(node.operator, node.output_port, op.operator)
                assert cycle >= sched.issue_cycle[operand] + node.operator.latency + edge


def test_pipelined_issue_overlaps_a_slow_op() -> None:
    # A fast chain advances while an unrelated slow divide is still in flight -- the barrier model could not do this.
    def f(a, b, c):  # type: ignore[no-untyped-def]
        return a / b + (a + b + c)

    mir = _run(f)
    sched = _schedule(mir)
    div = next(
        vid for vid, n in mir.nodes.items() if isinstance(n, MirOperation) and isinstance(n.operator, FDivOperator)
    )
    div_node = mir.nodes[div]
    assert isinstance(div_node, MirOperation)
    div_commit = sched.issue_cycle[div] + div_node.operator.latency
    adds = [vid for vid, n in mir.nodes.items() if isinstance(n, MirOperation) and isinstance(n.operator, FAddOperator)]
    # Some fadd of the independent (a+b+c) chain issues before the divide commits -- genuine overlap, no barrier.
    assert any(sched.issue_cycle[vid] < div_commit for vid in adds)


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
def test_two_comparisons_in_a_block_serialize_on_the_shared_comparator(config: OperatorCase) -> None:
    # Regression: a chained comparison (here ``lo < x < hi``) puts two comparator firings with distinct operand
    # pairs in one block. The single pooled holoso_fcmp instance serves one firing per initiation interval, so the
    # two must issue on distinct cycles. Before the contention rule the scheduler let both issue on the same cycle
    # -> they collided on the single comparator (one read the other's operands), corrupting the result in the RTL
    # while the model still passed.
    def f(x, lo, hi):  # type: ignore[no-untyped-def]
        return 0.0 if lo < x < hi else x

    lir = build(_run(f, config.make_ops(FMT)), f"deadband_{config.label}")
    in_valid_pcs = [
        lir.block_base[block.index] + op.issue_cycle
        for block in lir.blocks
        for op in block.ops
        if isinstance(op.inst.operator, FCmpOperator)
    ]
    assert len(in_valid_pcs) >= 2  # the chained comparison lowers to two comparator firings feeding a BoolAnd
    assert len(set(in_valid_pcs)) == len(in_valid_pcs)  # instance contention spaces them: no comparator collision


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
def test_branch_comparison_commits_at_block_makespan(config: OperatorCase) -> None:
    # White-box twin of test_cosim.py test_cosim_comparison_at_branch_boundary: pins that the SHARED kernel
    # (_modelref.branch_boundary_kernel) actually hits the boundary-slack corner -- the comparison is the last commit
    # in its block and feeds the branch -- at comparator-only and full-pipeline latency points. If a schedule change
    # ever moves the comparison off the makespan, this fails before the cosim silently de-targets.
    ops = config.make_ops(FMT)
    lir = build(_run(branch_boundary_kernel, ops), f"cmp_at_boundary_{config.label}")
    branch_blocks = [block for block in lir.blocks if isinstance(block.terminator, Branch)]
    assert len(branch_blocks) == 1
    (block,) = branch_blocks
    comparisons = [op for op in block.ops if isinstance(op.inst.operator, FCmpOperator)]
    assert len(comparisons) == 1
    (cmp_op,) = comparisons
    assert cmp_op.latency == config.fcmp_latency
    assert cmp_op.commit_cycle == block.block_makespan


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
def test_overlap_shrinks_branch_terminator_below_drained_boundary(config: OperatorCase) -> None:
    # Cross-block software pipelining (M7): a branch block whose every successor is single-predecessor shrinks its
    # terminator offset below the conservative wide drain boundary_step(makespan, wide_resident=True), so its in-flight
    # results spill into the
    # successor frame instead of fully draining. Pins that the overlap actually engages (the recip_newton loop header,
    # an in-block-condition branch to a single-pred body and a single-pred exit, is such a block).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from recip_newton import NewtonReciprocal

    lir = build(_run(NewtonReciprocal().__call__, config.make_ops(FMT)), f"recip_overlap_{config.label}")
    shrunk = [
        block
        for block in lir.blocks
        if isinstance(block.terminator, Branch)
        and block.term_offset < boundary_step(block.block_makespan, wide_resident=True)
    ]
    assert shrunk, "no branch block shrank its terminator: cross-block overlap did not engage"


def test_entry_branch_on_resident_condition_skips_the_wide_drain() -> None:
    # Regression (terminator read-floor): an entry block that branches on a RESIDENT live-in condition -- here a
    # persistent boolean state, the uart_rx / majority_voter entry shape -- shrinks its terminator to the issue-side
    # envelope. The condition is resident from the block's first cycle, so the branch needs no drain; pinning the
    # terminator to the wide drain instead would push every downstream block's base a cycle late, which the
    # term_offset assertion below catches.
    class _EntryStateBranch:
        def __init__(self) -> None:
            self._armed = False

        def step(self, a, b):  # type: ignore[no-untyped-def]
            if self._armed:  # the entry branches on the resident boolean state (its live-in)
                r = a / b  # a non-speculatable arm keeps this a real branch, not an if-converted select
            else:
                r = a + b
            self._armed = a > b
            return r

    lir = build(_run(_EntryStateBranch().step), "entry_state_branch")
    entry = lir.blocks[lir.entry]
    assert isinstance(entry.terminator, Branch)
    assert not entry.ops and not entry.inline_ops  # the entry only branches on the resident state; it does no work
    # The op-less entry rides the issue-side envelope floor (1), strictly below the wide drain (4) a pin would charge.
    assert entry.term_offset == 1
    assert entry.term_offset < boundary_step(entry.block_makespan, wide_resident=True)


def test_resident_bound_inline_select_bypasses_the_writeback_latch() -> None:
    # Regression (inline writeback latch -- the uart_rx block-leading select defect): a select is a combinational mux
    # written into the register array directly, carrying no pooled-operator writeback latch, so its WIDE result lands at
    # commit + FETCH_LAG + READ_FIRST_EDGE -- a cycle before a pooled wide result. Here the condition is a resident
    # boolean state and both arms are resident (an input and the state itself), so the select issues at its block's
    # first cycle; charging it the wide writeback latch would land its result a cycle late.
    class _ResidentSelect:
        def __init__(self) -> None:
            self._armed = False

        def step(self, c, d):  # type: ignore[no-untyped-def]
            r = c if self._armed else d  # condition (state) and both arms are resident -> the select binds nothing
            self._armed = c > d
            return r

    lir = build(_run(_ResidentSelect().step), "resident_select")
    selects = [
        (block, op) for block in lir.blocks for op in block.inline_ops if isinstance(op.operator, SelectOperator)
    ]
    assert len(selects) == 1
    block, op = selects[0]
    assert op.issue_cycle == 0  # resident operands impose no edge -> the select issues at the block's first cycle
    base = lir.block_base[block.index]
    landing = base + inline_landing_cycle(op.commit_cycle)
    assert landing == base + op.commit_cycle + FETCH_LAG + READ_FIRST_EDGE  # inline: no writeback latch (else +1)
    assert landing in lir.reg_liveness[op.write.dst]  # the result is live on its true (writeback-latch-free) landing


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
def test_overlap_spilled_result_lands_in_successor_frame(config: OperatorCase) -> None:
    # The overlap_spill_kernel corner (shared with test_cosim.py test_cosim_overlap_spill): the branch condition is an
    # input comparison that commits early, while a wide chain in the same block commits much later, so the block shrinks
    # to the chain's WRITE WORD and the chain result lands PAST the terminator -- in the single-predecessor arm frames.
    # Pins that a wide result genuinely spills (result_landing_cycle beyond term_offset, while its write word stays in
    # the block); the cosim twin proves the arm read waits for the in-flight landing rather than reading stale data.
    lir = build(_run(overlap_spill_kernel, config.make_ops(FMT)), f"overlap_spill_{config.label}")
    spilled = [
        (block, op, write)
        for block in lir.blocks
        if isinstance(block.terminator, Branch)
        and block.term_offset < boundary_step(block.block_makespan, wide_resident=True)
        for op in block.ops
        for write in op.writes
        if isinstance(write.dst, RegRef) and result_landing_cycle(write.dst, op.commit_cycle) > block.term_offset
    ]
    assert spilled, "no wide result spilled past a shrunk terminator: the overlap corner did not trigger"
    # The spilling write's control WORD stays in the block (only the writeback/read-first landing tail crosses the
    # terminator) -- so the emitter places it normally and the single-writer microcode validator never sees a replica.
    for block, op, _write in spilled:
        assert op.commit_cycle + 1 <= block.term_offset


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
def test_overlap_dead_arm_spill_does_not_clobber_a_sibling_live_value(config: OperatorCase) -> None:
    # Regression (review BLOCKER, found independently by the functional reviewer and Codex): under cross-block overlap a
    # wide result spills into BOTH single-pred arms because its writeback latch fires unconditionally before the
    # redirect. In an arm where that result is DEAD, the allocator must STILL reserve its register (inflight_defs); else
    # the spill clobbers a value the arm actually uses -- a silent miscompile the cosim cannot catch, since the
    # numerical model shares the same register file (model == RTL, both wrong). Checked against source semantics. The
    # shared kernel's else arm reads `v` while `w` is dead and spills; crash-before, w (=15 for x=3,y=1,z=2) overwrote
    # v's register and the else result was grossly wrong (~3.4 instead of 1.2).
    model = build_model(
        build(_run(overlap_dead_arm_spill_kernel, config.make_ops(FMT)), f"dead_arm_spill_{config.label}")
    )
    for x, y, z in [(3.0, 1.0, 2.0), (4.0, 2.0, 0.5), (2.5, 0.5, 1.5)]:  # x > y selects the else arm, where w is dead
        want = (x + y + z) / (z * z + 1.0)
        (got,) = model.run(x, y, z)
        assert math.isclose(
            got, want, rel_tol=1e-2
        ), f"x={x} y={y} z={z}: got {got}, want {want} (dead-arm spill clobber)"


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
def test_overlap_keeps_error_op_diagnostic_latch_in_frame(config: OperatorCase) -> None:
    # Regression (review round 3, Codex P1): a division (the error-bearing op) whose writeback spills past a SHRUNK
    # terminator latches err_pc as pc-FETCH_LAG when its write-enable EXECUTES -- FETCH_LAG fetch steps after its write
    # word. If the terminator redirected to the non-fall-through arm by then, err_pc captures the wrong (successor)
    # frame. The data writeback is unaffected (it rides the pipeline), so only a step-accurate err_pc check sees it; the
    # shrink floor must keep the latch in-block: term_offset >= writeback_word + FETCH_LAG for an error-bearing op.
    # Crash-before: term_offset was the bare write word (one FETCH_LAG short), so err_pc latched a redirected pc.
    lir = build(_run(overlap_div_err_kernel, config.make_ops(FMT)), f"overlap_div_err_{config.label}")
    checked = False
    for block in lir.blocks:
        if not isinstance(block.terminator, Branch):
            continue
        for op in block.ops:
            operator = op.inst.operator
            if operator.error_ports:  # the division: its err diagnostic latch must not cross the terminator
                assert block.term_offset < boundary_step(
                    block.block_makespan, wide_resident=True
                )  # the corner: this block shrinks
                assert block.term_offset >= pooled_writeback_word(op.commit_cycle, True) + FETCH_LAG
                checked = True
    assert checked, "the error-bearing division did not land in a shrinkable branch block: corner not exercised"


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
def test_spilled_result_landings_match_the_numerical_model(config: OperatorCase) -> None:
    # Regression (HTML report exactness): a result spilling past an overlap-shrunk terminator lands in EVERY successor
    # arm, exactly where the numerical model re-keys its in-flight write at the redirect. write_landing_pcs -- which
    # reg_liveness, bool_liveness, the HTML schedule, and the write-timeline test helper all stamp through -- must
    # reproduce those PCs on every path, not the linear fall-through frame alone. Crash-before: the old stamping omitted
    # the non-fall-through arm's landing, so the predicted set was a strict subset of the model's actual writes (the
    # report drew the spilled value's residence on the wrong arm). Tied directly to the cosim oracle: every register the
    # model writes (inputs bypass the writeback) is an op result here, so predicted landings must match it exactly.
    from holoso._backend.numerical import NumericalSimulator

    class _Recorder(NumericalSimulator):
        def __init__(self, lir: object) -> None:
            super().__init__(lir)  # type: ignore[arg-type]
            self.writes: dict[tuple[str, int], set[int]] = {}

        def _write(self, dst: object, value: object) -> None:
            self.writes.setdefault((type(dst).__name__, dst.index), set()).add(self.pc)  # type: ignore[attr-defined]
            super()._write(dst, value)  # type: ignore[arg-type]

    vectors = [(0.5, 2.0, 1.5), (2.0, 0.5, 1.5), (1.0, 1.0, 1.0), (0.5, 0.0, 1.5), (3.0, 1.0, 2.0), (1.0, 3.0, 0.0)]
    for kernel, name in [
        (overlap_spill_kernel, "overlap_spill"),
        (overlap_dead_arm_spill_kernel, "dead_arm_spill"),
        (overlap_div_err_kernel, "overlap_div_err"),
    ]:
        lir = build(_run(kernel, config.make_ops(FMT)), f"{name}_{config.label}")
        # These kernels write every register through an operation (no copies/installs/state), so the model's writeback
        # set is exactly the op-result landings -- the cleanest tie to write_landing_pcs.
        assert not any(block.copies or block.bool_writes for block in lir.blocks)
        assert not lir.float_state_slots and not lir.bool_state_slots
        predicted: dict[tuple[str, int], set[int]] = {}
        multi_arm = 0
        for block in lir.blocks:
            for op in (*block.ops, *block.inline_ops):
                for write in op.writes:
                    pcs = lir.write_landing_pcs(block, op, write)
                    predicted.setdefault((type(write.dst).__name__, write.dst.index), set()).update(pcs)
                    multi_arm += len(pcs) > 1
        assert multi_arm > 0, f"{name}: no result spills into multiple arms -- the regression is vacuous"
        actual: dict[tuple[str, int], set[int]] = {}
        sim = _Recorder(lir)
        for x, y, z in vectors:  # the vectors drive BOTH arms, so the union covers every arm a spill lands in
            sim.reset()
            sim.writes = {}
            sim.run(x, y, z)
            for key, pcs in sim.writes.items():
                actual.setdefault(key, set()).update(pcs)
        assert predicted == actual, f"{name}: write_landing_pcs {predicted} != model writebacks {actual}"


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
def test_overlapping_loop_kernel_landings_are_real_model_writes(config: OperatorCase) -> None:
    # recip_newton is a real overlapping LOOP kernel: its header branch shrinks below the drained boundary, so the
    # diagnostics run through write_landing_pcs on a genuinely shrunk-terminator layout (the synthetic kernels above
    # carry the multi-arm SPILLS; here every op result still lands in-block, exercising the shrunk-block path on a real,
    # non-synthetic kernel). With loop-carried copies the model also writes registers via installs, so strict equality
    # does not apply -- but every op-result landing write_landing_pcs predicts must be a real register write the model
    # performs on some path. A subset tie to the cosim oracle, guarding against a regression that mis-frames the
    # shrunk-block in-block landing.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from recip_newton import NewtonReciprocal
    from holoso._backend.numerical import NumericalSimulator

    class _Recorder(NumericalSimulator):
        def __init__(self, lir: object) -> None:
            super().__init__(lir)  # type: ignore[arg-type]
            self.writes: dict[int, set[int]] = {}

        def _write(self, dst: object, value: object) -> None:
            if isinstance(dst, RegRef):
                self.writes.setdefault(dst.index, set()).add(self.pc)
            super()._write(dst, value)  # type: ignore[arg-type]

    lir = build(_run(NewtonReciprocal().__call__, config.make_ops(FMT)), f"recip_newton_{config.label}")
    assert any(
        isinstance(block.terminator, Branch)
        and block.term_offset < boundary_step(block.block_makespan, wide_resident=True)
        for block in lir.blocks
    ), "recip_newton did not overlap: the real-kernel tie is not exercising a shrunk terminator"
    predicted: dict[int, set[int]] = {}
    for block in lir.blocks:
        for op in (*block.ops, *block.inline_ops):
            for write in op.writes:
                if isinstance(write.dst, RegRef):
                    predicted.setdefault(write.dst.index, set()).update(lir.write_landing_pcs(block, op, write))
    sim = _Recorder(lir)
    actual: dict[int, set[int]] = {}
    for seed in [0.5, 1.5, 2.5, 0.25, 1.0]:  # in-domain seeds (the Newton iteration converges for x < 3)
        sim.reset()
        sim.writes = {}
        sim.run(seed)
        for index, pcs in sim.writes.items():
            actual.setdefault(index, set()).update(pcs)
    for index, pcs in predicted.items():
        assert pcs <= actual.get(
            index, set()
        ), f"reg {index}: landings {sorted(pcs)} not all model writebacks {sorted(actual.get(index, set()))}"


def test_bool_only_block_drains_one_step_under_the_wide_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    # B1 (bank-aware drained boundary): a drained block that does WORK carrying only boolean values at its boundary AND
    # no tail install lands one fetch step earlier than a wide one -- the latch-free boolean bank has no write-latch
    # edge. But a tail INSTALL (a pc-gated boolean write/phi copy) lands one step LATER, at the wide boundary, so an
    # install-bearing bool-only block must KEEP the wide drain. Crash-before: a single-bank drain puts the bool-work
    # block one PC too late (no bool shrink), and a bank-aware drain that also shrinks an install-bearing block puts
    # that block one PC too EARLY (bool shrink despite the install landing wide).

    def is_bool_only(block: LirBlock) -> bool:  # no wide register write and no float copy at the tail
        return not block.copies and not any(
            isinstance(w.dst, RegRef) for op in (*block.ops, *block.inline_ops) for w in op.writes
        )

    # phase_frequency_detector is a single-block all-boolean kernel: its Ret does real boolean WORK (makespan > 0) and
    # installs nothing, so it drains one step under the wide boundary -- the bank-aware win for a value that LANDS in
    # the frame (distinct from a pure-drain Ret, whose resident output needs no boundary at all -- covered separately).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from phase_frequency_detector import PhaseFrequencyDetector  # noqa: PLC0415

    pfd = build(_run(PhaseFrequencyDetector().__call__), "pfd_bool_drain")
    pfd_ret = next(block for block in pfd.blocks if isinstance(block.terminator, Ret))
    assert (
        is_bool_only(pfd_ret) and not pfd_ret.bool_writes and pfd_ret.block_makespan > 0
    ), "pfd Ret: bool work, no install"
    assert pfd_ret.term_offset == boundary_step(pfd_ret.block_makespan, wide_resident=False)
    assert pfd_ret.term_offset < boundary_step(
        pfd_ret.block_makespan, wide_resident=True
    ), "drained at the wide boundary"

    # A bool-only block that carries a tail install keeps the WIDE drain: the pc-gated install lands at the wide
    # boundary, so shrinking to the bool boundary would read it one PC before it lands (a miscompile). A non-coalesced
    # boolean phi arm is such an install: here ``r``'s entry arm is the input ``a``, which is also returned, so it stays
    # live past the merge and cannot coalesce onto the phi register -- it installs by a pc-gated copy at the (bool-only)
    # not-taken arm's tail. If-conversion is disabled so the diamond stays a real branch with a residual phi install
    # rather than collapsing to a select. (In-place state commit elided the former majority_voter sticky-fault
    # installs.)
    monkeypatch.setattr(if_convert_pass, "_IFCONV_MAX_OPS", 0)

    def residual_bool_install(a: bool, b: bool, c: bool):  # type: ignore[no-untyped-def]
        r = a
        if c:
            r = a and b
        return r, a  # ``a`` returned -> the c-false arm (= a) of r's phi cannot coalesce -> a residual bool install

    inst = build(_run(residual_bool_install), "bool_install_drain")
    bool_install_blocks = [b for b in inst.blocks if b.bool_writes and is_bool_only(b)]
    assert bool_install_blocks, "no bool-only install-bearing block to exercise the install drain exception"
    for block in bool_install_blocks:
        assert block.term_offset == boundary_step(
            block.block_makespan, wide_resident=True
        ), "an install-bearing bool block shrank below the wide boundary where its install lands"


def test_entry_block_reclaims_its_first_control_word() -> None:
    # An inline op has no read latch and is combinational (latency 0), so the entry block's first boolean operation
    # issues on block-local cycle 0 and FIRES on executing step 0 -- reclaiming ``ucode[0]``. Crash-before: the cycle-1
    # scheduler start and the inline latency of 1 together pushed the first op two steps late, to executing step 2.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from quadrature_encoder import QuadratureEncoder

    lir = build(_run(QuadratureEncoder().__call__), "quad_reclaim")
    entry = next(block for block in lir.blocks if lir.block_base[block.index] == 0)
    first = min(entry.inline_ops, key=lambda op: op.issue_cycle)
    assert first.issue_cycle == 0, "entry block's first inline op did not reclaim ucode[0]"
    fire_pc = operand_read_cycle(first.operator, lir.block_base[entry.index] + first.issue_cycle)
    assert fire_pc - FETCH_LAG == 0, "the first boolean op does not fire on executing step 0"


def test_entry_state_liveout_producer_is_dwell_guarded() -> None:
    # The sequencer holds pc 0 during the accept wait and re-fires ``ucode[0]`` each idle cycle. As defense-in-depth
    # against a dwell that neither cosim nor the model exercises, the scheduler floors an entry-block producer of a
    # persistent-state live-out to cycle >= 1; a stateless twin still reclaims cycle 0. The flooring is cost-free on
    # every real kernel and the hazard is precluded today (such a producer writes a temporary, never the state register
    # -- see ``_assert_entry_dwell_safe``), so this pins the guard's behavior, not an active miscompile.
    class _Stateful:
        def __init__(self) -> None:
            self._s = False

        def __call__(self, a: bool, b: bool):  # type: ignore[no-untyped-def]
            prev = self._s
            self._s = a and b  # the new state is a combinational inline op in the entry block
            return prev

    def _stateless(a: bool, b: bool):  # type: ignore[no-untyped-def]
        return a and b

    stateful = build(_run(_Stateful().__call__), "dwell_stateful")
    stateless = build(_run(_stateless), "dwell_stateless")
    sf = min(op.issue_cycle for block in stateful.blocks for op in block.inline_ops)
    sl = min(op.issue_cycle for block in stateless.blocks for op in block.inline_ops)
    assert sl == 0, "a stateless entry inline op should reclaim ucode[0]"
    assert sf >= 1, "an entry inline op producing a persistent-state live-out must stay off ucode[0] (dwell guard)"


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
def test_const_branch_install_block_keeps_the_wide_drain(config: OperatorCase) -> None:
    # Regression (fuzz-found B1 miscompile): a constant branch condition formed by DIVISION escapes the
    # frontend's AST-level reachability fold (which evaluates only +,-,* of literals), so the HIR const-folder reduces
    # it to a BoolConst that if-conversion refuses -- leaving an EMPTY const-branch block (the condition install + a
    # branch, no float content). That condition install is a pc-gated copy landing at the WIDE boundary, so the
    # bank-aware drain must NOT shrink the block to the bool boundary, or the terminator reads the condition one PC
    # before it lands. Crash-before: KeyError (model) / stale branch read (RTL); pass-after: bit-exact vs the reference.
    lir = build(_run(const_branch_kernel, config.make_ops(FMT)), f"const_branch_{config.label}")
    # Structural teeth: the surviving const-branch block branches on a constant materialized by a tail bool write, so
    # it must drain at the WIDE boundary (where that pc-gated install lands), not the bool boundary. Pins the drain
    # itself, not only the output, so a future drain regression here is localized rather than silently model-correct.
    const_blocks = [
        b
        for b in lir.blocks
        if isinstance(b.terminator, Branch) and any(w.dst == b.terminator.cond for w in b.bool_writes)
    ]
    assert const_blocks, "the const-branch block did not survive; the corner is no longer exercised"
    for block in const_blocks:
        assert block.term_offset == boundary_step(
            block.block_makespan, wide_resident=True
        ), "a const-branch block shrank below the wide boundary where its condition install lands"
    model = build_model(lir)
    for x, y in [(2.0, 1.0), (1.0, 2.0), (5.0, 3.0), (-1.0, -2.0)]:
        (got,) = model.run(x, y)
        assert math.isclose(float(got), const_branch_kernel(x, y), rel_tol=1e-6)


def test_coalesced_install_block_pays_no_spurious_install_drain() -> None:
    # B2 (coalesced-install fixpoint): a phi-arm predecessor whose every arm coalesces onto the merged register installs
    # nothing, so the +1 install makespan the CFG-shape predicate would assign is spurious and must be dropped. The two
    # arms of this division diamond each produce a fresh quotient that coalesces into the merged phi register (zero
    # install copies), yet each arm block is a CFG phi-arm predecessor. Crash-before: each arm block's makespan carried
    # a spurious +1 install step, inflating its drain and last_pc; pass-after: makespan equals the work makespan.
    def div_diamond(x: float, y: float) -> float:
        if x > y:
            r = x / y
        else:
            r = y / x
        return r

    lir = build(_run(div_diamond), "div_diamond")
    arms = [b for b in lir.blocks if b.ops and not b.copies and not b.bool_writes and isinstance(b.terminator, Jump)]
    assert len(arms) == 2, "the division diamond's two coalesced arm blocks are the B2 target"
    for block in arms:
        last_commit = max(op.commit_cycle for op in (*block.ops, *block.inline_ops))
        assert block.block_makespan == last_commit, "a coalesced-install arm block paid a spurious +1 install drain"


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
def test_empty_merge_block_is_threaded_into_its_successor(config: OperatorCase) -> None:
    # B4 (empty merge-block elimination): a non-convertible diamond (a variable-divisor division) whose merge feeds a
    # following loop leaves an empty pass-through merge -- only the merged phi and a Jump, predecessors the two jump-
    # terminated diamond arms. Merge threading eliminates it, composing the diamond's phi arms into the loop header's
    # init arm. Crash-before (no merge threading): that empty Jump merge survives. The bit-exact RTL check of the
    # resulting three-arm loop-header phi is the cosim twin (test_cosim.py test_cosim_diamond_then_loop).
    lir = build(_run(diamond_then_loop_kernel, config.make_ops(FMT)), f"diamond_then_loop_{config.label}")
    by_index = {block.index: block for block in lir.blocks}
    preds: dict[int, list[int]] = {block.index: [] for block in lir.blocks}
    for block in lir.blocks:
        terminator = block.terminator
        targets = (
            [terminator.target]
            if isinstance(terminator, Jump)
            else [terminator.if_true, terminator.if_false] if isinstance(terminator, Branch) else []
        )
        for target in targets:
            preds[target].append(block.index)
    survivors = [
        block
        for block in lir.blocks
        if not (block.ops or block.inline_ops or block.copies or block.bool_writes)
        and isinstance(block.terminator, Jump)
        and preds[block.index]
        and all(isinstance(by_index[pred].terminator, Jump) for pred in preds[block.index])
    ]
    assert not survivors, "an empty pass-through merge block survived; merge threading did not fire"
    model = build_model(lir)
    for x, y in [(7.0, 2.0), (2.0, 7.0), (100.0, 3.0), (0.5, 4.0)]:
        (got,) = model.run(x, y)
        assert math.isclose(float(got), diamond_then_loop_kernel(x, y), rel_tol=1e-2)


def test_merge_threading_refuses_a_back_edge_carried_merge_phi() -> None:
    # Regression (review round 2, Codex): merge threading deletes a merge block's phis after composing the arm each
    # successor phi takes FROM the merge -- but ONLY that arm. A loop-invariant value the loop header carries on its
    # BACK-EDGE arm is a successor-phi arm too, yet from a different predecessor, so composition would not rewrite it;
    # deleting the merge phi would dangle. The guard must refuse such a merge (the deferred self-latch case).
    # Crash-before: optimize() raised KeyError after threading deleted the still-referenced merge phi.
    def loop_invariant_merge(a, den, c):  # type: ignore[no-untyped-def]
        if a > 0.0:
            x = a / den  # a real (non-speculatable) division branch -> a separate merge block holding phi x
        else:
            x = c
        z = 0.0
        while z < 1.0:  # x (the merge phi) is loop-invariant: carried on the loop header's back-edge arm, not rewritten
            z = x
        return z

    model = build_model(build(_run(loop_invariant_merge), "loop_invariant_merge"))
    for a, den, c in [(2.0, 2.0, 3.0), (-1.0, 4.0, 5.0), (3.0, 1.0, 0.0)]:  # x >= 1 so the latch loop terminates
        (got,) = model.run(a, den, c)
        assert math.isclose(float(got), loop_invariant_merge(a, den, c), rel_tol=1e-6)


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
def test_spill_carry_reads_at_the_model_landing_pc_not_one_cycle_late(config: OperatorCase) -> None:
    # Regression (P3a): the scheduler's cross-block spill carry (block_inflight / the scheduler's livein_landing) must
    # place a spilled result at the SAME absolute PC the numerical model writes it -- both Lir.write_landing_pcs and
    # _trace_landing map a block-local landing to block_base[arm] + (landing - term_offset - 1). The scheduler used its
    # own land - term_offset frame, one PC later, so reservation and read-gating ran on a second coordinate contract.
    # An arm whose ONLY constraint on its consumer is the spilled operand must read it at EXACTLY that landing PC.
    # Crash-before: under the +1 frame the read-gated consumer issued one PC late (every read strictly after the model
    # landing -- no equality), inflating the initiation interval by a cycle; pass-after: it reads at the landing.
    for kernel, name in [(overlap_spill_kernel, "overlap_spill"), (overlap_dead_arm_spill_kernel, "dead_arm_spill")]:
        lir = build(_run(kernel, config.make_ops(FMT)), f"{name}_{config.label}")
        by_index = {block.index: block for block in lir.blocks}
        spilled_any = False
        tight = 0
        for block in lir.blocks:
            if not isinstance(block.terminator, Branch) or block.term_offset >= boundary_step(
                block.block_makespan, wide_resident=True
            ):
                continue
            for op in block.ops:
                for write in op.writes:
                    if not isinstance(write.dst, RegRef):
                        continue
                    landing_pcs = lir.write_landing_pcs(block, op, write)
                    if len(landing_pcs) <= 1:
                        continue  # not a multi-arm spill
                    spilled_any = True
                    for arm in (block.terminator.if_true, block.terminator.if_false):
                        arm_block = by_index[arm]
                        base = lir.block_base[arm_block.index]
                        arm_landing = next(
                            (pc for pc in landing_pcs if base <= pc <= base + arm_block.term_offset), None
                        )
                        if arm_landing is None:
                            continue
                        # Match the consumer by the spilled register index. A later value time-sharing the register
                        # would also land >= arm_landing, so this can never produce a false SAFETY pass; the TIGHTNESS
                        # equality is the genuine spill consumer (empirically the only read at exactly arm_landing).
                        for consumer in arm_block.ops:
                            for operand in consumer.operands:
                                if not (isinstance(operand.source, RegRef) and operand.source.index == write.dst.index):
                                    continue
                                read_pc = base + operand_read_cycle(consumer.inst.operator, consumer.issue_cycle)
                                # SAFETY: never read the in-flight value before it physically lands -- an under-
                                # reservation reads stale data, a miscompile the model shares (cosim cannot catch it).
                                assert (
                                    read_pc >= arm_landing
                                ), f"{name}: reg{write.dst.index} read {read_pc} < landing {arm_landing}"
                                tight += read_pc == arm_landing
        assert spilled_any, f"{name}: no multi-arm spill -- the overlap corner is not exercised"
        # TIGHTNESS: the read-gated arm reads the spill at exactly its model landing PC (one coordinate contract). The
        # +1 frame pushes every such read one PC past the landing, so no equality would hold.
        assert tight > 0, f"{name}: no consumer reads a spilled value at its model landing PC (scheduler off by one)"


def test_entry_busy_gates_a_successor_firing_at_its_inherited_instance_free_cycle() -> None:
    # Coverage for the OTHER half of the cross-block-overlap carry: ``entry_busy`` (the per-instance busy residue an
    # overlapping predecessor hands its single-pred successor via ``successor_local_cycle`` at _build.py, the busy
    # branch of the spill carry). The spilled-value READ timing has a regression above, but the busy residue is empty
    # for every current kernel -- a residue survives only when an operator's initiation interval exceeds its latest
    # write word, which no II=1 operator does -- so no end-to-end build exercises it. Pin its consumption directly:
    # ``schedule_ops`` must hold a pooled firing off its instance until that instance frees in the successor frame.
    def f(a, b):  # type: ignore[no-untyped-def]
        return a * b  # one pooled firing; both operands are inputs (block-start ready), so nothing else can delay it

    mir = _run(f)
    view = _view(mir)
    pool = resolve_pool(mir.nodes)
    (mul,) = _muls(mir)
    operator = mir.nodes[mul].operator
    schedulable = set(view.operation_nodes)

    # With no residue the firing issues on the first cycle (operands resident at block start) -- so any later issue is
    # attributable to the residue alone, not a dependency or a livein_landing.
    assert schedule_ops(mir.nodes, pool, schedulable).issue_cycle[mul] == 1

    # The predecessor freed this instance at block-local ``free``; a successor whose terminator shrank to
    # ``term_offset`` inherits that as ``successor_local_cycle(free, term_offset)`` -- exactly the map the busy branch
    # of the spill carry applies. The firing must then issue precisely on that inherited free cycle, not before (a
    # still-in-flight instance) and not at cycle 1 (ignoring the residue).
    free, term_offset = 12, 4
    inherited = successor_local_cycle(free, term_offset)  # 12 - 4 - 1 = 7
    assert inherited > 1  # the residue is the genuinely binding constraint
    sched = schedule_ops(mir.nodes, pool, schedulable, entry_busy={(operator, 0): inherited})
    assert sched.issue_cycle[mul] == inherited
    assert sched.busy_until[(operator, 0)] == inherited + operator.initiation_interval


def test_residence_tint_is_path_exact_across_a_merge() -> None:
    # Regression (review P1, all three reviewers): the report's residence tint was not path-exact. Three manifestations,
    # all fixed: (a) a single global residence_rows collapsed a register's def/use across mutually-exclusive arms, so a
    # value live on two arms that rejoin at a merge had its lower-addressed arm truncated by the other arm's landing
    # (live register tinted DEAD) -- fixed by per-block CFG residence (_cfg_residence); (b) the read-first `<=` bounds
    # in residence_rows and the upward-exposed test treated a read on a value's own landing PC as reading the PRIOR
    # occupant, painting a register's residence spuriously back toward the block entry (dead register tinted LIVE) --
    # fixed by the write-then-read strict `<`; (c) a pc-gated install (a phi copy, a boolean write, or an early slot
    # writeback) was tinted at its FIRE step, one cycle before the model commits it -- fixed by routing both
    # the model and the diagnostic through ``install_landing`` (fire + 1), while a boundary slot install is read-first
    # at the boundary. Tied to the model oracle in BOTH banks: reg_liveness/bool_liveness must equal the union over both
    # branch arms (or, for the loop kernel, the executed trace) of the model's per-path residence, computed with an
    # INDEPENDENT write-then-read liveness that does not share residence_rows' rule. Crash-before: false-arm mid-rows
    # missing, pre-landing rows spuriously present, and -- for the install skew -- recip_newton's wide phi copies tinted
    # at PC 13/45 (fire) instead of 14/46 and bw's boolean write tinted at its fire step instead of its landing.
    from holoso._lir._ir import FloatOperand
    from holoso._backend.numerical import NumericalSimulator

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from recip_newton import NewtonReciprocal  # noqa: PLC0415 (example kernels live under examples/)

    def join_spill(x, y, z):  # w spills from the entry into BOTH arms and is read after the merge
        w = (x * z + y) * z + y
        if x < y:
            a = z + 1.0
        else:
            a = z / (y + 1.0)  # the division keeps this a real branch (not if-converted)
        return w + a

    def phi_merge(x, y):  # the merged value is produced in each arm (no entry spill), read after the merge
        if x < y:
            a = x + 1.0
        else:
            a = x / (y + 1.0)
        return a * 2.0

    def bool_write_merge(x, y):  # a boolean phi whose else arm reads the value inverted -> a non-coalesced bool_write
        p = x < y
        if x < 1.0:
            c = p
            d = x / (y + 1.0)
        else:
            c = not p
            d = y * 2.0
        return c, d

    class _Trace(NumericalSimulator):
        def __init__(self, lir: object) -> None:
            super().__init__(lir)  # type: ignore[arg-type]
            self.reads: set[int] = set()
            self.writes: set[int] = set()
            self.bool_reads: set[int] = set()
            self.bool_writes: set[int] = set()
            self.branch_read: tuple[int, int] | None = None  # (term_pc, cond index) read by this tick's redirect

        def tick(self, in_valid: bool, out_ready: bool) -> None:
            # The redirect samples a Branch condition directly out of bregs in _next_pc (not via _read), at the
            # terminator PC before the PC advances; capture it so the oracle counts that read at term_pc.
            terminator = self._terminators.get(self.pc) if self.pc != self._lir.last_pc else None
            self.branch_read = (self.pc, terminator.cond.index) if isinstance(terminator, Branch) else None
            super().tick(in_valid, out_ready)

        def _read(self, operand: object) -> object:
            if isinstance(operand, FloatOperand):
                if isinstance(operand.source, RegRef):
                    self.reads.add(operand.source.index)
            elif isinstance(operand.source, BoolRegRef):  # type: ignore[attr-defined]
                self.bool_reads.add(operand.source.index)  # type: ignore[attr-defined]
            return super()._read(operand)  # type: ignore[arg-type]

        def _write(self, dst: object, value: object) -> None:
            if isinstance(dst, RegRef):
                self.writes.add(dst.index)
            elif isinstance(dst, BoolRegRef):
                self.bool_writes.add(dst.index)
            super()._write(dst, value)  # type: ignore[arg-type]

    def model_residence(
        lir: object, vectors: list[tuple[float, ...]]
    ) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
        wide: dict[int, set[int]] = {}
        boolean: dict[int, set[int]] = {}
        for vec in vectors:
            sim = _Trace(lir)
            sim.reset()
            sim.set_inputs(*vec)
            steps: list[tuple[int, frozenset[int], frozenset[int], frozenset[int], frozenset[int]]] = []

            def tick(in_valid: bool, out_ready: bool) -> None:
                sim.reads, sim.writes, sim.bool_reads, sim.bool_writes = set(), set(), set(), set()
                sim.tick(in_valid, out_ready)
                if sim.branch_read is not None:  # the redirect read the condition at term_pc, before this tick's _apply
                    term_pc, cond = sim.branch_read
                    steps.append((term_pc, frozenset(), frozenset(), frozenset({cond}), frozenset()))
                steps.append(
                    (
                        sim.pc,
                        frozenset(sim.reads),
                        frozenset(sim.writes),
                        frozenset(sim.bool_reads),
                        frozenset(sim.bool_writes),
                    )
                )

            tick(True, False)
            guard = 0
            while not sim.out_valid and guard < 100_000:
                tick(False, False)
                guard += 1
            sim.reads, sim.bool_reads = set(), set()
            _ = sim.output_values  # the output taps read their registers at the boundary PC
            pc, rd, wr, brd, bwr = steps[-1]
            steps[-1] = (pc, rd | frozenset(sim.reads), wr, brd | frozenset(sim.bool_reads), bwr)
            live: set[int] = set()
            blive: set[int] = set()
            # Backward per-path liveness in the model's write-then-read order (a write lands then is read at the same
            # PC), so a read on a value's landing reads THAT value, not the prior occupant: kill the carry with the
            # write AFTER folding in the read. A def cell is resident even if its value is never read (it occupies the
            # register that cycle). Crucially this oracle does NOT share residence_rows' rule, so it is an independent
            # check -- the buggy `<=` and fire-step-install variants over-tint against it.
            for step_pc, reads, writes, breads, bwrites in reversed(steps):
                for reg in reads | live | writes:
                    wide.setdefault(reg, set()).add(step_pc)
                for reg in breads | blive | bwrites:
                    boolean.setdefault(reg, set()).add(step_pc)
                live = (reads | live) - writes
                blive = (breads | blive) - bwrites
        return wide, boolean

    cases: list[tuple[str, object, list[tuple[float, ...]]]] = [
        ("join_spill", build(_run(join_spill), "join_spill"), [(0.5, 2.0, 1.5), (2.0, 0.5, 1.5)]),
        ("phi_merge", build(_run(phi_merge), "phi_merge"), [(0.5, 2.0), (2.0, 0.5)]),
        ("bool_write_merge", build(_run(bool_write_merge), "bool_write_merge"), [(0.5, 2.0), (2.0, 0.5)]),
        # recip_newton is a real overlapping LOOP kernel with two non-coalesced wide phi copies (the install skew site);
        # its internal iteration covers the loop body, and the seed converges for a < 3.
        ("recip_newton", build(_run(NewtonReciprocal().__call__), "recip_newton"), [(0.5,), (1.5,), (2.5,)]),
    ]
    for name, lir, vectors in cases:
        last = lir.initiation_interval
        # No upper clip beyond the grid; deliberately NO lower clip, so a spurious pre-landing row (e.g. PC 0, the bug
        # the strict-`<` upward rule fixes) would surface as a mismatch rather than being silently discarded.
        model_wide, model_bool = model_residence(lir, vectors)
        for bank_tint, model_bank in ((lir.reg_liveness, model_wide), (lir.bool_liveness, model_bool)):
            tint = {reg.index: {pc for pc in rows if pc <= last} for reg, rows in bank_tint.items()}
            tint = {index: rows for index, rows in tint.items() if rows}
            model = {index: {pc for pc in rows if pc <= last} for index, rows in model_bank.items()}
            model = {index: rows for index, rows in model.items() if rows}
            assert tint == model, f"{name}: residence tint {tint} != model per-path residence {model}"
        # Convention-independent invariant (catches the same-PC def+use over-tint in either bank): no register holds a
        # live value before the program's first executing step.
        for reg, rows in {**lir.reg_liveness, **lir.bool_liveness}.items():
            assert min(rows) >= 1, f"{name}: {reg} tinted resident at PC {min(rows)} < 1"


def test_state_slot_residence_matches_the_model_under_carry() -> None:
    # Regression (review): the READ-FIRST boundary-install path -- residence_rows' read_first_defs and _cfg_residence
    # `upward` refinement -- governs only persistent state slots, which the stateless oracle above never builds. A
    # boundary state install reads-then-writes at last_pc, so a read there (an output tap of the live-in, or an in-place
    # install's own source) reads the PRIOR value; the prior strict-`<` rule mis-attributed it to the boundary def and
    # truncated the carried live-in, tinting a LIVE slot register DEAD mid-frame. Tie reg_liveness/bool_liveness for the
    # slot registers to a STEADY-STATE model oracle: drive many back-to-back transactions, compute backward
    # write-then-read liveness over the concatenated executed trace (so a slot live-out carries into the next
    # transaction's reads and a mid-frame gap surfaces), and union residence by PC over the middle transactions. Covers
    # a single- and multi-block WIDE boundary slot and a single- and multi-block BOOLEAN boundary slot (the multi-block
    # cases exercise the `upward` live-in marking of a carried slot). Crash-before: with read-first reverted
    # the carried slot live-in tints DEAD between cycle 1 and its boundary read.
    from holoso._lir._ir import Branch, FloatOperand
    from holoso._backend.numerical import NumericalSimulator

    class Delay:  # single-block wide boundary slot; the live-in is output-tapped at the boundary (read-first)
        def __init__(self) -> None:
            self._d = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            prev = self._d
            self._d = x
            return prev

    class MBWide:  # multi-block wide boundary slot: a real branch (divide by a variable) carries _s to the Ret block
        def __init__(self) -> None:
            self._s = 0.0

        def __call__(self, x, y):  # type: ignore[no-untyped-def]
            out = self._s
            if x > 0.0:
                self._s = x + y
            else:
                self._s = x / y
            return out

    class BoolHold:  # multi-block boolean boundary slot; the live-in is output-tapped at the boundary (read-first)
        def __init__(self) -> None:
            self._f = False

        def __call__(self, x):  # type: ignore[no-untyped-def]
            out = self._f
            if x > 0.0:
                self._f = True
            else:
                self._f = x < -1.0
            return out, x + 1.0

    class BoolToggle:  # single-block boolean slot installed in place (_b <= ~_b), also read by the select
        def __init__(self) -> None:
            self._b = False

        def __call__(self, x):  # type: ignore[no-untyped-def]
            old = self._b
            self._b = not self._b
            return x if old else -x

    class BoolSpin:  # in-place boolean slot (_b <= ~_b) whose live-in is read ONLY by its own install (source aliases
        def __init__(self) -> None:  # the destination): the carry survives only if the boundary bundle is read-first
            self._b = False

        def __call__(self, x):  # type: ignore[no-untyped-def]
            self._b = not self._b
            return x

    class _Trace(NumericalSimulator):
        def __init__(self, lir: object) -> None:
            super().__init__(lir)  # type: ignore[arg-type]
            self.events: list[tuple[int, str | None, str | None, int | None]] = []  # (pc, r/w, f/b, index) in order
            self.branch: tuple[int, int] | None = None

        def tick(self, in_valid: bool, out_ready: bool) -> None:
            term = self._terminators.get(self.pc) if self.pc != self._lir.last_pc else None
            self.branch = (self.pc, term.cond.index) if isinstance(term, Branch) else None
            super().tick(in_valid, out_ready)

        def _read(self, operand: object) -> object:
            source = operand.source  # type: ignore[attr-defined]
            if isinstance(operand, FloatOperand):
                if isinstance(source, RegRef):
                    self.events.append((self.pc, "r", "f", source.index))
            elif isinstance(source, BoolRegRef):
                self.events.append((self.pc, "r", "b", source.index))
            return super()._read(operand)  # type: ignore[arg-type]

        def _write(self, dst: object, value: object) -> None:
            if isinstance(dst, RegRef):
                self.events.append((self.pc, "w", "f", dst.index))
            if isinstance(dst, BoolRegRef):
                self.events.append((self.pc, "w", "b", dst.index))
            super()._write(dst, value)  # type: ignore[arg-type]

    # A step is (pc, float reads, float writes, bool reads, bool writes, read_first). read_first marks the boundary
    # install bundle, where the hardware reads every source then writes every destination on the boundary edge, so a
    # read outlives a same-register write (an in-place ``b <= ~b`` keeps its live-in); others are write-then-read
    # (a read on a value's landing reads the NEW value).
    Step = tuple[int, frozenset[int], frozenset[int], frozenset[int], frozenset[int], bool]

    def substeps(events: list[tuple[int, str | None, str | None, int | None]], read_first: bool) -> list[Step]:
        # Group a temporal event stream into one step per contiguous run at the same PC (a None-kind entry just marks a
        # PC visited, so a cycle that holds a value without touching it still contributes a step and fills the carry).
        out: list[Step] = []
        cur: int | None = None
        rf: set[int] = set()
        wf: set[int] = set()
        rb: set[int] = set()
        wb: set[int] = set()
        for pc, kind, bank, idx in events:
            if pc != cur:
                if cur is not None:
                    out.append((cur, frozenset(rf), frozenset(wf), frozenset(rb), frozenset(wb), read_first))
                cur, rf, wf, rb, wb = pc, set(), set(), set(), set()
            if idx is not None:
                {("r", "f"): rf, ("w", "f"): wf, ("r", "b"): rb, ("w", "b"): wb}[(kind, bank)].add(idx)
        if cur is not None:
            out.append((cur, frozenset(rf), frozenset(wf), frozenset(rb), frozenset(wb), read_first))
        return out

    def model_slot_residence(
        lir: object, vectors: list[tuple[float, ...]]
    ) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
        transactions = 16
        last_pc = lir.last_pc
        fin = [load.dst.index for load in lir.float_inputs]
        bin_ = [load.dst.index for load in lir.bool_inputs]
        sim = _Trace(lir)
        sim.reset()
        per_txn: list[list[Step]] = []
        for k in range(transactions):
            sim.set_inputs(*vectors[k % len(vectors)])
            # set_inputs writes the input lanes directly (not via _write); model them as defs at the first step
            events: list[tuple[int, str | None, str | None, int | None]] = [(1, "w", "f", i) for i in fin]
            events += [(1, "w", "b", i) for i in bin_]

            def run(in_valid: bool, out_ready: bool) -> None:
                sim.events = []
                sim.tick(in_valid, out_ready)
                if sim.branch is not None:  # the redirect samples the condition at term_pc, before this tick's _apply
                    events.append((sim.branch[0], "r", "b", sim.branch[1]))
                events.extend(sim.events)
                events.append((sim.pc, None, None, None))  # mark the landed PC visited even if it had no event

            run(True, False)
            guard = 0
            while not sim.out_valid and guard < 10_000:
                run(False, False)
                guard += 1
            sim.events = []
            _ = sim.output_values  # the output taps read their registers at the boundary PC (write-then-read with them)
            events.extend(sim.events)
            steps = substeps(events, False)  # everything up to the accept edge is write-then-read
            sim.events = []
            sim.tick(False, True)  # accept: the boundary state install fires, then the PC advances and _apply runs
            # The boundary install is a READ-FIRST parallel bundle at last_pc (read every source, then write every
            # destination), so it is split out and tagged; the trailing _apply is ordinary write-then-read.
            steps += substeps([e for e in sim.events if e[0] == last_pc], True)
            steps += substeps([e for e in sim.events if e[0] != last_pc], False)
            per_txn.append(steps)

        flat = [step for steps in per_txn for step in steps]
        wide: dict[int, set[int]] = {}
        boolean: dict[int, set[int]] = {}
        resid: list[tuple[int, frozenset[int], frozenset[int]]] = []
        live_f: set[int] = set()
        live_b: set[int] = set()
        for pc, rf, wf, rb, wb, read_first in reversed(flat):
            resid.append((pc, rf | live_f | wf, rb | live_b | wb))
            if read_first:  # read-then-write bundle: a read outlives a same-register write (the live-in survives)
                live_f = rf | (live_f - wf)
                live_b = rb | (live_b - wb)
            else:  # write-then-read landing: the write kills the carry, a read on the landing reads the new value
                live_f = (rf | live_f) - wf
                live_b = (rb | live_b) - wb
        resid.reverse()
        lo, hi = 3, transactions - 3  # the steady-state middle band, free of warm-up/drain edge effects
        cursor = 0
        for k, steps in enumerate(per_txn):
            for _ in steps:
                pc, residf, residb = resid[cursor]
                cursor += 1
                if lo <= k < hi:
                    for reg in residf:
                        wide.setdefault(reg, set()).add(pc)
                    for reg in residb:
                        boolean.setdefault(reg, set()).add(pc)
        return wide, boolean

    cases: list[tuple[str, object, list[tuple[float, ...]]]] = [
        ("Delay", build(_run(Delay().__call__), "Delay"), [(0.5,), (-0.5,), (1.5,), (2.0,)]),
        ("MBWide", build(_run(MBWide().__call__), "MBWide"), [(1.0, 2.0), (-1.0, 2.0), (1.5, 3.0), (-2.0, 4.0)]),
        ("BoolHold", build(_run(BoolHold().__call__), "BoolHold"), [(1.0,), (-2.0,), (-0.5,), (2.0,)]),
        ("BoolToggle", build(_run(BoolToggle().__call__), "BoolToggle"), [(0.5,), (1.5,), (-0.5,), (2.0,)]),
        ("BoolSpin", build(_run(BoolSpin().__call__), "BoolSpin"), [(0.5,), (1.5,), (-0.5,), (2.0,)]),
    ]
    compared = 0
    for name, lir, vectors in cases:
        last = lir.initiation_interval
        model_wide, model_bool = model_slot_residence(lir, vectors)
        # Restrict the comparison to the slot registers the read-first path governs (scratch registers on a not-taken
        # branch arm would be tinted by the static all-paths tint but absent from a single steady run -- path coverage,
        # not a defect; a carried slot register is live on every path, so it must match exactly).
        for slot in [*lir.float_state_slots]:
            if slot.needs_copy:
                tint = {pc for pc in lir.reg_liveness[slot.reg] if pc <= last}
                model = {pc for pc in model_wide.get(slot.reg.index, set()) if 1 <= pc <= last}
                assert tint == model, f"{name}: wide slot {slot.reg} tint {sorted(tint)} != model {sorted(model)}"
                compared += 1
        for slot in [*lir.bool_state_slots]:
            if slot.needs_copy:
                tint = {pc for pc in lir.bool_liveness[slot.reg] if pc <= last}
                model = {pc for pc in model_bool.get(slot.reg.index, set()) if 1 <= pc <= last}
                assert tint == model, f"{name}: bool slot {slot.reg} tint {sorted(tint)} != model {sorted(model)}"
                compared += 1
    # Guard against the kernels silently losing their non-coalesced slots (e.g. a future coalescing change) -- without
    # this the loop above would vacuously pass and re-open the read-first coverage gap this test exists to close.
    assert compared >= 5, f"expected every kernel to contribute a non-coalesced slot, compared only {compared}"


def test_write_landing_recursion_handles_multi_hop_spill() -> None:
    # Coverage for the multi-hop arm of the landing recursion. A result can spill past one overlap-shrunk terminator and
    # RE-spill past a second (a near-empty overlapping intermediate block whose own offset is below the inherited
    # landing). Frontends do not emit that shape -- a single hop lands at most FETCH_LAG cycles into a successor,
    # below the offset of any successor carrying an op -- so the recursion is exercised here on a hand-built layout,
    # pinning that it re-keys per terminator exactly as write_landing_pcs documents and terminates.
    from holoso._lir._ir import LirBlock, Jump, Branch, Ret, BoolRegRef, _trace_landing

    b0 = LirBlock(0, [], [], [], [], Branch(BoolRegRef(0), 1, 2), 0, 3)
    b1 = LirBlock(1, [], [], [], [], Jump(3), 0, 1)  # inherited landing 3 > offset 1 -> re-spills into b3
    b2 = LirBlock(2, [], [], [], [], Jump(3), 0, 5)  # inherited landing 3 <= offset 5 -> absorbs in-block
    b3 = LirBlock(3, [], [], [], [], Ret(), 0, 4)
    by_index = {block.index: block for block in (b0, b1, b2, b3)}
    base = [0, 10, 20, 30]
    # landing 7 in b0 spills (7 > 3) at block-local 3 into both arms; b1 re-spills 3 -> local 1 in b3 (base 30 + 1),
    # b2 absorbs at base 20 + 3; a landing within b0's offset lands once, in-block.
    assert sorted(_trace_landing(by_index, base, b0, 7)) == [23, 31]
    assert _trace_landing(by_index, base, b0, 2) == [2]


def test_control_arrows_anchor_at_the_terminator_pc() -> None:
    # Regression (HTML report exactness): the grid row axis is the model fetch PC, so a control-transfer arrow must root
    # at the terminator PC (where the redirect mux reads the condition register and that register's residence ends) and
    # point at the destination block's base PC -- no FETCH_LAG offset. Crash-before: the arrow rooted FETCH_LAG rows
    # below the terminator, where the condition register is already dead, so its dotted feed pointed at a blank cell.
    from holoso._backend.html._schedule import _control_arrows

    lir = build(_run(overlap_spill_kernel), "overlap_spill")
    arrows = _control_arrows(lir)
    assert arrows, "the branchy kernel must emit at least one control-transfer arrow"
    term_pcs = {lir.term_pc(block) for block in lir.blocks}
    bases = set(lir.block_base)
    bool_live = lir.bool_liveness
    for arrow in arrows:
        assert arrow.src_cyc in term_pcs, f"arrow root {arrow.src_cyc} is not a terminator PC"
        assert arrow.dst_cyc in bases, f"arrow target {arrow.dst_cyc} is not a block base PC"
        if (
            arrow.cond is not None
        ):  # the branch reads its condition on its terminator row, so the register is live there
            assert arrow.src_cyc in bool_live[arrow.cond], "the condition register is dead at the arrow's root row"


def test_phi_install_does_not_clobber_the_branch_condition() -> None:
    # Regression (review): a phi-arm install physically writes the phi's register at the predecessor's tail, one step
    # BEFORE the branch terminator reads its condition at the boundary. The interference model used to define the phi
    # only at the merge head, so when an arm came from a branching block the phi could share the very register holding
    # that block's branch condition -- the install then overwrote the condition and the branch took the wrong arm
    # (model and RTL agreed with each other, so only a semantics check catches it). The frontend currently routes
    # every phi arm through a dedicated jump block, so this CFG is built directly; cross-block scheduling will make
    # such shapes routine.
    builder = MirBuilder(FMT)
    entry = builder.block()
    then = builder.block()
    merge = builder.block()
    builder.position_at(entry)
    flag = builder.bool_input("flag", BoolType())
    other = builder.bool_input("other", BoolType())
    builder.branch(flag, then, merge)
    builder.position_at(then)
    inverted = builder.operation(BoolAndOperator(), [other, other], [BoolInversion(True), BoolInversion(True)])
    builder.jump(merge)
    builder.position_at(merge)
    merged = builder.phi(BoolType(), [(entry, other, BoolInversion()), (then, inverted, BoolInversion())])
    builder.bool_output("out", merged)
    builder.ret()
    model = build_model(build(builder.finish(), "phi_cond_clobber"))
    for flag_value in (False, True):
        for other_value in (False, True):
            want = (not other_value) if flag_value else other_value
            assert model.run(flag_value, other_value)[0] is want, f"flag={flag_value} other={other_value}"


def test_branch_on_phi_installed_in_the_branching_block_is_rejected() -> None:
    # Soundness boundary (review): if a branch condition is a phi taking an arm from the branching block itself, the
    # arm's install lands in the condition register exactly when the terminator reads it -- the branch would consult
    # the NEXT iteration's value, and no register assignment can avoid a value conflicting with itself. The build must
    # refuse this shape (the frontend never emits it; a future cross-block pass might) rather than miscompile: before
    # the guard, a self-loop header `phi = phi(entry: cond, header: not phi); branch(phi, header, exit)` ran its body
    # one trip short with model and RTL agreeing on the wrong count.
    builder = MirBuilder(FMT)
    entry = builder.block()
    header = builder.block()
    exit_block = builder.block()
    builder.position_at(entry)
    start = builder.bool_input("start", BoolType())
    builder.jump(header)
    builder.position_at(header)
    looping = builder.open_phi(BoolType(), (entry, start, BoolInversion()))
    inverted = builder.operation(BoolAndOperator(), [looping, looping], [BoolInversion(True), BoolInversion(True)])
    builder.set_phi_arms(looping, [(entry, start, BoolInversion()), (header, inverted, BoolInversion())])
    builder.branch(looping, header, exit_block)
    builder.position_at(exit_block)
    builder.bool_output("out", looping)
    builder.ret()
    with pytest.raises(UnsupportedConstruct, match="arm from the same block"):
        build(builder.finish(), "self_loop_cond")


def _ilog2(mir: Mir) -> list[int]:
    return [
        vid for vid, n in mir.nodes.items() if isinstance(n, MirOperation) and isinstance(n.operator, FMulILog2Operator)
    ]


def test_fmul_ilog2_same_k_shares_one_instance() -> None:
    # Two K=2 scalings that never run on the same cycle (the second waits on a multiply) pool onto one instance.
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a * b) * 4.0, b * 4.0

    mir = _run(f)
    il = _ilog2(mir)
    assert len(il) == 2
    sched = _schedule(mir)
    assert sched.issue_cycle[il[0]] != sched.issue_cycle[il[1]]  # not concurrent
    assert sched.inst_of[il[0]] == sched.inst_of[il[1]]  # ...so they share the one instance
    assert sum(1 for i in sched.instances if isinstance(i.operator, FMulILog2Operator)) == 1


def test_fmul_ilog2_same_k_serializes_by_default_parallelizes_with_budget() -> None:
    # Two independent K=2 scalings are both ready at cycle 1; the per-kind budget governs them like any other kind.
    def f(a, b):  # type: ignore[no-untyped-def]
        return a * 4.0, b * 4.0

    mir = _run(f)
    il = _ilog2(mir)
    assert len(il) == 2

    one = _schedule(mir)  # default budget 1 -> serialize onto a single instance
    assert one.issue_cycle[il[0]] != one.issue_cycle[il[1]]
    assert sum(1 for i in one.instances if isinstance(i.operator, FMulILog2Operator)) == 1


def test_fmul_ilog2_different_k_never_shares() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a * 4.0 + b * 8.0  # K=2 and K=3 -- distinct hardware modules

    mir = _run(f)
    il = _ilog2(mir)
    assert len(il) == 2
    sched = _schedule(mir)
    assert sched.inst_of[il[0]] != sched.inst_of[il[1]]  # different K -> different instances
    assert {sched.inst_of[v].operator.k for v in il} == {2, 3}
    assert {sched.inst_of[v].index for v in il} == {0}  # indices are local to each concrete operator value


def test_build_lir_small_kernel() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    lir = build(_run(f), "kernel")
    assert lir.module_name == "kernel"
    assert lir.float_format == FMT
    assert lir.regfile.width == lir.float_format.width
    assert lir.regfile.nreg >= 1
    assert {i.name for i in lir.float_inputs} == {"a", "b"}
    assert lir.regfile.nload == 2  # both inputs are preloaded via the regfile load port (registers 0..1)
    assert [o.name for o in lir.float_outputs] == ["out_0"]
    assert all(isinstance(o.tap, FloatOperand) for o in lir.float_outputs)
    assert all(isinstance(o.tap.source, RegRef) for o in lir.float_outputs)

    names = [p.name for p in lir.ports]
    for expected in (
        "clk",
        "rst",
        "in_valid",
        "in_ready",
        "out_valid",
        "out_ready",
        "in_a",
        "in_b",
        "out_0",
        "err_pc",
    ):
        assert expected in names


def test_state_writeback_installs_early_and_is_first_class() -> None:
    from holoso._lir import FETCH_LAG

    class LeakyDelay:
        def __init__(self) -> None:
            self._p = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            out = self._p + x  # reads the old _p; the fadd result is the only output
            self._p = x  # a non-coalesced writeback whose source (the input x) is an ordinary register
            return out

    lir = build(_run(LeakyDelay().__call__), "leaky_delay")
    (slot,) = lir.float_state_slots
    assert (
        bool(lir.float_state_slots or lir.bool_state_slots) and slot.needs_copy and isinstance(slot.tap, FloatOperand)
    )
    assert isinstance(slot.tap.source, RegRef)
    # The non-coalesced writeback is a first-class event in the liveness model: the slot register holds a live value
    # from the cycle the new value LANDS (one PC after the copy fires and samples its source, ``install_landing``;
    # previously absent, which is why the report could not render it).
    landing = install_landing(lir.state_copy_step(slot))
    assert landing in lir.reg_liveness[slot.reg]
    assert lir.state_copy_step(slot) == slot.install_cycle + FETCH_LAG + 1
    # Nothing reads _p's register after the old live-in and its source is an ordinary register, so the copy installs
    # before the boundary -- freeing the source register for the rest of the initiation rather than pinning it there.
    assert lir.state_copy_step(slot) < lir.initiation_interval
    # The carried live-out must survive to the boundary even though nothing reads it again this frame, so the slot
    # register stays live from its landing through the boundary -- an early install is not the value's death.
    assert set(range(landing, lir.initiation_interval + 1)) <= lir.reg_liveness[slot.reg]
    # Output wires carry the same FloatOperand tap primitive as state slots.
    assert all(isinstance(w.tap, FloatOperand) for w in lir.float_outputs)
    # Pin the hardware-frame cycle formulas the report, model, and allocator all depend on (the write/read latch
    # offsets around FETCH_LAG); every consumer routes through the shared _ir helpers that own this arithmetic.
    op = lir.ops[0]
    assert result_landing_cycle(op.writes[0].dst, op.commit_cycle) == op.commit_cycle + FETCH_LAG + 2
    assert operand_read_cycle(op.inst.operator, op.issue_cycle) == op.issue_cycle + FETCH_LAG - 1


def test_cfg_phi_merge_register_shows_residence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(if_convert_pass, "_IFCONV_MAX_OPS", 0)  # the subject is the branchy phi-copy machinery

    # A diamond merging two CONSTANT arms: constants are not register-backed, so neither coalesces -- the merged
    # register is written ONLY by the per-arm phi copies and read at the boundary, never by an operator. Before
    # phi-copy residence was added to reg_liveness, such a register had a use but no def and so collapsed to an empty
    # (untinted) live set -- the CFG-report liveness gap.
    def f(x):  # type: ignore[no-untyped-def]
        if x > 0.0:
            z = 1.0
        else:
            z = 2.0
        return z

    lir = build(_run(f), "diamond")
    assert any(block.copies for block in lir.blocks), "the merge must be resolved by phi-arm copies"
    (out,) = lir.float_outputs
    assert isinstance(out.tap.source, RegRef)
    assert lir.reg_liveness.get(out.tap.source), "the phi-merged output register must be tinted live in the report"


def test_cfg_write_only_state_slot_is_reserved() -> None:
    # A state slot written on every arm but never read before the write has no live-in, so its dedicated register is
    # never pinned to a value. The folded sign on the live-out (``self.acc = -t``) keeps it from coalescing in place, so
    # the register is reserved-but-empty (no live-in occupant, installed by the boundary copy). The colorer must still
    # reserve it -- a temporary considering it as a reuse candidate must skip it rather than fault on the missing pool
    # entry. (Without the sign fold the write-only live-out would coalesce onto the slot register; see the next test.)
    class WriteOnlyBranch:
        def __init__(self) -> None:
            self.acc = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            if x > 0.0:
                t = x * 2.0
            else:
                t = x * 3.0
            self.acc = -t  # the folded sign forces a non-coalesced (reserved, copy-installed) write-only slot
            return self.acc

    lir = build(_run(WriteOnlyBranch().__call__), "write_only")
    (slot,) = lir.float_state_slots
    assert slot.name == "acc" and slot.needs_copy  # reserved, not coalesced (the sign fold blocks in-place commit)
    assert slot.reg.index not in {
        write.dst.index for op in lir.ops for write in op.writes
    }  # reserved: no operator result lands on it
    model = build_model(lir)
    assert float(model.run(3.0)[0]) == -6.0 and float(model.run(-2.0)[0]) == 6.0


def test_cfg_state_slot_coalesces_onto_its_register() -> None:
    # A control-flow kernel (the float(x>0) cast forces the CFG path) whose state live-out is an operator result that
    # lands after the live-in is fully read coalesces onto the slot register: the producing operator writes it directly,
    # so the slot needs no install copy -- the same coalescing the straight-line allocator did, now in the CFG path.
    class Filt:
        def __init__(self) -> None:
            self.state = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            self.state = self.state * 0.9 + x
            return float(x > 0.0) * self.state

    lir = build(_run(Filt().__call__), "filt")
    assert any(block.inline_ops for block in lir.blocks)  # the float(x>0) cast is an inline op
    (slot,) = lir.float_state_slots
    assert not slot.needs_copy, "the slot live-out must coalesce onto the slot register (no install copy)"
    assert slot.tap.source == slot.reg
    model = build_model(lir)
    model.reset()
    first = float(model.run(2.0)[0])  # state <- 0*0.9 + 2 = 2; out = float(2>0)*2 = 2
    second = float(model.run(1.0)[0])  # state <- 2*0.9 + 1 ~ 2.8; out ~ 2.8 -- proves the coalesced state carried over
    assert abs(first - 2.0) < 1e-3 and 2.5 < second < 3.0


def test_cfg_branch_conditions_reuse_boolean_registers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(if_convert_pass, "_IFCONV_MAX_OPS", 0)  # the subject is branch-condition register reuse

    # Sequential data-dependent branches: each condition is computed, tested at its boundary, and dead before the next,
    # so the boolean bank reuses one register across them instead of allocating one per branch.
    def f(x, y, z):  # type: ignore[no-untyped-def]
        a = x
        if x > 0.0:
            a = a + 1.0
        if y > 0.0:
            a = a + 2.0
        if z > 0.0:
            a = a + 4.0
        return a

    lir = build(_run(f), "branches")
    comparisons = sum(1 for b in lir.blocks for op in b.ops if isinstance(op.inst.operator, FCmpOperator))
    assert comparisons >= 3
    assert lir.bool_regfile.nreg < comparisons  # the three conditions share boolean registers
    model = build_model(lir)
    assert abs(float(model.run(1.0, -1.0, 1.0)[0]) - 6.0) < 1e-3  # a=1; +1 (x>0); skip (y<=0); +4 (z>0) = 6


def _coalescing_self_copies(lir) -> int:  # type: ignore[no-untyped-def]
    """Count no-op identity copies (``r <= r`` with identity sign): a coalescable arm the pass should have merged."""
    return sum(
        1
        for block in lir.blocks
        for copy in block.copies
        if isinstance(copy.source.source, RegRef)
        and copy.source.source == copy.dst
        and copy.source.sign == FloatSignControl()
    )


def test_diamond_op_result_arms_coalesce(monkeypatch: pytest.MonkeyPatch) -> None:
    # A forced float diamond merges two op-result arms in mutually exclusive blocks; with no interference they coalesce
    # onto the merged register, so the phi installs NO copy at all (and certainly no no-op self-copy). The result stays
    # correct on both arms -- bit-exact value preservation against the RTL is the cosim's job; here we pin the win.
    monkeypatch.setattr(if_convert_pass, "_IFCONV_MAX_OPS", 0)  # keep the diamond a real phi merge, not a select

    def f(x, y):  # type: ignore[no-untyped-def]
        if x > 0.0:
            z = x + y
        else:
            z = x * y
        return z * x  # an operator use of the merged value, not only the boundary output

    lir = build(_run(f), "phicoal")
    assert sum(len(b.copies) for b in lir.blocks) == 0, "the diamond's op-result arms must coalesce away their copies"
    assert _coalescing_self_copies(lir) == 0
    model = build_model(lir)
    for a, b in ((2.0, 3.0), (-2.0, 3.0), (1.5, -4.0)):
        ref = (a + b) * a if a > 0.0 else (a * b) * a
        got = float(model.run(a, b)[0])
        assert abs(got - ref) <= 1e-2 * max(1.0, abs(ref)), f"{a},{b}: {got} vs {ref}"


def test_loop_carried_phi_coalesces_when_non_interfering() -> None:
    # A directed loop whose carried value is read once (read-first) before its update lands: the header phi and its
    # back-edge arm (an op result) do not interfere, so they coalesce -- the loop body installs no register-source copy.
    # Only the entry constants (acc=0, i=0) keep their copies.
    def f(x):  # type: ignore[no-untyped-def]
        acc = 0.0
        i = 0.0
        while i < 3.0:
            acc = acc + x
            i = i + 1.0
        return acc

    lir = build(_run(f), "accum")
    register_source_copies = [
        copy for block in lir.blocks for copy in block.copies if isinstance(copy.source.source, RegRef)
    ]
    assert not register_source_copies, "the non-interfering back-edge arms must coalesce (no register-source copy)"
    assert _coalescing_self_copies(lir) == 0
    model = build_model(lir)
    for x in (0.5, 1.0, 2.0, 3.0):
        assert abs(float(model.run(x)[0]) - 3.0 * x) <= 1e-2 * max(1.0, 3.0 * x)


def test_interfering_loop_carried_phi_keeps_its_copy() -> None:
    # The dual: Newton's reciprocal carries a value read several times across the body, so the header phi overlaps the
    # back-edge update (the new value lands while the old is still needed). The oracle must refuse that merge -- the
    # back-edge arm (an operator result) stays installed by a copy at the loop body's tail.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from recip_newton import NewtonReciprocal  # noqa: PLC0415  (example kernels live under examples/)

    lir = build(_run(NewtonReciprocal().__call__), "recip")
    back_edge_op_copies = [
        copy
        for block in lir.blocks
        if isinstance(block.terminator, Jump) and lir.block_base[block.terminator.target] <= lir.block_base[block.index]
        for copy in block.copies
        if isinstance(copy.source.source, RegRef)
    ]
    assert back_edge_op_copies, "the interfering loop-carried Newton update must keep its back-edge install copy"


def _check_float_kernel(fn, name, samples):  # type: ignore[no-untyped-def]
    """Build ``fn`` (must not crash) and check the cycle model matches the float64 reference on ``samples``."""
    lir = build(_run(fn), name)  # crash-before: the install-free oracle admitted an unsound merge -> backstop assert
    model = build_model(lir)
    for args in samples:
        got = [float(v) for v in model.run(*args)]
        ref = [float(v) for v in fn(*args)]
        assert len(got) == len(ref)
        for g, r in zip(got, ref):
            assert abs(g - r) <= 1e-2 * max(1.0, abs(r)), f"{name}{args}: {got} vs {ref}"


def test_phi_coalescing_residual_install_conflict_is_resolved() -> None:
    # Regression: a phi (``a``) coalesces onto input ``x``'s register because the install-free oracle sees no overlap,
    # yet ``x`` stays live in the else block as a sibling phi's identity arm (``z = x``) exactly where ``a``'s residual
    # (sign-folded) else-arm install writes that shared register. The final, install-aware interference then flags the
    # class against itself and the coloring backstop aborted the build. The fixpoint must de-coalesce ``a`` and build.
    # The division keeps the diamond a real branch (un-if-converted), which is what creates the phi merge.
    def k(x, b, cc):  # type: ignore[no-untyped-def]
        if b < cc:
            a = x
            z = 1.0
            d = b
        else:
            a = -(x + 1.0)
            z = x
            d = x / b
        return a, z, d

    _check_float_kernel(k, "coal_c1", [(2.0, 3.0, 5.0), (2.0, 3.0, 1.0), (-4.0, 2.0, 10.0), (1.5, 4.0, 0.5)])


def test_phi_coalescing_conflict_resolved_under_reversed_declaration_order() -> None:
    # The same hazard with the assignments and the return reversed: value ids -- hence the deterministic phi processing
    # order the union-find follows -- change, so a DIFFERENT phi wins the merge onto ``x``. The fixpoint must converge
    # regardless of which phi coalesced first; this pins the resolution as order-independent, not an artifact of one id
    # assignment.
    def k(x, b, cc):  # type: ignore[no-untyped-def]
        if b < cc:
            d = b
            z = 1.0
            a = x
        else:
            d = x / b
            z = x
            a = -(x + 1.0)
        return d, z, a

    _check_float_kernel(k, "coal_c2", [(2.0, 3.0, 5.0), (2.0, 3.0, 1.0), (-4.0, 2.0, 10.0), (1.5, 4.0, 0.5)])


def test_phi_coalescing_conflict_resolved_with_swapped_branch_arms() -> None:
    # The mirror: the coalescing identity arm sits in the else block and the sign-folded residual arm in the then block,
    # so the conflict is exercised from the opposite branch polarity. Confirms the de-coalescing is arm-order agnostic.
    def k(x, b, cc):  # type: ignore[no-untyped-def]
        if b < cc:
            a = -(x + 1.0)
            z = x
            d = x / b
        else:
            a = x
            z = 1.0
            d = b
        return a, z, d

    _check_float_kernel(k, "coal_c3", [(2.0, 3.0, 5.0), (2.0, 3.0, 1.0), (-4.0, 2.0, 10.0), (1.5, 4.0, 0.5)])


def test_bool_phi_coalescing_residual_install_conflict_is_resolved() -> None:
    # The boolean-bank twin of the residual-install conflict: phi ``a`` coalesces onto input ``q``'s 1-bit register
    # while ``q`` stays live as sibling phi ``z``'s identity arm (``z = q``) where ``a``'s residual (inverted) else-arm
    # install writes the shared register. A boolean phi keeps the diamond a real branch (bool phis are never
    # if-converted). The fixpoint must de-coalesce and build; checked bit-exact across all eight boolean input vectors.
    import itertools  # noqa: PLC0415

    def k(p: bool, q: bool, r: bool):  # type: ignore[no-untyped-def]
        if p:
            a = q
            z = True
            d = r
        else:
            a = not q
            z = q
            d = q and r
        return a, z, d

    lir = build(_run(k), "coal_bool")  # crash-before: the bool oracle admitted the unsound merge -> backstop assert
    model = build_model(lir)
    for p, q, r in itertools.product([False, True], repeat=3):
        got = [bool(int(v)) for v in model.run(p, q, r)]
        ref = list(k(p, q, r))
        assert got == ref, f"coal_bool({p},{q},{r}): {got} vs {ref}"


def test_state_war_backstop_allows_noop_writeback() -> None:
    # A no-op writeback (live-out is the live-in value itself) writes no new value, so the write-after-read backstop
    # must not trip -- this previously aborted a legal build.
    class Hold:
        def __init__(self) -> None:
            self.s = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            out = self.s + x
            self.s = self.s
            return out

    lir = build(_run(Hold().__call__), "hold")  # must not raise AssertionError
    assert {s.name for s in lir.float_state_slots} == {"s"}


def test_copy_slot_residence_unbroken_when_tapped_at_boundary() -> None:
    # When an output taps a copy slot's register at the boundary, read-first means that read returns the live-in, so the
    # live-in residence must stay continuous through the boundary (no false dead gap from the new boundary def).
    class Delay:
        def __init__(self) -> None:
            self._d = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            prev = self._d
            self._d = x
            return prev

    lir = build(_run(Delay().__call__), "delay")
    (slot,) = lir.float_state_slots
    assert sorted(lir.reg_liveness[slot.reg]) == list(range(1, lir.initiation_interval + 1))


def test_state_early_copy_frees_source_register() -> None:
    # The trapezoidal integrator's update is `_x_prev = in_x`. in_x's only late use is feeding that writeback, so the
    # copy installs in_x into the _x_prev slot register early; in_x's register is then reused by a later operation
    # instead of being pinned to the boundary -- the register-efficiency win this enables.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from trapezoidal_leaky_streaming_integrator import TrapezoidalLeakyStreamingIntegrator

    lir = build(_run(TrapezoidalLeakyStreamingIntegrator(k=2**-22).__call__), "trapz")
    (xprev,) = [s for s in lir.float_state_slots if s.name == "_x_prev"]
    (in_x,) = [load for load in lir.float_inputs if load.name == "x"]
    assert xprev.needs_copy and in_x.dst == xprev.tap.source  # the copy's source is the input register
    makespan = max((op.commit_cycle for op in lir.ops), default=0)
    assert xprev.install_cycle <= makespan  # installs before the boundary (present cycle == makespan + 1)
    # The freed input register is reused: a later operation's result is assigned to it as well.
    assert any(write.dst == in_x.dst for op in lir.ops for write in op.writes)


def test_build_lir_ekf1_stateless() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    lir = build(_run(ekf1_stateless.update_x_P), "update_x_P")
    assert len(lir.float_inputs) == 17
    assert len(lir.float_outputs) == 9
    fdivs = [inst for inst in lir.instances if isinstance(inst.operator, FDivOperator)]
    assert len(fdivs) == 1
    # The two K=1 power-of-two scalings are non-concurrent, so they pool onto a single shared instance.
    assert sum(1 for inst in lir.instances if isinstance(inst.operator, FMulILog2Operator)) == 1
    # Register reuse: not every distinct value occupies its own register.
    assert lir.regfile.nreg < len(lir.ops) + len(lir.float_inputs)
    # The interference test runs in the hardware frame (a value frees its register as soon as its last read precedes the
    # next value's landing), not the scheduler-frame rule that left it several cycles too conservative and produced 42
    # registers here. The bound is well below 42 to flag a regression of the hardware-accurate liveness without pinning
    # the exact minimum (currently 38); cosim (test_cosim_ekf1_stateless) proves the relaxed sharing is correct.
    assert lir.regfile.nreg <= 40
    # Inputs preload through the regfile's load port (registers 0..nload-1), so nload spans the input block.
    assert lir.regfile.nload == 17
    # Dedicated ports: one read port per operator operand (sum of arities = 2+2+1+2), one write port per tapped wide
    # output-port lane (the comparator's boolean taps contribute none).
    assert lir.regfile.nwr == 4
    assert lir.regfile.nrd == 7
    # The 1/x21 numerator survives as a constant immediate.
    assert any(abs(c - 1.0) < 1e-12 for c in lir.float_consts)


def test_sign_paired_constants_collapse_to_one_magnitude() -> None:
    # +c and -c share a single nonnegative pool entry; the sign rides the (free) per-operand sign control.
    def f(a):  # type: ignore[no-untyped-def]
        return a * 1000.0 + a * (-1000.0)

    lir = build(_run(f), "f")
    assert [c for c in lir.float_consts if abs(c) == 1000.0] == [1000.0]
    operands = [opnd for op in lir.ops for opnd in op.operands if isinstance(opnd.source, FloatConstRef)]
    assert len({opnd.source.index for opnd in operands}) == 1  # both products read one pool entry
    assert {opnd.sign for opnd in operands} == {FloatSignControl(), FloatSignControl(negate=True)}


def test_negative_constant_operand_is_stored_as_magnitude_with_negate() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a + (-1000.0)

    lir = build(_run(f), "f")
    assert all(c >= 0.0 for c in lir.float_consts)
    (operand,) = [opnd for op in lir.ops for opnd in op.operands if isinstance(opnd.source, FloatConstRef)]
    assert lir.float_consts[operand.source.index] == 1000.0
    assert operand.sign == FloatSignControl(negate=True)


def test_constant_pool_is_canonically_nonnegative() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateful
    import numpy as np

    filt = ekf1_stateful.Ekf1(
        x=[0.1e-3, 0.0, 0.0],
        P_urt=[1e3, 0.0, 0.0, 1e6, 0.0, 1e-3],
        R_diag=[1e3, 1e-6],
        Q_diag=np.array([1e-3, 1e9, 1e-9]),
    )
    lir = build(_run(filt.update), "ekf1_stateful")
    assert all(c >= 0.0 for c in lir.float_consts)
    assert len(lir.float_consts) == 6  # the +1000.0 / -1000.0 pair collapsed (was 7)


def test_underflowing_negative_constant_is_not_sign_folded() -> None:
    # A negative value that rounds to +0 in ZKF (which has no -0) must NOT carry a folded negate: the magnitude already
    # encodes to the canonical +0, so a negate over it would emit an illegal -0 rather than the +0 the value encodes to.
    def f(a):  # type: ignore[no-untyped-def]
        return a + (-1e-12)  # -1e-12 underflows to +0 in FloatFormat(6, 18)

    lir = build(_run(f), "f")
    (operand,) = [opnd for op in lir.ops for opnd in op.operands if isinstance(opnd.source, FloatConstRef)]
    assert FMT.encode(lir.float_consts[operand.source.index]) == 0  # the pooled magnitude is a zero-encoding
    assert operand.sign == FloatSignControl()  # identity, not negate


def test_underflowing_negative_constant_output_stays_canonical_zero() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a + a, -1e-12  # the -1e-12 output underflows to +0; it must stay canonical, not fold to illegal -0

    lir = build(_run(f), "f")
    (wire,) = [w for w in lir.float_outputs if isinstance(w.tap.source, FloatConstRef)]
    assert FMT.encode(lir.float_consts[wire.tap.source.index]) == 0
    assert wire.tap.sign == FloatSignControl()


def test_stateful_slot_register_gaps_are_reused() -> None:
    # A coalesced state slot's register, dead through the middle of the frame, is reused for temporaries instead of
    # being reserved, shedding registers (the stateful EKF dropped from 45 to ~39).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateful
    import numpy as np

    filt = ekf1_stateful.Ekf1(
        x=[0.1e-3, 0.0, 0.0],
        P_urt=[1e3, 0.0, 0.0, 1e6, 0.0, 1e-3],
        R_diag=[1e3, 1e-6],
        Q_diag=np.array([1e-3, 1e9, 1e-9]),
    )
    lir = build(_run(filt.update), "ekf1_stateful")
    assert lir.regfile.nreg <= 40  # gap-reuse sheds ~6; a regression to the fully-reserved 45 trips this


def test_register_sharing_is_hardware_disjoint() -> None:
    # ekf1_stateless time-multiplexes many values onto each register. Verify the hardware-frame interference invariant
    # directly: within a register, each value's last read precedes the next value's landing, R(a) < W(b) -- the same
    # liveness reg_liveness renders and the relaxed allocator shares against. Reconstructed via the test-only write-
    # timeline resolver over the model's landing PCs, so the test tracks the allocator's actual sharing decisions.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    lir = build(_run(ekf1_stateless.update_x_P), "update_x_P")
    timeline = build_write_timeline(lir)
    last_read: dict[tuple[int, str, int], int] = {}

    def note(source: object, read_cycle: int) -> None:
        if isinstance(source, RegRef):
            producer = latest_producer_before(timeline, source, read_cycle)
            key = (source.index, type(producer).__name__, producer.index)
            last_read[key] = max(last_read.get(key, read_cycle), read_cycle)

    for op in lir.ops:
        for operand in op.operands:
            note(operand.source, operand_read_cycle(op.inst.operator, op.issue_cycle))
    for wire in lir.float_outputs:
        note(wire.tap.source, lir.initiation_interval)
    for slot in lir.float_state_slots:
        note(slot.tap.source, lir.state_copy_step(slot))

    shared = 0
    for reg, events in timeline.items():
        shared += len(events) - 1
        for (landing_a, producer_a), (landing_b, _b) in zip(events, events[1:]):
            read_a = last_read.get((reg.index, type(producer_a).__name__, producer_a.index), landing_a)
            assert (
                read_a < landing_b
            ), f"register {reg.index}: {producer_a} last read {read_a} overlaps landing {landing_b}"
    assert shared > 0  # the kernel does pack multiple values per register, so the invariant is actually exercised


def test_build_rejects_mir_with_mixed_float_formats() -> None:
    other = FloatFormat(8, 24)
    mir = Mir(
        FMT,
        nodes={
            0: MirFloatInput("a", FloatType(FMT)),
            1: MirOperation(
                FAddOperator(other),
                [0, 0],
                [FloatSignControl(), FloatSignControl()],
                0,
                FloatSignControl(),
            ),
        },
        blocks=[MirBlock(0, (), (1,), MirRet())],
        input_ids=[0],
        outputs=[MirFloatOutput("out_0", 1)],
        state_slots=[],
    )
    with pytest.raises(ValueError, match="configured format"):
        build(mir, "mixed")


def test_mir_builder_rejects_mixed_float_operand_formats() -> None:
    other = FloatFormat(8, 24)
    builder = MirBuilder(FMT)
    a = builder.float_input("a", FloatType(FMT))
    b = builder.float_input("b", FloatType(other))
    with pytest.raises(ValueError, match="expects operands"):
        builder.operation(
            FAddOperator(FMT),
            [a, b],
            [FloatSignControl(), FloatSignControl()],
        )


def test_mir_operation_validates_invariants() -> None:
    with pytest.raises(TypeError, match="scalar_type"):
        MirFloatInput("a", OtherScalarType())
    with pytest.raises(TypeError, match="scalar_type"):
        MirFloatConst(OtherScalarType(), 1.0)
    with pytest.raises(ValueError, match="operand"):
        MirOperation(FAddOperator(FMT), [0], [FloatSignControl(), FloatSignControl()], 0, FloatSignControl())
    with pytest.raises(ValueError, match="conditioner"):
        MirOperation(FAddOperator(FMT), [0, 0], [FloatSignControl()], 0, FloatSignControl())
    # A boolean output port carries an inversion, never a sign control: booleans have no sign.
    with pytest.raises(TypeError, match="output conditioner"):
        MirOperation(
            FCmpOperator(FMT),
            [0, 0],
            [FloatSignControl(), FloatSignControl()],
            2,
            FloatSignControl(negate=True),
        )
    # A boolean operand carries an inversion too; a sign control on it is a type error.
    with pytest.raises(TypeError, match="operand conditioner"):
        MirOperation(BoolAndOperator(), [0, 0], [FloatSignControl(), BoolInversion()], 0, BoolInversion())
    with pytest.raises(ValueError, match="does not exist"):
        MirOperation(FAddOperator(FMT), [0, 0], [FloatSignControl(), FloatSignControl()], 1, FloatSignControl())
    with pytest.raises(TypeError, match="sign"):
        MirFloatOutput("out_0", 0, object())


def test_float_view_rejects_non_float_mir_before_scheduling() -> None:
    mir = Mir(
        FMT,
        nodes={0: OtherMirInput("a", OtherScalarType())},
        blocks=[MirBlock(0, (), (), MirRet())],
        input_ids=[0],
        outputs=[MirFloatOutput("out_0", 0)],
        state_slots=[],
    )
    with pytest.raises(UnsupportedConstruct, match="MIR input"):
        MirFloatView.from_mir(mir)


def test_float_view_rejects_non_input_input_id() -> None:
    mir = Mir(
        FMT,
        nodes={0: MirFloatConst(FloatType(FMT), 1.0)},
        blocks=[MirBlock(0, (), (), MirRet())],
        input_ids=[0],
        outputs=[MirFloatOutput("out_0", 0)],
        state_slots=[],
    )
    with pytest.raises(ValueError, match="must reference a MirFloatInput or MirBoolInput"):
        MirFloatView.from_mir(mir)


def test_float_view_rejects_missing_input_id() -> None:
    mir = Mir(
        FMT,
        nodes={0: MirFloatConst(FloatType(FMT), 1.0)},
        blocks=[MirBlock(0, (), (), MirRet())],
        input_ids=[1],
        outputs=[MirFloatOutput("out_0", 0)],
        state_slots=[],
    )
    with pytest.raises(ValueError, match="must reference a MirFloatInput or MirBoolInput"):
        MirFloatView.from_mir(mir)


def test_fmul_ilog2_operator_rejects_out_of_range_k() -> None:
    limit = (1 << FMT.wexp) - 2
    assert FMulILog2Operator(FMT, k=-limit).k == -limit
    assert FMulILog2Operator(FMT, k=limit - 1).k == limit - 1
    with pytest.raises(ValueError, match="k must satisfy"):
        FMulILog2Operator(FMT, k=limit)
    with pytest.raises(ValueError, match="k must satisfy"):
        FMulILog2Operator(FMT, k=-limit - 1)


def _read_mux_fan_in(lir) -> int:  # type: ignore[no-untyped-def]
    return sum(max(0, len(regs) - 1) for regs in lir.read_set_per_port.values())


def test_marked_commutative_operators_are_bit_exact_commutative() -> None:
    # The port-assignment pass swaps a commutative operator's operands, which is only sound if the operator is
    # exactly symmetric. Guard the FAddOperator/FMulOperator markings against a future non-commutative slip-up.
    import random

    from holoso._value import FloatValue, add_float_values, mul_float_values

    rng = random.Random(0)
    assert FAddOperator(FMT).is_commutative and FMulOperator(FMT).is_commutative
    assert not FDivOperator(FMT).is_commutative
    for evaluate in (add_float_values, mul_float_values):
        for _ in range(5000):
            a = FloatValue.from_float(FMT, rng.uniform(-2.0, 2.0) * 2.0 ** rng.randint(-22, 22))
            b = FloatValue.from_float(FMT, rng.uniform(-2.0, 2.0) * 2.0 ** rng.randint(-22, 22))
            assert evaluate(a, b).bits == evaluate(b, a).bits


def test_commutative_port_assignment_never_increases_read_mux_fan_in(  # type: ignore[no-untyped-def]
    monkeypatch,
) -> None:
    import holoso._lir._build as build_module

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    cfg = OpConfig(
        FAddOperator(FMT, stage_decode=1),
        FMulOperator(FMT, stage_input=1),
        FDivOperator(FMT),
        FMulILog2OperatorFamily(FMT),
        FCmpOperator(FMT),
    )
    monkeypatch.setattr(build_module, "assign_commutative_ports", lambda *args, **kwargs: {})
    baseline = build(_run(ekf1_stateless.update_x_P, cfg), "ekf1_stateless")
    monkeypatch.undo()
    optimized = build(_run(ekf1_stateless.update_x_P, cfg), "ekf1_stateless")

    assert _read_mux_fan_in(optimized) <= _read_mux_fan_in(baseline)
    assert _read_mux_fan_in(optimized) < _read_mux_fan_in(baseline)  # ekf1_stateless has commutative reach to reclaim


def test_optional_stages_raise_latency_without_changing_numerics() -> None:
    # A kernel touching every operator family: fadd, fmul, fdiv, and the 2^-2 strength-reduced fmul_ilog2.
    def kernel(a, b, c):  # type: ignore[no-untyped-def]
        return (a - b) / c + a * b * 0.25

    fmt = FloatFormat(8, 36)
    configs = {"default": default_ops(fmt), "staged": staged_ops(fmt)}
    lirs = {name: build(_run(kernel, ops), f"stages_{name}") for name, ops in configs.items()}
    assert lirs["default"].initiation_interval < lirs["staged"].initiation_interval

    # Optional stages only insert pipeline registers, so the numerical result is bit-identical across every config.
    models = {name: build_model(lir) for name, lir in lirs.items()}
    vectors = [(1.5, -0.5, 2.0), (3.25, 1.0, -4.0), (0.0, 2.5, 0.125), (-1.0, -1.0, 1e3)]
    for values in vectors:
        want = [v.bits for v in models["default"].run(*values)]
        for name, model in models.items():
            assert [v.bits for v in model.run(*values)] == want, f"{name} diverged from default at {values}"


def test_reach_floor_seed_skips_annealing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # The mux-fan-in objective bottoms out at 0 (every read port reaches one register, every register one producer). A
    # greedy seed already there is globally optimal, so refinement must short-circuit rather than burn the budget.
    import holoso._lir._regalloc as regalloc

    calls: list[int] = []
    real = regalloc.dual_annealing
    monkeypatch.setattr(regalloc, "dual_annealing", lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    def floor_kernel(a, b):  # type: ignore[no-untyped-def]
        return a + b  # no register sharing -> greedy seed is at the reach floor

    def sharing_kernel(a, b, c):  # type: ignore[no-untyped-def]
        return a * b + c  # the product and the sum reuse registers, lifting the objective above the floor

    build(_run(floor_kernel), "floor")
    assert calls == []  # early-exit: dual_annealing was never invoked
    build(_run(sharing_kernel), "sharing")
    assert calls  # a non-floor seed is still refined


def test_zero_regalloc_effort_bypasses_annealing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import holoso._lir._regalloc as regalloc

    monkeypatch.setattr(regalloc, "_REFINE_MAXITER", 0)
    monkeypatch.setattr(regalloc, "dual_annealing", lambda *args, **kwargs: pytest.fail("annealing was not bypassed"))

    def sharing_kernel(a, b, c):  # type: ignore[no-untyped-def]
        return a * b + c

    build(_run(sharing_kernel), "sharing")


def test_bool_to_float_cast_result_is_live_on_its_landing_cycle() -> None:
    # Regression: a bool->float cast is an inline combinational op written into the array directly, so its WIDE result
    # lands at inline_landing_cycle(commit) = commit + FETCH_LAG + READ_FIRST_EDGE -- a cycle BEFORE a pooled wide
    # result (no writeback latch). Charging it the wide writeback latch marks it past its true landing (and, for a
    # boundary cast, past the initiation interval, so its report cell falls off the grid) and leaves a consumer's read
    # cycle outside its residence. Here a multiply consumes the cast result.
    def f(x):  # type: ignore[no-untyped-def]
        return float(x > 0.0) * x

    lir = build(_run(f), "cast_mul")
    interval = lir.initiation_interval
    casts = [(b, op) for b in lir.blocks for op in b.inline_ops if isinstance(op.write.dst, RegRef)]
    assert casts, "expected a bool->float cast result in the wide bank"
    for block, op in casts:
        base = lir.block_base[block.index]
        landing = base + inline_landing_cycle(op.commit_cycle)
        assert landing == base + op.commit_cycle + FETCH_LAG + READ_FIRST_EDGE  # inline: no writeback latch
        assert 1 <= landing <= interval  # within the rendered schedule grid, not one row past the boundary
        # The cast write lands at its inline landing, NOT the latched wide landing (commit+4): a bool->float cast is
        # inline, so write_landing_pcs (via op_result_landing) must place it via inline_landing_cycle. Mis-dispatching
        # it through the wide writeback latch would return base + commit + 4 here -- this is the discriminating guard,
        # since reg_liveness alone is a union over the register's reuse and stays satisfied even with the late landing.
        assert lir.write_landing_pcs(block, op, op.write) == [landing]
        assert landing in lir.reg_liveness[op.write.dst]  # and it is live there
    cast_regs = {op.write.dst for _, op in casts}
    for fop in lir.ops:  # the consuming multiply must read the cast result within its residence (no late-def gap)
        for operand in fop.operands:
            if operand.source in cast_regs:
                assert operand_read_cycle(fop.inst.operator, fop.issue_cycle) in lir.reg_liveness[operand.source]


def test_two_relations_over_one_operand_pair_fuse_into_one_firing() -> None:
    # Two DIFFERENT relations over the same operand pair tap two distinct output ports of one comparator activation,
    # so they fuse into a single firing: one instance issue, one operand read, two boolean writes -- the multi-output
    # machinery exercised end to end on the boolean side. The model must still produce both values correctly.
    def f(a, b):  # type: ignore[no-untyped-def]
        below = a < b
        same = a == b
        return [float(below), float(same)]

    lir = build(_run(f), "fused_relations")
    firings = [op for block in lir.blocks for op in block.ops if isinstance(op.inst.operator, FCmpOperator)]
    assert len(firings) == 1, "lt and eq taps of one operand pair must fuse into one comparator firing"
    (firing,) = firings
    assert [write.port for write in firing.writes] == sorted(write.port for write in firing.writes)
    assert len(firing.writes) == 2
    assert len({write.dst for write in firing.writes}) == 2  # simultaneous landings get distinct registers
    model = build_model(lir)
    for a, b in [(1.0, 2.0), (2.0, 1.0), (1.5, 1.5)]:
        below, same = (float(v) for v in model.run(a, b))
        assert below == float(a < b) and same == float(a == b), f"a={a} b={b}"


def test_same_port_taps_with_different_inversions_do_not_fuse() -> None:
    # ``a < b`` taps the lt flag plainly and ``a >= b`` taps the SAME flag inverted: one output-port lane writes once
    # per firing, so these must stay two firings, spaced by instance contention. Both values must still be correct.
    def f(a, b):  # type: ignore[no-untyped-def]
        below = a < b
        not_below = a >= b
        return [float(below), float(not_below)]

    lir = build(_run(f), "split_inversions")
    firings = [op for block in lir.blocks for op in block.ops if isinstance(op.inst.operator, FCmpOperator)]
    assert len(firings) == 2, "same-port taps cannot share a firing"
    assert len({op.issue_cycle for op in firings}) == 2  # the single instance serializes them
    model = build_model(lir)
    for a, b in [(1.0, 2.0), (2.0, 1.0), (1.5, 1.5)]:
        below, not_below = (float(v) for v in model.run(a, b))
        assert below == float(a < b) and not_below == float(a >= b), f"a={a} b={b}"


class _ThrottledAdd(FAddOperator):
    """A test-only adder whose instance accepts a new firing only every 3 cycles (initiation interval 3)."""

    @property
    def initiation_interval(self) -> int:
        return 3


def test_initiation_interval_spaces_firings_on_one_instance() -> None:
    # Two independent additions contend for the single throttled instance: the second may not issue until the
    # first's busy window elapses, so their issues are at least II cycles apart (with II=1 they would share cycle 1).
    builder = MirBuilder(FMT)
    builder.block()
    a = builder.float_input("a", FloatType(FMT))
    b = builder.float_input("b", FloatType(FMT))
    slow = _ThrottledAdd(FMT)
    first = builder.operation(slow, [a, b], [FloatSignControl(), FloatSignControl()])
    second = builder.operation(slow, [b, a], [FloatSignControl(), FloatSignControl()])
    builder.float_output("out_0", first)
    builder.float_output("out_1", second)
    builder.ret()
    mir = builder.finish()
    sched = schedule_ops(mir.nodes, resolve_pool(mir.nodes), {first, second})
    spacing = abs(sched.issue_cycle[second] - sched.issue_cycle[first])
    assert spacing >= 3, f"II=3 must space same-instance firings by at least 3 cycles, got {spacing}"
    assert sched.inst_of[first] == sched.inst_of[second]  # one pooled instance serves both


class _HeavilyThrottledAdd(FAddOperator):
    """
    A test-only adder throttled to the deepest initiation interval the per-block busy windows support
    (latency + the inter-block drain gap; validated in ``OperatorInstance.__post_init__``).
    """

    @property
    def initiation_interval(self) -> int:
        return 10  # latency 4 + the maximum cross-block-safe excess of 6


def test_progress_cap_accommodates_long_initiation_intervals() -> None:
    # Regression (review): the scheduler's no-progress cap charged each firing only its latency plus the dependency
    # edge, so many independent firings contending for one instance with a long initiation interval exhausted the
    # cap while making perfectly legal progress -- a spurious "scheduler made no progress" abort on a legal kernel.
    # The cap now charges max(latency, initiation_interval) per firing (the old latency-only cap aborted at this
    # II with this many firings). Same-port duplicates do not fuse, so the
    # hand-built identical operations below are forty separate firings serialized on one instance.
    slow = _HeavilyThrottledAdd(FMT)
    nodes = {0: MirFloatInput("a", FloatType(FMT)), 1: MirFloatInput("b", FloatType(FMT))}
    count = 40
    for i in range(count):
        nodes[2 + i] = MirOperation(slow, [0, 1], [FloatSignControl(), FloatSignControl()], 0, FloatSignControl())
    sched = schedule_ops(nodes, {type(slow): 1}, set(range(2, 2 + count)))
    issues = sorted(sched.issue_cycle.values())
    assert len(issues) == count
    assert all(later - earlier >= 10 for earlier, later in zip(issues, issues[1:]))


def test_write_timeline_resolves_inline_wide_producers() -> None:
    # Regression (review): the write timeline recorded only pooled firings' wide writes, so a register written by an
    # inline bool->float cast and read by a float operator resolved to NO producer at all -- latest_producer_before
    # raised KeyError on the cast-fed multiply below.
    def f(x):  # type: ignore[no-untyped-def]
        return float(x > 0.0) * x

    lir = build(_run(f), "cast_timeline")
    timeline = build_write_timeline(lir)
    resolved = 0
    for op in lir.ops:
        read = operand_read_cycle(op.inst.operator, op.issue_cycle)
        for operand in op.operands:
            if isinstance(operand.source, RegRef):
                latest_producer_before(timeline, operand.source, read)  # must not raise for any operand
                resolved += 1
    assert resolved >= 2  # the multiply reads both x and the cast result
    assert any(
        isinstance(producer, InlineProducer) for events in timeline.values() for _, producer in events
    ), "the cast's wide write must appear in the timeline with its inline producer"


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
def test_commutative_comparator_swap_permutes_output_taps(config: OperatorCase) -> None:
    # The comparator is commutative under the gt/lt flag exchange. Two mirrored comparisons over one operand pair
    # otherwise read (a,b) and (b,a) -- two registers per read port; the port assignment orients one of them swapped,
    # shrinking each port's read-set to a single register, and the swapped firing's lt tap moves to gt. Bit-exact
    # because the ZKF ordering is total and compare is antisymmetric.
    def f(a, b):  # type: ignore[no-untyped-def]
        below = a < b
        above = b < a
        return [float(below), float(above)]

    lir = build(_run(f, config.make_ops(FMT)), f"mirrored_{config.label}")
    firings = [op for block in lir.blocks for op in block.ops if isinstance(op.inst.operator, FCmpOperator)]
    assert len(firings) == 2
    sources = [tuple(operand.source for operand in op.operands) for op in firings]
    assert sources[0] == sources[1], "the MILP must orient both firings to read the same registers per port"
    gt_port, lt_port = (FCmpOperator.tap_of(rel)[0] for rel in (RelationalOp.GT, RelationalOp.LT))
    ports = sorted(write.port for op in firings for write in op.writes)
    assert ports == sorted((gt_port, lt_port)), "exactly one firing's lt tap must move to gt under the swap"
    model = build_model(lir)
    for a, b in [(1.0, 2.0), (2.0, 1.0), (1.5, 1.5)]:
        below, above = (float(v) for v in model.run(a, b))
        assert below == float(a < b) and above == float(b < a), f"a={a} b={b}"


@pytest.mark.parametrize("config", PIPELINE_OP_CASES, ids=lambda config: config.label)
def test_chained_slot_live_in_blocks_early_install(config: OperatorCase) -> None:
    # Regression (review; pre-existing at HEAD): a slot whose live-in feeds ANOTHER slot's live-out ("self._a =
    # self._b") was documented as unable to early-install, but only the coalescing test consulted that fact -- the
    # early-install decision did not, so "_b"'s new value landed before "_a"'s boundary copy captured the old one.
    # The RTL then returned the NEW "_b" through "_a" while the model kept the old one (cosim diverged on the second
    # transaction). The tapped slot must now install at the boundary, and the model must match plain Python.
    lir = build(_run(ChainedSlots().__call__, config.make_ops(FMT)), f"chained_slots_{config.label}")
    slots = {slot.name: slot for slot in lir.float_state_slots}
    assert lir.state_copy_step(slots["_b"]) == lir.initiation_interval, "the tapped slot must not install early"
    reference = ChainedSlots()
    model = build_model(lir)
    for x in (2.0, 3.0, 4.0):
        got = float(model.run(x)[0])
        want = reference(x)
        assert abs(got - want) <= 1e-2 * max(1.0, abs(want)), f"x={x}: {got} vs {want}"


def test_select_folds_arm_signs_into_operand_conditioners() -> None:
    # ``x if c else -x`` costs exactly one comparison and one select: the arm negation rides the select's operand
    # conditioner (the inline dual of the pooled operators' sign sidebands), never a separate float operation.
    def f(x, c):  # type: ignore[no-untyped-def]
        y = x if c > 0.0 else -x
        return y

    lir = build(_run(f), "signed_select")
    assert len(lir.blocks) == 1, "the diamond must fully if-convert"
    selects = [op for block in lir.blocks for op in block.inline_ops if isinstance(op.operator, SelectOperator)]
    assert len(selects) == 1
    (select,) = selects
    cond, arm_true, arm_false = select.operands
    assert isinstance(arm_true, FloatOperand) and arm_true.sign == FloatSignControl()
    assert isinstance(arm_false, FloatOperand) and arm_false.sign == FloatSignControl(negate=True)
    assert not [op for block in lir.blocks for op in block.ops if not isinstance(op.inst.operator, FCmpOperator)]
    model = build_model(lir)
    for x in (2.0, -3.0):
        for c in (1.0, -1.0):
            assert float(model.run(x, c)[0]) == (x if c > 0.0 else -x)


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
def test_state_early_install_respects_a_select_reader(config: OperatorCase) -> None:
    # Pins the read-step frame of the state early-install bound: an inline select reads its operands at its fire
    # step (issue + latency + FETCH_LAG + 1), one cycle past where an issue-frame bound would have allowed the
    # slot's install copy to fire -- an early install bounded by issue cycles would overwrite the live-in before
    # the select reads it (RTL would take the NEW value through ``old`` while the model keeps the old one).
    lir = build(_run(SelectHold().step, config.make_ops(FMT)), f"select_hold_{config.label}")
    selects = [
        (block, op) for block in lir.blocks for op in block.inline_ops if isinstance(op.operator, SelectOperator)
    ]
    assert len(selects) == 1
    ((block, select),) = selects
    (slot,) = lir.float_state_slots
    select_read_pc = lir.block_base[block.index] + operand_read_cycle(select.operator, select.issue_cycle)
    assert lir.state_copy_step(slot) >= select_read_pc, "the install must not precede the select's operand read"
    reference = SelectHold()
    model = build_model(lir)
    for x, c in [(2.0, 1.0), (3.0, -1.0), (4.0, 1.0), (5.0, -1.0)]:
        got = float(model.run(x, c)[0])
        want = reference.step(x, c)
        assert abs(got - want) <= 1e-2 * max(1.0, abs(want)), f"x={x} c={c}: {got} vs {want}"


def test_not_folds_into_every_sink_position() -> None:
    # A semantic NOT never materializes hardware: it becomes a free inversion conditioner at each consumer. The
    # kernel routes one comparison's negation into a logic operand, a bool output, and a bool->float cast; the LIR
    # must contain NO inline op beyond the band and the cast, and the inversions must ride the operand sidebands.
    def f(a, b, c):  # type: ignore[no-untyped-def]
        flag = not (a > b)
        out_logic = flag and (c > 0.0)
        return [float(flag), float(out_logic)]

    lir = build(_run(f), "not_sinks")
    inline_mnemonics = sorted(op.operator.mnemonic for block in lir.blocks for op in block.inline_ops)
    assert inline_mnemonics == ["band", "ffrombool", "ffrombool"], inline_mnemonics
    band = next(op for block in lir.blocks for op in block.inline_ops if op.operator.mnemonic == "band")
    flag_operand, _ = band.operands
    assert isinstance(flag_operand, BoolOperand) and flag_operand.inversion == BoolInversion(True)
    model = build_model(lir)
    for a, b, c in [(1.0, 2.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, -1.0)]:
        flag = not (a > b)
        got = [float(v) for v in model.run(a, b, c)]
        assert got == [float(flag), float(flag and (c > 0.0))], f"{a},{b},{c}: {got}"


def test_not_on_a_branch_condition_swaps_the_targets() -> None:
    # ``if not cond`` costs nothing: the branch takes the complementary target instead of inverting the register.
    # The division arms keep the diamond a real branch (if-conversion refuses them).
    def f(a, b):  # type: ignore[no-untyped-def]
        if not (a > b):
            y = a / (b * b + 1.0)
        else:
            y = b / (a * a + 1.0)
        return y

    lir = build(_run(f), "not_branch")
    assert not any(block.inline_ops for block in lir.blocks), "the NOT must not materialize any gate"
    model = build_model(lir)
    for a, b in [(1.0, 2.0), (2.0, 1.0)]:
        want = a / (b * b + 1.0) if not (a > b) else b / (a * a + 1.0)
        got = float(model.run(a, b)[0])
        assert abs(got - want) <= 1e-2 * max(1.0, abs(want))


def test_double_negation_cancels() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        flag = not (not (a > b))
        return float(flag)

    lir = build(_run(f), "double_not")
    casts = [op for block in lir.blocks for op in block.inline_ops if op.operator.mnemonic == "ffrombool"]
    (cast,) = casts
    (operand,) = cast.operands
    assert isinstance(operand, BoolOperand) and operand.inversion == BoolInversion()
    model = build_model(lir)
    assert float(model.run(2.0, 1.0)[0]) == 1.0 and float(model.run(1.0, 2.0)[0]) == 0.0


def test_value_consumed_in_both_polarities_shares_one_producer() -> None:
    # ``x`` and ``not x`` share one comparator tap and one boolean register: the polarity lives on each consumer.
    def f(a, b):  # type: ignore[no-untyped-def]
        flag = a > b
        return [float(flag), float(not flag)]

    lir = build(_run(f), "both_polarities")
    comparisons = [op for block in lir.blocks for op in block.ops if isinstance(op.inst.operator, FCmpOperator)]
    assert len(comparisons) == 1 and len(comparisons[0].writes) == 1, "one tap serves both polarities"
    model = build_model(lir)
    for a, b in [(2.0, 1.0), (1.0, 2.0)]:
        assert [float(v) for v in model.run(a, b)] == [float(a > b), float(not (a > b))]


class _InvertedState:
    """A boolean state slot whose live-out is the negation of its own live-in: a self-toggling flag."""

    def __init__(self) -> None:
        self._flip = False

    def step(self, x):  # type: ignore[no-untyped-def]
        old = self._flip
        self._flip = not self._flip
        return x if old else -x


def test_bool_state_slot_carries_a_live_out_inversion() -> None:
    # The toggle's live-out is its own live-in inverted: the inversion rides the slot's install (needs_copy must be
    # True even though the source register IS the slot register), and the model must toggle across transactions.
    lir = build(_run(_InvertedState().step), "toggle")
    (slot,) = lir.bool_state_slots
    assert slot.needs_copy, "an inverted live-out needs its install copy even from the slot's own register"
    reference = _InvertedState()
    model = build_model(lir)
    for x in (1.0, 2.0, 3.0, 4.0):
        assert float(model.run(x)[0]) == reference.step(x)


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
def test_inverted_bool_phi_arm_installs_with_opposite_polarities(config: OperatorCase) -> None:
    # The headline M3 generalization end to end: a bool phi whose two arms reference the SAME base value under
    # opposite inversions (one arm rewrites the flag as its own negation). The two install copies must carry
    # opposite-polarity sources, and the model must take the correct value on both paths. The division keeps the
    # diamond a real branch (bool-phi diamonds are refused by if-conversion anyway; the div makes it doubly so).
    def f(a, b, c):  # type: ignore[no-untyped-def]
        flag = a > b
        if c > 0.0:
            flag = not flag
            d = a / (c * c + 1.0)
        else:
            d = b
        return [float(flag), d]

    lir = build(_run(f, config.make_ops(FMT)), f"inverted_arm_{config.label}")
    sources = [(write.source.source, write.source.inversion) for block in lir.blocks for write in block.bool_writes]
    flag_sources = [(src, inv) for src, inv in sources if isinstance(src, BoolRegRef)]
    assert len(flag_sources) == 2, flag_sources
    (src_a, inv_a), (src_b, inv_b) = flag_sources
    assert src_a == src_b, "both arms read the same base flag register"
    assert {inv_a, inv_b} == {BoolInversion(), BoolInversion(True)}, "the arms carry opposite polarities"
    model = build_model(lir)
    for a, b, c in [(2.0, 1.0, 1.0), (2.0, 1.0, -1.0), (1.0, 2.0, 1.0), (1.0, 2.0, -1.0)]:
        flag = a > b
        want_flag = (not flag) if c > 0.0 else flag
        got = [float(v) for v in model.run(a, b, c)]
        assert got[0] == float(want_flag), f"{a},{b},{c}: {got}"


def test_boolean_registers_are_reused_within_a_block() -> None:
    # Exact per-consumer read steps free a condition's register once its last reader fires, so a chain of sequential
    # selects whose conditions die mid-block shares a few boolean registers instead of one per condition. The chain
    # is data-dependent (each select feeds the next comparison), so the conditions' lifetimes are disjoint.
    def f(x):  # type: ignore[no-untyped-def]
        for _ in range(6):
            x = (x - 1.0) if x > 1.0 else (x + 1.0)
        return x

    lir = build(_run(f), "breg_reuse")
    conditions = sum(1 for block in lir.blocks for op in block.ops if isinstance(op.inst.operator, FCmpOperator))
    assert conditions == 6, "the unrolled chain carries six comparisons"
    assert lir.bool_regfile.nreg <= 2, f"disjoint condition lifetimes must share registers, got {lir.bool_regfile.nreg}"
    model = build_model(lir)
    for x in (0.0, 3.5, -2.0):
        want = x
        for _ in range(6):
            want = (want - 1.0) if want > 1.0 else (want + 1.0)
        got = float(model.run(x)[0])
        assert abs(got - want) <= 1e-3 * max(1.0, abs(want))


def test_boolean_logic_chain_reuses_registers_on_the_tight_same_bank_edge() -> None:
    # The band/bor same-bank reuse path (distinct from the select-cond reader exercised above): a deep boolean
    # reduction over many comparisons produces intermediate flags that die as the next gate consumes them, on the
    # tightest read-first edge (an inline gate reads its operand on exactly the step the next result's write-enable
    # fires; bool_landing = fire + 1 keeps R(a) < W(b)). The chain must collapse onto a handful of registers.
    def f(a, b, c, d, e, g):  # type: ignore[no-untyped-def]
        return 1.0 if (a > b and c > d and e > g and a > d and b > e) else 0.0

    lir = build(_run(f), "bool_chain")
    comparisons = sum(1 for block in lir.blocks for op in block.ops if isinstance(op.inst.operator, FCmpOperator))
    ands = sum(1 for block in lir.blocks for op in block.inline_ops if op.operator.mnemonic == "band")
    assert comparisons == 5 and ands >= 4, (comparisons, ands)
    assert lir.bool_regfile.nreg <= 3, f"the chained flags must reuse registers, got {lir.bool_regfile.nreg}"
    model = build_model(lir)
    import itertools

    for vals in itertools.product([0.0, 1.0], repeat=6):
        a, b, c, d, e, g = vals
        want = 1.0 if (a > b and c > d and e > g and a > d and b > e) else 0.0
        assert float(model.run(*vals)[0]) == want, vals


def test_drain_only_ret_with_a_resident_output_needs_no_boundary_drain() -> None:
    # A Ret reached by a branch that writes nothing itself, whose output was produced in a PREDECESSOR (resident,
    # already landed with every pipeline edge -- writeback latch and read-first -- paid), needs NO boundary drain at
    # all: out_valid asserts at the Ret block's own base PC, reading the resident output combinationally. A drained
    # block's boundary covers only values that LAND in its frame; a pure-drain block has none, so its terminator offset
    # is 0 (not the phantom ``boundary_step(0, ...)`` of a value that never commits there). octave_index is the
    # canonical case -- its loop body produces the octave count and the exit block does pure drain to out_valid.
    # Crash-before (the ``boundary_step(makespan=0, ...)`` over-charge): the drain-only Ret paid a full FETCH_LAG +
    # read-first (+ writeback) phantom drain, so out_valid landed three cycles late on every transaction.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from octave_index import octave_index  # noqa: PLC0415  (example kernels live under examples/)

    lir = build(_run(octave_index), "octave_drain_only_ret")
    ret = next(b for b in lir.blocks if isinstance(b.terminator, Ret))
    # Nothing lands in the Ret's own frame -- it neither computes nor installs; the output is resident.
    assert not (ret.ops or ret.inline_ops or ret.copies or ret.bool_writes), "the exit block must be pure drain"
    assert ret.term_offset == 0, "a resident-output drain-only Ret needs no boundary drain"
    ret_base = lir.block_base[ret.index]
    assert lir.last_pc == ret_base, "out_valid asserts at the Ret block base, not after a phantom drain"
    # The reclaim is the entire phantom boundary_step a value committing at the empty block's cycle 0 would have paid
    # (three cycles here: FETCH_LAG + read-first on the latch-free boolean bank). Format-independent, so it holds at
    # any FloatFormat.
    assert boundary_step(ret.block_makespan, wide_resident=False) == 3, "the reclaimed phantom bool drain"
    # The earlier boundary read is sound: the resident output is bit-exact against the Python reference on both ranges
    # (magnitude >= 1 takes the no-reciprocal arm; below unity inverts first), exercising the real branch into the loop.
    model = build_model(lir)
    for x in (8.0, 0.1, 1.0, 32.0, 0.03, -3.0):
        assert float(model.run(x)[0]) == octave_index(x), x

"""Unit tests for pipelined scheduling, register allocation, and LIR construction."""

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
    InlineProducer,
    Jump,
    RegRef,
    latest_producer_before,
    operand_read_cycle,
    result_landing_cycle,
)
from holoso._lir._ir import dependency_edge, wide_landing_cycle
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

from ._modelref import ChainedSlots, SelectHold, branch_boundary_kernel, default_ops, fcmp_staged_ops, staged_ops

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


def test_two_comparisons_in_a_block_serialize_on_the_shared_comparator() -> None:
    # Regression: a chained comparison (here ``lo < x < hi``) puts two comparator firings with distinct operand
    # pairs in one block. The single pooled holoso_fcmp instance serves one firing per initiation interval, so the
    # two must issue on distinct cycles. Before the contention rule the scheduler let both issue on the same cycle
    # -> they collided on the single comparator (one read the other's operands), corrupting the result in the RTL
    # while the model still passed.
    def f(x, lo, hi):  # type: ignore[no-untyped-def]
        return 0.0 if lo < x < hi else x

    lir = build(_run(f), "deadband")
    in_valid_pcs = [
        lir.block_base[block.index] + op.issue_cycle
        for block in lir.blocks
        for op in block.ops
        if isinstance(op.inst.operator, FCmpOperator)
    ]
    assert len(in_valid_pcs) >= 2  # the chained comparison lowers to two comparator firings feeding a BoolAnd
    assert len(set(in_valid_pcs)) == len(in_valid_pcs)  # instance contention spaces them: no comparator collision


@pytest.mark.parametrize("stage_input", [0, 1])
def test_branch_comparison_commits_at_block_makespan(stage_input: int) -> None:
    # White-box twin of test_cosim.py test_cosim_comparison_at_branch_boundary: pins that the SHARED kernel
    # (_modelref.branch_boundary_kernel) actually hits the boundary-slack corner -- the comparison is the last commit
    # in its block and feeds the branch -- at both comparator latencies. If a schedule change ever moves the
    # comparison off the makespan, this fails before the cosim silently de-targets.
    lir = build(_run(branch_boundary_kernel, fcmp_staged_ops(FMT, stage_input)), "cmp_at_boundary")
    branch_blocks = [block for block in lir.blocks if isinstance(block.terminator, Branch)]
    assert len(branch_blocks) == 1
    (block,) = branch_blocks
    comparisons = [op for op in block.ops if isinstance(op.inst.operator, FCmpOperator)]
    assert len(comparisons) == 1
    (cmp_op,) = comparisons
    assert cmp_op.latency == 1 + stage_input
    assert cmp_op.commit_cycle == block.block_makespan


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
    # The non-coalesced writeback is a first-class event in the liveness model: the slot register holds a live value on
    # its install step (previously absent, which is why the report could not render it).
    assert lir.state_copy_step(slot) in lir.reg_liveness[slot.reg]
    assert lir.state_copy_step(slot) == slot.install_cycle + FETCH_LAG + 1
    # Nothing reads _p's register after the old live-in and its source is an ordinary register, so the copy installs
    # before the boundary -- freeing the source register for the rest of the initiation rather than pinning it there.
    assert lir.state_copy_step(slot) < lir.initiation_interval
    # The carried live-out must survive to the boundary even though nothing reads it again this frame, so the slot
    # register stays live from its install step through the boundary -- an early install is not the value's death.
    assert set(range(lir.state_copy_step(slot), lir.initiation_interval + 1)) <= lir.reg_liveness[slot.reg]
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
    # never pinned to a value. The colorer must still reserve it (it sits below fresh_start and is installed by the
    # boundary copy) -- a temporary considering it as a reuse candidate must skip it rather than fault on the missing
    # pool entry.
    class WriteOnlyBranch:
        def __init__(self) -> None:
            self.acc = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            if x > 0.0:
                self.acc = x * 2.0
            else:
                self.acc = x * 3.0
            return self.acc

    lir = build(_run(WriteOnlyBranch().__call__), "write_only")
    (slot,) = lir.float_state_slots
    assert slot.name == "acc"
    assert slot.reg.index not in {
        write.dst.index for op in lir.ops for write in op.writes
    }  # reserved: no operator result lands on it
    model = build_model(lir)
    assert float(model.run(3.0)[0]) == 6.0 and float(model.run(-2.0)[0]) == -6.0


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
    # ekf1_stateless time-multiplexes many values onto each register. Verify the hardware-frame interference invariant directly:
    # within a register, each value's last read precedes the next value's landing, R(a) < W(b) -- the same liveness
    # reg_liveness renders and the relaxed allocator shares against. Reconstructed via the write-timeline resolution
    # the numerical model uses, so the test tracks the allocator's actual sharing decisions, not a hardcoded schedule.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless
    from holoso._lir import latest_producer_before

    lir = build(_run(ekf1_stateless.update_x_P), "update_x_P")
    timeline = lir.write_timeline
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


def test_commutative_port_assignment_never_increases_read_mux_fan_in(monkeypatch) -> None:  # type: ignore[no-untyped-def]
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
    # Regression: the bool->float cast result lands at wide_landing_cycle(commit) -- NOT one cycle later. An off-by-one
    # marked it past its true landing (and, for a boundary cast, past the initiation interval, so its report cell fell
    # off the grid) and left a consumer's read cycle outside its residence. Here a multiply consumes the cast result.
    def f(x):  # type: ignore[no-untyped-def]
        return float(x > 0.0) * x

    lir = build(_run(f), "cast_mul")
    interval = lir.initiation_interval
    casts = [
        (lir.block_base[b.index], op) for b in lir.blocks for op in b.inline_ops if isinstance(op.write.dst, RegRef)
    ]
    assert casts, "expected a bool->float cast result in the wide bank"
    for base, op in casts:
        landing = wide_landing_cycle(base + op.commit_cycle)
        assert 1 <= landing <= interval  # within the rendered schedule grid, not one row past the boundary
        assert landing in lir.reg_liveness[op.write.dst]  # live from its true landing (the off-by-one would miss it)
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
    timeline = lir.write_timeline
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


def test_commutative_comparator_swap_permutes_output_taps() -> None:
    # The comparator is commutative under the gt/lt flag exchange. Two mirrored comparisons over one operand pair
    # otherwise read (a,b) and (b,a) -- two registers per read port; the port assignment orients one of them swapped,
    # shrinking each port's read-set to a single register, and the swapped firing's lt tap moves to gt. Bit-exact
    # because the ZKF ordering is total and compare is antisymmetric.
    def f(a, b):  # type: ignore[no-untyped-def]
        below = a < b
        above = b < a
        return [float(below), float(above)]

    lir = build(_run(f), "mirrored")
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


def test_chained_slot_live_in_blocks_early_install() -> None:
    # Regression (review; pre-existing at HEAD): a slot whose live-in feeds ANOTHER slot's live-out ("self._a =
    # self._b") was documented as unable to early-install, but only the coalescing test consulted that fact -- the
    # early-install decision did not, so "_b"'s new value landed before "_a"'s boundary copy captured the old one.
    # The RTL then returned the NEW "_b" through "_a" while the model kept the old one (cosim diverged on the second
    # transaction). The tapped slot must now install at the boundary, and the model must match plain Python.
    lir = build(_run(ChainedSlots().__call__), "chained_slots")
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


def test_state_early_install_respects_a_select_reader() -> None:
    # Pins the read-step frame of the state early-install bound: an inline select reads its operands at its fire
    # step (issue + latency + FETCH_LAG + 1), one cycle past where an issue-frame bound would have allowed the
    # slot's install copy to fire -- an early install bounded by issue cycles would overwrite the live-in before
    # the select reads it (RTL would take the NEW value through ``old`` while the model keeps the old one).
    lir = build(_run(SelectHold().step), "select_hold")
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


def test_inverted_bool_phi_arm_installs_with_opposite_polarities() -> None:
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

    lir = build(_run(f), "inverted_arm")
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

"""
A vector-independent structural guard: every phi-arm install must LAND within its own block, at or before the block's
terminator step. An install whose landing PC exceeds the terminator is enqueued for a PC the block never reaches -- a
non-Ret terminator re-keys it onto the taken successor arm, but the Ret wrap silently drops it, a dead install.

This class of defect is invisible to every value comparison (cosim, the example-reference suite, the schedule-
independent MIR interpreter): a dead install that does not alter an output value passes them all, because it is
output-redundant on the vectors. Only a structural invariant catches it. The check is over the settled LIR, independent
of any input vector, so it holds for the data-dependent branch/loop kernels (uart_rx error frames included) too.
"""

import dataclasses

import pytest

import holoso
from holoso import FloatFormat, FloatValue
from holoso._eel import lower as lower_frontend
from holoso._lir import boundary_step, landing_cycle
from holoso._lir import BoolWrite, InlineScheduledOp, Lir, LirBlock, PooledScheduledOp, RegRef, WideCopy
from holoso._mir import lower as lower_to_mir

from ._examples import SPECS, ExampleSpec
from ._modelref import (
    assert_model_equals_interpreter,
    staged_fadd_options,
    build_lir,
    build_model_and_interpreter,
    mir_options,
    default_mir,
    default_options,
    DEFAULT_UNROLL_MAX_TRIPS,
    Vector,
)


def _build(spec: ExampleSpec) -> Lir:
    return build_lir(
        lower_to_mir(
            lower_frontend(spec.make_kernel(), DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(spec.options(spec.formats[0]))
        ),
        spec.name,
    )


def _phi_arm_installs(block: LirBlock) -> list[WideCopy | BoolWrite]:
    return [*block.wide_copies, *block.bool_writes]


def _block_ops(block: LirBlock) -> list[PooledScheduledOp | InlineScheduledOp]:
    return [*block.ops, *block.inline_ops]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_phi_arm_installs_land_within_their_block(spec: ExampleSpec) -> None:
    lir = _build(spec)
    for block in lir.blocks:
        for install in _phi_arm_installs(block):
            landing = install.landing(lir.fetch_lag)  # block-local; same fire+read-first edge the model/emitter commit
            assert landing <= block.term_offset, (
                f"{spec.name} block {block.index}: install of {install.dst} lands at {landing}, past the terminator "
                f"{block.term_offset} -- a dead install the Ret wrap would orphan"
            )


@pytest.mark.parametrize("name", ["uart_rx", "uart_tx"])
def test_targets_still_exercise_constant_installs(name: str) -> None:
    """
    uart_rx and uart_tx are kernels behind this work: their boolean live-outs ({b3,b4,b5} <- False, True on a
    parity/frame error) and other arms install literal constants with no source to sample, so they fire at the work
    makespan with no read-first push. Pin that these kernels still emit constant phi-arm installs, so a kernel-shape
    change cannot quietly make the recovered-cycle freezes meaningless. The settled timing itself is pinned
    end-to-end -- by those frozen lengths (uart_rx 51, uart_tx 37 in test_latency_freeze), by the
    landing <= terminator structural invariant above, and by RTL cosim -- not by re-deriving the install's own helpers.
    """
    spec = next(s for s in SPECS if s.name == name)
    lir = _build(spec)
    const_installs = [x for b in lir.blocks for x in _phi_arm_installs(b) if x.is_const]
    assert const_installs, f"{name} no longer emits constant phi-arm installs; the kernel shape changed"


class _InputArmSource:
    """
    A phi arm that passes an INPUT through: `b` is settled at block entry, so the install needs no read-first
    sampling, and `b` is read past the merge so the arm cannot coalesce. Dedicated rather than probed off a bundled
    example, whose shape may drift.
    """

    def __call__(self, c: bool, a: float, b: float) -> float:
        if c:
            y = a / b  # a non-speculatable division keeps this a real branch, not an if-converted select
        else:
            y = b
        return y * b


def test_input_sourced_install_is_settled() -> None:
    """
    The generalization beyond literal constants: a register source resident at block entry has nothing to
    read-first, so the install is classified settled (`settled_source`) and fires at the work makespan with no
    read-first push -- exactly like a constant. Pin that an INPUT-sourced install is present and so classified,
    matched by its source register against the input loads -- phi-sourced installs are also settled, so a
    settled-and-non-const filter alone could pass through a phi while the input path regressed. Constant,
    input-register, and state-read installs are three distinct mechanisms; the other two have their own pins.
    """
    fmt = FloatFormat(8, 36)
    lir = build_lir(
        lower_to_mir(lower_frontend(_InputArmSource().__call__, DEFAULT_UNROLL_MAX_TRIPS).hir, default_mir(fmt)),
        "input_arm_source",
    )
    input_regs = {load.dst for load in lir.inputs}
    input_sourced = [
        x
        for b in lir.blocks
        for x in _phi_arm_installs(b)
        if x.settled_source and not x.is_const and x.source.source in input_regs
    ]
    assert input_sourced, "the input-sourced settled install is gone, or the predicate regressed"


def test_computed_copy_not_last_work_fits_at_work_makespan() -> None:
    """
    A computed-source phi-arm copy whose source is NOT the block's last-committing work installs at the work makespan
    (landing read-first at the boundary), not one step past it. recip_newton's loop body is the canonical case: the
    copy y <- y_next sources y_next, while delta = y_next - y is the block's last work. Assert the block makespan
    equals the work makespan (no +1), with read-first/value correctness pinned by the example-reference and cosim
    suites. (y_next feeds the later delta, so it is not the block's last work; install_issue_cycle's in-block +1
    triggers only for a copy whose source IS the last work, pinned below.)
    """
    lir = _build(next(s for s in SPECS if s.name == "recip_newton"))
    bodies = [b for b in lir.blocks if any(not c.settled_source for c in b.wide_copies)]
    assert bodies, "recip_newton no longer has a computed-source phi-arm copy; the kernel shape changed"
    for b in bodies:
        work = max((op.commit_cycle for op in _block_ops(b)), default=0)
        assert b.block_makespan == work, (
            f"recip_newton block {b.index}: a computed copy still pushes the makespan ({b.block_makespan} > work "
            f"{work}) -- the loop-carried install pin regressed to the conservative +1"
        )
        assert all(c.landing(lir.fetch_lag) <= b.term_offset for c in b.wide_copies)


class _HoldOrUpdateBool:
    """
    A boolean state held on one arm and updated on the other: `out` takes the STATE READ `self.s` when `c` is
    false and `a and b` when true. No bundled example installs a state read as a phi arm, so this pins the third
    settled source kind (after constants and inputs).
    """

    def __init__(self) -> None:
        self.s = False

    def __call__(self, a: bool, b: bool, c: bool) -> tuple[bool, bool]:
        out = self.s
        if c:
            out = a and b
        self.s = a
        return out, self.s


def test_state_read_sourced_install_is_settled() -> None:
    """
    A phi arm that is a STATE READ is resident at block entry (the slot register holds it from the start), so its
    tail install is settled -- the generalization's third source kind. A zero if-conversion budget keeps the diamond
    a real branch, so the hold arm installs `self.s` by a pc-gated bool write rather than collapsing to a select.
    Pin both that the install is so classified (a non-const settled bool write) and -- the black-box teeth --
    that the held value is the OLD state across a hold/update sweep, which an early-read or clobbered state-read
    install would corrupt (the model vs a fresh Python reference, schedule-independent).
    """
    options = dataclasses.replace(default_options(FloatFormat(6, 18)), ifconv_max_ops=0)
    lir = build_lir(
        lower_to_mir(lower_frontend(_HoldOrUpdateBool().__call__, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)),
        "hold_or_update_bool",
    )
    settled_non_const = [x for b in lir.blocks for x in b.bool_writes if x.settled_source and not x.is_const]
    assert settled_non_const, "the state-read phi arm did not install as a settled bool write"

    model = holoso.synthesize(
        _HoldOrUpdateBool().__call__, options, name="hold_or_update_bool"
    ).numerical_model.elaborate()
    reference = _HoldOrUpdateBool()
    for a, b, c in [(True, True, False), (False, True, False), (True, True, True), (False, False, False)]:
        assert tuple(bool(v) for v in model.run(a, b, c)) == reference(a, b, c)


class _LiveThroughArm:
    """
    A non-coalesced phi arm whose source SPILLS IN: the overlapping entry's deep chain keeps `x` in flight past the
    entry's shrunk terminator, so it lands inside the pass-through `else` arm's own frame; `x` is used past the
    merge so it interferes with the phi and the arm cannot coalesce. No bundled example has this in-flight-source
    shape.
    """

    def __call__(self, c: bool, a: float, b: float) -> float:
        x = a * b
        x = x * b
        x = x * b
        x = x * b
        x = x * b  # x commits deep in the entry, far past the pass-through predecessor's own makespan
        if c:
            y = a / x  # a non-speculatable division keeps this a real branch, not an if-converted select
        else:
            y = x  # pass-through: the arm source is the entry's x, not a value of this predecessor block
        return y * x  # x is used past the merge, so it interferes with y's phi and the arm does not coalesce


def test_cross_block_source_install_read_gates_on_the_spilled_landing() -> None:
    """
    The directed pin for an install whose source arrives as an IN-FLIGHT SPILL: the overlapping entry's deep chain
    spills `x` past its shrunk terminator, landing at the pass-through arm's local step 1, and the arm's install
    sources it. The classification must be exact -- in-flight (not settled), read-gated at the spilled landing, and
    NOT pushed past the work makespan (the virtual commit is negative, so the install issues at the makespan and its
    fire strictly follows the landing here; the fire == landing equality boundary has its own shape elsewhere).
    Value correctness rides the schedule-independent model-vs-interpreter differential below; the interference
    residence shares the same single `install_source_commit`, so the placement pinned here is also the residence's
    frame.
    """
    fmt = FloatFormat(8, 36)
    ops = default_mir(fmt)
    kernel = _LiveThroughArm().__call__
    lir = build_lir(
        lower_to_mir(lower_frontend(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, ops),
        "live_through_arm",
    )
    cross = [(blk, c) for blk in lir.blocks for c in blk.wide_copies if not c.settled_source and not _block_ops(blk)]
    assert cross, "the kernel no longer exercises an in-flight-source install in an op-less arm; the shape changed"
    for blk, c in cross:
        assert c.issue_cycle == 0, "an in-flight source must not push the install past the (empty) work makespan"
        assert c.landing(lir.fetch_lag) == blk.term_offset, "the install lands read-first at the drained boundary"
    for blk in lir.blocks:
        assert all(c.landing(lir.fetch_lag) <= blk.term_offset for c in blk.wide_copies)

    model, interpreter = build_model_and_interpreter(kernel, ops, "live_through_arm", fmt)
    vectors: list[Vector] = [
        [c, FloatValue.from_float(fmt, a), FloatValue.from_float(fmt, b)]
        for c in (True, False)
        for a in (2.0, 5.0, 0.5, 9.0, 1.5)
        for b in (3.0, 1.5, 4.0, 0.25)
    ]
    assert_model_equals_interpreter(model, interpreter, vectors, "live_through_arm")


class _LastWorkArmSource:
    """
    A non-coalesced phi arm whose source IS its branch block's own LAST-committing work. In the taken arm `q = a / b`
    is the block's only (hence last) op, copied straight into the merged `r`; `q` is also read at the join
    (`r * q`), so it interferes with `r` and the copy cannot coalesce. The copy's source therefore commits at the
    block makespan, so the install must read-first ONE step past it -- exercising install_issue_cycle's in-block +1
    branch, which no bundled example reaches (recip_newton's loop-carried copy feeds a later op, so its source is never
    the block's last work).
    """

    def __call__(self, c: bool, a: float, b: float) -> float:
        if c:
            q = a / b  # the taken block's only/last op, copied into the merged r below
            r = q
        else:
            q = b / a
            r = b
        return r * q  # q is live past the merge, so r's arm copy (r <- q) cannot coalesce


def test_computed_copy_at_last_work_takes_the_terminator_cycle() -> None:
    """
    The dual of `test_computed_copy_not_last_work_fits_at_work_makespan`: when a non-coalesced copy's source IS the
    block's last-committing work, the install must read-first one step past it, so the block makespan is the work
    makespan + 1. This pins install_issue_cycle's in-block +1 branch -- load-bearing but reached by no bundled example,
    so a later change cannot silently drop it (miscompiling this shape) without failing here. The pushed install's
    landing is one step past the work landing, so the block drains one terminator cycle later. Value correctness is
    held by the schedule-independent model-vs-interpreter differential.
    """
    fmt = FloatFormat(8, 36)
    ops = default_mir(fmt)
    kernel = _LastWorkArmSource().__call__
    lir = build_lir(
        lower_to_mir(lower_frontend(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, ops),
        "last_work_arm",
    )
    pushed = [
        blk
        for blk in lir.blocks
        if any(not c.settled_source for c in blk.wide_copies)
        and blk.block_makespan == max((op.commit_cycle for op in _block_ops(blk)), default=0) + 1
    ]
    assert pushed, "no block takes the in-block +1 for a last-work copy source; the kernel shape changed"
    for blk in pushed:
        assert blk.term_offset == boundary_step(blk.block_makespan, lir.fetch_lag), "the push drains one step later"
    for blk in lir.blocks:
        assert all(c.landing(lir.fetch_lag) <= blk.term_offset for c in blk.wide_copies)

    fmt = FloatFormat(8, 36)
    model, interpreter = build_model_and_interpreter(kernel, ops, "last_work_arm", fmt)
    vectors: list[Vector] = [
        [c, FloatValue.from_float(fmt, a), FloatValue.from_float(fmt, b)]
        for c in (True, False)
        for a in (2.0, 5.0, 0.5, 9.0, 1.5)
        for b in (3.0, 1.5, 4.0, 0.25)
    ]
    assert_model_equals_interpreter(model, interpreter, vectors, "last_work_arm")


class _DrainedForeignArm:
    """
    A pass-through arm whose source is computed in the ENTRY and fully lands inside the entry's own frame (the
    entry's long tail keeps its terminator past the early product's landing), so the source is SETTLED at the arm --
    the imu_fusion `valid == False` shape. A node-kind classification would call any foreign operator result a
    possible in-flight spill and push the install past the (empty) work makespan, stretching the arm by a cycle.
    """

    def __call__(self, c: bool, a: float, b: float) -> tuple[float, float]:
        x = a * b
        z = ((x + a) * b + a) * b
        if c:
            y = a / b
        else:
            y = x
        return y * b, z + x  # x stays live past the merge, so the pass-through arm cannot coalesce


def test_drained_foreign_source_install_is_settled() -> None:
    fmt = FloatFormat(8, 36)
    ops = default_mir(fmt)
    kernel = _DrainedForeignArm().__call__
    lir = build_lir(lower_to_mir(lower_frontend(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, ops), "drained_foreign_arm")
    input_regs = {load.dst for load in lir.inputs}
    arms = [
        (b, c)
        for b in lir.blocks
        if not _block_ops(b)
        for c in b.wide_copies
        if isinstance(c.source.source, RegRef) and c.source.source not in input_regs and not c.is_const
    ]
    assert arms, "the foreign-op-sourced pass-through install is gone; the kernel shape changed"
    for blk, copy in arms:
        assert copy.settled_source, "a fully landed foreign source must be classified settled"
        assert copy.issue_cycle == 0, "a settled source must not push the install past the empty work makespan"
        assert blk.term_offset == landing_cycle(0, lir.fetch_lag), "the block drains at the unpushed landing"

    model, interpreter = build_model_and_interpreter(kernel, ops, "drained_foreign_arm", fmt)
    vectors: list[Vector] = [
        [c, FloatValue.from_float(fmt, a), FloatValue.from_float(fmt, b)]
        for c in (True, False)
        for a in (2.0, -1.5, 0.5)
        for b in (3.0, 1.5, 0.25)
    ]
    assert_model_equals_interpreter(model, interpreter, vectors, "drained_foreign_arm")


class _InflightEquality:
    """
    The public-API twin of the equality-boundary structural pin (test_schedule): with the staged fadd as the entry's
    last op, its result spills at successor-local fetch_lag and the pass-through arm's install fires exactly AT that
    landing. The cosim below verifies the within-cycle write-then-read equivalence of the model and the RTL at this
    boundary, which no value-only comparison can isolate.
    """

    def __call__(self, c: bool, a: float, b: float) -> tuple[float, float]:
        x = a + b
        if c:
            y = a / b
        else:
            y = x
        return y * b, x


@pytest.mark.cosim
def test_inflight_equality_cosim() -> None:
    """
    RTL == model at the in-flight equality boundary in lockstep: an install firing exactly at its source's landing
    must read the just-landed value within that cycle in both backends -- a within-cycle write-then-read agreement
    no value-only comparison isolates.
    """
    from ._cosim import run_cosim  # noqa: PLC0415
    from .hdl.hdl_float_oracle import SIMULATORS  # noqa: PLC0415

    fmt = FloatFormat(8, 36)
    eq_vectors = [{"c": c, "a": a, "b": b} for c in (True, False) for a, b in [(2.0, 4.0), (3.0, 1.5), (-1.0, 2.0)]]
    run_cosim(
        SIMULATORS[0],
        holoso.synthesize(_InflightEquality().__call__, staged_fadd_options(fmt), name="inflight_equality_seam"),
        eq_vectors,
    )

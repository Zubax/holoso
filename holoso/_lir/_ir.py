"""
The low-level IR (LIR): the scheduled, bound, register-allocated microprogram for the synthesized ZISC machine.

A Lir is controller-agnostic -- it describes which hardware operators issue on which cycle, reading/writing
which typed storage resources, with which folded port conditioners.
"""

from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeVar, assert_never

from .._operators import (
    BoolInversion,
    HardwareOperator,
    InlineHardwareOperator,
    PooledHardwareOperator,
    PortConditioner,
    WideConditioner,
)
from .._type import BoolType, FloatFormat, FloatType, IntFormat, IntType, ScalarType
from .._value import WideValue
from ._ports import ControlInputPort, ControlOutputPort, ControlPort, DataInputPort, DataOutputPort, Port

# The cycle-accurate timing model: one consistent physical story shared by the LIR cycle helpers below, the numerical
# model, the scheduler's dependency edges, the register allocator, and the HTML report, so a value's
# landing/read/copy/boundary cycle is computed in exactly one place and the consumers cannot drift.
# It is built from primitives:
#   - fetch_lag -- the microcode fetch leads the datapath by this many steps (a global frame offset), threaded from
#     the build entry and recorded on the Lir, so the schedule and every consumer share one value;
#   - READ_FIRST_EDGE -- a register read sees the value written one step earlier (write-then-read), so a result becomes
#     readable one step after it is written.
# Both register banks read combinationally and alike: an instance-backed operator's read-address word rides its
# issue step and the datapath samples the operand fetch_lag later, so every operand read lands at issue + fetch_lag
# independent of bank -- the read-side mirror of the uniform landing below. Every result -- pooled or inline, wide
# or boolean -- drives its register write combinationally from its producer's output at its commit step, so a
# result committed at C becomes readable at C + fetch_lag + READ_FIRST_EDGE, bank- and class-independent.
# The operator's own LATENCY is the orthogonal pipeline depth (a pooled instance has L stages, an inline op has none).
READ_FIRST_EDGE = 1


def landing_cycle(commit_cycle: int, fetch_lag: int) -> int:
    """
    The cycle a result committed at `commit_cycle` becomes readable in its register: the fetch lag plus the
    write-then-read edge. Bank- and class-independent -- every result drives its array write combinationally at its
    commit step, so a pooled or inline result on either bank lands alike.
    """
    return commit_cycle + fetch_lag + READ_FIRST_EDGE


def read_cycle(issue_cycle: int, fetch_lag: int) -> int:
    """
    The cycle an instance-backed operator samples its register operands: latch-free on both banks, fetch_lag after the
    read-address word rides the issue step.
    """
    return issue_cycle + fetch_lag


def inline_fire_cycle(commit_cycle: int, fetch_lag: int) -> int:
    """
    The cycle a PC-gated combinational statement fires: an inline operation (a select/mux, boolean logic, a float<->bool
    cast), or a pc-gated install/copy (a phi-arm copy, an early state install, a boolean write). It reads ALL
    its operands (or its source) and drives its destination register's write data on this single step, `fetch_lag`
    after its scheduler-frame placement. Its result becomes readable one `READ_FIRST_EDGE` later (`landing_cycle`).
    For a pc-gated install this is the coalescing equivalence the overlap layout relies on -- `install_landing` of
    this fire step equals `landing_cycle(commit_cycle)`, so a phi arm coalesced onto a direct operator write lands
    exactly where its copy would have. Whether the install sits at the work makespan or one step past lives in its
    PLACEMENT (`install_issue_cycle`), not in this fire step.
    """
    return commit_cycle + fetch_lag


def pooled_write_word(commit_cycle: int) -> int:
    """
    The fetch-PC-frame step on which a pooled lane's write commits -- its destination register's write opcode: the
    commit step itself. This is the lone helper in the fetch-PC frame (no `fetch_lag`) -- it places the microcode word
    the sequencer fetches, not the step the datapath acts on it. Shared by the emitter's microcode (where the word is
    placed) and the overlap layout (which keeps every write word inside the block), so the two cannot drift -- the same
    single-source-of-truth contract as the landing/read helpers above.
    """
    return commit_cycle


def operand_read_cycle(operator: HardwareOperator, issue_cycle: int, fetch_lag: int) -> int:
    """
    The hardware-frame cycle on which an operation samples its register operands (an operation reads all its operands
    on one cycle), the single definition shared by the register allocator's interference, the liveness views, and the
    numerical model so none can drift. A pooled instance reads its operands latch-free at `read_cycle` (both banks
    alike); an inline op fires -- and reads -- on its combinational fire step (`inline_fire_cycle`).
    """
    if isinstance(operator, PooledHardwareOperator):
        assert all(ty.is_wide for ty in operator.signature.operand_types), operator.mnemonic
        return read_cycle(issue_cycle, fetch_lag)
    return inline_fire_cycle(issue_cycle + operator.latency, fetch_lag)


def dependency_edge(producer: HardwareOperator, producer_port: int, consumer: HardwareOperator, fetch_lag: int) -> int:
    """
    The minimum same-block scheduling distance from a producer's commit to a consumer's issue (`issue_consumer >=
    commit_producer + edge`): the producer's result landing minus the consumer's operand-read timing, so the consumer
    reads no earlier than the producer's result becomes readable. Every producer -- pooled or inline, wide or boolean --
    lands at the one bank-independent `landing_cycle`. A POOLED consumer reads its operands latch-free at
    `read_cycle` (both banks alike); an INLINE consumer reads on its combinational fire step (`inline_fire_cycle`).
    One shared rule for the scheduler, the liveness views, and the model. There is NO floor below this spacing: the
    model commits every PC's landings before evaluating that PC's reads (`NumericalSimulator._apply`), so a consumer
    whose read PC equals the producer's landing PC reads the just-committed value -- write-then-read holds at the PC
    granularity. The zero-offset evaluation below is exact because every cycle helper is affine in its cycle argument
    with unit slope, so the difference at zero is the frame-independent spacing; a helper that ever loses that affinity
    breaks this derivation.
    """
    landing = landing_cycle(0, fetch_lag)
    if isinstance(consumer, PooledHardwareOperator):
        assert producer.signature.result_types[
            producer_port
        ].is_wide, f"{consumer.mnemonic}: pooled operators read only wide operands today"
        read = read_cycle(0, fetch_lag)
    else:
        read = inline_fire_cycle(consumer.latency, fetch_lag)
    return landing - read


def install_landing(fire_step: int) -> int:
    """
    The step a pc-gated install -- a phi copy, a boolean write, or an early state install -- commits its
    destination and becomes readable: one after the step it fires on. The model writes the destination into `_pending`
    one PC past the fire, so the numerical model and the liveness diagnostic route this +1 through this one helper and
    cannot drift. A boundary slot install is the lone exception: it reads-then-writes at `last_pc` and does not pass
    through here.
    """
    return fire_step + READ_FIRST_EDGE


def install_issue_cycle(work_makespan: int, source_commit: int | None) -> int:
    """
    The scheduler-frame placement of a block's tail install; `source_commit` is None for a SETTLED source (per
    `install_source_commit`: a constant, input, state read, phi result, or a foreign result already landed), which
    needs no read-first sampling. The install sits at the work makespan -- landing read-first at the boundary, after
    every in-block read, the latest placement (minimal destination residence). It is pushed one step past only when
    a locally COMPUTED source commits AT the makespan, which the install must fire after to read-first
    (`source_commit + READ_FIRST_EDGE` then exceeds the makespan) -- the block's own last-committing op. A
    spilled-in source carries a NEGATIVE virtual commit (the inverse of its landing; see `install_source_commit`),
    so the same `max` places its fire at or after that landing without a push. A settled source and a copy sourcing
    an EARLIER in-block op stay at the makespan and pay no terminator cycle; keeping them unpushed is load-bearing
    for the same-step parallel-bundle contract cross-referencing loop-header phis rely on. The block layout carries
    the same +1 into the block makespan exactly when an install issues past the work makespan, so the placement and
    the drain agree and an install cannot land past its block's terminator.
    """
    if source_commit is None:
        return work_makespan
    assert source_commit <= work_makespan, "install source_commit lies outside the install's own block frame"
    return max(work_makespan, source_commit + READ_FIRST_EDGE)


def boundary_step(makespan: int, fetch_lag: int) -> int:
    """
    The drained boundary / initiation-interval step: the cycle the block's latest boundary-resident result lands, where
    its live-outs and consumed-at-boundary values are resident. Bank-independent -- every result lands at the one
    `landing_cycle`. The single source of truth shared by the overlap layout (the terminator offset), the liveness
    boundary, and the numerical model, so the drain cannot drift between them.
    """
    return landing_cycle(makespan, fetch_lag)


def successor_local_cycle(block_local_cycle: int, term_offset: int) -> int:
    """
    Map a block-local cycle that crosses an overlap-shrunk terminator into the single-predecessor successor's frame.
    The successor frame begins at `term_pc + 1`, so a cycle at absolute `block_base + block_local_cycle` sits at
    `block_local_cycle - term_offset - 1` past the successor's base -- one continuous PC across the seam. This is the
    single coordinate map shared by the scheduler's spill carry (both the value landings and the per-instance busy
    residue), `_trace_landing`, and the numerical model's redirect re-keying, so they cannot drift apart.
    """
    return block_local_cycle - term_offset - 1


def _residence_rows(defs: list[int], uses: list[int], boundary: int) -> set[int]:
    """
    Collapse a register's definition and use cycles into the set of cycles on which it holds a live value: each value
    resides from its landing through its last use STRICTLY before the next definition, the boundary at latest, plus the
    landing cycle itself even when the value is never read (it still occupies the register that cycle). The strict bound
    is the write-then-read register semantics the numerical model commits: a read on a later value's landing cycle reads
    that NEW value, so it belongs to the next definition's residence, not the previous one -- a read on a value's OWN
    landing (the common producer->consumer case, where the consumer reads on the producer's landing PC) still counts for
    that value. Shared by the wide- and boolean-bank liveness so both banks compute residence in exactly one place.
    """
    writes = sorted(defs)
    rows: set[int] = set()
    for i, start in enumerate(writes):
        nxt = writes[i + 1] if i + 1 < len(writes) else boundary + 1
        last = max((use for use in uses if start <= use < nxt), default=start)
        rows.update(range(start, last + 1))
    return rows


@dataclass(frozen=True, slots=True)
class OperatorInstance:
    """
    One physical operator module, e.g. `u_fadd_0` or `u_fcmp_0`.

    `operator` is the fully specified pooled hardware operator it elaborates; `index` numbers the copies of that
    operator value. The scheduler pools firings by the hardware-operator instance: equal operators may time-share one
    module, each instance accepting a new firing every `initiation_interval` cycles.
    """

    operator: PooledHardwareOperator
    index: int  # 0-based within this concrete operator value

    @property
    def name(self) -> str:
        return f"{self.operator.mnemonic}_{self.index}"

    def __post_init__(self) -> None:
        # A pooled result commits at issue + latency, so latency >= 1 keeps its write opcode off the held `ucode[0]`
        # (the accept-dwell word) -- a latency-0 pooled operator would re-commit every idle cycle (see `transacting`).
        assert self.operator.latency >= 1, f"{self.operator.mnemonic}: pooled operator latency must be >= 1"
        # Every pooled operator passes through here, so its hand-synchronized per-port declarations are validated
        # once at the source: HDL port names align with the operands and the result types, and the commutation
        # permutation (when declared) is a type-preserving bijection -- a bad declaration fails here, not in emission.
        result_types = self.operator.signature.result_types
        assert len(self.operator.operand_hdl_ports) == self.operator.signature.arity, self.operator.mnemonic
        assert len(self.operator.output_hdl_ports) == len(result_types), self.operator.mnemonic
        permutation = self.operator.swap_output_permutation
        if permutation is not None:
            assert sorted(permutation) == list(range(len(result_types))), self.operator.mnemonic
            assert all(result_types[permutation[p]] == result_types[p] for p in range(len(permutation)))


# `(instance, operand position)`; the write side's `(instance, output port)` lanes are a different key.
type ReadPort = tuple[OperatorInstance, int]


@dataclass(frozen=True, slots=True)
class RegRef:
    """A read/write of wide data register `index` in the shared register bank."""

    index: int

    @property
    def stable_label(self) -> str:
        return f"r{self.index}"


@dataclass(frozen=True, slots=True)
class BoolRegRef:
    """A read/write of boolean register `index` in the 1-bit boolean register bank."""

    index: int

    @property
    def stable_label(self) -> str:
        return f"b{self.index}"


_BankReg = TypeVar("_BankReg", RegRef, BoolRegRef)  # one register bank's reference type (wide or boolean)


@dataclass(frozen=True, slots=True)
class _ConstRef:
    index: int

    @property
    def stable_label(self) -> str:
        return f"c{self.index}"


@dataclass(frozen=True, slots=True)
class WideConstRef(_ConstRef): ...


@dataclass(frozen=True, slots=True)
class _Operand:
    source: RegRef | _ConstRef


@dataclass(frozen=True, slots=True)
class WideOperand(_Operand):
    source: RegRef | WideConstRef
    conditioner: WideConditioner

    @property
    def stable_label(self) -> str:
        return self.conditioner.decorate(self.source.stable_label)


@dataclass(frozen=True, slots=True)
class _InputLoad:
    """An input port sampled into a typed register at in_valid."""

    name: str
    dst: RegRef | BoolRegRef
    scalar_type: ScalarType


@dataclass(frozen=True, slots=True)
class WideInputLoad(_InputLoad):
    dst: RegRef
    scalar_type: FloatType | IntType


@dataclass(frozen=True, slots=True)
class InPlace:
    """
    A state slot's live-out already sits in the slot register: its producing operator -- or, for a conditional or
    loop update, the arms of its phi -- wrote it there, so nothing is copied.
    """


@dataclass(frozen=True, slots=True)
class WideEarlyInstall:
    """
    A pc-gated copy of a wide slot's live-out `source` into the slot register at the ABSOLUTE scheduler-frame
    `cycle` -- a slot belongs to no block, so unlike a block copy's this cycle carries no base -- ahead of the
    boundary, as early as the old live-in is last read and the source is available, so the source register is free
    for the rest of the transaction.
    """

    source: WideOperand
    cycle: int

    def fire_step(self, fetch_lag: int) -> int:
        return inline_fire_cycle(self.cycle, fetch_lag)

    def landing(self, fetch_lag: int) -> int:
        return install_landing(self.fire_step(fetch_lag))


@dataclass(frozen=True, slots=True)
class WideBoundaryInstall:
    """
    A read-first copy of a wide slot's live-out `source` into the slot register at the accepted-output edge, so an
    output still reading the live-in sees the old value.
    """

    source: WideOperand


@dataclass(frozen=True, slots=True)
class WideStateSlot:
    """
    A persistent wide state register: reset to `reset_value`, holding the slot's live-in (carried over from the
    previous initiation) until `install` replaces it with the slot's live-out. A public attribute's observable
    `state_<name>` port is a separate output wire tapping the same value, not a property of the slot.
    """

    name: str
    reg: RegRef
    reset_value: WideValue  # the encoded machine word, which also names the slot's scalar family
    install: InPlace | WideEarlyInstall | WideBoundaryInstall


@dataclass(frozen=True, slots=True)
class BoolInputLoad(_InputLoad):
    dst: BoolRegRef
    scalar_type: BoolType = BoolType()


@dataclass(frozen=True, slots=True)
class BoolConstRef:
    """A boolean immediate (`True`/`False`); the bool bank has no constant pool, the value rides inline."""

    value: bool


type BoolSource = BoolRegRef | BoolConstRef


@dataclass(frozen=True, slots=True)
class BoolOperand:
    """
    A boolean operand: a boolean register read or an immediate True/False, with an optional folded inversion -- the
    1-bit dual of WideOperand's conditioner, free in fabric. An inverted immediate folds to its negated
    value at construction, so a constant operand always carries the identity inversion.
    """

    source: BoolSource
    inversion: BoolInversion = BoolInversion()

    def __post_init__(self) -> None:
        if isinstance(self.source, BoolConstRef) and self.inversion.invert:
            object.__setattr__(self, "source", BoolConstRef(not self.source.value))
            object.__setattr__(self, "inversion", BoolInversion())

    @property
    def stable_label(self) -> str:
        if isinstance(self.source, BoolConstRef):
            return "1" if self.source.value else "0"
        return self.inversion.decorate(self.source.stable_label)


@dataclass(frozen=True, slots=True)
class PortWrite:
    """
    One tapped output port of a firing: the `port`-th result lands in `dst` through the conditioner its scalar
    type admits -- a folded sign control for a float, an inversion for a boolean, nothing at all for an integer.
    Untapped ports of the firing simply have no PortWrite -- the module output is left unconnected.
    """

    port: int
    dst: RegRef | BoolRegRef
    conditioner: PortConditioner


@dataclass(frozen=True, slots=True)
class PooledScheduledOp:
    """
    One pooled-instance firing in the software-pipelined schedule: `inst` asserts `in_valid` on `issue_cycle`,
    and on `commit_cycle == issue_cycle + latency` every
    tapped output port lands in its destination register. The writes are sorted by port and pairwise distinct in
    both port and destination -- members of one firing land simultaneously, so the allocator always gives them
    distinct registers.
    """

    inst: OperatorInstance
    operands: list[WideOperand | BoolOperand]
    writes: list[PortWrite]
    issue_cycle: int
    latency: int
    immediates: tuple[int, ...]  # per-firing immediate values, aligned with the operator's immediate_ports

    @property
    def operator(self) -> PooledHardwareOperator:
        return self.inst.operator

    def __post_init__(self) -> None:
        assert self.writes, "a firing with no tapped output cannot exist (an unused operation has no MIR node)"
        ports = [write.port for write in self.writes]
        assert ports == sorted(set(ports)), f"write ports must be sorted and distinct: {ports}"
        assert len({write.dst for write in self.writes}) == len(self.writes), "write destinations must be distinct"

    @property
    def commit_cycle(self) -> int:
        return self.issue_cycle + self.latency


@dataclass(frozen=True, slots=True)
class InlineScheduledOp:
    """
    One inline-operator firing: a single PC-gated statement that reads its operands and drives its one result's write
    data combinationally on its fire step. Its result lands one read-first edge after the fire, at the bank- and
    class-independent `landing_cycle` -- the same landing as a pooled result.
    """

    operator: InlineHardwareOperator
    operands: list[WideOperand | BoolOperand]
    write: PortWrite
    issue_cycle: int
    latency: int

    @property
    def writes(self) -> list[PortWrite]:
        return [self.write]

    @property
    def immediates(self) -> tuple[int, ...]:
        return ()  # an inline operator is a pure combinational expression; it declares no immediate ports

    @property
    def commit_cycle(self) -> int:
        return self.issue_cycle + self.latency


@dataclass(frozen=True, slots=True)
class _OutputWire:
    """An output port: a named external sink driven at the last PC by a typed source tap."""

    name: str
    tap: WideOperand | BoolOperand
    scalar_type: ScalarType


@dataclass(frozen=True, slots=True)
class WideOutputWire(_OutputWire):
    tap: WideOperand
    scalar_type: FloatType | IntType


@dataclass(frozen=True, slots=True)
class BoolOutputWire(_OutputWire):
    tap: BoolOperand
    scalar_type: BoolType = BoolType()


@dataclass(frozen=True, slots=True)
class WideCopy:
    """
    A pc-gated move installing a phi arm's value into the merged register at a predecessor's tail: `dst`
    takes `source` on the block-relative `issue_cycle` (placed by `install_issue_cycle`). Used when a phi arm
    cannot coalesce onto the merged register. `settled_source` records whether `install_source_commit` answered
    None -- which a `RegRef` operand alone cannot reveal -- informational, consumed by tests only.
    """

    dst: RegRef
    source: WideOperand
    issue_cycle: int
    settled_source: bool

    @property
    def is_const(self) -> bool:
        return isinstance(self.source.source, WideConstRef)

    def fire_step(self, fetch_lag: int) -> int:
        return inline_fire_cycle(self.issue_cycle, fetch_lag)

    def landing(self, fetch_lag: int) -> int:
        return install_landing(self.fire_step(fetch_lag))


@dataclass(frozen=True, slots=True)
class BoolWrite:
    """
    A boolean register install of a phi arm (a bool const or another bool register, with the arm's folded inversion)
    on a block-relative cycle; see WideCopy.
    """

    dst: BoolRegRef
    source: BoolOperand
    issue_cycle: int
    settled_source: bool

    @property
    def is_const(self) -> bool:
        return isinstance(self.source.source, BoolConstRef)

    def fire_step(self, fetch_lag: int) -> int:
        return inline_fire_cycle(self.issue_cycle, fetch_lag)

    def landing(self, fetch_lag: int) -> int:
        return install_landing(self.fire_step(fetch_lag))


@dataclass(frozen=True, slots=True)
class Jump:
    target: int


@dataclass(frozen=True, slots=True)
class Branch:
    cond: BoolRegRef
    if_true: int
    if_false: int


@dataclass(frozen=True, slots=True)
class Ret:
    """The sole function exit: outputs and persistent state are resident at the block boundary."""


type Terminator = Jump | Branch | Ret


def terminator_arms(terminator: Terminator) -> list[int]:
    match terminator:
        case Jump(target=target):
            return [target]
        case Branch(if_true=if_true, if_false=if_false):
            return [if_true, if_false]
        case Ret():
            return []
        case _:
            assert_never(terminator)


@dataclass(frozen=True, slots=True)
class LirBlock:
    """
    One basic block of the scheduled microprogram, with block-relative cycles (block start is cycle 0). `ops`
    (pooled firings), `inline_ops`, `wide_copies`, and `bool_writes` are the block's datapath events;
    `terminator` redirects the fetch PC at the block boundary. `block_makespan` is the last commit cycle inside
    the block (0 if it has none). `term_offset` is the block-relative fetch cycle at which the terminator redirects
    the PC -- the block's boundary step -- and is the single source of truth for the terminator PC (the successor
    frame begins one step later, at `term_pc + 1`). For a block that drains (a multi-predecessor successor or a
    tail install) it is the latest cycle a value LANDS in the block's frame -- taken per landing event (every
    result, pooled or inline, wide or boolean, lands at the one bank-independent `landing_cycle`) -- but cross-block
    software pipelining shrinks it to the issue-side envelope when the block's in-flight results may spill into
    single-predecessor successors -- so a consumer reads it here rather than re-deriving the boundary.
    """

    index: int
    ops: list[PooledScheduledOp]
    inline_ops: list[InlineScheduledOp]
    wide_copies: list[WideCopy]
    bool_writes: list[BoolWrite]
    terminator: Terminator
    block_makespan: int
    term_offset: int


def _trace_landing(
    by_index: dict[int, LirBlock], block_base: list[int], block: LirBlock, landing_cycle: int
) -> list[int]:
    """
    Resolve a block-local `landing_cycle` to its absolute landing PC(s), following overlap spills across terminators
    exactly as the numerical model re-keys its in-flight writes at a redirect (see Lir.write_landing_pcs).
    """
    if landing_cycle <= block.term_offset:
        return [block_base[block.index] + landing_cycle]
    spilled = successor_local_cycle(landing_cycle, block.term_offset)
    arms = terminator_arms(block.terminator)
    return [pc for arm in arms for pc in _trace_landing(by_index, block_base, by_index[arm], spilled)]


@dataclass(frozen=True, slots=True)
class BoolBoundaryInstall:
    """
    A read-first copy of a boolean slot's live-out `source` into the slot register at the accepted-output edge, so an
    output or branch still reading the live-in sees the old value. The boolean bank has no early install.
    """

    source: BoolOperand


@dataclass(frozen=True, slots=True)
class BoolStateSlot:
    """
    A persistent boolean state register: reset to `reset_value`, holding the slot's live-in throughout the
    transaction until `install` replaces it with the slot's live-out.
    """

    name: str
    reg: BoolRegRef
    reset_value: bool
    install: InPlace | BoolBoundaryInstall


@dataclass(frozen=True, slots=True)
class RegFileLayout:
    nreg: int
    nrd: int
    nwr: int
    nload: int


@dataclass(frozen=True, slots=True)
class BoolRegFileLayout:
    """The boolean register bank: `nreg` 1-bit registers (branch conditions and boolean state)."""

    nreg: int


# The common surface of the two firing classes (operator/operands/writes/issue/commit), as the model consumes it.
type ScheduledOp = PooledScheduledOp | InlineScheduledOp


@dataclass(frozen=True, slots=True)
class Lir:
    module_name: str
    instances: list[OperatorInstance]
    wide_consts: list[WideValue]  # constant pool: index -> value
    float_format: FloatFormat
    int_format: IntFormat
    regfile: RegFileLayout
    inputs: list[WideInputLoad | BoolInputLoad]  # ordered as the function parameters
    ops: list[PooledScheduledOp]  # the pipelined pooled firings, flattened across blocks with ABSOLUTE issue cycles
    outputs: list[WideOutputWire | BoolOutputWire]
    wide_state_slots: list[WideStateSlot]  # persistent registers, ordered by attribute path
    # Control-flow overlay. A straight-line kernel has a single block ending in Ret; `blocks[0]` is the entry,
    # `block_base[i]` is block i's absolute start PC, and `last_pc` is the out_valid boundary (the single Ret).
    blocks: list[LirBlock]
    block_base: list[int]
    entry: int
    last_pc: int  # LASTPC: the fetch PC at which out_valid asserts (the single Ret block's boundary)
    min_initiation_interval: int  # shortest executable path latency; exact for branch-free kernels, else a lower bound
    bool_regfile: BoolRegFileLayout
    bool_state_slots: list[BoolStateSlot]  # persistent boolean registers, ordered by attribute path
    fetch_lag: int  # steps the control fetch leads the datapath; threaded from build(), one less than its fetch_stages

    @property
    def wide_register_width(self) -> int:
        """
        The wide bank's own width, which the RTL spells WREG. It equals the integer format's because one register
        holds either family whole -- an integer filling it exactly, a float occupying the low bits -- and that is a
        design decision rather than an identity, so the physical width is asked for by name and the two readings
        cannot drift apart unnoticed.
        """
        return self.int_format.width

    def _wide_widths(self) -> Iterator[int]:
        for carrier in [*self.inputs, *self.outputs]:
            if isinstance(carrier, (WideInputLoad, WideOutputWire)):
                yield carrier.scalar_type.width
        for value in [*self.wide_consts, *(slot.reset_value for slot in self.wide_state_slots)]:
            yield value.fmt.width
        firings: list[HardwareOperator] = [inst.operator for inst in self.instances]
        firings += [op.operator for block in self.blocks for op in block.inline_ops]
        for operator in firings:
            signature = operator.signature
            yield from (ty.width for ty in signature.operand_types + signature.result_types if ty.is_wide)

    def __post_init__(self) -> None:
        # Nothing wider than the bank may live in it, and the float format alone no longer answers whether anything
        # does: a kernel carrying no float sizes the word below it. Registers are untyped, so every typed carrier and
        # every operator port is the evidence instead.
        assert all(width <= self.wide_register_width for width in self._wide_widths())
        assert self.fetch_lag in (1, 2), self.fetch_lag
        assert len({inst.operator for inst in self.instances}) == len(
            {type(inst.operator) for inst in self.instances}
        ), "instance names index within the mnemonic, so one operator configuration per pooled class"
        # Cross-block instance reuse on a DRAINED edge -- onto a multi-predecessor successor (a merge, a loop
        # header, the Ret), which carries no per-instance busy residue -- needs the instance provably idle by the
        # time that successor first issues on it: the worst case is a firing committing at its block's makespan
        # (issue = makespan - latency), and the redirect-plus-fetch gap to the successor's first issue is at least
        # `latency + drain + 1` (the `drain` below). The drain is bank-independent -- every result lands at the
        # one landing -- so the worst-case gap is the same for every operator regardless of result bank. (A
        # single-predecessor successor inherits the residue explicitly via `entry_busy` and is sound for any
        # initiation interval.) The gap beyond the drain is the one-step terminator redirect into the successor
        # frame; every block's first pooled issue is block-local cycle 0, so it adds nothing past the redirect.
        # This bound guards those drained edges. Checked here, where the fetch lag is known, over every pooled instance.
        drain = boundary_step(0, self.fetch_lag)
        for inst in self.instances:
            bound = inst.operator.latency + drain + 1
            assert inst.operator.initiation_interval <= bound, (
                f"{inst.operator.mnemonic}: initiation_interval {inst.operator.initiation_interval} needs cross-block "
                f"busy tracking (max supported is latency + {drain + 1})"
            )
        # The numerical model evaluates firings in isolation and cannot witness a double issue; the residue across an
        # overlapped seam is asserted in the allocator, where it is known.
        for block in self.blocks:
            windows: dict[OperatorInstance, list[range]] = {}
            for op in block.ops:
                windows.setdefault(op.inst, []).append(
                    range(op.issue_cycle, op.issue_cycle + op.operator.initiation_interval)
                )
            for inst, spans in windows.items():
                spans.sort(key=lambda span: span.start)
                assert all(
                    earlier.stop <= later.start for earlier, later in zip(spans, spans[1:])
                ), f"block {block.index}: two firings busy on {inst.name} at once"

    @property
    def ports(self) -> list[Port]:
        ports: list[Port] = [
            ControlInputPort("clk", 1),
            ControlInputPort("rst", 1),
            ControlInputPort("in_valid", 1),
            ControlOutputPort("in_ready", 1),
            ControlOutputPort("out_valid", 1),
            ControlInputPort("out_ready", 1),
        ]
        ports += [DataInputPort(f"in_{load.name}", load.scalar_type) for load in self.inputs]
        ports += [DataOutputPort(wire.name, wire.scalar_type) for wire in self.outputs]
        ports.append(ControlOutputPort("err_pc", self.cyc_width))
        return ports

    @property
    def wide_inputs(self) -> list[WideInputLoad]:
        return [load for load in self.inputs if isinstance(load, WideInputLoad)]

    @property
    def bool_inputs(self) -> list[BoolInputLoad]:
        return [load for load in self.inputs if isinstance(load, BoolInputLoad)]

    @property
    def wide_outputs(self) -> list[WideOutputWire]:
        return [wire for wire in self.outputs if isinstance(wire, WideOutputWire)]

    @property
    def bool_outputs(self) -> list[BoolOutputWire]:
        return [wire for wire in self.outputs if isinstance(wire, BoolOutputWire)]

    @property
    def input_ports(self) -> list[DataInputPort]:
        return [port for port in self.ports if isinstance(port, DataInputPort)]

    @property
    def output_ports(self) -> list[DataOutputPort]:
        return [port for port in self.ports if isinstance(port, DataOutputPort)]

    @property
    def control_ports(self) -> list[ControlPort]:
        return [port for port in self.ports if isinstance(port, ControlPort)]

    @property
    def cyc_width(self) -> int:
        """Bit width of the err_pc diagnostic, which latches the executing step `pc - fetch_lag` of an error."""
        return max(1, (self.last_pc - self.fetch_lag).bit_length())

    @property
    def initiation_interval(self) -> int:
        """
        The out_valid boundary PC (`last_pc`). For a straight-line kernel this equals the observable
        in_valid->out_valid latency; with branches the per-path latency varies and is reported by the numerical model,
        while `min_initiation_interval` is the statically-known lower bound (exact when branch-free).
        """
        return self.last_pc

    def term_pc(self, block: LirBlock) -> int:
        """
        The absolute fetch PC at which `block`'s terminator redirects the PC: its base plus its `term_offset`. The
        single derivation consumed by the emitter's next-PC sequencer, the numerical model, the HTML report, and the
        boolean-condition liveness, so a terminator's address cannot drift between them.
        """
        return self.block_base[block.index] + block.term_offset

    def write_landing_pcs(self, block: LirBlock, op: ScheduledOp) -> list[int]:
        """
        Every absolute fetch PC at which `op`'s result committed in `block` lands -- one per execution path that can
        reach it. The landing is op-wide (every tapped write of one firing commits together), so it takes no `write`.
        Every result -- pooled or inline, on either bank -- lands at the one bank-independent `landing_cycle`.
        A landing at or before the block's terminator offset lands once, inside the block.
        A landing past an overlap-shrunk terminator spills into EACH successor arm's frame, at
        `block_base[arm] + (landing - term_offset - 1)`. This is exactly the numerical model's redirect re-keying of
        its in-flight writes, so the report places a spilled result where the hardware actually writes it on every
        path -- not in the linear fall-through frame. A drained block never spills, so a drained
        kernel returns one PC per write.

        The recursion re-keys at every terminator the landing crosses, mirroring the model exactly. A spilled result can
        therefore re-spill across a second shrunk terminator -- which needs a near-empty overlapping intermediate block,
        a shape current frontends do not emit, so in practice this resolves in one hop. The recursion is general-case
        insurance, and terminates because spills only cross single-predecessor forward edges (a finite DAG; a back-edge
        target is multi-predecessor and never overlaps).
        """
        local_landing = landing_cycle(op.commit_cycle, self.fetch_lag)
        by_index = {b.index: b for b in self.blocks}
        return _trace_landing(by_index, self.block_base, block, local_landing)

    @property
    def group_by_cycle(self) -> tuple[dict[int, list[PooledScheduledOp]], dict[int, list[PooledScheduledOp]]]:
        issues: dict[int, list[PooledScheduledOp]] = {}
        commits: dict[int, list[PooledScheduledOp]] = {}
        for op in self.ops:
            issues.setdefault(op.issue_cycle, []).append(op)
            commits.setdefault(op.commit_cycle, []).append(op)
        for group in (issues, commits):
            for ops in group.values():
                ops.sort(
                    key=lambda op: (
                        op.inst.operator.mnemonic,
                        op.inst.index,
                        op.writes[0].dst.index,
                        op.issue_cycle,
                    )
                )
        return issues, commits

    def _cfg_residence(
        self,
        defs: dict[_BankReg, list[int]],
        uses: dict[_BankReg, list[int]],
    ) -> dict[_BankReg, set[int]]:
        """
        Collapse a bank's absolute def/use PCs into the rows each register holds a live value, computed PER BASIC BLOCK
        (where the PC stream is straight-line, so `_residence_rows` is exact) with backward register liveness carrying
        a value across block boundaries. This is path-aware where a single global timeline is not: a value live on two
        mutually-exclusive arms that rejoin at a merge stays resident on BOTH arms, instead of the later-addressed arm's
        landing truncating the earlier one. For a straight-line kernel (one block) it reduces to a single
        `_residence_rows` over the whole frame.

        `defs`/`uses` are absolute fetch PCs (the report grid's row axis); each falls inside exactly one block's
        `[base, term_pc]` range (the ranges tile the frame contiguously in layout order). Within a block a live-in
        register is given a pseudo-def at the block base and a live-out one a pseudo-use at the terminator PC, so the
        per-block `_residence_rows` extends the carried value across the whole block.
        """
        order = sorted(range(len(self.blocks)), key=lambda i: self.block_base[i])
        sorted_bases = [self.block_base[i] for i in order]
        term_pc = {block.index: self.term_pc(block) for block in self.blocks}
        succ = {block.index: terminator_arms(block.terminator) for block in self.blocks}

        def block_of(pc: int) -> int:
            return order[bisect_right(sorted_bases, pc) - 1]

        block_defs: dict[int, dict[_BankReg, list[int]]] = {block.index: {} for block in self.blocks}
        block_uses: dict[int, dict[_BankReg, list[int]]] = {block.index: {} for block in self.blocks}
        for reg, pcs in defs.items():
            for pc in pcs:
                block_defs[block_of(pc)].setdefault(reg, []).append(pc)
        for reg, pcs in uses.items():
            for pc in pcs:
                block_uses[block_of(pc)].setdefault(reg, []).append(pc)

        # Per-block register liveness sets: `written` is defined in the block; `upward` is read STRICTLY before its
        # first def in the block, so it is needed at block entry (live-in). The strict `<` matters: a read on the
        # landing cycle of its own def reads the just-committed value (the model lands the write, then reads, at that
        # PC), not a live-in -- a same-PC def+use (e.g. an output tap or branch condition read on the cycle it lands)
        # must NOT be treated as live-in, or its residence would be painted spuriously back to the block entry.
        written: dict[int, set[_BankReg]] = {}
        upward: dict[int, set[_BankReg]] = {}
        for index in block_defs:
            ds, us = block_defs[index], block_uses[index]
            written[index] = set(ds)
            upward[index] = {reg for reg, reads in us.items() if reg not in ds or min(reads) < min(ds[reg])}
        live_in: dict[int, set[_BankReg]] = {index: set() for index in block_defs}
        live_out: dict[int, set[_BankReg]] = {index: set() for index in block_defs}
        changed = True
        while changed:  # backward dataflow over the block CFG; converges (monotone over a finite lattice)
            changed = False
            for index in block_defs:
                out: set[_BankReg] = set().union(*(live_in[s] for s in succ[index]), set())
                new_in = upward[index] | (out - written[index])
                if out != live_out[index] or new_in != live_in[index]:
                    live_out[index], live_in[index] = out, new_in
                    changed = True

        rows: dict[_BankReg, set[int]] = {}
        for index in block_defs:
            base, boundary = self.block_base[index], term_pc[index]
            active = written[index] | upward[index] | live_in[index] | live_out[index]
            for reg in active:
                d = block_defs[index].get(reg, []) + ([base] if reg in live_in[index] else [])
                u = block_uses[index].get(reg, []) + ([boundary] if reg in live_out[index] else [])
                resident = _residence_rows(d, u, boundary)
                if resident:
                    rows.setdefault(reg, set()).update(resident)
        return rows

    def _add_boundary_installs(self, rows: dict[_BankReg, set[int]], boundary_installed: list[_BankReg]) -> None:
        """
        A boundary install writes its slot on the accepted-output edge, after every read on the last PC, so the
        live-out it lands occupies the register on that PC alone and changes no other value's residence.
        """
        for reg in boundary_installed:
            rows.setdefault(reg, set()).add(self.last_pc)

    def _collect_op_events(
        self, reg_type: type[_BankReg], defs: dict[_BankReg, list[int]], uses: dict[_BankReg, list[int]]
    ) -> None:
        """
        Add one bank's per-block datapath op events to `defs`/`uses`: each result LANDING (stamped via
        `write_landing_pcs` at every successor-arm PC it spills into) and each operand READ. The single definition of
        op read/write timing, shared by both banks so reg_liveness and bool_liveness cannot drift.
        """
        block: (
            LirBlock  # explicit binding: the loop target's type is undecidable under the constrained-TypeVar reanalysis
        )
        for block in self.blocks:
            base_pc = self.block_base[block.index]
            block_ops: list[ScheduledOp] = [*block.ops, *block.inline_ops]
            for op in block_ops:
                read = operand_read_cycle(op.operator, base_pc + op.issue_cycle, self.fetch_lag)
                for write in op.writes:
                    if isinstance(write.dst, reg_type):
                        defs.setdefault(write.dst, []).extend(self.write_landing_pcs(block, op))
                for operand in op.operands:
                    if isinstance(operand.source, reg_type):
                        uses.setdefault(operand.source, []).append(read)

    @property
    def reg_liveness(self) -> dict[RegRef, set[int]]:
        """
        Map each wide register to the actual clock cycles on which it holds a live value.

        This is cycle-accurate to the emitted hardware, in the executing-step (hardware) frame. Timing comes from the
        shared helpers: an input lands on cycle 1; every operator result lands at `landing_cycle` (which for the last
        result is the initiation interval), selected per op by `write_landing_pcs`; an operand is read on
        `operand_read_cycle`; an output tap on the present cycle; and a slot install samples its source on its fire
        step -- the present cycle for a boundary install, earlier for an early one (the landing follows below). A slot
        register additionally stays live through the present cycle, since its live-out must reside there for the next
        initiation. Each row spans a value from when it lands in the array through its last read.

        Diagnostic only -- consumed by the reports (e.g., HTML schedule) and the tests, never by the emitter or the
        numerical model. Each op-result LANDING is stamped via `write_landing_pcs` at exactly the PC(s) the model
        writes it -- on every successor arm under overlap, not just the fall-through. A pc-gated install (a phi copy or
        an early state install) fires and samples its source on the copy step but lands its destination
        one PC later via `install_landing` -- the same +1 the model commits. A boundary install reads-then-writes
        at the boundary and lands there. Residence is then resolved per basic block by `_cfg_residence` (CFG-aware
        register liveness), so a value live on two mutually-exclusive arms that rejoin at a merge stays resident on
        BOTH arms. The result is cycle-exact to the numerical model on every register and every path.
        """
        present = self.initiation_interval  # hardware-frame present / boundary step
        defs: dict[RegRef, list[int]] = {}
        uses: dict[RegRef, list[int]] = {}
        boundary_installed: list[RegRef] = []
        for load in self.wide_inputs:
            defs.setdefault(load.dst, []).append(1)
        for slot in self.wide_state_slots:
            defs.setdefault(slot.reg, []).append(1)  # the live-in is resident in the slot register from the start
            match slot.install:
                case InPlace():
                    # An in-place live-out is an ordinary result already in the slot register; it must reside through
                    # the boundary to carry into the next initiation, even when nothing reads it again this frame.
                    uses.setdefault(slot.reg, []).append(present)
                case WideEarlyInstall() as install:
                    # An early install lands its destination one PC after its fire step and must reside through the
                    # boundary to carry; installing the new value early is not the slot's death.
                    step = install.fire_step(self.fetch_lag)
                    defs.setdefault(slot.reg, []).append(install.landing(self.fetch_lag))
                    uses.setdefault(slot.reg, []).append(present)
                    if isinstance(install.source.source, RegRef):
                        uses.setdefault(install.source.source, []).append(step)
                case WideBoundaryInstall(source=source):
                    boundary_installed.append(slot.reg)
                    if isinstance(source.source, RegRef):
                        uses.setdefault(source.source, []).append(present)
                case _:
                    assert_never(slot.install)
        for wire in self.wide_outputs:
            if isinstance(wire.tap.source, RegRef):
                uses.setdefault(wire.tap.source, []).append(present)
        for block in self.blocks:
            base_pc = self.block_base[block.index]
            for copy in block.wide_copies:  # phi copy fires here and samples its source; destination lands one PC later
                step = base_pc + copy.fire_step(self.fetch_lag)
                defs.setdefault(copy.dst, []).append(install_landing(step))
                if isinstance(copy.source.source, RegRef):
                    uses.setdefault(copy.source.source, []).append(step)
        self._collect_op_events(RegRef, defs, uses)
        rows = self._cfg_residence(defs, uses)
        self._add_boundary_installs(rows, boundary_installed)
        return rows

    @property
    def bool_liveness(self) -> dict[BoolRegRef, set[int]]:
        """
        Map each boolean register to the cycles on which it holds a live value, the boolean-bank counterpart of
        reg_liveness in the same executing-step frame. A boolean register is defined when a comparison,
        boolean-logic op, or float->bool cast commits its result, when a boolean phi/state install lands, and -- for a
        persistent slot -- at the live-in resident from cycle 1; it is read by a boolean-logic op or a bool->float cast
        taking it as an operand, by a branch testing it as a condition, by a phi/state install copying it, and at the
        boundary where a slot's live-out must persist for the next initiation. A boolean result that spills past an
        overlap-shrunk terminator is stamped on every successor arm via `write_landing_pcs`, exactly as the numerical
        model re-keys it; a phi/boolean write lands its destination one PC after its fire step via `install_landing`
        (the model's +1), while a boolean slot always installs read-first at the boundary. Residence is resolved by
        the same per-block `_cfg_residence` as reg_liveness, so a spilled or merged boolean is cycle-exact too.
        """
        present = self.initiation_interval
        defs: dict[BoolRegRef, list[int]] = {}
        uses: dict[BoolRegRef, list[int]] = {}
        boundary_installed: list[BoolRegRef] = []
        for slot in self.bool_state_slots:
            defs.setdefault(slot.reg, []).append(1)  # the live-in is resident from the start
            match slot.install:
                case InPlace():
                    uses.setdefault(slot.reg, []).append(present)  # an in-place live-out resides through the boundary
                case BoolBoundaryInstall(source=source):
                    boundary_installed.append(slot.reg)
                    if isinstance(source.source, BoolRegRef):
                        uses.setdefault(source.source, []).append(present)
                case _:
                    assert_never(slot.install)
        for load in self.bool_inputs:
            defs.setdefault(load.dst, []).append(1)
        for wire in self.bool_outputs:
            if isinstance(wire.tap.source, BoolRegRef):
                uses.setdefault(wire.tap.source, []).append(present)
        for block in self.blocks:
            base_pc = self.block_base[block.index]
            for bwrite in block.bool_writes:  # bool write fires and samples here; destination lands one PC later
                step = base_pc + bwrite.fire_step(self.fetch_lag)
                defs.setdefault(bwrite.dst, []).append(install_landing(step))
                if isinstance(bwrite.source.source, BoolRegRef):
                    uses.setdefault(bwrite.source.source, []).append(step)
            if isinstance(block.terminator, Branch):  # the next-PC case reads the condition at the block boundary PC
                uses.setdefault(block.terminator.cond, []).append(self.term_pc(block))
        self._collect_op_events(BoolRegRef, defs, uses)
        rows = self._cfg_residence(defs, uses)
        self._add_boundary_installs(rows, boundary_installed)
        return rows

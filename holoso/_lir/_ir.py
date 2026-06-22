"""
The low-level IR (LIR): the scheduled, bound, register-allocated microprogram for the synthesized ZISC machine.

A :class:`Lir` is controller-agnostic -- it describes which hardware operators issue on which cycle, reading/writing
which typed storage resources, with which folded sign controls.
"""

from bisect import bisect_right
from dataclasses import dataclass
from typing import TypeVar, assert_never

from .._operators import (
    BoolInversion,
    FloatSignControl,
    HardwareOperator,
    InlineHardwareOperator,
    PooledHardwareOperator,
    PortConditioner,
)
from .._type import BoolType, FloatFormat, FloatType, ScalarType, is_wide_type
from ._ports import ControlInputPort, ControlOutputPort, ControlPort, DataInputPort, DataOutputPort, Port

FETCH_STAGES = 3
FETCH_LAG = FETCH_STAGES - 1


# The cycle-accurate timing model: one consistent physical story shared by the LIR cycle helpers below, the numerical
# model, the scheduler's dependency edges, the register allocator, and the HTML report, so a value's
# landing/read/copy/boundary cycle is computed in exactly one place and the consumers cannot drift.
# It is built from primitives:
#   - FETCH_LAG -- the microcode fetch leads the datapath by this many steps (a global frame offset);
#   - a per-bank READ latch -- the wide bank presents an operand's read address one step early; the boolean bank reads
#     latch-free on the in_valid step;
#   - a per-bank WRITEBACK latch -- a wide result is registered one step before the array write; the boolean bank
#     writes directly at its commit step;
#   - READ_FIRST_EDGE -- a register read sees the value written one step earlier (write-then-read), so a result becomes
#     readable one step after it is written, on either bank.
# Timing is a property of the REGISTER BANK and of the operation class (instance-backed vs inline), never of an
# individual operator; the operator's own LATENCY is the orthogonal pipeline depth (a pooled instance has L stages, an
# inline combinational op has none).
READ_FIRST_EDGE = 1


@dataclass(frozen=True, slots=True)
class BankTiming:
    """
    The latching of one register bank. ``read_latch`` is how many steps early a read address is presented (the wide
    bank's read latch); ``writeback_latch`` is how many steps a result is registered before the array write (the wide
    bank's writeback latch). The boolean bank is latch-free, so both are zero.
    """

    read_latch: int
    writeback_latch: int


WIDE_BANK = BankTiming(read_latch=1, writeback_latch=1)
BOOL_BANK = BankTiming(read_latch=0, writeback_latch=0)


def bank_timing(is_wide: bool) -> BankTiming:
    return WIDE_BANK if is_wide else BOOL_BANK


def landing_cycle(commit_cycle: int, bank: BankTiming) -> int:
    """The cycle a result committed at ``commit_cycle`` becomes readable: writeback latch then read-first edge."""
    return commit_cycle + FETCH_LAG + bank.writeback_latch + READ_FIRST_EDGE


def read_cycle(issue_cycle: int, bank: BankTiming) -> int:
    """The cycle an instance-backed operator samples a register operand on ``bank``: read latch presents it early."""
    return issue_cycle + FETCH_LAG - bank.read_latch


def wide_landing_cycle(commit_cycle: int) -> int:
    """The cycle a wide result is readable in the array: its commit plus the writeback latch and the read-first edge."""
    return landing_cycle(commit_cycle, WIDE_BANK)


def bool_landing_cycle(commit_cycle: int) -> int:
    """The cycle a boolean result becomes readable: written directly at its commit step, visible the next step."""
    return landing_cycle(commit_cycle, BOOL_BANK)


def pooled_wide_read_cycle(issue_cycle: int) -> int:
    """The cycle an instance-backed operator samples a wide operand -- the read latch presents the address early."""
    return read_cycle(issue_cycle, WIDE_BANK)


def pooled_bool_read_cycle(issue_cycle: int) -> int:
    """The cycle an instance-backed operator samples a boolean operand: latch-free, directly on its in_valid step."""
    return read_cycle(issue_cycle, BOOL_BANK)


def inline_fire_cycle(commit_cycle: int, dst_is_wide: bool) -> int:
    """
    The cycle an inline combinational operation (boolean logic, a float<->bool cast) fires: it is one PC-gated
    statement that reads ALL its operands and writes its destination on this single step -- the commit step for a
    boolean destination, one later for a wide one (aligned with the destination bank's writeback latch).
    """
    return commit_cycle + FETCH_LAG + bank_timing(dst_is_wide).writeback_latch


def pooled_writeback_word(commit_cycle: int, dst_is_wide: bool) -> int:
    """
    The FETCH-PC-frame step on which a pooled lane drives its write-enable/address microcode word: the commit step plus
    the destination bank's writeback latch. This is the lone helper in the fetch-PC frame (no ``FETCH_LAG``) -- it
    places the microcode word the sequencer fetches, not the step the datapath acts on it. Shared by the emitter's
    microcode (where the word is placed) and the overlap layout (which keeps every write word inside the block), so the
    two cannot drift -- the same single-source-of-truth contract as the landing/read helpers above.
    """
    return commit_cycle + bank_timing(dst_is_wide).writeback_latch


def operand_read_cycle(operator: HardwareOperator, issue_cycle: int) -> int:
    """
    The hardware-frame cycle on which an operation samples its register operands (an operation reads all its operands
    on one cycle), the single definition shared by the register allocator's interference, the liveness views, and the
    numerical model so none can drift. A pooled instance reads through the wide read latch; no pooled operator reads
    a boolean operand yet, ENFORCED here -- when one appears it is presented latch-free on the in_valid step
    (``pooled_bool_read_cycle``) and this dispatch must grow per-operand granularity, reconciled with
    ``dependency_edge``. An inline operation fires -- and reads -- on its writeback step.
    """
    if isinstance(operator, PooledHardwareOperator):
        assert all(is_wide_type(ty) for ty in operator.signature.operand_types), operator.mnemonic
        return pooled_wide_read_cycle(issue_cycle)
    dst_is_wide = is_wide_type(operator.signature.result_types[0])
    return inline_fire_cycle(issue_cycle + operator.latency, dst_is_wide)


def dependency_edge(producer: HardwareOperator, producer_port: int, consumer: HardwareOperator) -> int:
    """
    The minimum same-block scheduling distance from a producer's commit to a consumer's issue (``issue_consumer >=
    commit_producer + edge``), derived from the landing of the producer's tapped output port's bank and the
    consumer's operand-read timing so the scheduler, the liveness views, and the model share one rule. The clamp's
    floor ``READ_FIRST_EDGE - consumer.latency`` holds the consumer's COMMIT at least one read-first edge after the
    producer's -- the same write-then-read edge a register read pays, expressed at commit level not read level -- so
    an inline consumer (latency 0) commits strictly after its producer, which the model's commit-ordered evaluation
    relies on. The zero-offset evaluation below is exact because every cycle helper is affine in its cycle argument
    with unit slope, so the difference at zero is the frame-independent spacing; a helper that ever loses that
    affinity breaks this derivation. No pooled operator reads a boolean operand yet, ENFORCED here in lockstep with
    ``operand_read_cycle`` (which charges every pooled consumer the wide read latch): the first bool-reading pooled
    operator must reconcile its presentation -- latch-free on the in_valid step, ``pooled_bool_read_cycle`` -- in
    both helpers at once.
    """
    producer_wide = is_wide_type(producer.signature.result_types[producer_port])
    landing = landing_cycle(0, bank_timing(producer_wide))
    if isinstance(consumer, PooledHardwareOperator):
        assert producer_wide, f"{consumer.mnemonic}: pooled operators read only wide operands today"
        read = pooled_wide_read_cycle(0)
    else:
        dst_is_wide = is_wide_type(consumer.signature.result_types[0])
        read = inline_fire_cycle(consumer.latency, dst_is_wide)
    return max(landing - read, READ_FIRST_EDGE - consumer.latency)


def copy_step_cycle(install_cycle: int) -> int:
    """
    The hardware fetch step on which a pc-gated install -- a non-coalesced slot writeback, a phi copy, or a boolean
    write -- fires and combinationally samples its source register. ``install_cycle`` is the scheduler-frame placement
    of the install (``makespan + 1`` at the boundary, earlier for an eagerly-installed float slot; see
    ``FloatStateSlot.install_cycle``), not a commit or landing cycle. A copy is a latch-free reg->reg move -- it carries
    no pooled read latch -- so it samples its source one READ_FIRST_EDGE past that placement in the executing frame,
    beyond the FETCH_LAG shift. That edge is bank-INDEPENDENT (a register read sees the prior step's value on either
    bank), which is why wide copies and boolean writes share this one formula; any bank-dependent writeback latch is
    absorbed into ``install_cycle`` upstream, never charged here. The destination's own write-then-read edge is added
    separately by ``install_landing``, giving the cosim-verified identity ``install_landing(copy_step_cycle(c)) ==
    wide_landing_cycle(c)``: an install placed at ``c`` lands exactly where a direct wide result committed at ``c``
    would, the coalescing equivalence the overlap layout relies on.
    """
    return install_cycle + FETCH_LAG + READ_FIRST_EDGE


def install_landing(fire_step: int) -> int:
    """
    The step a pc-gated install -- a phi copy, a boolean write, or an early (non-boundary) slot writeback -- commits its
    destination and becomes readable: one after the step it fires on and samples its source. The model writes the
    destination into ``_pending`` one PC past the fire, so the numerical model and the liveness diagnostic route this +1
    through this one helper and cannot drift. A boundary slot install is the lone exception: it reads-then-writes at
    ``last_pc`` and does not pass through here.
    """
    return fire_step + READ_FIRST_EDGE


def boundary_step(makespan: int, wide_resident: bool) -> int:
    """
    The drained boundary / initiation-interval step: the cycle the block's latest boundary-resident result lands, where
    its live-outs and consumed-at-boundary values are resident. ``wide_resident`` selects that latest value's bank: a
    block holding any wide value at its boundary pays the latched wide landing (the read-first write-latch edge), while
    an all-boolean boundary lands one step earlier on the latch-free boolean bank. The single source of truth shared by
    the overlap layout (the terminator offset), the liveness boundary, and the numerical model, so the per-bank drain
    cannot drift between them.
    """
    return landing_cycle(makespan, bank_timing(wide_resident))


def successor_local_cycle(block_local_cycle: int, term_offset: int) -> int:
    """
    Map a block-local cycle that crosses an overlap-shrunk terminator into the single-predecessor successor's frame.
    The successor frame begins at ``term_pc + 1``, so a cycle at absolute ``block_base + block_local_cycle`` sits at
    ``block_local_cycle - term_offset - 1`` past the successor's base -- one continuous PC across the seam. This is the
    single coordinate map shared by the scheduler's spill carry (both the value landings and the per-instance busy
    residue), ``_trace_landing``, and the numerical model's redirect re-keying, so they cannot drift apart.
    """
    return block_local_cycle - term_offset - 1


def residence_rows(
    defs: list[int], uses: list[int], present: int, read_first_defs: frozenset[int] = frozenset()
) -> set[int]:
    """
    Collapse a register's definition and use cycles into the set of cycles on which it holds a live value: each value
    resides from its landing through its last use STRICTLY before the next definition, the boundary at latest, plus the
    landing cycle itself even when the value is never read (it still occupies the register that cycle). The strict bound
    is the write-then-read register semantics the numerical model commits: a read on a later value's landing cycle reads
    that NEW value, so it belongs to the next definition's residence, not the previous one -- a read on a value's OWN
    landing (the common producer->consumer case, where the consumer reads on the producer's landing PC) still counts for
    that value.

    ``read_first_defs`` lists the definition PCs that are READ-FIRST rather than write-then-read: the boundary state
    install, where the hardware samples the register (the live-in, for an output tap or the install's own source) on
    the boundary edge BEFORE clocking in the new live-out. A read on such a def's PC therefore belongs to the PRIOR
    value, not the def landing there -- the opposite attribution from a normal landing. So the prior value keeps reads
    up to and INCLUDING a read-first next def, and a read-first def's own value keeps only reads strictly after it.
    Shared by the float- and boolean-bank liveness so both banks compute residence in exactly one place.
    """
    writes = sorted(defs)
    reads = sorted(uses)
    rows: set[int] = set()
    for i, start in enumerate(writes):
        nxt = writes[i + 1] if i + 1 < len(writes) else present + 1
        lo_excl = start in read_first_defs  # a read AT a read-first def reads the PRIOR value, not this one's landing
        hi_incl = nxt in read_first_defs  # ...so the prior value keeps reads up to and INCLUDING a read-first next def
        last = max(
            (
                use
                for use in reads
                if (use > start if lo_excl else use >= start) and (use <= nxt if hi_incl else use < nxt)
            ),
            default=start,
        )
        rows.update(range(start, last + 1))
    return rows


@dataclass(frozen=True, slots=True)
class OperatorInstance:
    """
    One physical operator module, e.g. ``u_fadd_326215ea_0`` or ``u_fcmp_7296114c_0``.

    ``operator`` is the fully specified pooled hardware operator it elaborates; ``index`` numbers the copies of that
    operator value. The scheduler pools firings by the hardware-operator instance: equal operators may time-share one
    module, each instance accepting a new firing every ``initiation_interval`` cycles.
    """

    operator: PooledHardwareOperator
    index: int  # 0-based within this concrete operator value

    def __post_init__(self) -> None:
        # Every pooled operator passes through here, so its three hand-synchronized per-port declarations are
        # validated once at the source: HDL port names align with the result types, and the commutation permutation
        # (when declared) is a type-preserving bijection -- a bad future declaration fails here, not in emission.
        result_types = self.operator.signature.result_types
        assert len(self.operator.output_hdl_ports) == len(result_types), self.operator.mnemonic
        permutation = self.operator.swap_output_permutation
        if permutation is not None:
            assert sorted(permutation) == list(range(len(result_types))), self.operator.mnemonic
            assert all(result_types[permutation[p]] == result_types[p] for p in range(len(permutation)))
        # Cross-block instance reuse has two regimes. A single-predecessor successor inherits the predecessor's per-
        # instance busy residue explicitly (``entry_busy``, the cross-block software-pipelining carry), so overlap onto
        # it is sound for ANY initiation interval. A DRAINED edge -- onto a multi-predecessor successor (a merge, a loop
        # header, the Ret), which carries no residue -- instead needs the instance provably idle by the time that
        # successor first issues on it: the worst case is a firing committing at its block's makespan (issue =
        # makespan - latency), and the redirect-plus-fetch gap to the successor's first issue is at least
        # ``latency + boundary_step(0, wide_resident) + 2``, where ``wide_resident`` is THIS operator's own result bank.
        # A wide-producing operator always fires in a wide-resident block (its float result holds the boundary), so its
        # drained successor lays at the latched wide drain; a purely-boolean-producing operator (a comparator) can fire
        # in an all-boolean block whose successor lays one PC earlier under the bank-aware drain -- the shorter gap, the
        # worst case for that operator. The makespan absorbs any entry_busy delay, since it tracks that firing's own
        # commit. This bound guards those drained edges; a deeper-throttled operator on a back-edge loop would
        # additionally need a post-layout re-entry-distance check, deferred until one exists.
        result_is_wide = any(is_wide_type(ty) for ty in result_types)
        drain = boundary_step(0, wide_resident=result_is_wide)
        # The gap beyond the drain is two steps: the terminator's redirect into the successor frame and the
        # successor's first issue step (READ_FIRST_EDGE-spaced); see the two-regime explanation above.
        redirect_and_first_issue = READ_FIRST_EDGE + 1
        bound = self.operator.latency + drain + redirect_and_first_issue
        assert self.operator.initiation_interval <= bound, (
            f"{self.operator.mnemonic}: initiation_interval {self.operator.initiation_interval} needs cross-block "
            f"busy tracking (max supported is latency + {drain + redirect_and_first_issue})"
        )


# An operator READ port: the ``(instance, operand-position)`` pair keying ``read_set_per_port``. Distinct from the
# WRITE-side ``(instance, output-port)`` lanes of ``write_set_per_register``/``_write_sets``, which are not read ports.
type ReadPort = tuple[OperatorInstance, int]


@dataclass(frozen=True, slots=True)
class RegRef:
    """A read/write of wide data register ``index`` in the shared register bank."""

    index: int

    @property
    def stable_label(self) -> str:
        return f"r{self.index}"

    @property
    def is_register(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class BoolRegRef:
    """A read/write of boolean register ``index`` in the 1-bit boolean register bank."""

    index: int

    @property
    def stable_label(self) -> str:
        return f"b{self.index}"

    @property
    def is_register(self) -> bool:
        return True


def _bank_of_ref(dst: RegRef | BoolRegRef) -> BankTiming:
    """The register bank a destination belongs to: wide for a ``RegRef``, the 1-bit boolean bank otherwise."""
    return WIDE_BANK if isinstance(dst, RegRef) else BOOL_BANK


def result_landing_cycle(dst: RegRef | BoolRegRef, commit_cycle: int) -> int:
    """
    The cycle a result lands per its destination bank -- the single dispatch every consumer (liveness, the numerical
    model, the report) routes through, so the per-bank rule cannot drift between them.
    """
    return landing_cycle(commit_cycle, _bank_of_ref(dst))


_BankReg = TypeVar("_BankReg", RegRef, BoolRegRef)  # one register bank's reference type (wide or boolean)


@dataclass(frozen=True, slots=True)
class ConstRef:
    """An immediate constant, ``index`` into one typed LIR constant pool."""

    index: int

    @property
    def stable_label(self) -> str:
        return f"c{self.index}"

    @property
    def is_register(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class FloatConstRef(ConstRef):
    """An immediate floating-point constant, ``index`` into the LIR float constant pool."""


@dataclass(frozen=True, slots=True)
class Operand:
    """An operator input: a register read or a constant immediate."""

    source: RegRef | ConstRef


@dataclass(frozen=True, slots=True)
class FloatOperand(Operand):
    """A float operator input: a wide-register read or float constant immediate, with folded sign control."""

    source: RegRef | FloatConstRef
    sign: FloatSignControl = FloatSignControl()

    @property
    def stable_label(self) -> str:
        return self.sign.decorate(self.source.stable_label)


@dataclass(frozen=True, slots=True)
class InputLoad:
    """An input port sampled into a typed register at in_valid."""

    name: str
    dst: RegRef | BoolRegRef


@dataclass(frozen=True, slots=True)
class FloatInputLoad(InputLoad):
    """A float input port sampled into a wide register at in_valid."""

    dst: RegRef


@dataclass(frozen=True, slots=True)
class FloatStateSlot:
    """
    A persistent float state register: reset to ``reset_value``, holding the slot's live-in (carried over from the
    previous initiation) until the install copy replaces it with the slot's live-out.

    ``tap`` is the live-out's source tap (register/constant + folded sign), the same primitive an output wire taps; here
    the sink is the slot register rather than a port. When the tap is exactly ``reg`` with an identity sign the live-out
    coalesced onto the slot register (its producing operator -- or, for a conditional/loop update, the arms of its phi
    -- wrote it in place) and the backend emits no copy; otherwise the backend fires a reg->reg copy for it, scheduled
    at ``install_cycle`` -- as early as the old live-in is last read and
    the source is available, the initiation boundary at the latest. The copy samples the tap on its fetch step
    (``copy_step_cycle(install_cycle)``) and the new live-out lands one fetch step later (``install_landing``) for an
    early install, or read-first at the boundary (``LASTPC``) for a boundary install. Installing before the boundary
    lets the source register be reused by unrelated operations for the rest of the initiation. A public attribute's
    observable ``state_<name>`` port is a separate output wire tapping the same value, not a property of the slot.
    """

    name: str
    reg: RegRef
    reset_value: float
    tap: FloatOperand
    install_cycle: int  # scheduler-frame install cycle; hardware fire = copy_step_cycle(it); makespan+1 = boundary

    @property
    def needs_copy(self) -> bool:
        return not (self.tap.source == self.reg and self.tap.sign == FloatSignControl())


@dataclass(frozen=True, slots=True)
class BoolInputLoad(InputLoad):
    """A boolean input port sampled into a boolean register at in_valid."""

    dst: BoolRegRef


@dataclass(frozen=True, slots=True)
class BoolConstRef:
    """A boolean immediate (``True``/``False``); the bool bank has no constant pool, the value rides inline."""

    value: bool


type BoolSource = BoolRegRef | BoolConstRef


@dataclass(frozen=True, slots=True)
class BoolOperand:
    """
    A boolean operand: a boolean register read or an immediate True/False, with an optional folded inversion -- the
    1-bit dual of :class:`FloatOperand`'s sign control, free in fabric. An inverted immediate folds to its negated
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
    One tapped output port of a firing: the ``port``-th result lands in ``dst`` through its type's conditioner (a
    folded sign control on a wide destination, an optional inversion on a boolean one). Untapped ports of the firing
    simply have no PortWrite -- the module output is left unconnected.
    """

    port: int
    dst: RegRef | BoolRegRef
    conditioner: PortConditioner


@dataclass(frozen=True, slots=True)
class PooledScheduledOp:
    """
    One pooled-instance firing in the software-pipelined schedule: ``inst`` asserts ``in_valid`` on ``issue_cycle``
    (its operands sampled per the bank read discipline), and on ``commit_cycle == issue_cycle + latency`` every
    tapped output port lands in its destination register. The writes are sorted by port and pairwise distinct in
    both port and destination -- members of one firing land simultaneously, so the allocator always gives them
    distinct registers.
    """

    inst: OperatorInstance
    operands: list[FloatOperand | BoolOperand]
    writes: list[PortWrite]
    issue_cycle: int
    latency: int

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
    One inline-operator firing: a single PC-gated statement that reads its operands and writes its one result on its
    fire step (the commit step for a boolean destination, one later for a wide one).
    """

    operator: InlineHardwareOperator
    operands: list[FloatOperand | BoolOperand]
    write: PortWrite
    issue_cycle: int
    latency: int

    @property
    def writes(self) -> list[PortWrite]:
        return [self.write]

    @property
    def commit_cycle(self) -> int:
        return self.issue_cycle + self.latency


@dataclass(frozen=True, slots=True)
class OutputWire:
    """An output port: a named external sink driven at the present step by a typed source tap."""

    name: str
    tap: FloatOperand | BoolOperand


@dataclass(frozen=True, slots=True)
class FloatOutputWire(OutputWire):
    """A float output port: the named external sink for a float source tap (register/constant + folded sign)."""

    tap: FloatOperand


@dataclass(frozen=True, slots=True)
class BoolOutputWire(OutputWire):
    """A boolean output port: the named external sink for a boolean register or immediate, with a folded inversion."""

    tap: BoolOperand


def scalar_type_of(
    node: FloatInputLoad | BoolInputLoad | FloatOutputWire | BoolOutputWire, fmt: FloatFormat
) -> ScalarType:
    """
    The scalar type carried by an input load or output wire: a float port carries the module format ``fmt``, a boolean
    port is a single bit. The single dispatch every consumer routes through (RTL ports and the numerical model alike),
    so the float/boolean choice cannot drift between them.
    """
    match node:
        case FloatInputLoad() | FloatOutputWire():
            return FloatType(fmt)
        case BoolInputLoad() | BoolOutputWire():
            return BoolType()
        case _:
            assert_never(node)


@dataclass(frozen=True, slots=True)
class FloatCopy:
    """
    A register-to-register move installing a phi arm's value into the merged register at a predecessor's tail: ``dst``
    takes ``source`` on the block-relative ``issue_cycle``. Used when a phi arm is not an operator result that can be
    coalesced directly onto the merged register (e.g. an input, a constant, or a value defined in another block).
    """

    dst: RegRef
    source: FloatOperand
    issue_cycle: int


@dataclass(frozen=True, slots=True)
class BoolWrite:
    """
    A boolean register install of a phi arm (a bool const or another bool register, with the arm's folded inversion)
    on a block-relative cycle.
    """

    dst: BoolRegRef
    source: BoolOperand
    issue_cycle: int


@dataclass(frozen=True, slots=True)
class Jump:
    """Unconditional control transfer to block ``target``."""

    target: int


@dataclass(frozen=True, slots=True)
class Branch:
    """Conditional control transfer on boolean register ``cond``."""

    cond: BoolRegRef
    if_true: int
    if_false: int


@dataclass(frozen=True, slots=True)
class Ret:
    """The sole function exit: outputs and persistent state are resident at the block boundary."""


type Terminator = Jump | Branch | Ret


def terminator_arms(terminator: Terminator) -> list[int]:
    """The successor block indices a terminator can redirect to: a jump's target, a branch's two arms, none for Ret."""
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
    One basic block of the scheduled microprogram, with block-relative cycles (block start is cycle 0). ``ops``
    (pooled firings), ``inline_ops``, ``copies``, and ``bool_writes`` are the block's datapath events; ``terminator``
    redirects the fetch PC at the block boundary. ``block_makespan`` is the last commit cycle inside the block (0 if
    it has none). ``term_offset`` is the block-relative fetch cycle at which the terminator redirects the PC -- the
    block's boundary step -- and is the single source of truth for the terminator PC (the successor frame begins one
    step later, at ``term_pc + 1``). It is the full drain ``boundary_step(block_makespan, wide_resident)`` for a block
    that drains (a multi-predecessor successor, a phi/const install, or a live-in branch condition) -- bank-aware, so a
    block carrying only boolean values across its boundary drains one step earlier than a wide one -- but cross-block
    software pipelining shrinks it to the issue-side envelope when the block's in-flight results may spill into single-
    predecessor successors -- so a consumer reads it here rather than re-deriving the boundary.
    """

    index: int
    ops: list[PooledScheduledOp]
    inline_ops: list[InlineScheduledOp]
    copies: list[FloatCopy]
    bool_writes: list[BoolWrite]
    terminator: Terminator
    block_makespan: int
    term_offset: int


def _trace_landing(
    by_index: dict[int, LirBlock], block_base: list[int], block: LirBlock, landing_cycle: int
) -> list[int]:
    """
    Resolve a block-local ``landing_cycle`` to its absolute landing PC(s), following overlap spills across terminators
    exactly as the numerical model re-keys its in-flight writes at a redirect (see :meth:`Lir.write_landing_pcs`).
    """
    if landing_cycle <= block.term_offset:
        return [block_base[block.index] + landing_cycle]
    spilled = successor_local_cycle(landing_cycle, block.term_offset)
    arms = terminator_arms(block.terminator)
    return [pc for arm in arms for pc in _trace_landing(by_index, block_base, by_index[arm], spilled)]


@dataclass(frozen=True, slots=True)
class BoolStateSlot:
    """
    A persistent boolean state register: reset to ``reset_value``, holding the slot's live-in throughout the
    transaction and installing its live-out (``live_out``, a boolean register or constant with a folded inversion)
    at the boundary, read-first
    -- so an output or branch that still reads the live-in sees the old value, exactly like a float slot. When the
    live-out already resides in the slot register uninverted it coalesced there (its producing operation, or the arms
    of its phi for a conditional/loop update, wrote it in place) and ``needs_copy`` is False -- no boundary copy.
    """

    name: str
    reg: BoolRegRef
    reset_value: bool
    live_out: BoolOperand

    @property
    def needs_copy(self) -> bool:
        """
        False only when the live-out already resides in the slot register UNINVERTED (an unwritten slot); a live-out
        under an inversion needs the install copy to apply it, even from the slot's own register.
        """
        return not (
            isinstance(self.live_out.source, BoolRegRef)
            and self.live_out.source == self.reg
            and not self.live_out.inversion.invert
        )


@dataclass(frozen=True, slots=True)
class RegFileLayout:
    """The shared wide data register file resource."""

    width: int
    nreg: int
    nrd: int
    nwr: int
    nload: int


@dataclass(frozen=True, slots=True)
class BoolRegFileLayout:
    """The boolean register bank: ``nreg`` 1-bit registers (branch conditions and boolean state)."""

    nreg: int


# The common surface of the two firing classes (operator/operands/writes/issue/commit), as the model consumes it.
type ScheduledOp = PooledScheduledOp | InlineScheduledOp


@dataclass(frozen=True, slots=True)
class Lir:
    module_name: str
    instances: list[OperatorInstance]
    float_consts: list[float]  # constant pool: index -> value
    float_format: FloatFormat
    regfile: RegFileLayout
    inputs: list[FloatInputLoad | BoolInputLoad]  # ordered as the function parameters
    ops: list[PooledScheduledOp]  # the pipelined pooled firings, flattened across blocks with ABSOLUTE issue cycles
    outputs: list[FloatOutputWire | BoolOutputWire]
    float_state_slots: list[FloatStateSlot]  # persistent registers, ordered as the instance attributes
    # Control-flow overlay. A straight-line kernel has a single block ending in Ret; ``blocks[0]`` is the entry,
    # ``block_base[i]`` is block i's absolute start PC, and ``last_pc`` is the out_valid boundary (the single Ret).
    blocks: list[LirBlock]
    block_base: list[int]
    entry: int
    last_pc: int  # LASTPC: the fetch PC at which out_valid asserts (the single Ret block's boundary)
    min_initiation_interval: int  # shortest executable path latency; exact for branch-free kernels, else a lower bound
    bool_regfile: BoolRegFileLayout
    bool_state_slots: list[BoolStateSlot]  # persistent boolean registers, ordered as the instance attributes

    def __post_init__(self) -> None:
        assert self.regfile.width == self.float_format.width

    @property
    def ports(self) -> list[Port]:
        fmt = self.float_format
        ports: list[Port] = [
            ControlInputPort("clk", 1),
            ControlInputPort("rst", 1),
            ControlInputPort("in_valid", 1),
            ControlOutputPort("in_ready", 1),
            ControlOutputPort("out_valid", 1),
            ControlInputPort("out_ready", 1),
        ]
        ports += [DataInputPort(f"in_{load.name}", scalar_type_of(load, fmt)) for load in self.inputs]
        ports += [DataOutputPort(wire.name, scalar_type_of(wire, fmt)) for wire in self.outputs]
        ports.append(ControlOutputPort("err_pc", self.cyc_width))
        return ports

    @property
    def float_inputs(self) -> list[FloatInputLoad]:
        return [load for load in self.inputs if isinstance(load, FloatInputLoad)]

    @property
    def bool_inputs(self) -> list[BoolInputLoad]:
        return [load for load in self.inputs if isinstance(load, BoolInputLoad)]

    @property
    def float_outputs(self) -> list[FloatOutputWire]:
        return [wire for wire in self.outputs if isinstance(wire, FloatOutputWire)]

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
    def present_step(self) -> int:
        """
        The hardware executing step on which the outputs are valid in the register array: the fetch PC reaches
        ``last_pc`` (the Ret boundary) and the executing step lags it by FETCH_LAG. For a straight-line kernel this is
        ``makespan + 2`` (the last commit plus the write latch); for a CFG it is the Ret block's resident step.
        """
        return self.last_pc - FETCH_LAG

    @property
    def cyc_width(self) -> int:
        """Bit width of the err_pc diagnostic: enough to hold any executing step ``0..present_step``."""
        return max(1, self.present_step.bit_length())

    @property
    def initiation_interval(self) -> int:
        """
        The out_valid boundary PC (``last_pc``). For a straight-line kernel this equals the observable
        in_valid->out_valid latency; with branches the per-path latency varies and is reported by the numerical model,
        while ``min_initiation_interval`` is the statically-known lower bound (exact when branch-free).
        """
        return self.last_pc

    def term_pc(self, block: LirBlock) -> int:
        """
        The absolute fetch PC at which ``block``'s terminator redirects the PC: its base plus its ``term_offset``. The
        single derivation consumed by the emitter's next-PC sequencer, the numerical model, the HTML report, and the
        boolean-condition liveness, so a terminator's address cannot drift between them.
        """
        return self.block_base[block.index] + block.term_offset

    def write_landing_pcs(self, block: LirBlock, dst: RegRef | BoolRegRef, commit_cycle: int) -> list[int]:
        """
        Every absolute fetch PC at which a result committed at ``block``-local ``commit_cycle`` lands in register
        ``dst`` -- one per execution path that can reach it. A landing at or before the block's terminator offset lands
        once, inside the block. A landing past an overlap-shrunk terminator spills into EACH successor arm's frame, at
        ``block_base[arm] + (landing - term_offset - 1)``. This is exactly the numerical model's redirect re-keying of
        its in-flight writes, so the report places a spilled result where the hardware actually writes it on every
        path -- not in the linear fall-through frame. A drained block never spills, so a drained
        kernel returns one PC per write.

        The recursion re-keys at every terminator the landing crosses, mirroring the model exactly. A spilled result can
        therefore re-spill across a second shrunk terminator -- which needs a near-empty overlapping intermediate block,
        a shape current frontends do not emit, so in practice this resolves in one hop. The recursion is general-case
        insurance, and terminates because spills only cross single-predecessor forward edges (a finite DAG; a back-edge
        target is multi-predecessor and never overlaps).
        """
        by_index = {b.index: b for b in self.blocks}
        return _trace_landing(by_index, self.block_base, block, result_landing_cycle(dst, commit_cycle))

    def state_copy_step(self, slot: FloatStateSlot) -> int:
        """
        The fetch-PC value -- equivalently the hardware-frame cycle -- on which a non-coalesced slot's writeback copy
        fires and reads its source. For a boundary install this is ``initiation_interval`` (LASTPC), where it reduces to
        the accepted-transaction edge and the live-out lands here read-first (the boundary read still sees the live-in).
        An early pc-gated install instead lands its destination one PC later, via ``install_landing`` -- the same +1 the
        model commits. Shared by liveness and the emitter so the two cannot drift.
        """
        return copy_step_cycle(slot.install_cycle)

    def float_state_install_is_boundary(self, slot: FloatStateSlot) -> bool:
        """
        Whether a non-coalesced float slot installs read-first at the accepted-output boundary (its ``state_copy_step``
        reaches LASTPC) rather than early via a pc-gated copy. The single early-vs-boundary test shared by the numerical
        model (which routes the install to the boundary edge vs a pc-gated step), the HTML report (which lands it
        read-first at LASTPC vs one PC later), and ``reg_liveness`` (which makes a boundary install read-first while an
        early one resides through the boundary to carry), so the read-first seam cannot drift between them.
        """
        return self.state_copy_step(slot) >= self.last_pc

    @property
    def read_set_per_port(self) -> dict[ReadPort, list[int]]:
        """
        For each operator read port -- identified by its ``(instance, operand-position)`` pair -- the sorted distinct
        register indices it ever reads across the schedule.

        Constant operands are excluded: they are immediates on the per-operand const-select path, not register reads.
        Ports that never read a register are absent. This drives the sparse per-port read mux: a port that reads a
        single register needs no mux at all, and one that reads several needs a mux spanning only those registers.
        """
        sets: dict[ReadPort, set[int]] = {}
        for op in self.ops:
            for pos, operand in enumerate(op.operands):
                if isinstance(operand.source, RegRef):
                    sets.setdefault((op.inst, pos), set()).add(operand.source.index)
        return {port: sorted(regs) for port, regs in sets.items()}

    @property
    def write_set_per_register(self) -> dict[int, list[tuple[OperatorInstance, int]]]:
        """
        For each WIDE register index, the ``(instance, output port)`` lanes that ever write it, in a canonical order.

        This drives the sparse per-register write select: a register written by a single lane needs no write-port
        mux. The input-load writers of registers ``0..nload-1`` are tracked separately via ``lir.float_inputs``
        (they are a distinct, address-free write source folded into the same select).
        """
        return self._write_sets(RegRef)

    @property
    def bool_write_set_per_register(self) -> dict[int, list[tuple[OperatorInstance, int]]]:
        """The boolean-bank counterpart of :attr:`write_set_per_register`, in the same canonical lane order."""
        return self._write_sets(BoolRegRef)

    def _write_sets(self, bank: type[RegRef] | type[BoolRegRef]) -> dict[int, list[tuple[OperatorInstance, int]]]:
        sets: dict[int, list[tuple[OperatorInstance, int]]] = {}
        for op in self.ops:
            for write in op.writes:
                if not isinstance(write.dst, bank):
                    continue
                writers = sets.setdefault(write.dst.index, [])
                if (op.inst, write.port) not in writers:
                    writers.append((op.inst, write.port))
        for writers in sets.values():
            writers.sort(key=lambda lane: (lane[0].operator.instance_stem, lane[0].index, lane[1]))
        return sets

    @property
    def write_select_fanin(self) -> int:
        """
        The ground-truth per-register write-select fan-in summed over both banks: for every register, the number of
        distinct drivers in its write chain beyond the first (``max(0, drivers - 1)``). The backend drives each register
        with one priority chain over exactly these drivers -- the input load, every pooled writeback lane, every inline
        (cast) write, every phi-arm copy/write, and a non-coalesced slot's install. A register's live-in carry is the
        chain's implicit hold (the unmatched-condition fall-through), not a mux input, so it is not counted. This is the
        true steering cost the sparse register file synthesizes, counting the phi-arm copies that
        ``write_set_per_register`` (pooled lanes only) omits, so it stays meaningful as coalescing trades copies for
        shared writeback lanes.
        """
        wide: dict[int, int] = {}
        boolc: dict[int, int] = {}
        for fload in self.float_inputs:
            wide[fload.dst.index] = wide.get(fload.dst.index, 0) + 1
        for bload in self.bool_inputs:
            boolc[bload.dst.index] = boolc.get(bload.dst.index, 0) + 1
        for reg, lanes in self.write_set_per_register.items():
            wide[reg] = wide.get(reg, 0) + len(lanes)
        for reg, lanes in self.bool_write_set_per_register.items():
            boolc[reg] = boolc.get(reg, 0) + len(lanes)
        for block in self.blocks:
            for inline_op in block.inline_ops:
                target = wide if isinstance(inline_op.write.dst, RegRef) else boolc
                target[inline_op.write.dst.index] = target.get(inline_op.write.dst.index, 0) + 1
            for copy in block.copies:
                wide[copy.dst.index] = wide.get(copy.dst.index, 0) + 1
            for bwrite in block.bool_writes:
                boolc[bwrite.dst.index] = boolc.get(bwrite.dst.index, 0) + 1
        for slot in self.float_state_slots:
            if slot.needs_copy:
                wide[slot.reg.index] = wide.get(slot.reg.index, 0) + 1
        for bslot in self.bool_state_slots:
            if bslot.needs_copy:
                boolc[bslot.reg.index] = boolc.get(bslot.reg.index, 0) + 1
        return sum(max(0, n - 1) for n in wide.values()) + sum(max(0, n - 1) for n in boolc.values())

    @property
    def group_by_cycle(self) -> tuple[dict[int, list[PooledScheduledOp]], dict[int, list[PooledScheduledOp]]]:
        """The schedule grouped into per-cycle issues and commits, each canonically ordered."""
        issues: dict[int, list[PooledScheduledOp]] = {}
        commits: dict[int, list[PooledScheduledOp]] = {}
        for op in self.ops:
            issues.setdefault(op.issue_cycle, []).append(op)
            commits.setdefault(op.commit_cycle, []).append(op)
        for group in (issues, commits):
            for ops in group.values():
                ops.sort(
                    key=lambda op: (
                        op.inst.operator.instance_stem,
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
        read_first: dict[_BankReg, set[int]] | None = None,
    ) -> dict[_BankReg, set[int]]:
        """
        Collapse a bank's absolute def/use PCs into the rows each register holds a live value, computed PER BASIC BLOCK
        (where the PC stream is straight-line, so ``residence_rows`` is exact) with backward register liveness carrying
        a value across block boundaries. This is path-aware where a single global timeline is not: a value live on two
        mutually-exclusive arms that rejoin at a merge stays resident on BOTH arms, instead of the later-addressed arm's
        landing truncating the earlier one. For a straight-line kernel (one block) it reduces to a single
        ``residence_rows`` over the whole frame.

        ``defs``/``uses`` are absolute fetch PCs (the report grid's row axis); each falls inside exactly one block's
        ``[base, term_pc]`` range (the ranges tile the frame contiguously in layout order). Within a block a live-in
        register is given a pseudo-def at the block base and a live-out one a pseudo-use at the terminator PC, so the
        per-block ``residence_rows`` extends the carried value across the whole block. ``read_first`` lists, per
        register, the READ-FIRST def PCs (a boundary state install): a read on such a PC reads the PRIOR value, so it
        both keeps that read out of the install's own residence and -- when the read is the register's earliest one in
        its block -- marks the register live-in there (the carried value, not the install, supplies that read).
        """
        read_first = read_first or {}
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

        # Per-block register liveness sets: ``written`` is defined in the block; ``upward`` is read STRICTLY before its
        # first def in the block, so it is needed at block entry (live-in). The strict ``<`` matters: a read on the
        # landing cycle of its own def reads the just-committed value (the model lands the write, then reads, at that
        # PC), not a live-in -- a same-PC def+use (e.g. an output tap or branch condition read on the cycle it lands)
        # must NOT be treated as live-in, or its residence would be painted spuriously back to the block entry.
        written: dict[int, set[_BankReg]] = {}
        upward: dict[int, set[_BankReg]] = {}
        for index in block_defs:
            ds, us = block_defs[index], block_uses[index]
            written[index] = set(ds)
            # A register is live-in (upward-exposed) if its earliest read is not supplied by an in-block def: read with
            # no def, or read strictly before the first def, or read AT a read-first def (which reads the prior value).
            upward[index] = {
                reg
                for reg, reads in us.items()
                if reg not in ds
                or min(reads) < min(ds[reg])
                or (min(reads) == min(ds[reg]) and min(ds[reg]) in read_first.get(reg, set()))
            }
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
                resident = residence_rows(d, u, boundary, frozenset(read_first.get(reg, set())))
                if resident:
                    rows.setdefault(reg, set()).update(resident)
        return rows

    def _collect_op_events(
        self, reg_type: type[_BankReg], defs: dict[_BankReg, list[int]], uses: dict[_BankReg, list[int]]
    ) -> None:
        """
        Add one bank's per-block datapath op events to ``defs``/``uses``: each result LANDING (stamped via
        ``write_landing_pcs`` at every successor-arm PC it spills into) and each operand READ. The single definition of
        op read/write timing, shared by both banks so :attr:`reg_liveness` and :attr:`bool_liveness` cannot drift.
        """
        block: (
            LirBlock  # explicit binding: the loop target's type is undecidable under the constrained-TypeVar reanalysis
        )
        for block in self.blocks:
            base_pc = self.block_base[block.index]
            block_ops: list[ScheduledOp] = [*block.ops, *block.inline_ops]
            for op in block_ops:
                read = operand_read_cycle(op.operator, base_pc + op.issue_cycle)
                for write in op.writes:
                    if isinstance(write.dst, reg_type):
                        defs.setdefault(write.dst, []).extend(self.write_landing_pcs(block, write.dst, op.commit_cycle))
                for operand in op.operands:
                    if isinstance(operand.source, reg_type):
                        uses.setdefault(operand.source, []).append(read)

    @property
    def reg_liveness(self) -> dict[RegRef, set[int]]:
        """
        Map each wide register to the actual clock cycles on which it holds a live value.

        This is cycle-accurate to the emitted hardware, in the executing-step (hardware) frame. Timing comes from the
        shared helpers: an input lands on cycle 1; an operator result lands on ``result_landing_cycle`` (which for the
        last result is the initiation interval); an operand is read on ``operand_read_cycle``; an output tap on the
        present cycle; and a non-coalesced slot's writeback fires and samples its source on ``state_copy_step`` -- the
        present cycle for a boundary copy, earlier for an early install (the landing follows below). A slot register
        additionally stays live through the present cycle, since its live-out must reside there for the next initiation.
        Each row spans a value from when it lands in the array through its last read.

        Diagnostic only -- consumed by the reports (e.g., HTML schedule) and the tests, never by the emitter or the
        numerical model. Each op-result LANDING is stamped via ``write_landing_pcs`` at exactly the PC(s) the model
        writes it -- on every successor arm under overlap, not just the fall-through. A pc-gated install (a phi copy or
        an early non-coalesced slot writeback) fires and samples its source on the copy step but lands its destination
        one PC later via ``install_landing`` -- the same +1 the model commits. A boundary install reads-then-writes
        at the boundary and lands there. Residence is then resolved per basic block by ``_cfg_residence`` (CFG-aware
        register liveness), so a value live on two mutually-exclusive arms that rejoin at a merge stays resident on BOTH
        arms. The result is cycle-exact to the numerical model on every register and every path.
        """
        present = self.initiation_interval  # hardware-frame present / boundary step
        defs: dict[RegRef, list[int]] = {}
        uses: dict[RegRef, list[int]] = {}
        read_first: dict[RegRef, set[int]] = {}
        for load in self.float_inputs:
            defs.setdefault(load.dst, []).append(1)
        for slot in self.float_state_slots:
            defs.setdefault(slot.reg, []).append(1)  # the live-in is resident in the slot register from the start
            if not slot.needs_copy:
                # A coalesced live-out is an ordinary result already in the slot register; it must reside through the
                # boundary to carry into the next initiation, even when nothing reads it again this frame.
                uses.setdefault(slot.reg, []).append(present)
            elif not self.float_state_install_is_boundary(slot):
                # An early pc-gated install lands its destination one PC after its fire step and must reside through the
                # boundary to carry; installing the new value early is not the slot's death.
                step = self.state_copy_step(slot)
                defs.setdefault(slot.reg, []).append(install_landing(step))
                uses.setdefault(slot.reg, []).append(present)
            else:
                # A boundary install reads-then-writes at the boundary: the hardware samples the live-in there (read
                # first) before clocking in the new live-out, so the boundary read belongs to the live-in (read_first),
                # and the live-out is resident at the boundary by its def alone -- no carry use, or a dead live-in would
                # be over-tinted across the whole frame.
                step = self.state_copy_step(slot)
                defs.setdefault(slot.reg, []).append(step)
                read_first.setdefault(slot.reg, set()).add(step)
        for wire in self.float_outputs:
            if isinstance(wire.tap.source, RegRef):
                uses.setdefault(wire.tap.source, []).append(present)
        for slot in self.float_state_slots:  # the live-out tap is read on the install step to persist the slot
            if isinstance(slot.tap.source, RegRef):
                uses.setdefault(slot.tap.source, []).append(self.state_copy_step(slot))
        for block in self.blocks:
            base_pc = self.block_base[block.index]
            for copy in block.copies:  # phi copy fires here and samples its source; destination lands one PC later
                step = base_pc + copy_step_cycle(copy.issue_cycle)
                defs.setdefault(copy.dst, []).append(install_landing(step))
                if isinstance(copy.source.source, RegRef):
                    uses.setdefault(copy.source.source, []).append(step)
        self._collect_op_events(RegRef, defs, uses)
        return self._cfg_residence(defs, uses, read_first)

    @property
    def bool_liveness(self) -> dict[BoolRegRef, set[int]]:
        """
        Map each boolean register to the cycles on which it holds a live value, the boolean-bank counterpart of
        :attr:`reg_liveness` in the same executing-step frame. A boolean register is defined when a comparison,
        boolean-logic op, or float->bool cast commits its result, when a boolean phi/state install lands, and -- for a
        persistent slot -- at the live-in resident from cycle 1; it is read by a boolean-logic op or a bool->float cast
        taking it as an operand, by a branch testing it as a condition, by a phi/state install copying it, and at the
        boundary where a slot's live-out must persist for the next initiation. A boolean result that spills past an
        overlap-shrunk terminator is stamped on every successor arm via ``write_landing_pcs``, exactly as the numerical
        model re-keys it; a phi/boolean write lands its destination one PC after its fire step via ``install_landing``
        (the model's +1), while a boolean slot always installs read-first at the boundary. Residence is resolved by the
        same per-block ``_cfg_residence`` as :attr:`reg_liveness`, so a spilled or merged boolean is cycle-exact too.
        """
        present = self.initiation_interval
        defs: dict[BoolRegRef, list[int]] = {}
        uses: dict[BoolRegRef, list[int]] = {}
        read_first: dict[BoolRegRef, set[int]] = {}
        for slot in self.bool_state_slots:
            defs.setdefault(slot.reg, []).append(1)  # the live-in is resident from the start
            if slot.needs_copy:
                # A boolean slot always installs read-first at the boundary: the live-out's def alone marks it resident
                # there (no carry use, which would over-tint a dead live-in), and any boundary read of the slot register
                # is the live-in (read_first). The install samples its source on the boundary edge.
                defs.setdefault(slot.reg, []).append(present)
                read_first.setdefault(slot.reg, set()).add(present)
                if isinstance(slot.live_out.source, BoolRegRef):
                    uses.setdefault(slot.live_out.source, []).append(present)
            else:
                uses.setdefault(slot.reg, []).append(present)  # a coalesced live-out must reside through the boundary
        for load in self.bool_inputs:
            defs.setdefault(load.dst, []).append(1)
        for wire in self.bool_outputs:
            if isinstance(wire.tap.source, BoolRegRef):
                uses.setdefault(wire.tap.source, []).append(present)
        for block in self.blocks:
            base_pc = self.block_base[block.index]
            for bwrite in block.bool_writes:  # bool write fires and samples here; destination lands one PC later
                step = base_pc + copy_step_cycle(bwrite.issue_cycle)
                defs.setdefault(bwrite.dst, []).append(install_landing(step))
                if isinstance(bwrite.source.source, BoolRegRef):
                    uses.setdefault(bwrite.source.source, []).append(step)
            if isinstance(block.terminator, Branch):  # the next-PC case reads the condition at the block boundary PC
                uses.setdefault(block.terminator.cond, []).append(self.term_pc(block))
        self._collect_op_events(BoolRegRef, defs, uses)
        return self._cfg_residence(defs, uses, read_first)

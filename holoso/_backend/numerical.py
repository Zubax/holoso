"""
The numerical backend: a pure-Python functional model of a generated module, bit-exact to its RTL.

``NumericalModel`` is a software interpreter for the scheduled LIR microprogram. It replays the schedule applying the
same per-operator ZKF rounding the hardware does, so one ``__call__`` reproduces one module transaction bit-for-bit:
flat input values in port order to a flat tuple of output values in port order. A stateful module additionally carries
its persistent state registers between calls, exactly as the hardware retains them between initiations; ``reset()``
reloads the snapshot the hardware loads at module reset.
"""

from dataclasses import dataclass, field

from .._value import FloatValue
from .._lir import BoolInputLoad, BoolOutputWire, CombScheduledOp, FloatConstRef, FloatInputLoad, FloatOperand
from .._lir import FloatOutputWire, RegRef
from .._lir import FloatScheduledOp, Lir
from .._lir import InputProducer, OperationProducer, StateProducer, latest_producer_before
from .._lir import BoolConstRef, BoolOperand, Branch, Jump, Ret
from .._lir import landing_cycle, wide_operand_read_cycle
from .._operators import *
from .._type import FloatFormat

type ModelInput = FloatValue | float | bool
type ModelOutput = FloatValue | bool


def _apply_sign(value: FloatValue, sign: FloatSignControl) -> FloatValue:
    """Apply a folded sign control exactly as ``holoso_fsgnop`` does in the RTL."""
    return sign.apply_value(value)


def _coerce_input(value: ModelInput, fmt: FloatFormat, index: int) -> FloatValue:
    if isinstance(value, FloatValue):
        if value.fmt != fmt:
            raise ValueError(f"input {index} has {value.fmt}, expected {fmt}")
        return value
    if type(value) is float:
        return FloatValue.from_float(fmt, value)
    raise TypeError(f"input {index} must be FloatValue or float, got {type(value).__name__}")


def _coerce_bool_input(value: ModelInput, index: int) -> bool:
    if type(value) is bool:
        return value
    raise TypeError(f"input {index} must be bool, got {type(value).__name__}")


def _resident[T](timeline: list[tuple[int, T]], read_cycle: int) -> T:
    """
    The value a register holds when read at hardware-frame ``read_cycle``: the latest write whose landing cycle does
    not exceed it. The timeline is seeded with the value carried in from the previous block (landing 0) and extended in
    landing order as the block's operators commit, so a register reused for several values yields the live one.
    """
    value = timeline[0][1]
    for landing, written in timeline:
        if landing <= read_cycle:
            value = written
        else:
            break
    return value


@dataclass(slots=True)
class NumericalModel:
    """
    A bit-exact functional model of a generated module: one ``__call__`` is one module transaction.

    Call it with the input values in module-port order; it returns the output values in module-port order. The result
    matches the generated Verilog bit-for-bit because every operator evaluates the same ZKF bits as the hardware. A
    stateful module carries its persistent state registers between calls (``reset()`` reloads the reset snapshot),
    mirroring how the hardware retains state across initiations and reloads it at reset. The model holds the scheduled
    ``Lir`` and is picklable, so a generated testbench can embed it. For a control-flow kernel one ``__call__`` follows
    the taken path block by block, so the result is correct for any branch outcome.

    TODO: the model reproduces output bits exactly but not yet the per-transaction cycle latency (which is data-
          dependent once branches/loops shortcut the PC). Predicting that latency would enable cycle-accurate
          lockstep cosimulation; today the bench checks output bits at the out_valid handshake instead.
    """

    lir: Lir
    _state: list[FloatValue] = field(init=False)
    _bool_state: list[bool] = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reload every persistent state register with its reset snapshot, as the hardware does at module reset."""
        fmt = self.lir.float_format
        self._state = [FloatValue.from_float(fmt, slot.reset_value) for slot in self.lir.float_state_slots]
        self._bool_state = [slot.reset_value for slot in self.lir.bool_state_slots]

    def __call__(self, *inputs: ModelInput) -> tuple[ModelOutput, ...]:
        lir = self.lir
        if len(inputs) != len(lir.inputs):
            raise ValueError(f"expected {len(lir.inputs)} inputs, got {len(inputs)}")
        fmt = lir.float_format
        float_values: list[FloatValue] = []
        bool_values: list[bool] = []
        for index, (load, raw_input) in enumerate(zip(lir.inputs, inputs, strict=True)):
            match load:
                case FloatInputLoad():
                    float_values.append(_coerce_input(raw_input, fmt, index))
                case BoolInputLoad():
                    bool_values.append(_coerce_bool_input(raw_input, index))
        if lir.is_control_flow:
            return self._run_cfg(float_values, bool_values)
        consts = [FloatValue.from_float(fmt, const_value) for const_value in lir.float_consts]

        # Per-register write timeline: (commit_cycle, producer) in increasing commit order. A slot register starts the
        # initiation holding its live-in (the value carried over from the previous call) and may be overwritten by a
        # coalesced operator later. Inputs are sampled at cycle 0; each op commits at its commit_cycle. Operands resolve
        # against this so a register reused for several values yields the value live at the operand's read cycle.
        writes = lir.write_timeline

        op_values: dict[int, FloatValue] = {}

        def source_value(source: RegRef | FloatConstRef, read_cycle: int) -> FloatValue:
            if isinstance(source, FloatConstRef):
                return consts[source.index]
            producer = latest_producer_before(writes, source, read_cycle)
            match producer:
                case InputProducer(index=index):
                    return float_values[index]
                case OperationProducer(index=index):
                    return op_values[index]
                case StateProducer(index=index):
                    return self._state[index]

        def eval_tap(operand: FloatOperand, cycle: int) -> FloatValue:
            # One evaluation for every source tap: an operator operand, an output wire, or a state slot's live-out.
            return _apply_sign(source_value(operand.source, cycle), operand.sign)

        # Evaluate in commit order: a producer commits before any consumer issues, so its value is ready in op_values.
        for j in sorted(range(len(lir.ops)), key=lambda k: (lir.ops[k].commit_cycle, lir.ops[k].issue_cycle)):
            op = lir.ops[j]
            operands = [eval_tap(o, lir.operand_read_cycle(op)) for o in op.operands]
            op_values[j] = _apply_sign(op.inst.operator.evaluate(*operands), op.result_sign)

        # Hardware-frame read cycles, matching reg_liveness and the RTL: an output is resident in the array on the
        # boundary step (the last result lands at the initiation interval), and each slot's live-out source is sampled
        # on its writeback step -- before any later operation reuses that register (a value becomes readable on its
        # landing cycle, so a read on the writeback step still resolves to the source that landed at or before it).
        outputs = tuple(eval_tap(wire.tap, lir.initiation_interval) for wire in lir.float_outputs)
        self._state = [eval_tap(slot.tap, lir.state_copy_step(slot)) for slot in lir.float_state_slots]
        return outputs

    def _run_cfg(self, float_values: list[FloatValue], bool_values: list[bool]) -> tuple[ModelOutput, ...]:
        """
        Execute a control-flow microprogram by following the taken path block by block. Each block evaluates its
        scheduled float ops (in commit order), installs its phi-arm copies and boolean writes, then its terminator
        selects the successor; at Ret the outputs are read and the persistent state is taken from the slot registers.
        """
        lir = self.lir
        fmt = lir.float_format
        consts = [FloatValue.from_float(fmt, const_value) for const_value in lir.float_consts]
        regs: dict[int, FloatValue] = {
            load.dst.index: input_value for load, input_value in zip(lir.float_inputs, float_values)
        }
        for fslot, state_value in zip(lir.float_state_slots, self._state):
            regs[fslot.reg.index] = state_value
        bregs: dict[int, bool] = {
            load.dst.index: input_value for load, input_value in zip(lir.bool_inputs, bool_values)
        }
        for bslot, bool_state_value in zip(lir.bool_state_slots, self._bool_state):
            bregs[bslot.reg.index] = bool_state_value

        def fval(operand: FloatOperand) -> FloatValue:
            source = operand.source
            base = consts[source.index] if isinstance(source, FloatConstRef) else regs[source.index]
            return _apply_sign(base, operand.sign)

        def bval(operand: BoolOperand) -> bool:
            source = operand.source
            return source.value if isinstance(source, BoolConstRef) else bregs[source.index]

        blocks = {block.index: block for block in lir.blocks}
        current = lir.entry
        for _ in range(1_000_000):  # bounded against a non-terminating kernel; ample for any real loop trip count
            block = blocks[current]
            # Per-register write timeline within this block, in the executing-step (hardware) frame: seeded with the
            # value carried in from the previous block (landing 0) and extended as each operator commits. An operand is
            # resolved at its read cycle, so a register reused for several values within the block yields the one live
            # at the read -- the same read-first resolution the straight-line path applies globally via write_timeline.
            ftl: dict[int, list[tuple[int, FloatValue]]] = {reg: [(0, value)] for reg, value in regs.items()}
            btl: dict[int, list[tuple[int, bool]]] = {reg: [(0, value)] for reg, value in bregs.items()}

            def fread(operand: FloatOperand, read_cycle: int) -> FloatValue:
                source = operand.source
                base = (
                    consts[source.index]
                    if isinstance(source, FloatConstRef)
                    else _resident(ftl[source.index], read_cycle)
                )
                return _apply_sign(base, operand.sign)

            def bread(operand: BoolOperand, read_cycle: int) -> bool:
                source = operand.source
                return source.value if isinstance(source, BoolConstRef) else _resident(btl[source.index], read_cycle)

            # All operations -- float arithmetic and combinational ops across both banks -- evaluate in a single
            # commit-sorted pass, so a producer is computed before any consumer; the read-cycle resolution above makes
            # the reads register-reuse-correct even when a consumer commits after a later writer of one of its operands.
            scheduled: list[FloatScheduledOp | CombScheduledOp] = [*block.ops, *block.comb_ops]
            scheduled.sort(key=lambda o: (o.commit_cycle, 0 if isinstance(o, FloatScheduledOp) else 1, o.dst.index))
            for op in scheduled:
                operator = op.inst.operator if isinstance(op, FloatScheduledOp) else op.operator
                read = wide_operand_read_cycle(operator, op.issue_cycle)
                land = landing_cycle(op.commit_cycle)
                if isinstance(op, FloatScheduledOp):
                    result = op.inst.operator.evaluate(*(fread(operand, read) for operand in op.operands))
                    ftl.setdefault(op.dst.index, []).append((land, _apply_sign(result, op.result_sign)))
                    continue
                match op.operator:
                    case FComparisonOperator() as cmp:
                        left, right = op.operands
                        assert isinstance(left, FloatOperand) and isinstance(right, FloatOperand)
                        btl.setdefault(op.dst.index, []).append(
                            (land, cmp.evaluate(fread(left, read), fread(right, read)))
                        )
                    case BoolLogicOperator() as logic:
                        inputs: list[bool] = []
                        for operand in op.operands:
                            assert isinstance(operand, BoolOperand)
                            inputs.append(bread(operand, read))
                        btl.setdefault(op.dst.index, []).append((land, logic.evaluate(*inputs)))
                    case FloatToBoolOperator() as to_bool:
                        (operand,) = op.operands
                        assert isinstance(operand, FloatOperand)
                        btl.setdefault(op.dst.index, []).append((land, to_bool.evaluate(fread(operand, read))))
                    case BoolToFloatOperator() as to_float:
                        (operand,) = op.operands
                        assert isinstance(operand, BoolOperand)
                        assert isinstance(op.dst, RegRef)
                        ftl.setdefault(op.dst.index, []).append((land, to_float.evaluate(bread(operand, read))))
                    case _:
                        raise RuntimeError(f"no model evaluation for combinational operator {op.operator!r}")
            # The values resident at the block boundary: the last write to each register (or the carried-in value). The
            # phi-arm copies, the terminator, and the Ret reads all fire at the drained tail, so they read these.
            regs = {reg: timeline[-1][1] for reg, timeline in ftl.items()}
            bregs = {reg: timeline[-1][1] for reg, timeline in btl.items()}
            # Phi-arm installs are a parallel copy bundle (a swap reads both old values), so read every source before
            # writing any destination -- both for the float copies and the boolean writes.
            copied = [fval(copy.source) for copy in block.copies]
            for copy, copy_value in zip(block.copies, copied):
                regs[copy.dst.index] = copy_value
            written = [bval(write.source) for write in block.bool_writes]
            for write, write_value in zip(block.bool_writes, written):
                bregs[write.dst.index] = write_value
            match block.terminator:
                case Jump(target=target):
                    current = target
                case Branch(cond=cond, if_true=if_true, if_false=if_false):
                    current = if_true if bregs[cond.index] else if_false
                case Ret():
                    # Read-first at the boundary: the slot register is read-only in the body, so an output (or any read
                    # of a slot's live-in, e.g. ``return old``) sees the OLD slot value here; the new persistent state
                    # is each slot's live-out -- a distinct value (a phi/operator/input/const), tapped with its sign.
                    def out_value(wire: FloatOutputWire | BoolOutputWire) -> ModelOutput:
                        match wire:
                            case FloatOutputWire():
                                return fval(wire.tap)
                            case BoolOutputWire():
                                return bval(wire.tap)

                    outputs = tuple(out_value(wire) for wire in lir.outputs)
                    self._state = [fval(slot.tap) for slot in lir.float_state_slots]
                    self._bool_state = [bval(slot.live_out) for slot in lir.bool_state_slots]
                    return outputs
        raise RuntimeError("control flow did not reach a return; the CFG may not terminate")

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(load.name for load in self.lir.inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(wire.name for wire in self.lir.outputs)

    @property
    def float_format(self) -> FloatFormat:
        return self.lir.float_format

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}({self.lir.module_name!r},"
            f" inputs={self.input_names}, outputs={self.output_names},"
            f" float_format={self.float_format})"
        )


def generate(lir: Lir) -> NumericalModel:
    """Build the bit-exact functional model from a finished :class:`Lir`."""
    return NumericalModel(lir)

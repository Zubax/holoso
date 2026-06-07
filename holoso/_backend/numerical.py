"""
The numerical backend: a pure-Python functional model of a generated module, bit-exact to its RTL.

``NumericalModel`` is a software interpreter for the scheduled LIR microprogram. It replays the schedule applying the
same per-operator ZKF rounding the hardware does, so one ``__call__`` reproduces one module transaction bit-for-bit:
flat input values in port order to a flat tuple of output values in port order. A stateful module additionally carries
its persistent state registers between calls, exactly as the hardware retains them between initiations; ``reset()``
reloads the snapshot the hardware loads at module reset.
"""

from dataclasses import dataclass, field

from .._value import compare_float_values, FloatValue
from .._lir import FloatConstRef, FloatOperand, FloatRegRef, Lir
from .._lir import InputProducer, OperationProducer, StateProducer, latest_producer_before
from .._lir import BoolConstRef, BoolOperand, Branch, Jump, Ret
from .._operators import FloatSignControl
from .._type import FloatFormat

type ModelInput = FloatValue | float


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
        fmt = self.lir.float_regfile.fmt
        self._state = [FloatValue.from_float(fmt, slot.reset_value) for slot in self.lir.float_state_slots]
        self._bool_state = [slot.reset_value for slot in self.lir.bool_state_slots]

    def __call__(self, *inputs: ModelInput) -> tuple[FloatValue, ...]:
        lir = self.lir
        if len(inputs) != len(lir.float_inputs):
            raise ValueError(f"expected {len(lir.float_inputs)} inputs, got {len(inputs)}")
        fmt = lir.float_regfile.fmt
        in_values = [_coerce_input(value, fmt, index) for index, value in enumerate(inputs)]
        if lir.is_control_flow:
            return self._run_cfg(in_values)
        consts = [FloatValue.from_float(fmt, value) for value in lir.float_consts]

        # Per-register write timeline: (commit_cycle, producer) in increasing commit order. A slot register starts the
        # initiation holding its live-in (the value carried over from the previous call) and may be overwritten by a
        # coalesced operator later. Inputs are sampled at cycle 0; each op commits at its commit_cycle. Operands resolve
        # against this so a register reused for several values yields the value live at the operand's read cycle.
        writes = lir.float_write_timeline

        op_values: dict[int, FloatValue] = {}

        def value(source: FloatRegRef | FloatConstRef, read_cycle: int) -> FloatValue:
            if isinstance(source, FloatConstRef):
                return consts[source.index]
            producer = latest_producer_before(writes, source, read_cycle)
            match producer:
                case InputProducer(index=index):
                    return in_values[index]
                case OperationProducer(index=index):
                    return op_values[index]
                case StateProducer(index=index):
                    return self._state[index]

        def eval_tap(operand: FloatOperand, cycle: int) -> FloatValue:
            # One evaluation for every source tap: an operator operand, an output wire, or a state slot's live-out.
            return _apply_sign(value(operand.source, cycle), operand.sign)

        # Evaluate in commit order: a producer commits before any consumer issues, so its value is ready in op_values.
        for j in sorted(
            range(len(lir.float_ops)), key=lambda k: (lir.float_ops[k].commit_cycle, lir.float_ops[k].issue_cycle)
        ):
            op = lir.float_ops[j]
            operands = [eval_tap(o, lir.operand_read_cycle(op)) for o in op.operands]
            op_values[j] = _apply_sign(op.inst.operator.evaluate(*operands), op.result_sign)

        # Hardware-frame read cycles, matching float_liveness and the RTL: an output is resident in the array on the
        # boundary step (the last result lands at the initiation interval), and each slot's live-out source is sampled
        # on its writeback step -- before any later operation reuses that register (a value becomes readable on its
        # landing cycle, so a read on the writeback step still resolves to the source that landed at or before it).
        outputs = tuple(eval_tap(wire.tap, lir.initiation_interval) for wire in lir.float_outputs)
        self._state = [eval_tap(slot.tap, lir.state_copy_step(slot)) for slot in lir.float_state_slots]
        return outputs

    def _run_cfg(self, in_values: list[FloatValue]) -> tuple[FloatValue, ...]:
        """
        Execute a control-flow microprogram by following the taken path block by block. Each block evaluates its
        scheduled float ops (in commit order), installs its phi-arm copies and boolean writes, then its terminator
        selects the successor; at Ret the outputs are read and the persistent state is taken from the slot registers.
        """
        lir = self.lir
        fmt = lir.float_regfile.fmt
        consts = [FloatValue.from_float(fmt, value) for value in lir.float_consts]
        fregs: dict[int, FloatValue] = {load.dst.index: value for load, value in zip(lir.float_inputs, in_values)}
        for slot, value in zip(lir.float_state_slots, self._state):
            fregs[slot.reg.index] = value
        bregs: dict[int, bool] = {slot.reg.index: value for slot, value in zip(lir.bool_state_slots, self._bool_state)}

        def fval(operand: FloatOperand) -> FloatValue:
            source = operand.source
            base = consts[source.index] if isinstance(source, FloatConstRef) else fregs[source.index]
            return _apply_sign(base, operand.sign)

        def bval(operand: BoolOperand) -> bool:
            source = operand.source
            return source.value if isinstance(source, BoolConstRef) else bregs[source.index]

        blocks = {block.index: block for block in lir.blocks}
        current = lir.entry
        for _ in range(1_000_000):  # bounded against a non-terminating kernel; ample for any real loop trip count
            block = blocks[current]
            # Operator results land in fresh registers (no in-block reuse), so commit-order evaluation reading the
            # current register file is hazard-free.
            for op in sorted(block.float_ops, key=lambda o: (o.commit_cycle, o.dst.index)):
                result = op.inst.operator.evaluate(*(fval(operand) for operand in op.operands))
                fregs[op.dst.index] = _apply_sign(result, op.result_sign)
            # Comparators read the (now committed) float registers and write the boolean bank; the relation reduces
            # the exact three-way ZKF order to a boolean (exact, not via a lossy float decode).
            for bop in sorted(block.bool_ops, key=lambda o: (o.commit_cycle, o.dst.index)):
                left, right = (fval(operand) for operand in bop.operands)
                bregs[bop.dst.index] = bop.relation.holds(compare_float_values(left, right))
            # Phi-arm installs are a parallel copy bundle (a swap reads both old values), so read every source before
            # writing any destination -- both for the float copies and the boolean writes.
            copied = [fval(copy.source) for copy in block.float_copies]
            for copy, copy_value in zip(block.float_copies, copied):
                fregs[copy.dst.index] = copy_value
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
                    outputs = tuple(fval(wire.tap) for wire in lir.float_outputs)
                    self._state = [fval(slot.tap) for slot in lir.float_state_slots]
                    self._bool_state = [bval(slot.live_out) for slot in lir.bool_state_slots]
                    return outputs
        raise RuntimeError("control flow did not reach a return; the CFG may not terminate")

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(load.name for load in self.lir.float_inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(wire.name for wire in self.lir.float_outputs)

    @property
    def float_format(self) -> FloatFormat:
        return self.lir.float_regfile.fmt

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}({self.lir.module_name!r},"
            f" inputs={self.input_names}, outputs={self.output_names},"
            f" float_format={self.float_format})"
        )


def generate(lir: Lir) -> NumericalModel:
    """Build the bit-exact functional model from a finished :class:`Lir`."""
    return NumericalModel(lir)

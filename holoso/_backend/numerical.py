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
from .._lir import FloatConstRef, FloatOperand, FloatRegRef, Lir
from .._lir import InputProducer, OperationProducer, StateProducer, latest_producer_before
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
    ``Lir`` and is picklable, so a generated testbench can embed it.

    TODO: When branching is implemented, the numerical model will need to be extended such that it also predicts the
          cycle latency of each transaction. This is necessary for cycle-accurate verification.
    """

    lir: Lir
    _state: list[FloatValue] = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reload every persistent state register with its reset snapshot, as the hardware does at module reset."""
        fmt = self.lir.float_regfile.fmt
        self._state = [FloatValue.from_float(fmt, slot.reset_value) for slot in self.lir.float_state_slots]

    def __call__(self, *inputs: ModelInput) -> tuple[FloatValue, ...]:
        lir = self.lir
        if len(inputs) != len(lir.float_inputs):
            raise ValueError(f"expected {len(lir.float_inputs)} inputs, got {len(inputs)}")
        fmt = lir.float_regfile.fmt
        consts = [FloatValue.from_float(fmt, value) for value in lir.float_consts]
        in_values = [_coerce_input(value, fmt, index) for index, value in enumerate(inputs)]

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
            operands = [eval_tap(o, op.issue_cycle) for o in op.operands]
            op_values[j] = _apply_sign(op.inst.operator.evaluate(*operands), op.result_sign)

        # Scheduler-frame settle cycle: the final value is live one cycle after the last commit. This is the model's
        # logical frame, distinct from the hardware present_step (makespan + 2); the model is value- not cycle-exact.
        settle = lir.makespan + 1
        outputs = tuple(eval_tap(wire.tap, settle) for wire in lir.float_outputs)
        # Advance the persistent state for the next call: each slot's live-out tap, read from old state then committed.
        self._state = [eval_tap(slot.tap, settle) for slot in lir.float_state_slots]
        return outputs

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

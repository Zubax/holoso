"""
The numerical backend: a pure-Python functional model of a generated module, bit-exact to its RTL.

``NumericalModel`` is a software interpreter for the scheduled LIR microprogram. It replays the schedule applying the
same per-operator ZKF rounding the hardware does, so one ``__call__`` reproduces one module transaction bit-for-bit:
flat input values in port order to a flat tuple of output values in port order. It holds the ``Lir`` and pickles via
default pickle (every field is a frozen dataclass of plain values), so a generated testbench can embed it as its oracle.
"""

from dataclasses import dataclass

from .._value import FloatValue
from .._lir import FloatConstRef, FloatRegRef, Lir
from .._lir import InputProducer, OperationProducer, latest_producer_before
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


@dataclass(frozen=True, slots=True)
class NumericalModel:
    """
    A pure, stateless, bit-exact functional model of a generated module: one ``__call__`` is one module transaction.

    Call it with the input values in module-port order; it returns the output values in module-port order. The result
    matches the generated Verilog bit-for-bit because every operator evaluates the same ZKF bits as the hardware. The
    model holds the scheduled ``Lir`` and is picklable, so a generated testbench can embed it.

    TODO: When branching is implemented, the numerical model will need to be extended such that it also predicts the
          cycle latency of each transaction. This is necessary for cycle-accurate verification.
    """

    lir: Lir

    def __call__(self, *inputs: ModelInput) -> tuple[FloatValue, ...]:
        lir = self.lir
        if len(inputs) != len(lir.float_inputs):
            raise ValueError(f"expected {len(lir.float_inputs)} inputs, got {len(inputs)}")
        fmt = lir.float_regfile.fmt
        consts = [FloatValue.from_float(fmt, value) for value in lir.float_consts]
        in_values = [_coerce_input(value, fmt, index) for index, value in enumerate(inputs)]

        # Per-register write timeline: (commit_cycle, producer) in increasing commit order. Inputs are sampled at
        # cycle 0; each op commits at its commit_cycle. Operands resolve against this so a register reused for several
        # values over its lifetime yields the value that is live at the operand's read (issue) cycle, not the final one.
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

        # Evaluate in commit order: a producer commits before any consumer issues, so its value is ready in op_values.
        for j in sorted(
            range(len(lir.float_ops)), key=lambda k: (lir.float_ops[k].commit_cycle, lir.float_ops[k].issue_cycle)
        ):
            op = lir.float_ops[j]
            operands = [_apply_sign(value(o.source, op.issue_cycle), o.sign) for o in op.operands]
            op_values[j] = _apply_sign(op.inst.operator.evaluate(*operands), op.result_sign)

        present = lir.makespan + 1  # outputs present one cycle after the last commit; they read the final live value
        return tuple(_apply_sign(value(wire.source, present), wire.sign) for wire in lir.float_outputs)

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

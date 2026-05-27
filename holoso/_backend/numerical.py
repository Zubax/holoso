"""
The numerical backend: a pure-Python functional model of a generated module, bit-exact to its RTL.

``NumericalModel`` is a software interpreter for the scheduled LIR microprogram. It replays the schedule applying the
same per-operator ZKF rounding the hardware does, so one ``__call__`` reproduces one module transaction bit-for-bit:
flat input scalars in port order to a flat tuple of output scalars in port order. It holds the ``Lir`` and pickles via
default pickle (every field is a frozen dataclass of plain values), so a generated testbench can embed it as its oracle.
"""

from dataclasses import dataclass

from .._lir import FloatConstRef, FloatRegRef, Lir
from .._lir import InputProducer, OperationProducer, float_write_timeline, latest_producer_before
from .._operators import FloatSignControl
from .._type import FloatFormat


def _apply_sign(value: float, sign: FloatSignControl) -> float:
    """Apply a folded sign control exactly as ``holoso_fsgnop`` does in the RTL."""
    return sign.apply_float(value)


@dataclass(frozen=True, slots=True)
class NumericalModel:
    """
    A pure, stateless, bit-exact functional model of a generated module: one ``__call__`` is one module transaction.

    Call it with the input scalars in module-port order; it returns the output scalars in module-port order. The result
    matches the generated Verilog bit-for-bit because every operator's result is rounded to the ZKF format just as the
    hardware rounds it. The model holds the scheduled ``Lir`` and is picklable, so a generated testbench can embed it.

    TODO: When branching is implemented, the numerical model will need to be extended such that it also predicts the
          cycle latency of each transaction. This is necessary for cycle-accurate verification.
    """

    lir: Lir

    def __call__(self, *inputs: float) -> tuple[float, ...]:
        lir = self.lir
        if len(inputs) != len(lir.float_inputs):
            raise ValueError(f"expected {len(lir.float_inputs)} inputs, got {len(inputs)}")
        fmt = lir.float_regfile.fmt
        consts = lir.float_consts
        in_values = [fmt.round(x) for x in inputs]

        # Per-register write timeline: (commit_cycle, producer) in increasing commit order. Inputs are sampled at
        # cycle 0; each op commits at its commit_cycle. Operands resolve against this so a register reused for several
        # values over its lifetime yields the value that is live at the operand's read (issue) cycle, not the final one.
        writes = float_write_timeline(lir)

        op_values: dict[int, float] = {}

        def value(source: FloatRegRef | FloatConstRef, read_cycle: int) -> float:
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
            op_values[j] = fmt.round(_apply_sign(op.inst.operator.evaluate(*operands), op.result_sign))

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


def generate(lir: Lir) -> NumericalModel:
    """Build the bit-exact functional model from a finished :class:`Lir`."""
    return NumericalModel(lir)

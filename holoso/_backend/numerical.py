"""
The numerical backend: a pure-Python functional model of a generated module, bit-exact to its RTL.

``NumericalModel`` is a software interpreter for the scheduled LIR microprogram. It replays the schedule applying the
same per-operator ZKF rounding the hardware does, so one ``__call__`` reproduces one module transaction bit-for-bit:
flat input scalars in port order to a flat tuple of output scalars in port order. It holds the ``Lir`` and pickles via
default pickle (every field is a frozen dataclass of plain values), so a generated testbench can embed it as its oracle.
"""

from dataclasses import dataclass

from .._format import FloatFormat
from .._lir import ConstRef, Lir, RegRef
from .._operators import Sgnop

# A value source on a register's write timeline: an input (by input index) or an operator result (by op index).
type _Producer = tuple[str, int]


def _apply_sgnop(value: float, sgnop: Sgnop) -> float:
    """Apply a folded sign-op exactly as ``holoso_fsgnop`` does in the RTL: absolute value first, then negate."""
    if Sgnop.ABS in sgnop:
        value = abs(value)
    if Sgnop.NEG in sgnop:
        value = -value
    return value


def _latest_before(writes: list[tuple[int, _Producer]], read_cycle: int) -> _Producer:
    """
    The producer of the value a register holds when read at ``read_cycle``: the latest write committed strictly before
    it. The register file is read-first (a value committed at cycle ``c`` is readable from ``c + 1``), so this correctly
    distinguishes a still-live value from a later reuse of the same physical register on a different cycle.
    """
    chosen: _Producer | None = None
    for commit_cycle, producer in writes:  # writes are in increasing commit-cycle order
        if commit_cycle < read_cycle:
            chosen = producer
        else:
            break
    assert chosen is not None, "operand read resolves to no prior writer; the schedule is inconsistent"
    return chosen


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
        if len(inputs) != len(lir.inputs):
            raise ValueError(f"expected {len(lir.inputs)} inputs, got {len(inputs)}")
        fmt = lir.fmt
        consts = lir.consts
        in_values = [fmt.round(x) for x in inputs]

        # Per-register write timeline: (commit_cycle, producer) in increasing commit order. Inputs are sampled at
        # cycle 0; each op commits at its commit_cycle. Operands resolve against this so a register reused for several
        # values over its lifetime yields the value that is live at the operand's read (issue) cycle, not the final one.
        writes: dict[int, list[tuple[int, _Producer]]] = {}
        for i, load in enumerate(lir.inputs):
            writes.setdefault(load.dst.index, []).append((0, ("in", i)))
        for j, op in enumerate(lir.ops):
            writes.setdefault(op.dst.index, []).append((op.commit_cycle, ("op", j)))
        for events in writes.values():
            events.sort()

        op_values: dict[int, float] = {}

        def value(source: RegRef | ConstRef, read_cycle: int) -> float:
            if isinstance(source, ConstRef):
                return consts[source.index]
            kind, index = _latest_before(writes[source.index], read_cycle)
            return in_values[index] if kind == "in" else op_values[index]

        # Evaluate in commit order: a producer commits before any consumer issues, so its value is ready in op_values.
        for j in sorted(range(len(lir.ops)), key=lambda k: (lir.ops[k].commit_cycle, lir.ops[k].issue_cycle)):
            op = lir.ops[j]
            operands = [_apply_sgnop(value(o.source, op.issue_cycle), o.sgnop) for o in op.operands]
            op_values[j] = fmt.round(_apply_sgnop(op.inst.op.evaluate(*operands), op.y_sgnop))

        present = lir.makespan + 1  # outputs present one cycle after the last commit; they read the final live value
        return tuple(_apply_sgnop(value(wire.source, present), wire.sgnop) for wire in lir.outputs)

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(load.name for load in self.lir.inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(wire.name for wire in self.lir.outputs)

    @property
    def float_format(self) -> FloatFormat:
        return self.lir.fmt


def generate(lir: Lir) -> NumericalModel:
    """Build the bit-exact functional model from a finished :class:`Lir`."""
    return NumericalModel(lir)

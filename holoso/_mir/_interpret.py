"""
The MIR interpreter: a bit-exact, schedule-independent reference model of a kernel.

Where the numerical backend (``holoso._backend.numerical``) replays the scheduled, register-allocated LIR cycle by
cycle, this interpreter evaluates the unscheduled MIR dataflow graph directly: it walks the CFG, evaluates each
operation once through the operator's own bit-exact ``evaluate``, and resolves phis by the edge actually taken. It owns
no registers, no operator instances, and no cycle counter, so it is independent of scheduling, instance binding,
register allocation, and the cross-block overlap machinery. Comparing it against the numerical model therefore isolates
exactly that LIR layer, bit-for-bit and with no tolerance: a divergence is a scheduling/binding/regalloc/overlap
miscompile -- the class the RTL-vs-model cosimulation is structurally blind to, since the model shares the very LIR the
RTL is emitted from.

It is a first-class verification peer of the numerical model, not a test helper, and deliberately imports nothing from
``holoso._lir`` -- that independence is the whole point and a guard test enforces it.
"""

from typing import assert_never

from .._operators import BoolInversion, FloatSignControl, PortConditioner
from .._util import ValueId
from .._type import FloatFormat, LogicalPort
from .._value import FloatValue
from ._ir import (
    Mir,
    MirBlock,
    MirBoolConst,
    MirBoolInput,
    MirBoolOutput,
    MirBoolStateRead,
    MirBoolStateSlot,
    MirBranch,
    MirFloatConst,
    MirFloatInput,
    MirFloatOutput,
    MirFloatStateRead,
    MirFloatStateSlot,
    MirJump,
    MirOperation,
    MirOutput,
    MirPhi,
    MirRet,
    MirStateSlot,
)

type InterpreterInput = FloatValue | float | bool
type InterpreterOutput = FloatValue | bool
type _Value = FloatValue | bool


def _apply_conditioner(conditioner: PortConditioner, value: _Value) -> _Value:
    """Apply a port's folded sideband: a sign control on a float value, an inversion on a boolean one."""
    if isinstance(conditioner, FloatSignControl):
        assert isinstance(value, FloatValue), "a float sign control applies only to a FloatValue"
        return conditioner.apply_value(value)
    assert isinstance(value, bool), "a boolean inversion applies only to a bool"
    return conditioner.apply(value)


def _coerce_float(value: InterpreterInput, fmt: FloatFormat, index: int) -> FloatValue:
    if isinstance(value, FloatValue):
        if value.fmt != fmt:
            raise ValueError(f"input {index} has {value.fmt}, expected {fmt}")
        return value
    if type(value) is float:
        return FloatValue.from_float(fmt, value)
    raise TypeError(f"input {index} must be FloatValue or float, got {type(value).__name__}")


def _coerce_bool(value: InterpreterInput, index: int) -> bool:
    if type(value) is bool:
        return value
    raise TypeError(f"input {index} must be bool, got {type(value).__name__}")


class MirInterpreter:
    """
    A runnable bit-exact reference for one selected MIR graph. ``run`` evaluates one whole transaction (inputs to
    outputs) and advances the persistent slot state, so a caller drives an ordered sequence of transactions the same
    way it drives :meth:`NumericalSimulator.run`; ``reset`` reloads the slot reset snapshot. Construct one per kernel.
    """

    def __init__(self, mir: Mir) -> None:
        self._mir = mir
        self._blocks: dict[int, MirBlock] = {block.id: block for block in mir.blocks}
        self._state: dict[str, _Value] = {}
        self.reset()

    @property
    def inputs(self) -> list[LogicalPort]:
        """The logical input ports in parameter order, each with its scalar type."""
        return [LogicalPort(node.name, node.scalar_type) for node in self._input_nodes()]

    @property
    def outputs(self) -> list[LogicalPort]:
        """The logical output ports in return order, each with its scalar type."""
        return [LogicalPort(out.name, self._mir.nodes[out.value].scalar_type) for out in self._mir.outputs]

    def reset(self) -> None:
        """Reload every persistent slot with its reset snapshot, as at rst (the live-in of the next transaction)."""
        fmt = self._mir.float_format
        state: dict[str, _Value] = {}
        for slot in self._mir.state_slots:
            match slot:
                case MirFloatStateSlot():
                    state[slot.name] = FloatValue.from_float(fmt, float(slot.reset_value))
                case MirBoolStateSlot():
                    state[slot.name] = bool(slot.reset_value)
                case _:
                    assert False, f"unhandled state slot {type(slot).__name__}"
        self._state = state

    def run(self, *inputs: InterpreterInput, max_blocks: int = 10_000_000) -> list[InterpreterOutput]:
        """
        Evaluate one whole transaction: bind the inputs and the entry-global leaves, walk the CFG to the ``Ret``,
        read the outputs, then advance the persistent state (read-first). ``max_blocks`` bounds a non-terminating loop.
        """
        env = self._initial_env(inputs)
        current = self._mir.entry
        previous: int | None = None
        steps = 0
        while True:
            block = self._blocks[current]
            self._resolve_phis(block, previous, env)
            for op_id in block.operations:
                operation = self._mir.nodes[op_id]
                assert isinstance(operation, MirOperation), f"node {op_id} in block.operations is not a MirOperation"
                operands = [
                    _apply_conditioner(conditioner, env[operand])
                    for operand, conditioner in zip(operation.operands, operation.operand_conditioners, strict=True)
                ]
                results = operation.operator.evaluate(*operands)
                env[op_id] = _apply_conditioner(operation.output_conditioner, results[operation.output_port])
            terminator = block.terminator
            match terminator:
                case MirRet():
                    break
                case MirJump(target=target):
                    previous, current = current, target
                case MirBranch(cond=cond, if_true=if_true, if_false=if_false):
                    condition = env[cond]
                    assert isinstance(condition, bool), "a branch condition must evaluate to a bool"
                    previous, current = current, (if_true if condition else if_false)
                case _:
                    assert_never(terminator)
            steps += 1
            if steps > max_blocks:
                raise RuntimeError(f"MIR interpretation did not reach Ret within {max_blocks} steps")

        outputs = [self._read_output(out, env) for out in self._mir.outputs]
        self._writeback_state(env)
        return outputs

    def _input_nodes(self) -> list[MirFloatInput | MirBoolInput]:
        nodes: list[MirFloatInput | MirBoolInput] = []
        for vid in self._mir.input_ids:
            node = self._mir.nodes[vid]
            assert isinstance(node, (MirFloatInput, MirBoolInput)), f"input {vid} is not an input node"
            nodes.append(node)
        return nodes

    def _initial_env(self, inputs: tuple[InterpreterInput, ...]) -> dict[ValueId, _Value]:
        input_nodes = self._input_nodes()
        if len(inputs) != len(input_nodes):
            raise ValueError(f"expected {len(input_nodes)} inputs, got {len(inputs)}")
        fmt = self._mir.float_format
        env: dict[ValueId, _Value] = {}
        for index, (vid, input_node, raw) in enumerate(zip(self._mir.input_ids, input_nodes, inputs, strict=True)):
            match input_node:
                case MirFloatInput():
                    env[vid] = _coerce_float(raw, fmt, index)
                case MirBoolInput():
                    env[vid] = _coerce_bool(raw, index)
                case _:
                    assert_never(input_node)
        for vid, node in self._mir.nodes.items():
            match node:
                case MirFloatConst(value=value):
                    env[vid] = FloatValue.from_float(fmt, float(value))
                case MirBoolConst(value=value):
                    env[vid] = bool(value)
                case MirFloatStateRead(name=name) | MirBoolStateRead(name=name):
                    env[vid] = self._state[name]
                case _:
                    pass  # inputs (bound above), operations and phis (bound during the walk)
        return env

    def _resolve_phis(self, block: MirBlock, previous: int | None, env: dict[ValueId, _Value]) -> None:
        """Bind every phi in ``block`` as a parallel snapshot of the arm taken from ``previous`` (loop swaps need this)."""
        if not block.phis:
            return
        assert previous is not None, f"block {block.id} reached with phis but no predecessor (entry phi?)"
        snapshot: dict[ValueId, _Value] = {}
        for phi_id in block.phis:
            phi = self._mir.nodes[phi_id]
            assert isinstance(phi, MirPhi), f"node {phi_id} in phis is not a phi"
            arm = next((entry for entry in phi.arms if entry[0] == previous), None)
            assert arm is not None, f"phi {phi_id} has no arm for predecessor {previous}"
            _pred, value, conditioner = arm
            snapshot[phi_id] = _apply_conditioner(conditioner, env[value])
        env.update(snapshot)

    def _read_output(self, out: MirOutput, env: dict[ValueId, _Value]) -> InterpreterOutput:
        value = env[out.value]
        match out:
            case MirFloatOutput():
                assert isinstance(value, FloatValue)
                return out.sign.apply_value(value)
            case MirBoolOutput():
                assert isinstance(value, bool)
                return out.inversion.apply(value)
            case _:
                assert False, f"unhandled output {type(out).__name__}"

    def _writeback_state(self, env: dict[ValueId, _Value]) -> None:
        """Read every slot's conditioned live-out, then commit all at once -- the read-first parallel slot semantics."""
        new_state: dict[str, _Value] = {}
        for slot in self._mir.state_slots:
            new_state[slot.name] = self._read_slot(slot, env)
        self._state.update(new_state)

    def _read_slot(self, slot: MirStateSlot, env: dict[ValueId, _Value]) -> _Value:
        value = env[slot.live_out]
        match slot:
            case MirFloatStateSlot():
                assert isinstance(value, FloatValue)
                return slot.sign.apply_value(value)
            case MirBoolStateSlot():
                assert isinstance(value, bool)
                return slot.inversion.apply(value)
            case _:
                assert False, f"unhandled state slot {type(slot).__name__}"

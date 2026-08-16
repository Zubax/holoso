"""
The HIR evaluator: a host-precision reference model of an unoptimized frontend-emitted HIR graph -- the compiled
half of the front-end differential oracle, whose other half is CPython executing the original kernel.

Where the MIR interpreter (`holoso._mir._interpret`) is bit-exact in the target format to isolate the LIR layer,
this evaluator runs in the compiler's own arithmetic -- host floats, unbounded ints, each `Operator.evaluate`
itself -- upstream of HIR optimization, so a divergence from CPython indicts the front-end alone. It deliberately
imports nothing from `_mir`/`_lir`/`_eel`; a guard test enforces the independence.

`NoNumber` is carried as a poison value rather than an abort: the front-end's `and`/`or` gates are eager, so
the graph computes operands CPython may short-circuit past, and convicting one of those would fail kernels CPython
answers happily (`x > 1.0 and 1.0 / y > 0.0` at `y == 0`). Poison flows through consumers, except that an
operator's declared absorbing element swallows it -- an AND gate with a constant-0 input outputs 0 whatever garbage
sits on the other input -- and only a poison reaching an observable sink raises: a branch condition, an output, or
a state live-out, the last with the whole transaction's state left uncommitted. A poisoned BRANCH CONDITION inside
a short-circuited gate operand is the one shape the absorbing rule cannot reach (control cannot carry poison), so
such a kernel fails here even where CPython and the gate-level hardware would both answer through the gate;
accepted as a documented limitation rather than speculating both arms into loops and nontermination.

Values live in the environment as `Const` nodes, so the charter's constant arithmetic governs every intermediate
(no NaN, no negative zero). The SELECT operators are strict like any other operator -- they cannot ignore a
poisoned unselected arm -- which is sound within this evaluator's charter: the frontends emit conditionals as real
branches, and if-conversion runs only in the optimizer, downstream of the graphs evaluated here.
"""

from dataclasses import dataclass
from typing import assert_never

from .._util import BlockId, ValueId
from ._const import BoolConst, Const, FloatConst, IntConst
from ._ir import Block, Branch, Hir, InPort, Jump, Operation, Phi, Ret, StateRead
from ._operators import NoNumber
from ._types import BoolType, FloatType, IntType, Type

type Scalar = float | bool | int


@dataclass(frozen=True, slots=True)
class _Poison:
    what: str


type _Value = Const | _Poison


def _coerce(value: Scalar, ty: Type, index: int) -> Const:
    if isinstance(ty, FloatType) and type(value) is float:
        return FloatConst(value)
    if isinstance(ty, BoolType) and type(value) is bool:
        return BoolConst(value)
    if isinstance(ty, IntType) and type(value) is int:
        return IntConst(value)
    raise TypeError(f"input {index} expects {ty}, got {type(value).__name__} {value!r}")


def _scalar(const: Const) -> Scalar:
    assert isinstance(const, (FloatConst, BoolConst, IntConst)), f"constant {const!r} carries no scalar"
    return const.value


def _evaluate(operation: Operation, operands: list[_Value]) -> _Value:
    consts = [operand for operand in operands if isinstance(operand, Const)]
    if len(consts) != len(operands):
        absorbing = operation.operator.absorbing()
        if absorbing is not None and absorbing in consts:
            return absorbing
        return next(operand for operand in operands if isinstance(operand, _Poison))
    try:
        return operation.operator.evaluate(consts)
    except NoNumber as error:
        return _Poison(error.what)


class HirEvaluator:
    """
    A runnable reference for one HIR graph: run evaluates one whole transaction (inputs to outputs) and
    advances the persistent slots -- read-first, committed only after every sink proved to name a number -- so a
    caller drives an ordered sequence of transactions; reset reloads the reset snapshot. One per kernel.
    """

    def __init__(self, hir: Hir) -> None:
        self._hir = hir
        self._blocks: dict[BlockId, Block] = {block.id: block for block in hir.blocks}
        self._state: dict[str, Const] = {}
        self.reset()

    def reset(self) -> None:
        self._state = {slot.name: slot.reset_value for slot in self._hir.state_slots}

    @property
    def state(self) -> dict[str, Scalar]:
        return {name: _scalar(const) for name, const in self._state.items()}

    def run(self, *inputs: Scalar, max_blocks: int = 10_000_000) -> list[Scalar]:
        """
        Evaluate one whole transaction: bind the inputs and the entry-global leaves, walk the CFG to the `Ret`,
        read the outputs, then advance the persistent state. `max_blocks` bounds a non-terminating loop. Raises
        NoNumber when a poison reaches a sink; the transaction then commits nothing.
        """
        env = self._initial_env(inputs)
        current = self._hir.entry
        previous: BlockId | None = None
        steps = 0
        while True:
            block = self._blocks[current]
            self._resolve_phis(block, previous, env)
            for op_id in block.operations:
                operation = self._hir.nodes[op_id]
                assert isinstance(operation, Operation), f"node {op_id} in block.operations is not an Operation"
                env[op_id] = _evaluate(operation, [env[operand] for operand in operation.operands])
            terminator = block.terminator
            match terminator:
                case Ret():
                    break
                case Jump(target=target):
                    previous, current = current, target
                case Branch(cond=cond, if_true=if_true, if_false=if_false):
                    condition = env[cond]
                    if isinstance(condition, _Poison):
                        raise NoNumber(f"the branch condition of block {current}: {condition.what}")
                    assert isinstance(condition, BoolConst), "a branch condition must evaluate to a boolean"
                    previous, current = current, (if_true if condition.value else if_false)
                case _:
                    assert_never(terminator)
            steps += 1
            if steps > max_blocks:
                raise RuntimeError(f"HIR evaluation did not reach Ret within {max_blocks} blocks")

        outputs: list[Scalar] = []
        for out in self._hir.outputs:
            value = env[out.value]
            if isinstance(value, _Poison):
                raise NoNumber(f"the output {out.name!r}: {value.what}")
            outputs.append(_scalar(value))
        new_state: dict[str, Const] = {}
        for slot in self._hir.state_slots:
            live_out = env[slot.live_out]
            if isinstance(live_out, _Poison):
                raise NoNumber(f"the live-out of state slot {slot.name!r}: {live_out.what}")
            new_state[slot.name] = live_out
        self._state = new_state
        return outputs

    def _initial_env(self, inputs: tuple[Scalar, ...]) -> dict[ValueId, _Value]:
        if len(inputs) != len(self._hir.input_ids):
            raise ValueError(f"expected {len(self._hir.input_ids)} inputs, got {len(inputs)}")
        env: dict[ValueId, _Value] = {}
        for index, (vid, raw) in enumerate(zip(self._hir.input_ids, inputs, strict=True)):
            port = self._hir.nodes[vid]
            assert isinstance(port, InPort), f"input {vid} is not an InPort"
            env[vid] = _coerce(raw, port.type, index)
        for vid, node in self._hir.nodes.items():
            match node:
                case Const():
                    env[vid] = node
                case StateRead(slot=slot):
                    assert slot in self._state, f"state read of unknown slot {slot!r}"
                    env[vid] = self._state[slot]
                case _:
                    pass  # inputs are bound above; operations and phis are bound during the walk
        return env

    def _resolve_phis(self, block: Block, previous: BlockId | None, env: dict[ValueId, _Value]) -> None:
        """Bind every phi as a parallel snapshot of the arm taken from `previous` (loop swaps need this)."""
        if not block.phis:
            return
        assert previous is not None, f"block {block.id} reached with phis but no predecessor (entry phi?)"
        snapshot: dict[ValueId, _Value] = {}
        for phi_id in block.phis:
            phi = self._hir.nodes[phi_id]
            assert isinstance(phi, Phi), f"node {phi_id} in block.phis is not a Phi"
            arm = next((value for pred, value in phi.arms if pred == previous), None)
            assert arm is not None, f"phi {phi_id} has no arm for predecessor {previous}"
            snapshot[phi_id] = env[arm]
        env.update(snapshot)

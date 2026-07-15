"""
The FIR data model: a private, non-SSA control-flow graph over mutable Places, produced by the builder with no
analysis applied. Ops are generic Python-level operations (PyBin, PyCompare, PyCall...) that the analyzer later
types, folds, and resolves; every op and terminator carries an origin stack for located diagnostics. Each
FunctionUnit has one entry and ONE canonical exit: `return` is an assignment to the hidden ReturnPlace plus a jump,
so early returns need no flags and inlined callees can retarget their exit jumps to per-invocation continuations.
Temporaries are write-once by ANF construction; named locals rebind freely (the analyzer owns definedness).
"""

import enum
from dataclasses import dataclass, field

from ..._util import RelationalOp
from ._opsem import BinOp, UnOp
from ._signature import ParameterContract, ReturnContract
from ._value import StaticValue


@dataclass(frozen=True, slots=True)
class Origin:
    """One source frame; expansion pushes further frames so diagnostics point at the user call site."""

    function: str
    line: int
    column: int


type OriginStack = tuple[Origin, ...]


@dataclass(frozen=True, slots=True)
class BindingId:
    """
    A unique local binding slot within one FunctionUnit template. Synthetic ANF temporaries are write-once and
    spelled ``%N``; source-named bindings rebind freely. Comprehension targets get their own BindingId, distinct
    from any same-named function local -- that is the whole PEP 709 isolation story at the IR level.
    """

    name: str
    serial: int

    @property
    def is_temp(self) -> bool:
        return self.name.startswith("%")

    def __str__(self) -> str:
        return self.name if self.is_temp else f"{self.name}.{self.serial}"


@dataclass(frozen=True, slots=True)
class Local:
    binding: BindingId

    def __str__(self) -> str:
        return str(self.binding)


@dataclass(frozen=True, slots=True, eq=False)
class StateLeaf:
    """
    A persistent state scalar: the owning component by identity plus the attribute path down to the leaf.
    Equality and hash key on the owner's IDENTITY (the Reference doctrine): the generated value-based forms would
    conflate equal-valued distinct components and refuse unhashable owners in place-keyed maps.
    """

    component: object
    path: tuple[str, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateLeaf):
            return NotImplemented
        return self.component is other.component and self.path == other.path

    def __hash__(self) -> int:
        return hash((id(self.component), self.path))

    def __str__(self) -> str:
        return f"state:{'.'.join(self.path)}"


@dataclass(frozen=True, slots=True)
class ReturnPlace:
    def __str__(self) -> str:
        return "return"


type Place = Local | StateLeaf | ReturnPlace


class SelectMode(enum.Enum):
    """
    Distinguishes the value semantics of an eager two-way select: Python's ``a and b``/``a or b`` return one of
    the ORIGINAL operands (never a bool), while a chained comparison contributes a bool link.
    """

    AND = "and"
    OR = "or"


@dataclass(slots=True)
class LoadConst:
    dst: BindingId
    value: StaticValue
    origin: OriginStack


@dataclass(slots=True, eq=False)
class LoadRef:
    """
    Bind an identity reference to an object outside the value domain (a callable, module, class, component, or
    None). ``eq=False``: comparing ops would otherwise run the referent's own ``==`` (foreign semantics).
    """

    dst: BindingId
    obj: object
    origin: OriginStack


@dataclass(slots=True)
class LoadPlace:
    dst: BindingId
    place: Place
    origin: OriginStack


@dataclass(slots=True)
class StorePlace:
    place: Place
    src: BindingId
    origin: OriginStack


@dataclass(slots=True)
class UnbindPlace:
    """``checked`` distinguishes user ``del`` (unbinding an unbound name is an error, as in Python) from the
    builder's own comprehension-scope clearing on entry (silent)."""

    place: Place
    checked: bool
    origin: OriginStack


@dataclass(slots=True)
class PyBin:
    """``inplace`` marks augmented assignment: Python's ``+=`` mutates aliases of a mutable aggregate where plain
    rebinding does not, so the analyzer must see the difference to accept scalars and reject aggregates."""

    dst: BindingId
    op: BinOp
    lhs: BindingId
    rhs: BindingId
    inplace: bool
    origin: OriginStack


@dataclass(slots=True)
class PyUn:
    dst: BindingId
    op: UnOp
    operand: BindingId
    origin: OriginStack


@dataclass(slots=True)
class PyCompare:
    dst: BindingId
    op: RelationalOp
    lhs: BindingId
    rhs: BindingId
    origin: OriginStack


@dataclass(slots=True)
class PyNot:
    dst: BindingId
    operand: BindingId
    origin: OriginStack


@dataclass(slots=True)
class PyTruth:
    """Python truthiness of the operand, as a bool temp; Branch conditions and selects consume these."""

    dst: BindingId
    operand: BindingId
    origin: OriginStack


@dataclass(slots=True)
class PySelect:
    """
    Eager value select: both arms are already evaluated (pinned eager boolean semantics). AND yields ``rhs`` when
    ``cond`` is true else ``lhs``; OR yields ``lhs`` when ``cond`` is true else ``rhs`` -- matching Python's
    operand-returning ``and``/``or`` with ``cond = truth(lhs)``.
    """

    dst: BindingId
    mode: SelectMode
    cond: BindingId
    lhs: BindingId
    rhs: BindingId
    origin: OriginStack


@dataclass(slots=True)
class PyCall:
    """A lazy call: the callee is a temp holding whatever the analyzer resolves; only executable calls dispatch."""

    dst: BindingId
    callee: BindingId
    args: tuple[BindingId, ...]
    kwargs: tuple[tuple[str, BindingId], ...]
    origin: OriginStack


@dataclass(slots=True)
class PyAttr:
    dst: BindingId
    obj: BindingId
    name: str
    origin: OriginStack


@dataclass(slots=True)
class PyStoreAttr:
    """A generic attribute write (``self.x = v``); the analyzer maps it onto a StateLeaf place or rejects."""

    obj: BindingId
    name: str
    src: BindingId
    origin: OriginStack


@dataclass(slots=True)
class PySubscript:
    dst: BindingId
    obj: BindingId
    index: BindingId
    origin: OriginStack


@dataclass(slots=True)
class PyLen:
    dst: BindingId
    obj: BindingId
    origin: OriginStack


@dataclass(slots=True)
class BuildTuple:
    dst: BindingId
    items: tuple[BindingId, ...]
    origin: OriginStack


@dataclass(slots=True)
class BuildList:
    dst: BindingId
    items: tuple[BindingId, ...]
    origin: OriginStack


type Op = (
    LoadConst
    | LoadRef
    | LoadPlace
    | StorePlace
    | UnbindPlace
    | PyBin
    | PyUn
    | PyCompare
    | PyNot
    | PyTruth
    | PySelect
    | PyCall
    | PyAttr
    | PyStoreAttr
    | PySubscript
    | PyLen
    | BuildTuple
    | BuildList
)


@dataclass(frozen=True, slots=True)
class BlockId:
    index: int

    def __str__(self) -> str:
        return f"b{self.index}"


@dataclass(slots=True)
class Jump:
    target: BlockId
    origin: OriginStack


@dataclass(slots=True)
class Branch:
    """Two-way branch on a truth temp (the builder inserts PyTruth for every condition)."""

    cond: BindingId
    then_target: BlockId
    else_target: BlockId
    origin: OriginStack


@dataclass(slots=True)
class StaticFor:
    """
    A loop template over a to-be-static iterable: the analyzer clones ``body_entry``..(back edge) once per trip,
    binding ``target`` to each element; a residual (dynamic-trip) iterable is a located rejection. The body
    subgraph exits by jumping back here (re-testing) and leaves via ``exit_target``.
    """

    target: Place
    iterable: BindingId
    body_entry: BlockId
    exit_target: BlockId
    body_blocks: frozenset[BlockId]
    origin: OriginStack


@dataclass(slots=True)
class Fail:
    """Control reaches a failure: a raise, or a read that Python would refuse. Never falls through."""

    message: str
    origin: OriginStack


@dataclass(slots=True)
class UnitExit:
    """The canonical exit: the single point where ReturnPlace and the state leaves become the unit's results."""

    origin: OriginStack


type Terminator = Jump | Branch | StaticFor | Fail | UnitExit


@dataclass(slots=True)
class Block:
    id: BlockId
    ops: list[Op] = field(default_factory=list)
    terminator: Terminator | None = None


@dataclass(slots=True)
class FunctionUnit:
    """One function template: parameters are pre-bound named locals; ``exit`` is the only UnitExit block."""

    name: str
    params: list[BindingId]
    blocks: dict[BlockId, Block]
    entry: BlockId
    exit: BlockId
    bound_self: object | None
    param_contracts: dict[str, ParameterContract]  # per datapath parameter, by runtime spelling; seeds the analyzer
    return_contract: ReturnContract | None  # None for an inlined callee (not a port boundary)


def _op_reads(op: Op) -> list[BindingId]:
    match op:
        case LoadConst() | LoadRef() | LoadPlace():
            return []
        case StorePlace(src=src):
            return [src]
        case PyStoreAttr(obj=obj, src=src):
            return [obj, src]
        case UnbindPlace():
            return []
        case PyBin(lhs=lhs, rhs=rhs) | PyCompare(lhs=lhs, rhs=rhs):
            return [lhs, rhs]
        case PyUn(operand=operand) | PyNot(operand=operand) | PyTruth(operand=operand):
            return [operand]
        case PySelect(cond=cond, lhs=lhs, rhs=rhs):
            return [cond, lhs, rhs]
        case PyCall(callee=callee, args=args, kwargs=kwargs):
            return [callee, *args, *(v for _, v in kwargs)]
        case PyAttr(obj=obj):
            return [obj]
        case PySubscript(obj=obj, index=index):
            return [obj, index]
        case PyLen(obj=obj):
            return [obj]
        case BuildTuple(items=items) | BuildList(items=items):
            return list(items)


def op_dst(op: Op) -> BindingId | None:
    match op:
        case StorePlace() | UnbindPlace() | PyStoreAttr():
            return None
        case _:
            return op.dst


def verify(unit: FunctionUnit) -> None:
    """
    Structural invariants only (the analyzer owns semantic, path-sensitive ones like definedness): every block is
    terminated and every jump target exists; exactly one UnitExit, at ``unit.exit``; every ANF temporary has
    exactly one writer, and every read temporary has a writer somewhere in the unit.
    """
    assert unit.entry in unit.blocks and unit.exit in unit.blocks
    temp_writers: dict[BindingId, int] = {}
    for block in unit.blocks.values():
        assert block.terminator is not None, f"{block.id}: unterminated block"
        for op in block.ops:
            dst = op_dst(op)
            if dst is not None and dst.is_temp:
                temp_writers[dst] = temp_writers.get(dst, 0) + 1
        targets: list[BlockId]
        match block.terminator:
            case Jump(target=target):
                targets = [target]
            case Branch(then_target=then_target, else_target=else_target):
                targets = [then_target, else_target]
            case StaticFor(body_entry=body_entry, exit_target=exit_target):
                targets = [body_entry, exit_target]
            case Fail() | UnitExit():
                targets = []
        for target in targets:
            assert target in unit.blocks, f"{block.id}: dangling target {target}"
        assert isinstance(block.terminator, UnitExit) == (block.id == unit.exit), f"{block.id}: misplaced UnitExit"
    over_written = [str(t) for t, n in temp_writers.items() if n > 1]
    assert not over_written, f"temps written more than once: {over_written}"
    for block in unit.blocks.values():
        for op in block.ops:
            for read in _op_reads(op):
                if read.is_temp:
                    assert read in temp_writers, f"{block.id}: read of unwritten temp {read}"
        match block.terminator:
            case Branch(cond=cond) if cond.is_temp:
                assert cond in temp_writers, f"{block.id}: branch on unwritten temp {cond}"
            case StaticFor(iterable=iterable) if iterable.is_temp:
                assert iterable in temp_writers, f"{block.id}: loop over unwritten temp {iterable}"
            case _:
                pass


def _format_op(op: Op) -> str:
    match op:
        case LoadConst(dst=dst, value=value):
            return f"{dst} = const {value}"
        case LoadRef(dst=dst, obj=obj):
            label = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None) or type(obj).__name__
            return f"{dst} = ref {label}"  # address-free: reprs of plain objects would break golden determinism
        case LoadPlace(dst=dst, place=place):
            return f"{dst} = load {place}"
        case StorePlace(place=place, src=src):
            return f"store {place} <- {src}"
        case UnbindPlace(place=place, checked=checked):
            return f"{'del' if checked else 'clear'} {place}"
        case PyBin(dst=dst, op=bin_op, lhs=lhs, rhs=rhs, inplace=inplace):
            return f"{dst} = py.bin[{bin_op.value}{'=' if inplace else ''}] {lhs}, {rhs}"
        case PyUn(dst=dst, op=un_op, operand=operand):
            return f"{dst} = py.un[{un_op.value}] {operand}"
        case PyCompare(dst=dst, op=rel, lhs=lhs, rhs=rhs):
            return f"{dst} = py.cmp[{rel.name.lower()}] {lhs}, {rhs}"
        case PyNot(dst=dst, operand=operand):
            return f"{dst} = py.not {operand}"
        case PyTruth(dst=dst, operand=operand):
            return f"{dst} = py.truth {operand}"
        case PySelect(dst=dst, mode=mode, cond=cond, lhs=lhs, rhs=rhs):
            return f"{dst} = py.select[{mode.value}] {cond} ? {lhs} : {rhs}"
        case PyCall(dst=dst, callee=callee, args=args, kwargs=kwargs):
            rendered = [str(a) for a in args] + [f"{k}={v}" for k, v in kwargs]
            return f"{dst} = py.call {callee}({', '.join(rendered)})"
        case PyAttr(dst=dst, obj=obj, name=name):
            return f"{dst} = py.attr {obj}.{name}"
        case PyStoreAttr(obj=obj, name=name, src=src):
            return f"py.store_attr {obj}.{name} <- {src}"
        case PySubscript(dst=dst, obj=obj, index=index):
            return f"{dst} = py.subscript {obj}[{index}]"
        case PyLen(dst=dst, obj=obj):
            return f"{dst} = py.len {obj}"
        case BuildTuple(dst=dst, items=items):
            return f"{dst} = py.tuple ({', '.join(str(i) for i in items)})"
        case BuildList(dst=dst, items=items):
            return f"{dst} = py.list [{', '.join(str(i) for i in items)}]"


def _format_terminator(terminator: Terminator) -> str:
    match terminator:
        case Jump(target=target):
            return f"jump {target}"
        case Branch(cond=cond, then_target=then_target, else_target=else_target):
            return f"branch {cond} ? {then_target} : {else_target}"
        case StaticFor(target=target, iterable=iterable, body_entry=body_entry, exit_target=exit_target):
            return f"static_for {target} in {iterable} body={body_entry} exit={exit_target}"
        case Fail(message=message):
            return f"fail {message!r}"
        case UnitExit():
            return "unit_exit"


def pretty(unit: FunctionUnit) -> str:
    """Deterministic dump: blocks in id order, one op per line -- the golden-test surface."""
    lines = [f"unit {unit.name}({', '.join(str(p) for p in unit.params)})"]
    for block_id in sorted(unit.blocks, key=lambda b: b.index):
        block = unit.blocks[block_id]
        suffix = " (entry)" if block_id == unit.entry else ""
        lines.append(f"{block_id}:{suffix}")
        for op in block.ops:
            lines.append(f"    {_format_op(op)}")
        assert block.terminator is not None
        lines.append(f"    {_format_terminator(block.terminator)}")
    return "\n".join(lines) + "\n"

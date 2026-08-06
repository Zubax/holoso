"""
The Eel tree: the one representation shared by all three front-end stages (desugar, partial evaluation, emit).

Eel is compact-atom A-normal form: every operand of every operation is an atom (a temp reference, a local-name
reference, or a constant); every non-atomic subexpression is hoisted by the desugarer to a fresh temp in CPython
evaluation order. Temps are anonymous locals, not SSA values: a guarded construct (a desugared conditional
expression or comparison chain) assigns its result temp in both arms.

The tree is pure syntax: no numpy, no function objects, and no HIR beyond one opacity -- a residual
``IntrinsicCall`` carries the operator the partial evaluator selected, typed ``object`` here and downcast only by
emit, so the boundary is an IMPORT boundary rather than a representational one.
The partial evaluator keeps the pairing between an EelFunction and its FunctionType on the side. Every node carries
an Origin — a source location plus the inline frame chain, which is empty at desugar and grows as the partial
evaluator inlines calls — so any rejection raised inside library-stub code is re-attributed to the user's call site.

Store targets stay syntactic (a root name plus a selector chain, with only index expressions hoisted to atoms):
the ownership model exempts store-target-prefix reads from sharing, which requires the prefix to remain a path
rather than a hoisted aggregate read. A store rooted in an environment name is representable so the partial
evaluator can reject it with the escaped-root diagnostic instead of a syntax error.

Residual-only forms (never produced by the desugarer, introduced by the partial evaluator, consumed by emit):
IntrinsicCall (a resolved HIR operator riding opaquely — also the spelling of every PE-inserted conversion),
SlotRead/SlotWrite over flattened state-slot paths, ResidualWhile with its explicit loop-carried phis and the
ResidualBreak/ResidualContinue terminators of its body, ResidualFrame with its result rows and the
ResidualFrameReturn terminators of its sites, ResidualReturn against the OutputDecl table, and the mandatory
ScalarType annotations on residual Assign/Param.
Residual Eel is fully scalar: no TupleExpr survives
partial evaluation; a multi-output kernel returns one typed atom per OutputDecl row. A ResidualReturn is a
TERMINATOR — one per return path, each preceded by the per-leaf SlotWrite commits of that path — and emission
joins all sites in a single exit block, so HIR's sole Ret invariant holds. A returned leaf whose value matches
the same public state slot's live-out at every site is elided from the table. A bare scalar return flattens
to leaf path (0,), matching the existing out_0 port convention.
"""

from dataclasses import dataclass
from enum import Enum

from .._errors import SourceLocation


@dataclass(frozen=True, slots=True)
class CallFrame:
    callee: str
    site: SourceLocation


@dataclass(frozen=True, slots=True)
class Origin:
    location: SourceLocation
    frames: tuple[CallFrame, ...] = ()


class UnaryOp(Enum):
    NEG = "-"
    POS = "+"
    NOT = "not"
    INVERT = "~"


class BinaryOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    FLOORDIV = "//"
    MOD = "%"
    POW = "**"
    MATMUL = "@"
    LSHIFT = "<<"
    RSHIFT = ">>"
    BITAND = "&"
    BITOR = "|"
    BITXOR = "^"
    AND = "and"  # the eager boolean gates: both operands always evaluate, deviating from Python deliberately
    OR = "or"


class CompareOp(Enum):
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    NE = "!="


class ParamKind(Enum):
    POSITIONAL_ONLY = "positional-only"
    POSITIONAL_OR_KEYWORD = "positional-or-keyword"
    KEYWORD_ONLY = "keyword-only"


class ScalarType(Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"


@dataclass(frozen=True, slots=True)
class TempRef:
    origin: Origin
    index: int


@dataclass(frozen=True, slots=True)
class LocalRef:
    origin: Origin
    name: str


@dataclass(frozen=True, slots=True)
class Const:
    origin: Origin
    value: bool | int | float


type Atom = TempRef | LocalRef | Const


@dataclass(frozen=True, slots=True)
class TempBind:
    origin: Origin
    index: int


@dataclass(frozen=True, slots=True)
class LocalBind:
    origin: Origin
    name: str


type Binding = TempBind | LocalBind


@dataclass(frozen=True, slots=True)
class PosArg:
    value: Atom


@dataclass(frozen=True, slots=True)
class StarArg:
    value: Atom


@dataclass(frozen=True, slots=True)
class KwArg:
    name: str
    value: Atom


type Argument = PosArg | StarArg | KwArg


@dataclass(frozen=True, slots=True)
class AttrSel:
    name: str


@dataclass(frozen=True, slots=True)
class IndexSel:
    index: Atom


type Selector = AttrSel | IndexSel

type SlotPath = tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class Unary:
    origin: Origin
    op: UnaryOp
    operand: Atom


@dataclass(frozen=True, slots=True)
class Binary:
    origin: Origin
    op: BinaryOp
    left: Atom
    right: Atom


@dataclass(frozen=True, slots=True)
class Compare:
    origin: Origin
    op: CompareOp
    left: Atom
    right: Atom


@dataclass(frozen=True, slots=True)
class Call:
    """
    Arguments keep the syntactic form (positional/starred/keyword) for signature binding at the partial
    evaluator; the wrapper list is canonical order — positional and starred in source order, then keywords in
    source order — which is also CPython's evaluation order, pinned by the hoisted temps preceding the call.
    """

    origin: Origin
    callee: Atom
    args: tuple[Argument, ...]


@dataclass(frozen=True, slots=True)
class AttrRead:
    origin: Origin
    base: Atom
    attr: str


@dataclass(frozen=True, slots=True)
class IndexRead:
    origin: Origin
    base: Atom
    index: Atom


@dataclass(frozen=True, slots=True)
class SliceRead:
    origin: Origin
    base: Atom
    lo: Atom | None
    hi: Atom | None


@dataclass(frozen=True, slots=True)
class SliceSel:
    """One sliced axis of a multi-dimensional subscript; bounds are hoisted atoms or open."""

    lo: "Atom | None"
    hi: "Atom | None"


@dataclass(frozen=True, slots=True)
class MultiIndexRead:
    """
    A tuple subscript ``m[i, j]`` / ``m[:, k]``: one selector per axis, in source order. Tensor-only — the
    partial evaluator rejects it on sequences. Stores never use this form: slice assignment is banned, and a
    pure-index tuple store rides an ordinary IndexSel with a tuple-valued atom.
    """

    origin: Origin
    base: Atom
    axes: tuple[Atom | SliceSel, ...]


@dataclass(frozen=True, slots=True)
class TupleExpr:
    """A starred item splices an aggregate in place; the partial evaluator expands it once shapes are known."""

    origin: Origin
    items: tuple[Atom | StarArg, ...]


@dataclass(frozen=True, slots=True)
class ListExpr:
    origin: Origin
    items: tuple[Atom | StarArg, ...]


@dataclass(frozen=True, slots=True)
class EnvRead:
    """
    A non-local name: free=True resolves through the closure cell of the same name; free=False resolves through
    the function's globals and then builtins at snapshot time — exactly CPython's dynamic global scope.
    """

    origin: Origin
    name: str
    free: bool


@dataclass(frozen=True, slots=True)
class Comp:
    """
    A list comprehension: its own scope. The target is a fresh desugar-renamed name (never colliding with a
    function local); the iterable is evaluated once, outside, in the enclosing naming context.
    """

    origin: Origin
    target: str
    iterable: Atom
    body: tuple[Stmt, ...]
    element: Atom


@dataclass(frozen=True, slots=True)
class IntrinsicCall:
    """
    Residual-only: a call resolved to a single HIR operator, riding opaquely (the tree stays HIR-free; only emit
    downcasts). PE-inserted conversions (int-to-float promotion arms, casts) use this same spelling.
    """

    origin: Origin
    operator: object
    args: tuple[Atom, ...]


@dataclass(frozen=True, slots=True)
class SlotRead:
    origin: Origin
    slot: SlotPath


type Expr = (
    Atom
    | Unary
    | Binary
    | Compare
    | Call
    | AttrRead
    | IndexRead
    | SliceRead
    | MultiIndexRead
    | TupleExpr
    | ListExpr
    | EnvRead
    | Comp
    | IntrinsicCall
    | SlotRead
)


@dataclass(frozen=True, slots=True)
class Assign:
    origin: Origin
    target: Binding
    value: Expr
    stype: ScalarType | None = None  # None from desugar; mandatory on residual assignments


@dataclass(frozen=True, slots=True)
class Unpack:
    origin: Origin
    targets: tuple[Binding, ...]
    value: Atom


@dataclass(frozen=True, slots=True)
class Store:
    origin: Origin
    root: LocalRef | EnvRead
    path: tuple[Selector, ...]
    value: Atom


@dataclass(frozen=True, slots=True)
class AugStore:
    """
    The old element is read at the store step, after the value temp — CPython reads it before the RHS, but the
    divergence is unobservable: expressions cannot mutate aggregates and a same-region walrus rebinding an index
    or root name is rejected by the desugarer.
    """

    origin: Origin
    root: LocalRef | EnvRead
    path: tuple[Selector, ...]
    op: BinaryOp
    value: Atom


@dataclass(frozen=True, slots=True)
class AugAssign:
    """
    Kept as a distinct node because the desugarer has no types: on a list or tensor, ``x += e`` is an in-place
    mutation under the ownership model, while on a scalar or tuple it is a rebind; the partial evaluator decides
    by kind.
    """

    origin: Origin
    target: LocalBind
    op: BinaryOp
    value: Atom


@dataclass(frozen=True, slots=True)
class If:
    origin: Origin
    cond: Atom
    then: tuple[Stmt, ...]
    orelse: tuple[Stmt, ...]


@dataclass(frozen=True, slots=True)
class While:
    """The header re-executes before every test: the condition's temps live there."""

    origin: Origin
    header: tuple[Stmt, ...]
    cond: Atom
    body: tuple[Stmt, ...]


@dataclass(frozen=True, slots=True)
class For:
    origin: Origin
    target: LocalBind
    iterable: Atom
    body: tuple[Stmt, ...]


@dataclass(frozen=True, slots=True)
class LoopPhi:
    """``index`` is the phi's own temp, ``entry`` its value on loop entry, ``back`` its per-iteration value."""

    origin: Origin
    index: int
    stype: ScalarType
    entry: Atom
    back: Atom


@dataclass(frozen=True, slots=True)
class ResidualWhile:
    """
    Residual-only: a genuine back-edge loop. Every test binds the phis and re-runs the header, so the header
    also runs on the final failing test and its temps stay readable after the loop (it dominates the exit).
    """

    origin: Origin
    phis: tuple[LoopPhi, ...]
    header: tuple[Stmt, ...]
    cond: Atom
    body: tuple[Stmt, ...]


@dataclass(frozen=True, slots=True)
class Return:
    origin: Origin
    value: Atom | None


@dataclass(frozen=True, slots=True)
class ResidualReturn:
    """Residual-only: one typed scalar atom per OutputDecl row, in table order; bare when outputs are empty."""

    origin: Origin
    values: tuple[Atom, ...]


@dataclass(frozen=True, slots=True)
class SlotWrite:
    origin: Origin
    slot: SlotPath
    value: Atom


@dataclass(frozen=True, slots=True)
class Break:
    origin: Origin


@dataclass(frozen=True, slots=True)
class Continue:
    origin: Origin


@dataclass(frozen=True, slots=True)
class ResidualBreak:
    """
    Residual-only terminator: the edge from a ``ResidualWhile`` body to the loop exit. Values leaving on this
    edge ride ordinary join temps the partial evaluator assigns in the break arm and at header end, so the
    emitter's loop-exit join stays mechanical.
    """

    origin: Origin


@dataclass(frozen=True, slots=True)
class ResidualContinue:
    """Residual-only terminator: the edge from a ``ResidualWhile`` body to the back edge (the next test)."""

    origin: Origin


@dataclass(frozen=True, slots=True)
class FrameRow:
    """One residual result column of a ``ResidualFrame``; ``index`` is the temp the caller reads."""

    origin: Origin
    index: int
    stype: ScalarType


@dataclass(frozen=True, slots=True)
class ResidualFrame:
    """
    Residual-only: an inlined callee whose return sites include one inside the callee's own residual loop,
    so the sites cannot rejoin as sibling arms and instead converge at the frame exit. Each
    ``ResidualFrameReturn`` carries one atom per row, in row order; statically uniform result leaves never
    enter the rows -- they stay in the caller's value model.
    """

    origin: Origin
    rows: tuple[FrameRow, ...]
    body: Block


@dataclass(frozen=True, slots=True)
class ResidualFrameReturn:
    """Residual-only terminator: one return site of a ``ResidualFrame``, one atom per frame row."""

    origin: Origin
    values: tuple[Atom, ...]


@dataclass(frozen=True, slots=True)
class Raise:
    """
    A raise with a static message: literal string parts interleaved with hoisted value atoms, formatted by the
    partial evaluator with statically known values. The exception type is display text only.
    """

    origin: Origin
    exc_type: str
    parts: tuple[str | Atom, ...]


type Stmt = (
    Assign
    | Unpack
    | Store
    | AugStore
    | AugAssign
    | If
    | While
    | For
    | ResidualWhile
    | Return
    | ResidualReturn
    | SlotWrite
    | Break
    | Continue
    | ResidualBreak
    | ResidualContinue
    | ResidualFrame
    | ResidualFrameReturn
    | Raise
)

type Block = tuple[Stmt, ...]


@dataclass(frozen=True, slots=True)
class SlotDecl:
    slot: SlotPath
    stype: ScalarType
    reset: bool | int | float


@dataclass(frozen=True, slots=True)
class OutputDecl:
    path: tuple[int | str, ...]
    stype: ScalarType


@dataclass(frozen=True, slots=True)
class Param:
    origin: Origin
    name: str
    kind: ParamKind
    stype: ScalarType | None = None  # None from desugar; mandatory on residual parameters


@dataclass(frozen=True, slots=True)
class EelFunction:
    """
    Defaults and annotations are data, read from the function object by the partial evaluator, never syntax.
    ``slots`` and ``outputs`` are empty from the desugarer and populated on the residual function.
    """

    origin: Origin
    name: str
    params: tuple[Param, ...]
    body: Block
    slots: tuple[SlotDecl, ...] = ()
    outputs: tuple[OutputDecl, ...] = ()

"""Test-only verification helpers."""

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from holoso import FAddOperator, FCmpOperator, FDivOperator, FMulILog2OperatorFamily, FMulOperator, OpConfig
from holoso._backend.numerical import NumericalSimulator, generate as generate
from holoso._frontend import lower as lower_frontend
from holoso._hir import optimize
from holoso._lir import Lir, build
from holoso._mir import MirInterpreter, lower as lower_to_mir
from holoso._type import FloatFormat
from holoso._value import FloatValue
from holoso._frontend._ast_support import Path, port_name

type Vector = list[FloatValue | bool]


@dataclass(frozen=True, slots=True)
class OperatorCase:
    label: str
    make_ops: Callable[[FloatFormat], OpConfig]
    fcmp_latency: int


def build_model(lir: Lir) -> NumericalSimulator:
    return generate(lir).elaborate()


def build_model_and_interpreter(
    kernel: Callable[..., object], ops: OpConfig, name: str
) -> tuple[NumericalSimulator, MirInterpreter]:
    """
    Drive one kernel through the internal pipeline and return (numerical model, MIR interpreter) over the SAME MIR --
    the single source of truth for the differential-oracle tests. The model descends through ``build`` (the
    scheduled/allocated LIR, where the verified bug class lives); the interpreter is taken straight off the MIR
    (upstream of ``build``), so the two share everything except the LIR layer.
    """
    mir = lower_to_mir(optimize(lower_frontend(kernel)), ops)
    return build_model(build(mir, name)), MirInterpreter(mir)


def show_value(value: FloatValue | bool) -> str:
    return f"{float(value):.6g}" if isinstance(value, FloatValue) else str(value)


def assert_model_equals_interpreter(
    model: NumericalSimulator, interpreter: MirInterpreter, vectors: list[Vector], label: str
) -> None:
    assert model.inputs == interpreter.inputs, f"{label}: input ports differ (name or type)"
    assert model.outputs == interpreter.outputs, f"{label}: output ports differ (name or type)"
    for vector in vectors:
        model_out = model.run(*vector)
        interp_out = interpreter.run(*vector)
        assert model_out == interp_out, (
            f"{label}: model != interpreter for inputs {[show_value(v) for v in vector]}: "
            f"{[show_value(v) for v in model_out]} vs {[show_value(v) for v in interp_out]}"
        )


def flatten_value(root: object) -> list[tuple[Path, Any]]:
    leaves: list[tuple[Path, Any]] = []

    def walk(node: object, path: Path) -> None:
        if isinstance(node, (list, tuple)) and not isinstance(node, str):
            for index, item in enumerate(node):
                walk(item, [*path, index])
        elif dataclasses.is_dataclass(node) and not isinstance(node, type):
            for field in dataclasses.fields(node):
                walk(getattr(node, field.name), [*path, field.name])
        else:
            leaves.append((path, node))

    if (isinstance(root, (list, tuple)) and not isinstance(root, str)) or (
        dataclasses.is_dataclass(root) and not isinstance(root, type)
    ):
        walk(root, [])
    else:
        leaves.append(([0], root))
    return leaves


def evaluate_reference(fn: Callable[..., object], inputs: Mapping[str, float]) -> list[float]:
    result = fn(**inputs)
    return [float(value) for _, value in flatten_value(result)]


def output_names(root: object) -> list[str]:
    return [port_name(path) for path, _ in flatten_value(root)]


def unit_roundoff(fmt: FloatFormat) -> float:
    return 2.0 ** -(fmt.wman - 1)


def default_tolerance(
    fmt: FloatFormat, op_count: int, magnitude: float = 1.0, rel_factor: float = 16.0
) -> tuple[float, float]:
    u = unit_roundoff(fmt)
    rtol = rel_factor * max(op_count, 1) * u
    atol = rtol * abs(magnitude)
    return rtol, atol


def within(actual: float, expected: float, rtol: float, atol: float) -> bool:
    """Whether ``actual`` is within ``atol + rtol*|expected|`` of ``expected`` (infinities must match exactly)."""
    if math.isinf(expected) or math.isinf(actual) or math.isnan(expected) or math.isnan(actual):
        return actual == expected
    return abs(actual - expected) <= atol + rtol * abs(expected)


def bounded(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def log_uniform_positive(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def random_legal_bits(fmt: FloatFormat, rng: np.random.Generator) -> int:
    """A uniformly random finite, legal ZKF bit pattern (normals and +0; no inf/subnormal/negative zero)."""
    span = 1 << fmt.width
    while True:
        bits = int(rng.integers(0, span, dtype=np.uint64))
        if fmt.is_legal(bits) and fmt.is_finite(bits):
            return bits


def spd_matrix(rng: np.random.Generator, n: int, diag_lo: float = 0.5, diag_hi: float = 2.0) -> np.ndarray:
    lower = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            lower[i, j] = rng.uniform(diag_lo, diag_hi) if i == j else rng.uniform(-1.0, 1.0)
    return lower @ lower.T


def encode_inputs(fmt: FloatFormat, values: dict[str, float | bool]) -> dict[str, int]:
    return {name: int(value) if type(value) is bool else fmt.encode(value) for name, value in values.items()}


def format_edge_bits(fmt: FloatFormat) -> list[int]:
    """
    Canonical legal-ZKF edge bit patterns for one format: zero, ±0.5, ±1, ±smallest-normal, ±largest-finite.
    Built directly from the bit layout so the extremes stay exact even where they would overflow a Python float.
    """
    frac_bits = fmt.wman - 1
    sign_bit = 1 << (fmt.width - 1)
    max_exp = (1 << fmt.wexp) - 2  # the all-ones exponent is infinity, so the largest finite exponent is one below it
    magnitudes = [
        0,  # canonical zero (ZKF has no negative zero, so it has no signed counterpart)
        fmt.encode(0.5),
        fmt.encode(1.0),
        1 << frac_bits,  # smallest normal: exponent 1, zero fraction
        (max_exp << frac_bits) | ((1 << frac_bits) - 1),  # largest finite: max exponent, all-ones fraction
    ]
    edges: list[int] = []
    for magnitude in magnitudes:
        edges.append(magnitude)
        if magnitude != 0:
            edges.append(sign_bit | magnitude)
    return edges


def default_ops(fmt: FloatFormat) -> OpConfig:
    return fcmp_staged_ops(fmt, 0)


def fcmp_staged_ops(fmt: FloatFormat, stage_input: int) -> OpConfig:
    """Default operators with only the comparator's stage knob varied (latency ``1 + stage_input``)."""
    return OpConfig(
        FAddOperator(fmt),
        FMulOperator(fmt),
        FDivOperator(fmt),
        FMulILog2OperatorFamily(fmt),
        FCmpOperator(fmt, stage_input=stage_input),
    )


def fcmp_s1_ops(fmt: FloatFormat) -> OpConfig:
    return fcmp_staged_ops(fmt, 1)


def branch_boundary_kernel(a, b, c):  # type: ignore[no-untyped-def]
    """
    The boundary-slack corner kernel shared by the cosim test and its white-box schedule twin: the comparison is the
    LAST commit in its block and feeds the branch, so its result lands in the condition register exactly one step
    before the terminator reads it. The two tests must exercise the same kernel, so it lives here. The division in
    the else arm is unspeculatable, which keeps the diamond a REAL branch under default if-conversion -- the corner
    under test exists only on a branchy schedule.
    """
    t = a * b + c
    if t > c:
        y = t + 1.0
    else:
        y = (t - 1.0) / (b * b + 1.0)  # structurally nonzero divisor: the bench asserts err_pc == 0 per vector
    return y


def overlap_spill_kernel(x, y, z):  # type: ignore[no-untyped-def]
    """
    Cross-block software-pipelining corner shared by the cosim test and its white-box twin. The branch CONDITION
    (``x < y``) depends only on inputs, so it commits early; a wide chain (``w``) computed in the same block commits
    much later. The block's terminator therefore shrinks to ``w``'s write word (not the early condition), and ``w``
    SPILLS past the terminator into BOTH (single-predecessor) arms, which read it -- so a consumer in an arm must wait
    for ``w``'s in-flight landing in the successor frame. The unspeculatable division in the else arm keeps the diamond
    a real branch under default if-conversion (so the spill crosses a genuine branch, replicated onto both arms).
    """
    w = (x * z + y) * z + y  # a wide chain whose result outlives the early comparison's commit
    if x < y:
        r = w + 1.0  # then-arm reads the spilled w
    else:
        r = w / (y * y + 1.0)  # else-arm reads the spilled w; the division keeps this a branch (structurally nonzero)
    return r


def overlap_dead_arm_spill_kernel(x, y, z):  # type: ignore[no-untyped-def]
    """
    Cross-block overlap SOUNDNESS corner: a value live ONLY in one arm shares no register hazard with a value the
    sibling arm spills onto it. ``v`` is computed in the entry block and used only in the else arm; the wide chain
    ``w`` commits late and spills past the shrunk terminator into BOTH arms (its write-enable fires unconditionally
    before the redirect). In the else arm ``w`` is DEAD -- if the allocator reuses ``w``'s register for ``v`` there,
    the unconditional spill of ``w`` clobbers ``v`` before the arm reads it (a silent miscompile the cosim cannot
    catch, since the numerical model shares the same register file). The else arm's value must therefore be checked
    against the source semantics, not just RTL==model. The unspeculatable division keeps this a real branch.
    """
    v = x + y  # lives across the branch, read only in the else arm
    w = (x * z + y) * z + y  # wide chain commits late -> spills into both arms; DEAD in the else arm
    if x < y:
        r = w * 2.0  # then-arm uses the spilled w
    else:
        a = v + z  # else-arm reads v (must survive w's dead-arm spill); w is unused here
        r = a / (z * z + 1.0)
    return r


def const_branch_kernel(x, y):  # type: ignore[no-untyped-def]
    """
    Empty const-branch block corner shared by the cosim test and its white-box twin. The inner condition ``1.0 / 5.0 >
    0.0`` is constant-true but formed by DIVISION, which escapes the frontend's AST-level reachability fold (it
    evaluates only +,-,* of literals), so the HIR const-folder reduces it to a BoolConst that if-conversion refuses --
    leaving an EMPTY const-branch block (the condition install + a branch, no float content). That const materialization
    is a pc-gated install read AT the terminator and lands at the drained boundary, so the drain must keep that
    boundary for it; shrinking below it made the branch read the condition one PC before it landed.
    """
    r = x
    if x > y:
        if (1.0 / 5.0) > 0.0:  # constant-true via division: an empty const-branch block, no float content
            r = x + 1.0
        else:
            r = x + 2.0
    return r


def diamond_then_loop_kernel(x, y):  # type: ignore[no-untyped-def]
    """
    Empty merge-block elimination (B4) corner shared by the cosim test and its white-box twin. The variable-divisor
    division keeps the diamond a REAL branch (unspeculatable), so its merge stays a separate block; that merge holds
    only the merged phi (no operation) and jumps into the following loop header, making it an empty pass-through merge
    whose predecessors (the two diamond arms) are both jump-terminated. Merge threading eliminates it, composing the
    diamond's phi arms into the loop header's init arm -- producing a THREE-arm loop-header phi (two forward init arms
    plus the back-edge), a shape no other kernel exercises against RTL. The loop drives the data-dependent latency.
    """
    if x > y:
        r = x / y
    else:
        r = y / x
    while r > 1.0:
        r = r - 1.0
    return r


def overlap_div_err_kernel(x, y, z):  # type: ignore[no-untyped-def]
    """
    Cross-block overlap err_pc corner (shared by the white-box twin and the directed err_pc cosim). A division -- the
    one error-bearing op -- commits late, so its result spills past the shrunk terminator. The data write lands in the
    taken arm correctly, but the err_pc diagnostic latches ``pc - FETCH_LAG`` when the write-
    enable executes, FETCH_LAG steps after its write word; if the terminator redirected to the NON-fall-through arm by
    then, err_pc would capture the successor frame instead of the division's step. The shrink floor must keep that
    latch in-block. ``x < z`` selects the non-fall-through (true) arm, the only arm with a PC discontinuity; ``y == 0``
    makes the division error. The else arm's division keeps this a real branch under default if-conversion.
    """
    q = x / y
    if x < z:
        r = q + 1.0  # non-fall-through arm: a PC redirect coincides with the division's err-latch cycle
    else:
        r = q / (z * z + 1.0)  # structurally nonzero divisor; keeps the diamond a real branch
    return r


def staged_ops(fmt: FloatFormat) -> OpConfig:
    """
    A deeply pipelined configuration, distinct enough from the default to exercise the schedule, register allocation,
    and handshake at a longer latency. Deliberately hardcoded -- it is a test fixture chosen for coverage, not a
    derived enumeration of operator knobs, so it stays valid as new (not necessarily stage-shaped) knobs are added.
    """
    return OpConfig(
        FAddOperator(
            fmt, stage_input=1, stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1, stage_output=1
        ),
        FMulOperator(fmt, stage_input=1, stage_product=1, stage_pack=1, stage_output=1),
        FDivOperator(fmt, stage_input=1, stage_pack=1, stage_output=1),
        FMulILog2OperatorFamily(fmt, stage_input=1, stage_decode=1),
        FCmpOperator(fmt, stage_input=1),
    )


PIPELINE_OP_CASES = (
    OperatorCase("default", default_ops, 1),
    OperatorCase("staged", staged_ops, 2),
)

COMPARATOR_OP_CASES = (
    OperatorCase("default", default_ops, 1),
    OperatorCase("fcmp_s1", fcmp_s1_ops, 2),
    OperatorCase("staged", staged_ops, 2),
)


class ChainedSlots:
    """
    Chained persistent slots: ``_a`` captures ``_b``'s OLD value while ``_b`` advances, behind a long float tail.
    Shared by the schedule-level regression test and its RTL cosim twin -- the two must exercise the same kernel.
    """

    def __init__(self) -> None:
        self._a = 0.0
        self._b = 0.0

    def __call__(self, x):  # type: ignore[no-untyped-def]
        self._a = self._b
        self._b = x + 1.0
        return self._a * 2.0 + (x * 1.5) / (x - 0.5)


class SelectHold:
    """
    A Ret-block select is the slot live-in's LAST reader while the new live-out commits early: pins the read-step
    frame of the state early-install bound. Shared by the white-box schedule test and its RTL cosim twin.
    """

    def __init__(self) -> None:
        self._h = 1.0

    def step(self, x, c):  # type: ignore[no-untyped-def]
        old = self._h
        self._h = x + 1.0
        y = old if c > 0.0 else x
        return y * 2.0 + (x * 1.5) / (x * x + 0.5)  # structurally nonzero divisor (the bench asserts err_pc == 0)


def phi_swap_loop(x, n):  # type: ignore[no-untyped-def]
    """
    A while loop whose two carried values genuinely SWAP across the back edge (``a, b = b, a``), producing two
    loop-header phis whose back-edge arms cross-reference each other (phi_a's arm is phi_b and vice versa). The header
    must resolve its phis as a PARALLEL snapshot -- read both old values, then bind both; sequential resolution (bind
    ``a``, then read the new ``a`` for ``b``) would collapse the swap into ``a == b`` and miscompile. A separate counter
    ``n`` drives termination, so the swap is pure and the trip count is the integer part of ``n``. Both the numerical
    model and the MIR interpreter resolve phis in parallel, so this kernel is checked against the float64 Python
    reference -- which swaps correctly -- turning a sequential-phi regression in EITHER oracle into a divergence
    (interp==model alone could not catch it, since a shared sequential bug would still agree). With integer-valued
    inputs every output is exact in the format.
    """
    a = x
    b = x + 1.0
    i = n
    while i > 0.0:
        a, b = b, a
        i = i - 1.0
    return a * 2.0 + b


def overlap_drained_passthrough_kernel(x, y, z):  # type: ignore[no-untyped-def]
    """
    A wide chain ``w`` computed in the overlapping entry block spills past the shrunk terminator into a then arm that
    does NO work and merely passes ``w`` through as the merged value, so ``w`` is the live-out of a fully-DRAINED,
    no-work arm. This exercises the drained-block-receiving-a-spill path and pins the ``term_offset <= drained boundary``
    boundary invariant: the spill lands within the successor's drained-boundary cap because the predecessor's issue-side envelope
    already tracks ``w``'s late write word, so the successor-local spill is only the fixed fetch/latch gap regardless
    of the chain depth (a reviewer hypothesized a chain-depth-scaled spill could exceed the cap; it cannot, and this
    kernel locks that in). The else arm's unspeculatable division keeps the diamond a real branch.
    """
    w = ((((x * z + y) * z + y) * z + y) * z + y) * z + y
    if x < y:
        r = w  # then arm: pass-through, no work; w spills in and is the live-out to the merge
    else:
        r = w / (z * z + 1.0)
    return r * 2.0


def overlap_livein_branch_arm_kernel(x, y, z):  # type: ignore[no-untyped-def]
    """
    The wide chain ``w`` spills from the overlapping entry into an arm that ITSELF branches on a LIVE-IN condition ``c``
    (computed in the entry block, not the arm) -- exercising the overlap interaction the plain dead-arm shape never
    reaches: a block that receives a spill and branches on a RESIDENT live-in condition shrinks its terminator to the
    issue-side envelope (the resident condition adds no read floor) rather than pinning to the drained boundary. Every divisor
    is structurally nonzero, so each diamond stays a real branch.
    """
    c = z > 2.0  # a live-in boolean condition, computed in the entry block (both arms reachable over the input range)
    w = ((((x * z + y) * z + y) * z + y) * z + y) * z + y  # wide chain, spills past the shrunk entry terminator
    if x < y:
        if c:  # this arm branches on the LIVE-IN c while receiving w's spill
            r = w + 1.0
        else:
            r = w / (z * z + 1.0)
    else:
        r = w / (y * y + 1.0)
    return r


class SlotSwap:
    """
    Two persistent slots that SWAP each transaction (``self._a, self._b = self._b, self._a``), forcing the parallel,
    read-first state writeback to exchange their two registers from old values -- the register-swap correctness the
    forward chained-slot SHIFT never exercises. Checked against the float64 reference (Python swaps correctly), so a
    shared sequential-writeback bug in BOTH oracles would still surface as a divergence.
    """

    def __init__(self) -> None:
        self._a = 1.0
        self._b = -2.0

    def step(self, x):  # type: ignore[no-untyped-def]
        old_a = self._a
        old_b = self._b
        self._a = old_b  # swap: a <- old b
        self._b = old_a  # swap: b <- old a
        return old_a * 2.0 + old_b * 4.0 + x  # exact for integer x; reads both OLD slot values to observe the swap

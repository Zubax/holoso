"""
The register allocator, white-box where the annealer is the one place a wrong cost delta hides from every
black-box oracle: the incremental objective against a full recomputation, the descent's local optimality, the
effort-0 contract, the annealer's objective against the emitted mux arms, and the binding's and orientation's
behavior through the model and the emitted machine.
"""

import random
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

import holoso._lir._regalloc as regalloc_module
from holoso import FAddOptions, FCmpOptions, FDivOptions, FloatFormat, FMulOptions, FSortOptions
from holoso._eel import lower
from holoso._lir import (
    BoolOperand,
    BoolRegRef,
    InlineWriteSource,
    Lir,
    OpWriteSource,
    RegRef,
    Boundary,
    WideConstRef,
    write_events,
    write_sources_per_register,
)
from holoso._lir._ir import InPlace
from holoso._lir._sources import read_arms
from holoso._lir._sources import WideOperandTemplate
from holoso._lir._regalloc import (
    ColoringProblem,
    Firing,
    FixedProducer,
    InlineWriter,
    InputWriter,
    MoveWriter,
    RegallocTuning,
    _Snapshot,
    _State,
    _compact,
    _descend,
    _greedy,
    _seed_incidence,
    color,
)
from holoso._mir import lower as lower_to_mir
from holoso._operators import FAddOperator, FCmpOperator, FDivOperator, FMulOperator, SelectOperator
from holoso._operators import BoolInversion, FloatSignControl
from holoso._type import FloatType
from holoso._value import coerce_scalar
from ._modelref import (
    DEFAULT_UNROLL_MAX_TRIPS,
    FROZEN_TUNING,
    SharedLiveOut,
    SharedLiveOutBool,
    assert_model_equals_interpreter,
    build_lir,
    build_model,
    build_model_and_interpreter,
    default_mir,
    default_options,
    mir_options,
    with_instances,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from cordic_sincos import CordicSinCos  # noqa: E402

_FMT = FloatFormat(8, 36)
_FADD = FAddOperator(_FMT, FAddOptions())
_FMUL = FMulOperator(_FMT, FMulOptions(), 0)
_FDIV = FDivOperator(_FMT, FDivOptions())
_SELECT = SelectOperator(FloatType(_FMT))
_SIGN = FloatSignControl()


def _select_writer(rng: random.Random, values: list[int]) -> InlineWriter:
    """A select over a known boolean register and two wide arms, each a value of the bank or a signed pool word."""

    def arm() -> WideOperandTemplate:
        if rng.random() < 0.2:
            return WideOperandTemplate(WideConstRef(rng.randrange(2)), FloatSignControl(negate=rng.random() < 0.5))
        return WideOperandTemplate(rng.choice(values), _SIGN)

    condition = BoolOperand(BoolRegRef(rng.randrange(2)), BoolInversion(rng.random() < 0.5))
    return InlineWriter(_SELECT, (condition, arm(), arm()), _SIGN)


def _synthetic(rng: random.Random, effort: int) -> ColoringProblem:
    """
    A random bank: three pinned inputs, movable values under a sparse interference graph, firings of the three
    operator kinds on two instances (each class realized twice, none co-issued) reading values
    or pool words, each value written by one lane (or by two, as a coalesced class is), inline results on some of
    the values no firing produces (repeating a template on several values, so their keys collide once those values
    share a register), and residual arm moves, some moving a value into itself.
    """
    values = list(range(24))
    inputs, movable = values[:3], values[3:]
    interferes: dict[int, set[int]] = {v: set() for v in values}
    for a in values:
        for b in values:
            if a < b and rng.random() < 0.2:
                interferes[a].add(b)
                interferes[b].add(a)
    firings: list[Firing] = []
    written: set[int] = set()
    for i in range(18):
        operator = rng.choice([_FADD, _FMUL, _FDIV])
        reads = [rng.choice(values) if rng.random() < 0.8 else WideConstRef(rng.randrange(3)) for _ in range(2)]
        unwritten = [v for v in movable if v not in written]
        target = rng.choice(unwritten) if unwritten and rng.random() < 0.85 else rng.choice(movable)
        written.add(target)
        firings.append(
            Firing(
                leader=1000 + i,
                operator=operator,
                block=0,
                issue=i,
                seed_instance=rng.randrange(2),
                reads=reads,
                writes=[(0, target)],
            )
        )
    producers: dict[int, list[FixedProducer]] = {v: [] for v in values}
    producers.update({v: [InputWriter(v)] for v in inputs})
    templates = [_select_writer(rng, values) for _ in range(4)]
    for v in movable:
        if v not in written and rng.random() < 0.7:
            producers[v].append(rng.choice(templates) if rng.random() < 0.5 else _select_writer(rng, values))
        if rng.random() < 0.25:
            source = v if rng.random() < 0.3 else rng.choice(values)
            producers[v].append(MoveWriter(WideOperandTemplate(source, FloatSignControl(negate=rng.random() < 0.5))))
    return ColoringProblem(
        movable=movable,
        pinned={v: v for v in inputs},
        interferes=interferes,
        fixed_producers=producers,
        reserved=frozenset(),
        fresh_start=len(inputs),
        firings=firings,
        instances={_FADD: 2, _FMUL: 2, _FDIV: 2},
        tuning=RegallocTuning(effort=effort, register_price=2.0),
    )


def _seed_state(problem: ColoringProblem) -> _State:
    seed = _greedy(problem, *_seed_incidence(problem))
    return _State(problem, _Snapshot(seed, [False] * len(problem.firings), [f.seed_instance for f in problem.firings]))


def _flippable(problem: ColoringProblem) -> list[int]:
    return [i for i, firing in enumerate(problem.firings) if firing.operator.is_commutative]


def _bindable(problem: ColoringProblem) -> list[int]:
    return [i for i, firing in enumerate(problem.firings) if problem.instances[firing.operator] > 1]


def test_incremental_objective_matches_a_full_recomputation() -> None:
    rng = random.Random(1)
    problem = _synthetic(rng, effort=0)
    state = _seed_state(problem)
    flippable = _flippable(problem)
    bindable = _bindable(problem)
    for step in range(3000):
        before = state.cost
        draw = rng.random()
        if draw < 0.15:
            i = rng.choice(flippable)
            state.flip_firing(i)
            state.flip_firing(i)
            assert state.cost == before
            state.flip_firing(i)
        elif draw < 0.3:
            i = rng.choice(bindable)
            target = 1 - state.instance[i]
            if not state.instance_free(i, target):
                continue
            state.move_instance(i, target)
            state.move_instance(i, 1 - target)
            assert state.cost == before
            state.move_instance(i, target)
        elif draw < 0.4:
            i, j = rng.sample(bindable, 2)
            same_class = problem.firings[i].operator == problem.firings[j].operator
            if not same_class or state.instance[i] == state.instance[j] or not state.swap_instances(i, j):
                continue
            assert state.swap_instances(i, j) and state.cost == before
            assert state.swap_instances(i, j)
        else:
            vid = rng.choice(problem.movable)
            reg = rng.randrange(state.fresh() + 1)
            old = state.assign[vid]
            if reg == old or not state.fits(vid, reg):
                continue
            state.move_value(vid, reg)
            state.move_value(vid, old)
            assert state.cost == before
            state.move_value(vid, reg)
        if step % 100 == 0:
            fresh = _State(problem, state.snapshot())
            assert (state.read_arms, state.write_arms, state.open) == (fresh.read_arms, fresh.write_arms, fresh.open)


@pytest.mark.parametrize("effort", [0, 10])
def test_the_descent_leaves_no_improving_move(effort: int) -> None:
    problem = _synthetic(random.Random(2), effort)
    coloring = color(problem)
    flips = [coloring.swap[firing.leader] for firing in problem.firings]
    instances = [coloring.instance[firing.leader] for firing in problem.firings]
    state = _State(problem, _Snapshot(dict(coloring.assign), flips, instances))
    cost = state.cost
    for vid in problem.movable:
        old = state.assign[vid]
        for reg in [*state.occupied(), state.fresh()]:
            if reg != old and state.fits(vid, reg):
                state.move_value(vid, reg)
                assert state.cost >= cost, (vid, reg)
                state.move_value(vid, old)
    for i in _flippable(problem):
        state.flip_firing(i)
        assert state.cost >= cost, i
        state.flip_firing(i)
    bindable = _bindable(problem)
    for i in bindable:
        old = state.instance[i]
        for target in range(2):
            if target != old and state.instance_free(i, target):
                state.move_instance(i, target)
                assert state.cost >= cost, (i, target)
                state.move_instance(i, old)
    for i in bindable:
        for j in bindable:
            if (
                i < j
                and problem.firings[i].operator == problem.firings[j].operator
                and state.instance[i] != state.instance[j]
            ):
                if state.swap_instances(i, j):
                    assert state.cost >= cost, (i, j)
                    assert state.swap_instances(i, j)


def test_two_singly_written_values_share_a_register_at_the_configured_price() -> None:
    # Two values, each written by its own lane and read by nothing, that do not interfere: apart they cost two open
    # registers (4.0 at price 2), together one register with two writers (one arm plus 2.0). The greedy seed keeps
    # them apart (a fresh register is free of arms); the descent must find the merge, a move an affinity-only
    # candidate list never proposes because neither value's ports or writers reach the other's register.
    problem = ColoringProblem(
        movable=[0, 1],
        pinned={},
        interferes={0: set(), 1: set()},
        fixed_producers={0: [], 1: []},
        reserved=frozenset(),
        fresh_start=0,
        firings=[
            Firing(
                100, _FDIV, block=0, issue=0, seed_instance=0, reads=[WideConstRef(0), WideConstRef(1)], writes=[(0, 0)]
            ),
            Firing(
                101, _FMUL, block=0, issue=1, seed_instance=0, reads=[WideConstRef(0), WideConstRef(1)], writes=[(0, 1)]
            ),
        ],
        instances={_FDIV: 1, _FMUL: 1},
        tuning=RegallocTuning(effort=0, register_price=2.0),
    )
    coloring = color(problem)
    assert (coloring.nreg, coloring.read_arms, coloring.write_arms) == (1, 0, 1)


def test_effort_zero_is_the_descended_greedy_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _synthetic(random.Random(3), effort=0)
    monkeypatch.setattr(regalloc_module, "_anneal", lambda *args: pytest.fail("effort 0 must not anneal"))
    coloring = color(problem)
    state = _seed_state(problem)
    _descend(state, regalloc_module._Decisions.of(problem))
    assign, nreg = _compact(problem, state.assign)
    assert (coloring.assign, coloring.nreg) == (assign, nreg)
    assert coloring.swap == {firing.leader: state.flip[i] for i, firing in enumerate(problem.firings)}


def test_repeated_residual_arms_are_one_writer() -> None:
    # Four constant arms of one phi, two per constant, so the residual installs are two moves each listed twice on the
    # merged register. The emitter keys a move by its operand, so each is one arm, and the phi may then share the
    # input's register once the input is dead; a writer attached twice on one register survived the merge's undo and
    # hid that improvement from the descent. The comparisons come first, so every branching block is empty and no arm
    # threads into it, where the input would still be live.
    def kernel(x: float) -> float:
        below0 = x < 0.0
        below1 = x < 1.0
        below2 = x < 2.0
        if below0:
            return 1.0
        if below1:
            return 2.0
        if below2:
            return 1.0
        return 2.0

    options = replace(default_options(FloatFormat(6, 18)), ifconv_max_ops=0)
    mir = lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options))
    lir = build_lir(mir, "piecewise", FROZEN_TUNING)
    assert len(lir.blocks) > 1, "the premise needs the arms as residual installs, not a select"
    assert lir.regfile.nreg == 1


class _SharedLiveOutResets:
    """Two slots with different resets ending the transaction holding one value."""

    def __init__(self) -> None:
        self.a = 0.0
        self.b = 1.0

    def step(self, x: float, y: float) -> float:
        q = x * y + self.b + self.a
        self.a = q
        self.b = q
        return q * y


def test_shared_live_out_slots_persist_through_the_boundary_install() -> None:
    # The coexistence shape in both banks -- a boundary-installing slot whose own register also takes opcode writes,
    # the premise test_backend's elaboration test pins. Its RTL bench takes the numerical model as its oracle, so
    # the model itself is checked here over several
    # transactions -- the wide kernel against the schedule-independent interpreter (it accumulates, so a float64
    # reference would drift at e6m18), the boolean twin against plain Python (exact).
    fmt = FloatFormat(6, 18)
    model, interpreter = build_model_and_interpreter(SharedLiveOut().step, default_mir(fmt), "shared_live_out", fmt)
    vectors = [
        [coerce_scalar(port.scalar_type, x, port.name) for port in model.inputs] for x in (1.0, 2.5, -3.0, 4.0, 0.5)
    ]
    assert_model_equals_interpreter(model, interpreter, vectors, "shared_live_out")
    bool_lir = build_lir(
        lower_to_mir(lower(SharedLiveOutBool().step, DEFAULT_UNROLL_MAX_TRIPS).hir, default_mir(fmt)), "shared_bool"
    )
    bool_model = build_model(bool_lir)
    reference = SharedLiveOutBool()
    for x, y in [(True, False), (True, True), (False, True), (True, True), (False, False), (True, False), (True, True)]:
        got = {port.name: bool(value) for port, value in zip(bool_model.outputs, bool_model.run(x, y), strict=True)}
        want = reference.step(x, y)
        for name, value in got.items():  # a return aliasing public state is served by that state port alone
            expected = (
                want[int(name[4:])] if name.startswith("out_") else getattr(reference, name.removeprefix("state_"))
            )
            assert value == expected, (x, y, name)


def test_two_slots_ending_on_one_value_hold_it_once_and_copy_once() -> None:
    fmt = FloatFormat(6, 18)
    kernel = _SharedLiveOutResets().step
    lir = build_lir(lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, default_mir(fmt)), "shared_resets")
    assert sum(not isinstance(slot.install, InPlace) for slot in lir.wide_state_slots) == 1
    model, interpreter = build_model_and_interpreter(kernel, default_mir(fmt), "shared_resets", fmt)
    vectors = [
        [coerce_scalar(port.scalar_type, value, port.name) for port in model.inputs]
        for value in (0.5, -1.25, 2.0, 0.0, 3.5)
    ]
    assert_model_equals_interpreter(model, interpreter, vectors, "shared_resets")


def _sorter_witness(a: float, b: float, c: float) -> float:
    x = min(a, b)
    y = max(x, c)
    z = min(y, a)
    return max(z, b)


def _casts_witness(a: float, b: float, c: float) -> float:
    return float(a < b) + float(b < c) + float(c < a)


def _select_witness(x: float) -> float:
    for _ in range(3):
        x = x * 2.0 if x < 1.0 else -0.5
    return x


# The build asserts the wide bank's objective against the emitted arms on every kernel; these witnesses pin that the
# merges the objective must price alike are really exercised. The sorter chain is the lane-keying witness: two of its
# firings land the min and the max lane in one register, which an instance-keyed writer count would price as one
# writer. cordic_sincos is the inline-result witness, keyed by the registers its operands resolve to; the casts are the
# boolean-register witness (two casts of predicates sharing a boolean register into one wide register are one arm,
# which the wide bank sees because the boolean bank is allocated first); the selects are the signed constant witness
# (three selects over `-0.5` and one register are one arm); SharedLiveOut is the slot witness (a boundary-installing
# slot whose register also takes opcode writes).
_WITNESSES: dict[str, tuple[Callable[[], Callable[..., object]], FSortOptions | None]] = {
    "sorter": (lambda: _sorter_witness, FSortOptions()),
    "cordic_sincos": (lambda: CordicSinCos().__call__, None),
    "casts": (lambda: _casts_witness, None),
    "selects": (lambda: _select_witness, None),
    "shared_live_out": (lambda: SharedLiveOut().step, None),
}


def _merged_inline_arms(lir: Lir) -> tuple[int, int]:
    """Wide inline write events against the distinct (register, source) arms they select: fewer arms is the merge."""
    events = [e for e in write_events(lir) if isinstance(e.dst, RegRef) and isinstance(e.source, InlineWriteSource)]
    return len(events), len({(e.dst, e.source) for e in events})


@pytest.mark.parametrize("instances", [1, 2, 3])
@pytest.mark.parametrize("name", list(_WITNESSES))
def test_the_steering_witnesses_exercise_their_merges(name: str, instances: int) -> None:
    # The instance counts reach the annealer's rebinding moves, whose objective the build checks against the arms.
    make_kernel, fsort = _WITNESSES[name]
    options = default_options(_FMT)
    if fsort is not None:
        options = replace(options, operator=replace(options.operator, fsort=fsort))
    options = with_instances(options, instances)
    lir = build_lir(lower_to_mir(lower(make_kernel(), DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), name)
    if name in ("cordic_sincos", "casts", "selects") and instances == 1:
        events, arms = _merged_inline_arms(lir)
        assert events > arms, "the witness needs one emitted arm serving several inline results"
    if name == "selects":
        signed = [
            operand
            for e in write_events(lir)
            if isinstance(e.source, InlineWriteSource)
            for operand in e.source.operands
            if isinstance(operand.source, WideConstRef) and not operand.conditioner.is_identity
        ]
        assert signed, "the witness needs a signed pool word among the merged operands"
    if name == "shared_live_out":
        written = {e.dst for e in write_events(lir)}
        assert any(
            isinstance(slot.install, Boundary) and slot.reg in written for slot in lir.wide_state_slots
        ), "the witness needs a boundary-installing slot whose register also takes opcode writes"
    if name == "sorter":
        lanes = {
            dst: {(source.inst, source.port) for source in sources if isinstance(source, OpWriteSource)}
            for dst, sources in write_sources_per_register(write_events(lir)).items()
        }
        assert any(
            len(ports) > len({inst for inst, _ in ports}) for ports in lanes.values()
        ), "the witness needs a shared lane"


def test_a_swapped_comparator_keeps_every_relation() -> None:
    # Four relations over one operand pair: `a < b` with `a == b`, its inversion `a >= b` (a second firing of the same
    # pair, since a lane writes once per firing), and the mirrored `b < a` (a third firing reading (b, a), which the
    # annealer orients the other way round so every firing reads one register pair per port, its lt tap moving to gt).
    # The model must agree with the interpreter on every relation, equal inputs included, through the permuted lane
    # and the inverted tap.
    def kernel(a: float, b: float) -> tuple[float, float, float, float]:
        return float(a < b), float(a >= b), float(b < a), float(a == b)

    fmt = FloatFormat(6, 18)
    lir = build_lir(
        lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, default_mir(fmt)), "relations", FROZEN_TUNING
    )
    firings = [op for block in lir.blocks for op in block.ops if isinstance(op.inst.operator, FCmpOperator)]
    assert len(firings) == 3, "the premise needs the mirrored comparison as its own firing"
    assert len({tuple(operand.source for operand in op.operands) for op in firings}) == 1, "one firing must be swapped"
    model, interpreter = build_model_and_interpreter(kernel, default_mir(fmt), "relations", fmt, FROZEN_TUNING)
    rng = random.Random(4)
    vectors = []
    for _ in range(64):
        a = rng.uniform(-4.0, 4.0)
        b = a if rng.random() < 0.25 else rng.uniform(-4.0, 4.0)
        vectors.append([coerce_scalar(port.scalar_type, x, port.name) for port, x in zip(model.inputs, (a, b))])
    assert_model_equals_interpreter(model, interpreter, vectors, "relations")


def _k_shared(
    a: float, b: float, c: float, d: float, e: float, f: float, g: float, h: float
) -> tuple[float, float, float]:
    return c * d + a * b, a * e + c * f, c * h + a * g


def _k_chain(a: float, b: float, c: float, d: float, e: float, f: float) -> tuple[float, float]:
    p = a * b
    q = c * d
    return p * e + q * a, q * f + p * c


def _k_saturated(
    a: float, b: float, c: float, d: float, e: float, f: float, g: float, h: float, i: float
) -> tuple[float, float, float]:
    # Three products ready together take the three instances in leader order (p, q, r), and the next three too
    # (c*g, e*h, a*i), pairing every instance with two products sharing nothing; the pairing that shares an operand
    # per instance is a rotation of the second group, reachable only by swaps since both groups are saturated.
    p = a * b
    q = c * d
    r = e * f
    return p + c * g, q + e * h, r + a * i


def _class_read_arms(lir: Lir, mnemonic: str) -> int:
    return sum(max(0, n - 1) for (inst, _), n in read_arms(lir).items() if inst.operator.mnemonic == mnemonic)


# Kernels where the scheduler's first-free binding is provably wrong, with the multiplier read arms the frozen tuning
# reaches; the figures the same annealer reaches with binding moves disabled (the seed binding, orientation on) are
# k_chain 5, k_shared 6, k_saturated 6.
@pytest.mark.parametrize(
    "name,kernel,instances,expected",
    [("k_chain", _k_chain, 2, 2), ("k_shared", _k_shared, 2, 4), ("k_saturated", _k_saturated, 3, 3)],
)
def test_rebinding_beats_the_first_free_seed(
    name: str, kernel: Callable[..., object], instances: int, expected: int
) -> None:
    options = default_options(_FMT)
    options = replace(options, operator=replace(options.operator, fmul=FMulOptions(instances=instances)))
    lir = build_lir(
        lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), name, FROZEN_TUNING
    )
    assert {inst.name for inst in lir.instances if inst.operator.mnemonic == "fmul"} == {
        f"fmul_{i}" for i in range(instances)
    }
    assert _class_read_arms(lir, "fmul") == expected


def test_comparators_swap_and_rebind_together() -> None:
    # Six comparisons over two shared operands on two comparator instances, two of them mirrored: the optimum reads
    # one operand register per port on each instance, which takes both a flip of each mirrored firing and rebinding
    # across the instances (the seed binding, orientation on, reads 6 arms; the annealer reaches 4).
    def kernel(a: float, b: float, c: float, d: float, e: float, f: float, g: float, h: float) -> tuple[float, ...]:
        return float(d < c), float(a < b), float(a < e), float(c < f), float(h < c), float(a < g)

    options = default_options(_FMT)
    options = replace(options, operator=replace(options.operator, fcmp=FCmpOptions(instances=2)))
    lir = build_lir(
        lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), "cmp_shared", FROZEN_TUNING
    )
    assert _class_read_arms(lir, "fcmp") == 4


def test_a_rejected_swap_is_undone_without_assertions() -> None:
    """
    `-O` strips every assert, so the rollback of a rejected instance swap must be a plain call whose result is
    asserted, never the assert itself: with the rollback gone the descent keeps the swap and cycles. Run the
    two-multiplier chain kernel at effort 0 in an optimized interpreter and require it to finish.
    """
    script = (
        "from dataclasses import replace\n"
        "import holoso\n"
        "from tests._modelref import default_options\n"
        "from tests.test_regalloc import _k_chain\n"
        "options = default_options(holoso.FloatFormat(6, 18))\n"
        "options = replace(options, regalloc_effort=0, "
        "operator=replace(options.operator, fmul=holoso.FMulOptions(instances=2)))\n"
        "holoso.synthesize(_k_chain, options)\n"
    )
    try:
        subprocess.run(
            [sys.executable, "-O", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            timeout=120,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the descent did not terminate under -O")

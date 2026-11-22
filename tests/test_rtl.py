"""
The operator blocks in Amaranth simulation: the integer ones against their own references, where a new transaction
enters in every cycle and every output is checked exactly `timing.latency` cycles later, so an immediate, a sign or a
flag delayed apart from its payload fails as surely as a wrong value; and every pooled primitive against the block it
runs on, driven through the ports it declares.
"""

import itertools
import random
import re
from collections.abc import Callable, Sequence

import pytest
import zkf
from amaranth.hdl import ClockDomain, Module, Shape, Value
from amaranth.lib import wiring
from amaranth.sim import Simulator, SimulatorContext

from holoso import FloatFormat, OperatorOptions, Options
from holoso._operators import HardwareOperator, PooledPrimitive
from holoso._operators._int._common import IntBlock
from holoso._operators._int._iabss import IAbss
from holoso._operators._int._iadds import IAdds
from holoso._operators._int._idivs import IDivs
from holoso._operators._int._imuls import IMuls
from holoso._operators._int._ipopcnt import IPopcnt
from holoso._operators._int._ishft import IShft
from holoso._value import FloatValue, IntValue

from . import _options as lean
from ._modelref import build_ops
from ._primitives import concrete_primitive_kinds, primitives_of, random_operand

type Operands = tuple[int, ...]

SMALL_WIDTHS = (2, 3, 4)
"""Simulated exhaustively, as is every unary operator up to `UNARY_EXHAUSTIVE_WIDTH`."""

UNARY_EXHAUSTIVE_WIDTH = 10

WIDTHS = (16, 24, 33, 44)

PORT_STAGES = ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2))
"""`(stage_input, stage_output)` beside the default `(0, 0)`: a register on each side, then one side moved to 0 or 2."""

STAGED_WIDTH = 33

RANDOM_COUNT = 400

WMULTIPLIER = 18


def simulate(
    block: IntBlock,
    transactions: Sequence[Operands],
    driven: Callable[[Operands], Sequence[bool]] | None = None,
    unfrozen: IntBlock | None = None,
) -> None:
    """
    Each transaction is the arguments of `evaluate`: the inputs of the block before it was specialized, `unfrozen`,
    in signature order, of which only those the block still has are driven. `driven` tells which outputs a
    transaction's immediates drive; the others are not checked. The pipeline drains under random inputs, so that the
    last results cannot pass merely because the inputs were held.
    """
    members = block.signature.members
    reference = (block if unfrozen is None else unfrozen).signature.members
    arguments = [name for name, member in reference.items() if member.flow == wiring.In]
    inputs = [name for name, member in members.items() if member.flow == wiring.In]
    outputs = [name for name, member in members.items() if member.flow == wiring.Out]
    assert set(inputs) <= set(arguments)
    assert outputs == [name for name, member in reference.items() if member.flow == wiring.Out]
    timing = block.timing
    assert isinstance(timing, zkf.Timing)
    latency = timing.latency
    rng = random.Random(repr(block.config))
    noise = [
        tuple(noise_value(rng, Shape.cast(reference[name].shape)) for name in arguments) for _ in range(latency + 2)
    ]
    plan = [*transactions, *noise]

    dut = block.component()
    top = Module()
    top.domains.sync = ClockDomain("sync")
    top.submodules.dut = dut
    sim = Simulator(top)
    sim.add_clock(1e-6)
    sampled: list[tuple[int, ...]] = []

    # Processes rather than a testbench: the design then settles once per cycle and not once per input set.
    async def driver(ctx: SimulatorContext) -> None:
        with ctx.critical():
            for operands in plan:
                for name, value in zip(arguments, operands, strict=True):
                    if name in inputs:
                        ctx.set(getattr(dut, name), value)
                await ctx.tick()

    async def monitor(ctx: SimulatorContext) -> None:
        async for _, _, *values in ctx.tick().sample(*(getattr(dut, name) for name in outputs)):
            sampled.append(tuple(values))

    sim.add_process(driver)
    sim.add_process(monitor)
    sim.run()

    assert len(sampled) >= len(transactions) + latency
    for index, operands in enumerate(transactions):
        result = block.evaluate(*operands)
        expected = tuple(result) if isinstance(result, tuple) else (result,)
        checked = driven(operands) if driven is not None else (True,) * len(outputs)
        want = {name: int(value) for name, value, check in zip(outputs, expected, checked, strict=True) if check}
        got = {name: value for name, value, check in zip(outputs, sampled[index + latency], checked) if check}
        assert got == want, f"{block.config}: transaction {index} {dict(zip(arguments, operands))}"


def noise_value(rng: random.Random, shape: Shape) -> int:
    bits = rng.getrandbits(shape.width)
    return bits - (1 << shape.width) if shape.signed and bits >> (shape.width - 1) else bits


def word_range(w: int) -> range:
    return range(-(1 << (w - 1)), 1 << (w - 1))


def edges(w: int) -> list[int]:
    low, high = -(1 << (w - 1)), (1 << (w - 1)) - 1
    return sorted({0, 1, -1, low, high, low + 1, high - 1})


def word(rng: random.Random, w: int) -> int:
    """An edge value, uniform bits, or a uniform bit length, in equal shares: uniform bits alone are nearly all huge."""
    match rng.randrange(3):
        case 0:
            return rng.choice(edges(w))
        case 1:
            return rng.randint(-(1 << (w - 1)), (1 << (w - 1)) - 1)
        case _:
            magnitude = rng.getrandbits(rng.randrange(w))
            return -magnitude if rng.getrandbits(1) else magnitude


def unary(w: int, extra: Sequence[int] = ()) -> list[Operands]:
    if w <= UNARY_EXHAUSTIVE_WIDTH:
        return [(x,) for x in word_range(w)]
    rng = random.Random(w)
    return [(x,) for x in (*edges(w), *extra, *(word(rng, w) for _ in range(RANDOM_COUNT)))]


def binary(w: int, extra: Sequence[tuple[int, int]] = ()) -> list[tuple[int, int]]:
    if w in SMALL_WIDTHS:
        return list(itertools.product(word_range(w), word_range(w)))
    rng = random.Random(w)
    pairs = [*itertools.product(edges(w), edges(w)), *extra]
    return pairs + [(word(rng, w), word(rng, w)) for _ in range(RANDOM_COUNT)]


def with_immediate(pairs: Sequence[tuple[int, int]], seed: int) -> list[Operands]:
    """Each pair under both values of a one-bit immediate, shuffled so that the immediate changes at random cycles."""
    transactions: list[Operands] = [(immediate, a, b) for a, b in pairs for immediate in (0, 1)]
    random.Random(seed).shuffle(transactions)
    return transactions


PMUL_REGISTERS = {2: 3, 3: 3, 4: 3, 16: 3, 24: 4, 33: 4, 44: 4}
"""
The register positions of each word's product grid on 18-bit multipliers, which `zkf.Pmul` takes in order: a word
within one multiplier has the output, the operand capture and the product; a wider one adds a level of the grid's sum.
"""


# ---------------------------------------------------------------------------------------------------------------------


def iadds_transactions(w: int) -> list[Operands]:
    low, high = -(1 << (w - 1)), (1 << (w - 1)) - 1
    rng = random.Random(w)
    equal = [(x, x) for x in edges(w)] + [(x, x) for x in (word(rng, w) for _ in range(20))]
    return with_immediate(binary(w, [(low, high), (high, low), *equal]), w)


def iadds_driven(operands: Operands) -> tuple[bool, ...]:
    sub = bool(operands[0])
    return True, sub, sub, sub


@pytest.mark.parametrize("fast", (False, True))
@pytest.mark.parametrize("w", (*SMALL_WIDTHS, *WIDTHS))
def test_iadds(w: int, fast: bool) -> None:
    simulate(IAdds(wint=w, fast=fast), iadds_transactions(w), iadds_driven)


@pytest.mark.parametrize("stage_input,stage_output", PORT_STAGES)
def test_iadds_stages(stage_input: int, stage_output: int) -> None:
    block = IAdds(wint=STAGED_WIDTH, stage_input=stage_input, stage_output=stage_output)
    simulate(block, iadds_transactions(STAGED_WIDTH), iadds_driven)


def test_iadds_combined() -> None:
    block = IAdds(wint=24, stage_input=2, stage_output=0, fast=True)
    simulate(block, iadds_transactions(24), iadds_driven)


@pytest.mark.parametrize(
    "block",
    [IAdds(wint=8, stage_input=2).specialized(sub=False), IShft(wint=8, stage_input=2).specialized(right=True)],
    ids=["iadds", "ishft"],
)
def test_a_held_immediate_takes_no_register(block: IntBlock) -> None:
    # A value that never changes is the same at every stage, so no stage registers it.
    verilog = block.compile()
    assert not re.search(r"\b(sub|right)_in_q", verilog), "the immediate rides the input stages"


@pytest.mark.parametrize("sub", (False, True))
@pytest.mark.parametrize("w,stage_input,stage_output", [(3, 0, 0), (STAGED_WIDTH, 0, 0), (STAGED_WIDTH, 2, 1)])
def test_iadds_specialized(w: int, stage_input: int, stage_output: int, sub: bool) -> None:
    # The block an instance renders when all its firings add, or all subtract: `sub` is held inside, not driven.
    block = IAdds(wint=w, stage_input=stage_input, stage_output=stage_output)
    transactions = [operands for operands in iadds_transactions(w) if operands[0] == sub]
    simulate(block.specialized(sub=sub), transactions, iadds_driven, unfrozen=block)


# ---------------------------------------------------------------------------------------------------------------------


def imuls_transactions(w: int) -> list[Operands]:
    # Each pair of powers of two lands exactly on the rail: past MAX by one, or on MIN exactly.
    rails = [(sa << k, sb << (w - 1 - k)) for k in range(1, w - 1) for sa in (1, -1) for sb in (1, -1)]
    return list(binary(w, rails))


IMULS_CASES = [(w, stages) for w in (*SMALL_WIDTHS, *WIDTHS) for stages in range(PMUL_REGISTERS[w] + 1)]


@pytest.mark.parametrize("w,stage_core_mul", IMULS_CASES)
def test_imuls(w: int, stage_core_mul: int) -> None:
    simulate(IMuls(wint=w, wmultiplier=WMULTIPLIER, stage_core_mul=stage_core_mul), imuls_transactions(w))


@pytest.mark.parametrize("stage_input,stage_output", PORT_STAGES)
def test_imuls_stages(stage_input: int, stage_output: int) -> None:
    block = IMuls(wint=STAGED_WIDTH, wmultiplier=WMULTIPLIER, stage_input=stage_input, stage_output=stage_output)
    simulate(block, imuls_transactions(STAGED_WIDTH))


def test_imuls_combined() -> None:
    """Narrow multipliers make a grid of many products, every one of its six registers taken."""
    w, wmultiplier = 33, 9
    block = IMuls(wint=w, wmultiplier=wmultiplier, stage_core_mul=6, stage_input=0, stage_output=2)
    simulate(block, imuls_transactions(w))


# ---------------------------------------------------------------------------------------------------------------------


def idivs_transactions(w: int) -> list[Operands]:
    if w in SMALL_WIDTHS:
        return list(binary(w))
    rng = random.Random(w)
    high = (1 << (w - 1)) - 1
    exact = []  # a zero remainder in each combination of signs
    for _ in range(40):
        den = rng.getrandbits(rng.randrange(1, w - 1)) or 1
        quo = rng.randint(0, high // den)
        exact += [(sn * quo * den, sd * den) for sn in (1, -1) for sd in (1, -1)]
    return list(binary(w, exact))


@pytest.mark.parametrize("w", (*SMALL_WIDTHS, *WIDTHS))
def test_idivs(w: int) -> None:
    simulate(IDivs(wint=w), idivs_transactions(w))


@pytest.mark.parametrize("stage_input,stage_output", PORT_STAGES)
def test_idivs_stages(stage_input: int, stage_output: int) -> None:
    block = IDivs(wint=STAGED_WIDTH, stage_input=stage_input, stage_output=stage_output)
    simulate(block, idivs_transactions(STAGED_WIDTH))


def test_idivs_combined() -> None:
    block = IDivs(wint=24, stage_input=2, stage_output=0, keep_attrs={"keep": "true"})
    simulate(block, idivs_transactions(24))


# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("w", (*SMALL_WIDTHS, *WIDTHS))
def test_iabss(w: int) -> None:
    simulate(IAbss(wint=w), unary(w))


@pytest.mark.parametrize("stage_input,stage_output", PORT_STAGES)
def test_iabss_stages(stage_input: int, stage_output: int) -> None:
    simulate(IAbss(wint=STAGED_WIDTH, stage_input=stage_input, stage_output=stage_output), unary(STAGED_WIDTH))


def test_iabss_combined() -> None:
    simulate(IAbss(wint=44, stage_input=2, stage_output=2), unary(44))


# ---------------------------------------------------------------------------------------------------------------------


def ishft_transactions(w: int) -> list[Operands]:
    if w in SMALL_WIDTHS:
        return with_immediate(binary(w), w)
    rng = random.Random(w)
    # Every amount out to twice the power of two past the width, which covers both ends of the count field.
    bound = 2 << (w - 1).bit_length()
    amounts = [s for s in range(-bound, bound + 1) if s in word_range(w)]
    swept = [(x, s) for x in [*edges(w), word(rng, w), word(rng, w)] for s in amounts]
    return with_immediate(binary(w, swept), w)


@pytest.mark.parametrize("w", (*SMALL_WIDTHS, *WIDTHS, 64))
def test_ishft(w: int) -> None:
    simulate(IShft(wint=w), ishft_transactions(w))


@pytest.mark.parametrize("stage_input,stage_output", PORT_STAGES)
def test_ishft_stages(stage_input: int, stage_output: int) -> None:
    block = IShft(wint=STAGED_WIDTH, stage_input=stage_input, stage_output=stage_output)
    simulate(block, ishft_transactions(STAGED_WIDTH))


def test_ishft_combined() -> None:
    simulate(IShft(wint=24, stage_input=2, stage_output=2), ishft_transactions(24))


@pytest.mark.parametrize("right", (False, True))
@pytest.mark.parametrize("w,stage_input,stage_output", [(3, 0, 0), (STAGED_WIDTH, 0, 0), (STAGED_WIDTH, 2, 1)])
def test_ishft_specialized(w: int, stage_input: int, stage_output: int, right: bool) -> None:
    # The block an instance renders when all its firings shift one way: `right` is held inside, not driven.
    block = IShft(wint=w, stage_input=stage_input, stage_output=stage_output)
    transactions = [operands for operands in ishft_transactions(w) if operands[0] == right]
    simulate(block.specialized(right=right), transactions, unfrozen=block)


# ---------------------------------------------------------------------------------------------------------------------


def ipopcnt_transactions(w: int) -> list[Operands]:
    rng = random.Random(w)
    powers = [s * m << k for k in range(w) for m in (1, 3) for s in (1, -1)]
    odd_multiples = [-((2 * rng.getrandbits(w - 2) + 1) << k) for k in range(w - 1)]
    return unary(w, [v for v in powers + odd_multiples if v in word_range(w)])


@pytest.mark.parametrize("w", (*SMALL_WIDTHS, 5, 10, *WIDTHS, 64))
def test_ipopcnt(w: int) -> None:
    """The widths reach every shape of the reduction: a lone counter, a bare pair, carry-save levels, many counters."""
    simulate(IPopcnt(wint=w), ipopcnt_transactions(w))


@pytest.mark.parametrize("stage_input,stage_output", PORT_STAGES)
def test_ipopcnt_stages(stage_input: int, stage_output: int) -> None:
    block = IPopcnt(wint=STAGED_WIDTH, stage_input=stage_input, stage_output=stage_output)
    simulate(block, ipopcnt_transactions(STAGED_WIDTH))


def test_ipopcnt_combined() -> None:
    simulate(IPopcnt(wint=64, stage_input=2, stage_output=0), ipopcnt_transactions(64))


# ---------------------------------------------------------------------------------------------------------------------


def test_default_latency() -> None:
    """The defaults register nothing at the ports, leaving every block but the divider combinational."""
    w = 33
    assert IAdds(wint=w).timing == zkf.Timing(0, 1)
    assert IMuls(wint=w, wmultiplier=WMULTIPLIER).timing == zkf.Timing(0, 1)
    assert IDivs(wint=w).timing == zkf.Timing(1 + (w + 1) // 2, 1)
    assert IAbss(wint=w).timing == zkf.Timing(0, 1)
    assert IShft(wint=w).timing == zkf.Timing(0, 1)
    assert IPopcnt(wint=w).timing == zkf.Timing(0, 1)


@pytest.mark.parametrize(
    "block",
    [
        IAdds(wint=12, fast=True),
        IMuls(wint=12, wmultiplier=WMULTIPLIER, stage_core_mul=2),
        IDivs(wint=12),
        IAbss(wint=12),
        IShft(wint=12),
        IPopcnt(wint=12),
    ],
    ids=lambda block: block.operator,
)
def test_compile(block: IntBlock) -> None:
    assert f"module _k_zkf_{block.operator}_" in block.compile(prefix="_k_")


def test_idivs_keep_attrs_name_the_module_but_not_the_configuration() -> None:
    plain = IDivs(wint=12)
    kept = IDivs(wint=12, keep_attrs={"keep": "true"})
    assert kept == plain and kept.name != plain.name


# ---------------------------------------------------------------------------------------------------------------------


def _every_pooled_primitive() -> list[PooledPrimitive]:
    float_options = {name: getattr(lean, name)() for name in ("fadd", "fmul", "fdivsqrt", "fmul_ilog2", "filog2")}
    float_options |= {name: getattr(lean, name)() for name in ("fcmp", "frint", "ffma", "fexp2", "flog2", "fcordic")}
    options = Options(
        operator=OperatorOptions(**float_options, ffromint=lean.ffromint()),
        ffmt=FloatFormat(6, 18),
        wint_min=33,
        wmultiplier=18,
    )
    ops = build_ops(options, options.wint_min)
    operators: list[HardwareOperator] = [
        ops.fadd,
        ops.fmul,
        ops.fdivsqrt,
        ops.filog2,
        ops.fmul_ilog2,
        ops.fcmp,
        ops.frint,
        ops.ffma,
    ]
    operators += [ops.fexp2, ops.flog2, ops.fcordic, ops.ffromint]
    operators += [ops.iadds, ops.imuls, ops.idivs, ops.iabss, ops.ishft, ops.ipopcnt]
    return [primitive for operator in operators for primitive in primitives_of(operator)]


def _rtl_answer(primitive: PooledPrimitive, operands: Sequence[FloatValue | IntValue]) -> int:
    """
    The bits the operator's block answers on the primitive's output port, issued once with its operands on the operand
    ports its drive declares and its immediates on theirs, the inputs it leaves undeclared held at zero.
    """
    dut = primitive.operator.block.component()
    members = primitive.operator.block.signature.members
    inputs = {name: operand.bits for name, operand in zip(primitive.drive.operand_ports, operands, strict=True)}
    inputs |= {name: int(value) for name, value in primitive.immediates.items()}
    inputs |= {name: 1 for name in ("in_valid", "out_ready") if name in members}
    output = Value.cast(getattr(dut, primitive.output_name))
    top = Module()
    top.domains.sync = ClockDomain("sync")
    top.submodules.dut = dut
    sim = Simulator(top)
    sim.add_clock(1e-6)
    answer: list[int] = []

    async def bench(ctx: SimulatorContext) -> None:
        for name, value in inputs.items():
            ctx.set(Value.cast(getattr(dut, name)), value)
        for _ in range(primitive.latency):
            await ctx.tick()
            if "in_valid" in members:
                ctx.set(dut.in_valid, 0)
        answer.append(ctx.get(output) & ((1 << len(output)) - 1))

    sim.add_testbench(bench)
    sim.run()
    return answer[0]


def test_every_primitive_answers_as_the_block_wired_by_its_declaration() -> None:
    # The emitter wires a firing's operands to the operand ports its drive declares and lands the result from the output
    # port its primitive names, while `evaluate` calls the block's reference directly; this ties the two together, so a
    # port declared in the wrong order or a result read off the wrong port cannot pass the numerical model.
    primitives = _every_pooled_primitive()
    assert {type(primitive).__name__ for primitive in primitives} == concrete_primitive_kinds()
    rng = random.Random(0)
    for primitive in primitives:
        for _ in range(3):
            operands = [random_operand(rng, scalar_type) for scalar_type in primitive.signature.operand_types]
            want = primitive.evaluate(*operands)
            bits = int(want) if isinstance(want, bool) else want.bits
            assert _rtl_answer(primitive, operands) == bits, (primitive, operands)

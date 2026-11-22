"""Elaboration tests for the generated Verilog backend (structural correctness under Icarus)."""

import dataclasses
import itertools
import math
import random
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import zkf
from amaranth.hdl import Shape
import pytest

from holoso import (
    BoolType,
    FDivsqrtOptions,
    FFromIntOptions,
    FloatFormat,
    FloatValue,
    FMulILog2Options,
    IAbssOptions,
    IAddsOptions,
    IDivsOptions,
    IMulsOptions,
    IntFormat,
    IntValue,
    IPopcntOptions,
    IShftOptions,
    OperatorOptions,
    Options,
    UnsupportedConstruct,
    synthesize,
)
from holoso._operators import (
    FAtan2Primitive,
    FCosPrimitive,
    FHypotPrimitive,
    FMaxPrimitive,
    FMinPrimitive,
    FCordicOperator,
    FDivPrimitive,
    FMulILog2Operator,
    FSinPrimitive,
    FSqrtPrimitive,
    FRintPrimitive,
    HardwareOperator,
    IAddPrimitive,
    IShlPrimitive,
    IShrPrimitive,
    ICmpPrimitive,
    ISubPrimitive,
)
from holoso._operators._float import FCmpOperator, FDivsqrtOperator, FFromIntOperator, FILog2Operator, FRintOperator
from holoso._operators._int import (
    IAbsOperator,
    IAddOperator,
    IDivOperator,
    IMulOperator,
    IPopcntOperator,
    IShftOperator,
)
from holoso._backend.verilog._microcode import (
    build_microcode,
    f_input,
    read_codebook,
    immediate_inputs,
    write_codebook,
)
from holoso import SynthesisResult
from holoso._type import FloatType, ScalarType
from holoso._backend.verilog import generate
from holoso._eel import lower
from holoso._lir import BoolRegRef, Boundary, Lir, RegRef, WideStateSlot, write_events
from holoso._mir import MirOptions, Mir, lower as lower_to_mir

from ._modelref import (
    build_lir,
    build_ops,
    mir_options,
    default_ifmt,
    DEFAULT_UNROLL_MAX_TRIPS,
    SharedLiveOut,
    SharedLiveOutBool,
)
from . import _options as lean
from ._primitives import concrete_primitive_kinds, primitives_of, random_operand

_requires_iverilog = pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")


def _ops(fmt: FloatFormat) -> MirOptions:
    return mir_options(
        Options(
            operator=OperatorOptions(
                fadd=lean.fadd(),
                fmul=lean.fmul(),
                fdivsqrt=lean.fdivsqrt(),
                fmul_ilog2=lean.fmul_ilog2(),
                fcmp=lean.fcmp(),
            ),
            ffmt=fmt,
            wmultiplier=18,
        )
    )


def _run(target: object, ops: MirOptions, fmt: FloatFormat) -> Mir:
    return lower_to_mir(lower(target, DEFAULT_UNROLL_MAX_TRIPS).hir, ops)


def _compile(name: str, verilog: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    vpath = tmp_path / f"{name}.v"
    vpath.write_text(verilog)
    cmd = ["iverilog", "-g2005", "-s", name, "-o", str(tmp_path / f"{name}.out"), str(vpath)]
    return subprocess.run(cmd, capture_output=True, text=True)


def _elaborate(name: str, verilog: str, tmp_path: Path) -> None:
    result = _compile(name, verilog, tmp_path)
    assert result.returncode == 0, result.stderr


def test_operator_instance_names_are_mnemonic_and_copy_index() -> None:
    def scale(a: float, b: float) -> float:
        return a * 4.0 + b * 8.0

    fmt = FloatFormat(6, 18)
    lir = build_lir(_run(scale, _ops(fmt), fmt), "scale")
    names = re.findall(r"^_scale_zkf_mul_ilog2_[0-9a-f]{12} u_(\w+) \(", generate(lir, {}).verilog, re.MULTILINE)
    assert names == ["fmul_ilog2_0"]  # both exponents ride one pooled scaler, named by mnemonic and copy index


@_requires_iverilog
def test_comparisons_share_one_pooled_fcmp_instance() -> None:
    # Comparisons live in mutually-exclusive blocks and execute sequentially, so they share a single comparator
    # (the one-instance-per-operator pooling convention), its operands riding the ordinary microcode read-mux
    # lanes -- not one instance per comparison.
    def kernel(x: float) -> float:
        if x > 1.0:
            y = x + 1.0
        elif x < -1.0:
            y = x - 1.0
        else:
            y = x
        return y

    verilog = generate(build_lir(_run(kernel, _ops(FloatFormat(8, 24)), FloatFormat(8, 24)), "two_cmp"), {}).verilog
    assert len(re.findall(r" u_fcmp_\d+ \($", verilog, re.MULTILINE)) == 1


def _integer_operators(ifmt: IntFormat) -> list[HardwareOperator]:
    return [
        IAddOperator.build(ifmt, IAddsOptions()),
        IAddOperator.build(ifmt, IAddsOptions(fast=True)),
        IDivOperator.build(ifmt, IDivsOptions()),
        IAbsOperator.build(ifmt, IAbssOptions()),
        IShftOperator.build(ifmt, IShftOptions()),
        IPopcntOperator.build(ifmt, IPopcntOptions()),
        *(IMulOperator.build(ifmt, IMulsOptions(stage_core_mul=stages), 18) for stages in range(4)),
    ]


def _mixed_format_operators(ffmt: FloatFormat, ifmt: IntFormat) -> list[HardwareOperator]:
    return [
        FFromIntOperator.build(ffmt, ifmt, lean.ffromint()),
        FFromIntOperator.build(
            ffmt,
            ifmt,
            lean.ffromint(
                stage_input=1,
                stage_output=1,
                stage_core=FFromIntOptions.CoreStages.NORMALIZE | FFromIntOptions.CoreStages.PACK,
            ),
        ),
        FRintOperator.build(ffmt, ifmt, lean.frint()),
        FRintOperator.build(ffmt, ifmt, lean.frint(stage_input=2)),
        FMulILog2Operator.build(ffmt, ifmt, lean.fmul_ilog2()),
        FMulILog2Operator.build(
            ffmt, ifmt, lean.fmul_ilog2(stage_input=1, stage_core=FMulILog2Options.CoreStages.DECODE)
        ),
        FILog2Operator.build(ffmt, ifmt, lean.filog2()),
    ]


def _contract_operators() -> list[HardwareOperator]:
    ffmt, ifmt = FloatFormat(6, 18), IntFormat(24)
    floats = OperatorOptions(
        fadd=lean.fadd(), fmul=lean.fmul(), fcmp=lean.fcmp(), ffma=lean.ffma(), fexp2=lean.fexp2(), flog2=lean.flog2()
    )
    ops = build_ops(Options(operator=floats, ffmt=ffmt, wmultiplier=18), ifmt.width)
    first_digit = FDivsqrtOptions.CoreStages.FIRST_DIGIT
    return [
        ops.fadd,
        ops.fmul,
        ops.fcmp,
        ops.ffma,
        ops.fexp2,
        ops.flog2,
        FDivsqrtOperator.build(ffmt, lean.fdivsqrt()),
        FDivsqrtOperator.build(ffmt, lean.fdivsqrt(fast=True, stage_input=2, stage_output=1, stage_core=first_digit)),
        *(
            FCordicOperator.build(ffmt, lean.fcordic(**knobs), 18)
            for knobs in ({}, {"stage_input": 1}, {"stage_core_mul": 2})
        ),
        *_mixed_format_operators(ffmt, ifmt),
        *_integer_operators(ifmt),
    ]


def test_an_instance_block_serves_exactly_what_its_primitives_vary_on() -> None:
    # Whatever subset of an operator's primitives an instance runs, its block keeps an immediate exactly where they
    # disagree on it and every other port as the unspecialized block has it (the microcode encodes a kept immediate by
    # that shape, and the emitter wires every error port), and it answers and times each of them as the block built
    # for that primitive alone does. The drives of an operator keep one timing or one transaction in flight at each.
    rng = random.Random(0)
    operators = _contract_operators()
    assert {type(p).__name__ for operator in operators for p in primitives_of(operator)} == concrete_primitive_kinds()
    for operator in operators:
        primitives = primitives_of(operator)
        unspecialized = operator.block.signature.members
        for size in range(1, len(primitives) + 1):
            for subset in itertools.combinations(primitives, size):
                where = (operator.name, operator.block.name, [type(primitive).__name__ for primitive in subset])
                block = operator.instance_block(frozenset(subset))
                members = block.signature.members
                varied = {name for name in operator.immediate_ports if len({p.immediates[name] for p in subset}) > 1}
                assert {name for name in operator.immediate_ports if name in members} == varied, where
                assert all(name in members for name in operator.error_ports), where
                for name, member in members.items():
                    assert (member.flow, member.shape) == (unspecialized[name].flow, unspecialized[name].shape), where
                timing = block.timing
                for primitive in subset:
                    drive = primitive.drive
                    if isinstance(timing, zkf.Timing):
                        assert timing == drive.timing, where
                    else:
                        (kept,) = varied
                        assert timing[primitive.immediates[kept]] == drive.timing, where
                    assert all(port in members for port in drive.operand_ports), where
                    assert primitive.output_name in members, where
                    for _ in range(2):
                        operands = [random_operand(rng, ty) for ty in primitive.signature.operand_types]
                        assert primitive.evaluate(*operands, block=block) == primitive.evaluate(*operands), where
        timings = {primitive.drive.timing for primitive in primitives}
        assert len(timings) == 1 or all(t.initiation_interval >= t.latency >= 1 for t in timings), operator.name


def _net(scalar_type: ScalarType) -> str:
    return f"[{scalar_type.width - 1}:0] " if scalar_type.is_wide else ""


def _renderable_blocks(operator: HardwareOperator) -> list[zkf.Block]:
    """Every block an instance of `operator` can render: its own, and the one for each set of primitives it may run."""
    primitives = primitives_of(operator)
    subsets = (frozenset(s) for size in range(1, len(primitives) + 1) for s in itertools.combinations(primitives, size))
    return [operator.block, *(operator.instance_block(subset) for subset in subsets)]


def _pooled_probe(name: str, operators: list[HardwareOperator]) -> str:
    """
    A module instantiating each block an instance of each operator can render (`_renderable_blocks`) through the ports,
    widths, immediates and error ports the operator declares for itself, so a port it declares and its block lacks
    fails right here, followed by the blocks' converted modules.
    """
    blocks: dict[str, tuple[HardwareOperator, zkf.Block]] = {}
    for operator in operators:
        for block in _renderable_blocks(operator):
            blocks.setdefault(block.name, (operator, block))
    lines = [f"module {name};", "    wire clk = 1'b0;", "    wire rst = 1'b0;"]
    for index, (operator, block) in enumerate(blocks.values()):
        members = block.signature.members
        connections = [".clk(clk)", ".rst(rst)"] if block.clocked else []
        connections += [f".{port}(1'b0)" for port in ("in_valid", "out_ready") if port in members]
        for port, ty in operator.operand_ports.items():
            if port in members:
                lines.append(f"    wire {_net(ty)}u{index}_{port} = {ty.width}'d0;")
                connections.append(f".{port}(u{index}_{port})")
        for port, ty in operator.output_ports.items():
            if port in members:
                lines.append(f"    wire {_net(ty)}u{index}_{port};")
                connections.append(f".{port}(u{index}_{port})")
        for port in (name for name in operator.immediate_ports if name in members):
            connections.append(f".{port}({Shape.cast(members[port].shape).width}'d0)")
        for port in operator.error_ports:
            lines.append(f"    wire u{index}_{port};")
            connections.append(f".{port}(u{index}_{port})")
        lines.append(f"    _{name}_{block.name} u{index} ({', '.join(connections)});")
    return "\n".join([*lines, "endmodule", "", *(block.compile(prefix=f"_{name}_") for _, block in blocks.values())])


@_requires_iverilog
@pytest.mark.parametrize("width", (2, 3, 24, 33, 44))
def test_integer_operators_render_the_ports_they_declare(width: int, tmp_path: Path) -> None:
    # An odd width is mandatory because that is where the divider's ceiling can slip.
    name = f"int_probe_w{width}"
    _elaborate(name, _pooled_probe(name, _integer_operators(IntFormat(width))), tmp_path)


@_requires_iverilog
@pytest.mark.parametrize("wexp,wman,wint", ((6, 18, 44), (8, 36, 24), (3, 4, 12), (6, 18, 17)))
def test_mixed_format_operators_render_the_ports_they_declare(wexp: int, wman: int, wint: int, tmp_path: Path) -> None:
    # Several triples because the integer side is sized independently of the float one.
    name = f"mixed_probe_e{wexp}m{wman}i{wint}"
    operators = _mixed_format_operators(FloatFormat(wexp, wman), IntFormat(wint))
    _elaborate(name, _pooled_probe(name, operators), tmp_path)


@_requires_iverilog
@pytest.mark.parametrize("wexp,wman,fast", ((6, 18, True), (8, 24, True), (8, 24, False), (3, 5, False), (8, 37, True)))
def test_fdivsqrt_renders_the_ports_it_declares(wexp: int, wman: int, fast: bool, tmp_path: Path) -> None:
    # Both significand parities and both recurrences, which the kernel-level and cocotb coverage never all reach: the
    # core's digit count, the reach of its first-digit stage and the frame of each single-operation build depend on
    # WMAN's parity, and the radix-8 recurrence (`fast`) does not take a significand below 9 bits.
    name = f"fdivsqrt_probe_e{wexp}m{wman}_f{int(fast)}"
    fmt = FloatFormat(wexp, wman)
    first_digit = FDivsqrtOptions.CoreStages.FIRST_DIGIT
    operators: list[HardwareOperator] = [
        FDivsqrtOperator.build(fmt, lean.fdivsqrt(fast=fast)),
        FDivsqrtOperator.build(fmt, lean.fdivsqrt(fast=fast, stage_core=first_digit)),
        FDivsqrtOperator.build(fmt, lean.fdivsqrt(fast=fast, stage_input=2, stage_output=1, stage_core=first_digit)),
    ]
    _elaborate(name, _pooled_probe(name, operators), tmp_path)


@_requires_iverilog
def test_cordic_renders_the_ports_it_declares(tmp_path: Path) -> None:
    name = "fcordic_probe"
    _elaborate(name, _pooled_probe(name, [FCordicOperator.build(FloatFormat(6, 18), lean.fcordic(), 18)]), tmp_path)


def _rotating(x: float) -> tuple[float, float]:
    return math.sin(x), math.cos(x)


def _vectoring(y: float, x: float) -> tuple[float, float]:
    return math.atan2(y, x), math.hypot(y, x)


def _rotating_a_vectored_angle(y: float, x: float) -> float:
    return math.cos(math.atan2(y, x))


def _rotating_beside_vectoring(a: float, y: float, x: float) -> tuple[float, float]:
    return math.sin(a), math.atan2(y, x)


def _two_angles_one_measured(p: float, q: float, y: float, x: float) -> tuple[float, float, float]:
    return math.atan2(p, q), math.atan2(y, x), math.hypot(y, x)


def _rotating_beside_a_measured_angle(a: float, y: float, x: float) -> tuple[float, float, float]:
    return math.sin(a), math.atan2(y, x), math.hypot(y, x)


def _mirrored_angles_one_measured(y: float, x: float) -> tuple[float, float, float]:
    return math.atan2(y, x), math.atan2(-y, x), math.hypot(y, x)


@pytest.mark.parametrize(
    "kernel, instances, builds",
    [
        (_rotating, 1, [(False, True)]),
        (_vectoring, 1, [(True, True)]),
        (_rotating_a_vectored_angle, 1, [(None, False)]),
        (_rotating_beside_vectoring, 2, [(False, True), (True, False)]),
        (_two_angles_one_measured, 1, [(True, True)]),
        (_two_angles_one_measured, 2, [(True, True), (True, False)]),
        (_rotating_beside_a_measured_angle, 1, [(None, True)]),
        (_mirrored_angles_one_measured, 2, [(True, True), (True, False)]),
    ],
    ids=lambda value: value.__name__.strip("_") if callable(value) else None,
)
def test_a_cordic_instance_renders_the_block_its_firings_need(
    kernel: Callable[..., object], instances: int, builds: list[tuple[bool | None, bool]]
) -> None:
    # A rotation and a vectoring issued together take an instance each, so each renders a block for its own drive
    # alone; an instance running both keeps the immediate. A vectoring whose magnitude nothing reads leaves it out, so
    # an instance builds the magnitude exactly where one of its firings needs it -- neither more (wasted fabric) nor
    # less.
    # Each `(vectoring, length)` is the ZKF specialization the instance must render.
    options = Options(
        operator=OperatorOptions(
            fmul=lean.fmul(), fmul_ilog2=lean.fmul_ilog2(), fcordic=lean.fcordic(instances=instances)
        ),
        ffmt=FloatFormat(6, 18),
        wmultiplier=18,
    )
    name = f"cordic_block_{kernel.__name__.strip('_')}_x{instances}"
    lir = build_lir(lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), name)
    cordics = [(inst, block) for inst, block in lir.instances.items() if inst.operator.name == "fcordic"]
    operator = cordics[0][0].operator
    assert isinstance(operator, FCordicOperator)
    expected = [operator.block.specialized(vectoring=vectoring, length=length) for vectoring, length in builds]
    assert sorted(block.name for _, block in cordics) == sorted(block.name for block in expected)


@pytest.mark.parametrize("stages", [0, 1])
def test_each_cordic_primitive_runs_at_the_timing_of_its_own_drive(stages: int) -> None:
    # A primitive's timing is that of the block built for its drive alone; an instance that both rotates and vectors
    # runs the block built for both, so scheduling rests on ZKF timing each drive alike in both builds.
    operator = FCordicOperator.build(FloatFormat(6, 18), lean.fcordic(stage_input=stages), 18)
    timing = operator.block.timing
    assert not isinstance(timing, zkf.Timing)
    assert FSinPrimitive(operator).drive.timing == FCosPrimitive(operator).drive.timing == timing[False]
    assert FAtan2Primitive(operator).drive.timing == FHypotPrimitive(operator).drive.timing == timing[True]
    assert FSinPrimitive(operator).latency != FAtan2Primitive(operator).latency, "the premise: the drives differ"


def _sine_and_cosine(a: float) -> tuple[float, float]:
    return math.sin(a), math.cos(a)


def _angle_and_magnitude(y: float, x: float) -> tuple[float, float]:
    return math.atan2(y, x), math.hypot(y, x)


def _sorted_and_ordered(a: float, b: float) -> tuple[float, float, bool]:
    return min(a, b), max(a, b), a < b


def _difference_and_order(a: int, b: int) -> tuple[int, bool, bool, bool]:
    return a - b, a < b, a == b, a > b


def _truncated_both_ways(x: float) -> tuple[int, float]:
    return int(x), float(int(x))


def _quotient_and_remainder(a: int, b: int) -> tuple[int, int]:
    return a // b, a % b


@pytest.mark.parametrize(
    "kernel, operator, taps",
    [
        (_sine_and_cosine, "fcordic", 2),
        (_angle_and_magnitude, "fcordic", 2),
        (_sorted_and_ordered, "fcmp", 3),
        (_difference_and_order, "iadds", 4),
        (_truncated_both_ways, "frint", 2),
        (_quotient_and_remainder, "idivs", 2),
    ],
)
def test_the_results_one_drive_answers_are_one_firing(kernel: Callable[..., object], operator: str, taps: int) -> None:
    options = Options(
        operator=OperatorOptions(
            fadd=lean.fadd(),
            fmul=lean.fmul(),
            fcmp=lean.fcmp(),
            fcordic=lean.fcordic(),
            frint=lean.frint(),
            ffromint=lean.ffromint(),
            fmul_ilog2=lean.fmul_ilog2(),
            idivs=IDivsOptions(),
        ),
        ffmt=FloatFormat(6, 18),
        wmultiplier=18,
        wint_min=16,
    )
    lir = build_lir(lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), "one_drive")
    (firing,) = [op for block in lir.blocks for op in block.ops if op.inst.operator.name == operator]
    assert len(firing.writes) == taps


def _twin_angles_by_sign(y: float, x: float) -> tuple[float, float, float]:
    return math.atan2(abs(y), x), math.atan2(abs(-y), x) * 2.0, math.hypot(y, x)


def _twin_angles_by_truth(y: float, x: float) -> tuple[float, float, float]:
    return math.atan2(float(bool(y)), x), math.atan2(float(bool(abs(y))), x) * 2.0, math.hypot(float(bool(y)), x)


@pytest.mark.parametrize("kernel", [_twin_angles_by_sign, _twin_angles_by_truth])
def test_equivalent_angles_and_their_magnitude_are_one_cordic_firing(kernel: Callable[..., object]) -> None:
    # Two spellings of one angle become one operation once lowering collapses their signs or truth casts, and the
    # magnitude of the same pair rides that very firing.
    options = Options(
        operator=OperatorOptions(
            fadd=lean.fadd(), fmul=lean.fmul(), fmul_ilog2=lean.fmul_ilog2(), fcordic=lean.fcordic()
        ),
        ffmt=FloatFormat(6, 18),
        wmultiplier=18,
    )
    lir = build_lir(lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), "twins")
    firings = [op for block in lir.blocks for op in block.ops if op.inst.operator.name == "fcordic"]
    assert len(firings) == 1


@_requires_iverilog
def test_small_kernel_elaborates(tmp_path: Path) -> None:
    def kernel(a: float, b: float) -> float:
        return (a - b) * 0.25 + a * b

    fmt = FloatFormat(8, 24)
    lir = build_lir(_run(kernel, _ops(fmt), fmt), "kernel")
    _elaborate("kernel", generate(lir, {}).verilog, tmp_path)


@_requires_iverilog
def test_kernel_with_division_elaborates(tmp_path: Path) -> None:
    def blend(a: float, b: float, c: float) -> float:
        return a / b + c * 2.0

    fmt = FloatFormat(6, 18)
    lir = build_lir(_run(blend, _ops(fmt), fmt), "blend")
    _elaborate("blend", generate(lir, {}).verilog, tmp_path)


@_requires_iverilog
def test_constant_only_module_elaborates(tmp_path: Path) -> None:
    # No inputs and an all-constant output leave the wide bank empty, so the register array must be omitted rather
    # than declared zero-length.
    def const_only() -> float:
        return 3.5

    fmt = FloatFormat(8, 24)
    lir = build_lir(_run(const_only, _ops(fmt), fmt), "const_only")
    _elaborate("const_only", generate(lir, {}).verilog, tmp_path)


def test_boolean_output_port_is_one_bit_and_assigned() -> None:
    class Trigger:
        def __init__(self) -> None:
            self.high = 1.0
            self.low = -1.0
            self.y = False

        def __call__(self, x: float) -> bool:
            if x > self.high:
                self.y = True
            elif x < self.low:
                self.y = False
            return self.y

    fmt = FloatFormat(8, 24)
    lir = build_lir(_run(Trigger().__call__, _ops(fmt), fmt), "bool_trigger")
    (port,) = [port for port in lir.output_ports if port.name == "state_y"]
    assert isinstance(port.scalar_type, BoolType)
    assert port.width == 1
    verilog = generate(lir, {}).verilog
    assert re.search(r"\boutput wire state_y\b", verilog)
    assert re.search(r"\bassign state_y = (?:1'b[01]|bregs\[\d+\]);", verilog)


def test_boolean_input_port_is_one_bit_and_loaded() -> None:
    def passthrough(flag: bool) -> bool:
        return flag

    fmt = FloatFormat(8, 24)
    lir = build_lir(_run(passthrough, _ops(fmt), fmt), "bool_input")
    assert [load.name for load in lir.inputs] == ["flag"]
    assert isinstance(lir.bool_inputs[0].dst, BoolRegRef)
    assert not isinstance(lir.bool_inputs[0].dst, RegRef)
    (port,) = lir.input_ports
    assert port.name == "in_flag"
    assert isinstance(port.scalar_type, BoolType)
    assert port.width == 1
    verilog = generate(lir, {}).verilog
    assert re.search(r"\binput  wire in_flag\b", verilog)
    assert re.search(r"\bbregs\[\d+\] <= in_flag;", verilog)
    assert re.search(r"\bassign out_0 = bregs\[\d+\];", verilog)


@_requires_iverilog
def test_boolean_only_stateful_module_elaborates(tmp_path: Path) -> None:
    class Toggle:
        def __init__(self) -> None:
            self.flag = False

        def __call__(self) -> bool:
            self.flag = not self.flag
            return self.flag

    fmt = FloatFormat(8, 24)
    lir = build_lir(_run(Toggle().__call__, _ops(fmt), fmt), "bool_toggle")
    assert lir.input_ports == []
    (port,) = lir.output_ports
    assert port.name == "state_flag"
    assert isinstance(port.scalar_type, BoolType)
    verilog = generate(lir, {}).verilog
    assert re.search(r"\bassign state_flag = (?:1'b[01]|~?bregs\[\d+\]);", verilog)  # the tap may ride an inversion
    assert not re.search(r"\bregs\[\d+\] <=", verilog)
    _elaborate("bool_toggle", verilog, tmp_path)


def test_parameter_name_colliding_with_control_port_is_rejected() -> None:
    # A parameter named 'valid'/'ready' becomes data port in_valid/in_ready, colliding with the control ports and
    # producing un-elaboratable Verilog; LIR construction must reject it instead of emitting duplicate ports.
    def collide(valid: float, ready: float) -> float:
        return valid + ready

    fmt = FloatFormat(6, 18)
    with pytest.raises(UnsupportedConstruct):
        build_lir(_run(collide, _ops(fmt), fmt), "collide")


def test_kernel_without_outputs_is_rejected() -> None:
    def empty(x: float) -> tuple[()]:
        return ()

    fmt = FloatFormat(6, 18)
    with pytest.raises(UnsupportedConstruct):
        _run(empty, _ops(fmt), fmt)


@_requires_iverilog
def test_state_slot_folded_sign_coexists_with_sibling_port(tmp_path: Path) -> None:
    # A public attribute `y_d` becomes the port state_y_d; a sibling slot `y` whose boundary copy carries a folded sign
    # is emitted as an inline holoso_fsgnop() call in the state install. Both must elaborate cleanly together.
    class Collide:
        def __init__(self) -> None:
            self.y = 0.0
            self.y_d = 0.0
            self._p = 0.0

        def __call__(self, x: float) -> float:
            self.y_d = self._p
            self.y = -self._p
            self._p = x
            return self.y

    fmt = FloatFormat(8, 24)
    lir = build_lir(_run(Collide().__call__, _ops(fmt), fmt), "collide_state")
    _elaborate("collide_state", generate(lir, {}).verilog, tmp_path)


@_requires_iverilog
def test_ekf1_stateless_elaborates(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    fmt = FloatFormat(6, 18)
    lir = build_lir(_run(ekf1_stateless.update_x_P, _ops(fmt), fmt), "update_x_P")
    _elaborate("update_x_P", generate(lir, {}).verilog, tmp_path)


@_requires_iverilog
def test_ekf1_stateful_elaborates(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateful

    fmt = FloatFormat(6, 18)
    filt = ekf1_stateful.Ekf1(
        x=[0.0, 0.0, 0.0], P_urt=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0], R_diag=[1.0, 1.0], Q_diag=np.array([1.0, 1.0, 1.0])
    )
    lir = build_lir(_run(filt.update, _ops(fmt), fmt), "ekf1_stateful")
    _elaborate("ekf1_stateful", generate(lir, {}).verilog, tmp_path)


def _two_division_kernel(a: float, b: float, c: float, d: float) -> float:
    return a / b + c / d  # two divisions share one fdivsqrt instance and land in two distinct registers


def test_error_gate_ors_over_multiple_landing_registers() -> None:
    # An error-bearing operator (fdivsqrt) whose result lands in >=2 distinct registers reconstructs its commit window
    # as the OR, over those registers, of `uc_op_<reg> == <its source code>`. No bundled example produces a multi-term
    # err gate, and the numerical model does not simulate `err`, so cosim cannot reach it -- pin the reconstruction.
    options = Options(
        operator=OperatorOptions(fadd=lean.fadd(), fdivsqrt=lean.fdivsqrt()), ffmt=FloatFormat(6, 18), wmultiplier=18
    )
    verilog = synthesize(_two_division_kernel, options, name="two_div").verilog_output.verilog
    err = next(line.strip() for line in verilog.splitlines() if line.strip().startswith("assign err ="))
    assert err.count("uc_op_") >= 2 and " | " in err and "_error" in err, err


def test_wide_multi_output_operator_elaborates_with_per_port_lanes(tmp_path: Path) -> None:
    from holoso._lir import (
        Lir,
        LirBlock,
        PooledScheduledOp,
        Exit,
        Jump,
        WideInputLoad,
        WideOperand,
    )
    from holoso._lir._ir import PortWrite, WideOutputWire, landing_cycle
    from holoso._operators import FloatSignControl

    _FETCH_LAG = 2  # datapath lag matching the 3-stage control fetch: one less than fetch_stages

    fmt = FloatFormat(6, 18)
    comparator = FCmpOperator.build(fmt, lean.fcmp())
    op = PooledScheduledOp(
        drive=FMinPrimitive(comparator).drive,
        instance=0,
        operands=[WideOperand(RegRef(0), FloatSignControl()), WideOperand(RegRef(1), FloatSignControl())],
        taps={
            FMinPrimitive(comparator): PortWrite(RegRef(2), None),
            FMaxPrimitive(comparator): PortWrite(RegRef(3), None),
        },
        issue_cycle=1,
    )
    lir = Lir(
        module_name="sort_probe",
        instances={op.inst: op.inst.operator.block},
        wide_consts=[],
        float_format=fmt,
        int_format=default_ifmt(fmt),
        fetch_lag=_FETCH_LAG,
        wide_register_count=4,
        inputs=[WideInputLoad("a", RegRef(0), FloatType(fmt)), WideInputLoad("b", RegRef(1), FloatType(fmt))],
        outputs=[
            WideOutputWire("out_0", WideOperand(RegRef(2), FloatSignControl()), FloatType(fmt)),
            WideOutputWire("out_1", WideOperand(RegRef(3), FloatSignControl()), FloatType(fmt)),
        ],
        wide_state_slots=[],
        blocks=[LirBlock(0, [op], [], [], Jump(Exit()), landing_cycle(op.commit_cycle, _FETCH_LAG))],
        bool_register_count=0,
        bool_state_slots=[],
    )
    verilog = generate(lir, {}).verilog
    for port in ("min", "max"):
        assert f"_{port}_q" not in verilog, "the per-port result register must not be emitted"
        assert re.search(rf"wire\s+\[WFLT-1:0\]\s+s_fcmp_0_{port}\s*;", verilog), "per-port combinational result wire"
        assert re.search(
            rf"regs\[\d+\] <= s_fcmp_0_{port}\b", verilog
        ), "the wide write must read the combinational output wire directly"
    assert ".min(" in verilog and ".max(" in verilog
    if shutil.which("iverilog") is None:
        pytest.skip("iverilog not installed")
    _elaborate("sort_probe", verilog, tmp_path)


def _and_gate(a: bool, b: bool, /) -> bool:
    return a and b


def _madd_only(a: float, b: float, c: float) -> float:
    return a * b + c


@_requires_iverilog
def test_unused_register_bank_is_omitted(tmp_path: Path) -> None:
    # A purely-boolean kernel uses no wide bank, and an arithmetic kernel with no booleans uses no boolean bank. The
    # count localparam is stated either way; what must not appear is the register array itself, which at zero length
    # is illegal Verilog.
    bool_lir = build_lir(_run(_and_gate, _ops(FloatFormat(8, 24)), FloatFormat(8, 24)), "and_gate")
    assert bool_lir.wide_register_count == 0
    bool_v = generate(bool_lir, {}).verilog
    assert "NREG      =   0;" in bool_v
    assert "reg  [WREG-1:0] regs" not in bool_v and "[0:-1]" not in bool_v
    _elaborate("and_gate", bool_v, tmp_path)

    float_lir = build_lir(_run(_madd_only, _ops(FloatFormat(8, 24)), FloatFormat(8, 24)), "madd_only")
    assert float_lir.bool_register_count == 0
    float_v = generate(float_lir, {}).verilog
    assert "NBREG     =   0;" in float_v
    assert "bregs" not in float_v and "[0:-1]" not in float_v
    _elaborate("madd_only", float_v, tmp_path)


@_requires_iverilog
@pytest.mark.parametrize("bank", ["wide", "bool"])
def test_a_boundary_install_coexisting_with_opcode_writes_elaborates(bank: str, tmp_path: Path) -> None:
    # The boundary install outranks the opcode arm on the same register.
    # It is safe because every opcode write lands by its exit PC, so none executes alongside the install; the wide
    # kernel is cosimulated in test_cosim.py, so here only the premise and the elaboration are checked.
    from holoso._lir import write_events

    kernel = SharedLiveOut().step if bank == "wide" else SharedLiveOutBool().step
    name = f"shared_live_out_{bank}"
    lir = build_lir(_run(kernel, _ops(FloatFormat(6, 18)), FloatFormat(6, 18)), name)
    written = {event.dst for event in write_events(lir)}
    coexisting = [
        slot
        for slot in lir.boundary_installs
        if isinstance(slot, WideStateSlot) == (bank == "wide") and slot.reg in written
    ]
    assert coexisting, "the premise needs a boundary-installing slot whose own register also takes opcode writes"
    _elaborate(name, generate(lir, {}).verilog, tmp_path)


_INT_OPTIONS = Options(
    operator=OperatorOptions(
        fadd=lean.fadd(),
        fcmp=lean.fcmp(),  # a min alone leaves the max lane untapped
        imuls=IMulsOptions(),
        ffromint=lean.ffromint(),
        frint=lean.frint(),
    ),
    ffmt=FloatFormat(6, 18),
    wint_min=34,  # wider than the float, so a port sized at WFLT would silently lose its top bits
    wmultiplier=18,
)


class _IntegerKernel:
    """One instance of every wide site an integer reaches: both conversions, both slot installs, a negative reset."""

    def __init__(self) -> None:
        self._n = -3  # negative, so the reset literal exercises two's complement and not only the width
        self._prev = 0

    def step(self, a: int, b: int, n: int, x: float) -> tuple[int, int, int, int, int, bool, float, int]:
        # Exporting the slot's OLD value keeps it live to the boundary, which is what makes the install a
        # boundary copy -- the one arm that taps a slot through a conditioner rather than an opcode write.
        previous = self._prev
        self._n = self._n + a
        self._prev = b
        return (
            self._n,
            previous,
            abs(a - b) * (a & b) ^ ~a,
            (a << 3) // 7,
            a >> n,
            a > b,
            float(a) + min(x, float(b)),
            int(math.floor(x)),
        )


def _instantiation(verilog: str, mnemonic: str) -> str:
    found = re.search(rf"^\w+ u_{mnemonic}_\d+ \(\n.*?\n\);", verilog, re.S | re.MULTILINE)
    assert found is not None, f"{mnemonic} is not instantiated"
    return found.group()


@pytest.fixture(scope="module")
def _integer_result() -> SynthesisResult:
    return synthesize(_IntegerKernel().step, _INT_OPTIONS, name="int_kernel")


_INT34_MIN, _INT34_MAX = -8589934592, 8589934591  # spelled out: the reference must not share the code under test


def _clamp34(value: int) -> int:
    return min(max(value, _INT34_MIN), _INT34_MAX)


def _decode(value: object) -> int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, FloatValue):
        return float(value)
    assert isinstance(value, IntValue)
    return int(value)


def test_an_integer_kernel_model_matches_python_beyond_the_float_width(_integer_result: SynthesisResult) -> None:
    """The vectors exceed the signed 24-bit float word in both signs, where a WFLT-sized lane would truncate."""

    def wrap(value: int) -> int:
        return ((value - _INT34_MIN) % (2 * -_INT34_MIN)) + _INT34_MIN

    sim = _integer_result.numerical_model.elaborate()
    n_state, prev_state = -3, 0
    # Each float sum is exactly representable at e6m18 so the reference stays independent of the operator model.
    vectors = [(2**30, -(2**30), 3, 1.5), (-(2**30), 2**30, 1, -(2.0**29)), (2**25, -(2**25), 5, 0.25), (5, 3, 0, 0.5)]
    for a, b, n, x in vectors:
        n_state = _clamp34(n_state + a)
        expected: list[int | float | bool] = [
            n_state,
            prev_state,
            _clamp34(_clamp34(_clamp34(abs(_clamp34(a - b))) * (a & b)) ^ ~a),
            wrap(a << 3) // 7,
            a >> n,
            a > b,
            float(a) + min(x, float(b)),
            math.floor(x),
        ]
        prev_state = b
        assert [_decode(value) for value in sim.run(a, b, n, x)] == expected, (a, b, n, x)


def test_an_integer_port_binds_no_sign_sideband_and_declares_its_own_width(_integer_result: SynthesisResult) -> None:
    """
    The sign field exists only on a float operand port, and the read mux feeding an integer one must be as wide as the
    register rather than as the float -- the silent half, which elaborates either way and drops the top bits.
    """
    verilog = _integer_result.verilog_output.verilog
    ffromint, frint, iadds = (_instantiation(verilog, name) for name in ("ffromint", "frint", "iadds"))
    assert "holoso_fsgnop(" not in ffromint  # integer operand, float result
    assert frint.count("holoso_fsgnop(") == 1 and ".a(holoso_fsgnop(" in frint  # float operand, integer result
    assert "holoso_fsgnop(" not in iadds
    assert re.search(r"reg  \[WINT-1:0\] s_iadds_\w+_a;", verilog)
    assert re.search(r"wire \[WINT-1:0\] s_frint_\w+_y_int;", verilog)
    assert re.search(r"reg  \[WFLT-1:0\] s_frint_\w+_a;", verilog)


def test_only_a_float_port_is_allocated_a_microcode_sign_field(_integer_result: SynthesisResult) -> None:
    """
    The literal field set for this known kernel: float operand ports only -- no integer port and no result, a result
    never being conditioned at its producer.
    """
    assert set(re.findall(r"\buc_\w+?_sgn\b", _integer_result.verilog_output.verilog)) == {
        "uc_fadd_0_a_sgn",
        "uc_fadd_0_b_sgn",
        "uc_fcmp_0_a_sgn",
        "uc_fcmp_0_b_sgn",
        "uc_frint_0_a_sgn",
    }


def test_an_integer_state_slot_resets_to_its_own_word(_integer_result: SynthesisResult) -> None:
    """
    Silent otherwise: the float codec answers a legal-looking literal for every integer, so the slot comes up
    holding the encoding of a float rather than its own reset, at the float's width rather than the register's.
    """
    snapshots = [
        text.strip() for text in _integer_result.verilog_output.verilog.splitlines() if "reset snapshot" in text
    ]
    assert any("<= 34'h3fffffffd;" in text for text in snapshots), snapshots  # -3 in two's complement at WREG
    sim = _integer_result.numerical_model.elaborate()
    assert _decode(sim.run(10, 2, 0, 0.0)[0]) == -3 + 10
    sim.reset()
    assert _decode(sim.run(1, 2, 0, 0.0)[0]) == -3 + 1, "the reset must restore -3, not clear the register"


def test_the_boundary_installed_integer_slot_taps_its_conditioner() -> None:
    # The boundary install taps the slot through its conditioner -- the only integer tap on that emitter arm, which
    # no public metadata exposes -- and emits the boundary copy from the tap source to the slot register.
    lir = build_lir(
        lower_to_mir(lower(_IntegerKernel().step, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(_INT_OPTIONS)),
        "int_kernel",
    )
    (slot,) = [s for s in lir.wide_state_slots if isinstance(s.install, Boundary)]
    assert isinstance(slot.live_out.source, RegRef)
    verilog = generate(lir, {}).verilog
    assert f"if (out_valid && out_ready) regs[{slot.reg.index}] <= regs[{slot.live_out.source.index}];" in verilog


def test_an_integer_port_declares_itself_signed(_integer_result: SynthesisResult) -> None:
    """A wider signed consumer zero-fills an unsigned port, so a negative integer arrives as a large positive one."""
    declared = {
        name: qualifiers
        for qualifiers, name in re.findall(r"wire (.*?)(\w+),$", _integer_result.verilog_output.verilog, re.M)
    }
    assert declared["in_a"] == "signed [33:0] " and declared["out_0"] == "signed [33:0] "
    assert declared["in_x"] == "[23:0] ", "a float is a bit pattern, not a signed number"


@_requires_iverilog
def test_an_integer_kernel_emits_rtl_that_elaborates(_integer_result: SynthesisResult, tmp_path: Path) -> None:
    """Every wide site keyed on float rather than on the port's own family, so none of this could be rendered."""
    _elaborate("int_kernel", _integer_result.verilog_output.verilog, tmp_path)


def _offset_by_one(x: float) -> float:
    return x + 1.0


def test_a_float_write_fills_the_wide_high_bits_with_dont_care() -> None:
    """The adopted bit-fill policy: don't-care high bits when WREG > WFLT, no fill machinery at all at gap 0."""
    gapped = synthesize(
        _offset_by_one,
        Options(operator=OperatorOptions(fadd=lean.fadd()), ffmt=FloatFormat(6, 18), wint_min=33, wmultiplier=18),
        name="Gap9",
    )
    assert "{{(WREG-WFLT){1'bx}}, s_fadd" in gapped.verilog_output.verilog
    flat = synthesize(
        _offset_by_one,
        Options(operator=OperatorOptions(fadd=lean.fadd()), ffmt=FloatFormat(6, 18), wmultiplier=18),
        name="Gap0",
    )
    assert "1'bx" not in flat.verilog_output.verilog


@_requires_iverilog
def test_a_boolean_synthesis_attribute_renders_as_an_integer(tmp_path: Path) -> None:
    # A Python bool is an int, and an attribute mapping holding one must print as Verilog does a number.
    options = Options(
        operator=OperatorOptions(fadd=lean.fadd(), fmul=lean.fmul()),
        ffmt=FloatFormat(6, 18),
        wmultiplier=18,
        ucode_rom_attrs={"keep": True},
    )
    verilog = synthesize(_madd_only, options, name="bool_attr").verilog_output.verilog
    assert "(* keep = 1 *)" in verilog
    _elaborate("bool_attr", verilog, tmp_path)


def _divide_and_root(a: float, b: float, c: float) -> float:
    return a / b + math.sqrt(c)


def _rotate_and_vector(a: float, y: float, x: float) -> float:
    return math.sin(a) + math.atan2(y, x)


def _rotate_and_measure(a: float, y: float, x: float) -> float:
    return math.sin(a) + math.atan2(y, x) + math.hypot(y, x)


def _two_roundings(x: float, y: float) -> float:
    return float(math.floor(x)) + float(math.ceil(y))


def _add_subtract_and_shift(a: int, b: int, n: int) -> tuple[int, bool, int]:
    return (a + b) - (a << n), a < b, b >> n


# What each operation drives on its operator's immediate, as the blocks document the encodings.
_SELECTED: dict[type, int] = {
    FDivPrimitive: 0,
    FSqrtPrimitive: 1,
    FSinPrimitive: 0,
    FCosPrimitive: 0,
    FAtan2Primitive: 1,
    FHypotPrimitive: 1,
    IAddPrimitive: 0,
    ISubPrimitive: 1,
    ICmpPrimitive: 1,
    IShlPrimitive: 0,
    IShrPrimitive: 1,
}


@pytest.mark.parametrize(
    "kernel, checks",
    [
        (_divide_and_root, {FDivPrimitive, FSqrtPrimitive}),
        (_rotate_and_vector, {FSinPrimitive, FAtan2Primitive}),
        (_rotate_and_measure, {FSinPrimitive, FAtan2Primitive, FHypotPrimitive}),
        (_two_roundings, {FRintPrimitive}),
        (_add_subtract_and_shift, {IAddPrimitive, ISubPrimitive, ICmpPrimitive, IShlPrimitive, IShrPrimitive}),
    ],
)
def test_each_firing_drives_its_operation_on_the_immediate_on_its_issue_step(
    kernel: Callable[..., object], checks: set[type]
) -> None:
    # The numerical model never reads the microcode, so an immediate value on the wrong step, or the wrong value, passes
    # every model comparison and shows only in an RTL simulation. Where the instance keeps a timing per immediate value,
    # the entry at the value driven is the latency the firing was scheduled at.
    fmt = FloatFormat(6, 18)
    operator = OperatorOptions(
        fadd=lean.fadd(),
        fmul=lean.fmul(),
        fdivsqrt=lean.fdivsqrt(),
        fmul_ilog2=lean.fmul_ilog2(),
        fcmp=lean.fcmp(),
        fcordic=lean.fcordic(),
        frint=lean.frint(),
        ffromint=lean.ffromint(),
    )
    options = Options(operator=operator, ffmt=fmt, wmultiplier=18, wint_min=16)
    lir = build_lir(lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), "immediate")
    events = write_events(lir)
    fields = build_microcode(lir, read_codebook(lir), write_codebook(events), events)
    driven: dict[str, set[int]] = {}
    checked: set[type] = set()
    for block in lir.blocks:
        for op in block.ops:
            ports = immediate_inputs(lir, op.inst)
            if not ports:
                continue
            (port,) = ports
            step = lir.block_base[block.index] + op.issue_cycle
            for primitive in op.taps:
                checked.add(type(primitive))
                if isinstance(primitive, FRintPrimitive):
                    expected = int(primitive.rounding)
                else:
                    expected = _SELECTED[type(primitive)]
                assert fields[f_input(op.inst.name, port)].values[step] == expected, (
                    op.inst.name,
                    type(primitive).__name__,
                )
                driven.setdefault(op.inst.name, set()).add(expected)
                timing = lir.instances[op.inst].timing
                if not isinstance(timing, zkf.Timing):
                    scheduled = zkf.Timing(op.commit_cycle - op.issue_cycle, op.initiation_interval)
                    assert timing[expected] == scheduled, (op.inst.name, type(primitive).__name__)
    assert any(len(values) > 1 for values in driven.values()), "the premise: an instance driving two values"
    assert checked >= checks, "the premise: every primitive of the kernel checked on an instance driving two values"


def _floor_alone(x: float) -> int:
    return math.floor(x)


def _floor_beside_ceil(x: float) -> tuple[int, int]:
    return math.floor(x), math.ceil(x)


def _sum_alone(a: int, b: int) -> int:
    return a + b


def _difference_alone(a: int, b: int) -> int:
    return a - b


def _sum_beside_difference(a: int, b: int) -> tuple[int, int]:
    return a + b, a - b


def _left_alone(a: int, n: int) -> int:
    return a << n


def _left_beside_right(a: int, n: int) -> tuple[int, int]:
    return a << n, a >> n


@pytest.mark.parametrize(
    "kernel, operator, held",
    [
        (_floor_alone, "frint", True),
        (_floor_beside_ceil, "frint", False),
        (_sum_alone, "iadds", True),
        (_difference_alone, "iadds", True),
        (_sum_beside_difference, "iadds", False),
        (_left_alone, "ishft", True),
        (_left_beside_right, "ishft", False),
    ],
)
def test_an_instance_holds_inside_the_immediate_its_primitives_agree_on(
    kernel: Callable[..., object], operator: str, held: bool
) -> None:
    # The block an instance renders drops the input the firings bound to it all give one value, and with it the
    # microcode field that would drive it; where they disagree, the input stays and the field drives it per firing.
    options = Options(
        operator=OperatorOptions(fadd=lean.fadd(), frint=lean.frint(), ffromint=lean.ffromint()),
        ffmt=FloatFormat(6, 18),
        wmultiplier=18,
        wint_min=16,
    )
    lir = build_lir(lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), "held")
    (inst,) = [inst for inst in lir.instances if inst.operator.name == operator]
    (immediate,) = inst.operator.immediate_ports
    events = write_events(lir)
    fields = build_microcode(lir, read_codebook(lir), write_codebook(events), events)
    assert (immediate in lir.instances[inst].signature.members) != held
    assert (f_input(inst.name, immediate) in fields) != held


def _difference_beside_an_order(a: int, b: int, c: int, d: int) -> tuple[int, bool]:
    return a - b, c < d


def test_an_instance_block_keeping_an_immediate_its_firings_agree_on_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A subtraction and an ordering each specialize alone, and agree on `sub` together; a hook keeping the input for
    # the pair would leave fabric that nothing switches, which the instance check refuses as an internal error.
    alone = IAddOperator.instance_block
    monkeypatch.setattr(
        IAddOperator,
        "instance_block",
        lambda self, primitives: self.block if len(primitives) > 1 else alone(self, primitives),
    )
    options = Options(operator=OperatorOptions(fadd=lean.fadd()), ffmt=FloatFormat(6, 18), wmultiplier=18, wint_min=16)
    with pytest.raises(AssertionError, match="iadds"):
        synthesize(_difference_beside_an_order, options, name="unswitched")


def test_the_model_evaluates_each_firing_on_the_block_its_instance_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    # The MIR interpreter evaluates every operation on the operator's block, so a specialization computing otherwise
    # shows only where the numerical model evaluates on the block the instance renders: here an adder that only adds,
    # built as if it only subtracted, which the block refuses to evaluate.
    monkeypatch.setattr(IAddOperator, "instance_block", lambda self, primitives: self.block.specialized(sub=True))
    options = Options(operator=OperatorOptions(fadd=lean.fadd()), ffmt=FloatFormat(6, 18), wmultiplier=18, wint_min=16)
    simulator = synthesize(_sum_alone, options, name="misbuilt").numerical_model.elaborate()
    with pytest.raises(AssertionError, match="serves its own immediate alone"):
        simulator.run(1, 2)


def _root(x: float) -> float:
    return math.sqrt(x)


def _rotate_beside_vector(a: float, y: float, x: float) -> tuple[float, float]:
    return math.sin(a), math.atan2(y, x)


def test_an_instance_renders_the_block_for_the_operations_bound_to_it() -> None:
    # A divider that only takes roots has no divisor port, no read field for one and no immediate; two CORDIC instances,
    # one rotating and one vectoring, each render the block for their own operation only.
    fmt = FloatFormat(6, 18)
    operator = OperatorOptions(
        fadd=lean.fadd(), fmul=lean.fmul(), fdivsqrt=lean.fdivsqrt(), fcmp=lean.fcmp(), fcordic=lean.fcordic()
    )
    base = Options(operator=operator, ffmt=fmt, wmultiplier=18)
    lir = build_lir(lower_to_mir(lower(_root, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(base)), "root")
    (divider,) = [inst for inst in lir.instances if inst.operator.name == "fdivsqrt"]
    assert {"b", "op_sqrt"}.isdisjoint(lir.instances[divider].signature.members)
    events = write_events(lir)
    fields = build_microcode(lir, read_codebook(lir), write_codebook(events), events)
    assert not any(name.startswith(f"uc_{divider.name}_b") for name in fields)
    two = dataclasses.replace(base, operator=dataclasses.replace(operator, fcordic=lean.fcordic(instances=2)))
    lir = build_lir(lower_to_mir(lower(_rotate_beside_vector, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(two)), "pair")
    cordics = sorted((inst for inst in lir.instances if inst.operator.name == "fcordic"), key=lambda inst: inst.index)
    assert len(cordics) == 2, "the premise: the rotation and the vectoring issue together"
    members = [set(lir.instances[inst].signature.members) for inst in cordics]
    assert all("vectoring" not in ports for ports in members)
    assert sorted("b" in ports for ports in members) == [False, True]

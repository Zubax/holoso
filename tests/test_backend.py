"""Elaboration tests for the generated Verilog backend (structural correctness under Icarus)."""

import math
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from holoso import (
    BoolType,
    FAddOptions,
    FCmpOptions,
    FCordicOptions,
    FDivsqrtOptions,
    FExp2Options,
    FFromIntOptions,
    FILog2Options,
    FLog2Options,
    FloatFormat,
    FloatValue,
    FMulILog2Options,
    FMulOptions,
    FSortOptions,
    FRintOptions,
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
    CordicMode,
    FCordicOperator,
    FExp2Operator,
    FFromIntOperator,
    FILog2Operator,
    FLog2Operator,
    FMulILog2Operator,
    FSortOperator,
    FSortPrimitive,
    FDivsqrtOperator,
    FRintOperator,
    HardwareOperator,
    IAbsOperator,
    IAddOperator,
    IDivOperator,
    IMulOperator,
    IPopcntOperator,
    IShftOperator,
)
from holoso import SynthesisResult
from holoso._type import FloatType, IntType, ScalarType
from holoso._backend.verilog import generate
from holoso._eel import lower
from holoso._lir import BoolRegRef, Boundary, Lir, RegRef, WideStateSlot
from holoso._mir import MirOptions, Mir, lower as lower_to_mir

from .hdl.hdl_float_oracle import HDL_DIR, sources
from ._modelref import (
    build_lir,
    mir_options,
    default_ifmt,
    DEFAULT_UNROLL_MAX_TRIPS,
    SharedLiveOut,
    SharedLiveOutBool,
)

_requires_iverilog = pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")


def _ops(fmt: FloatFormat) -> MirOptions:
    return mir_options(
        Options(
            OperatorOptions(
                fadd=FAddOptions(),
                fmul=FMulOptions(),
                fdivsqrt=FDivsqrtOptions(),
                fmul_ilog2=FMulILog2Options(),
                fcmp=FCmpOptions(),
            ),
            ffmt=fmt,
        )
    )


def _run(target: object, ops: MirOptions, fmt: FloatFormat) -> Mir:
    return lower_to_mir(lower(target, DEFAULT_UNROLL_MAX_TRIPS).hir, ops)


def _compile(name: str, verilog: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    vpath = tmp_path / f"{name}.v"
    vpath.write_text(verilog)
    cmd = [
        "iverilog",
        "-g2012",
        "-I",
        str(HDL_DIR),
        "-s",
        name,
        "-o",
        str(tmp_path / f"{name}.out"),
        str(vpath),
        *(str(s) for s in sources()),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _elaborate(name: str, verilog: str, tmp_path: Path) -> None:
    result = _compile(name, verilog, tmp_path)
    assert result.returncode == 0, result.stderr


def test_operator_instance_names_are_mnemonic_and_copy_index() -> None:
    def scale(a: float, b: float) -> float:
        return a * 4.0 + b * 8.0

    fmt = FloatFormat(6, 18)
    lir = build_lir(_run(scale, _ops(fmt), fmt), "scale")
    names = re.findall(r"\bholoso_fmul_ilog2\s+#\([^;]+?\)\s+u_([A-Za-z_][A-Za-z0-9_]*)\s+\(", generate(lir).verilog)
    assert names == ["fmul_ilog2_0"]  # both exponents ride one pooled scaler, named by mnemonic and copy index


@_requires_iverilog
def test_comparisons_share_one_pooled_fcmp_instance() -> None:
    # Comparisons live in mutually-exclusive blocks and execute sequentially, so they share a single holoso_fcmp
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

    verilog = generate(build_lir(_run(kernel, _ops(FloatFormat(8, 24)), FloatFormat(8, 24)), "two_cmp")).verilog
    assert verilog.count("holoso_fcmp #") == 1


def test_streaming_wrapper_rejects_wrong_latency(tmp_path: Path) -> None:
    # A wrong LATENCY must be caught by the zkf_cmp register-stage-count guard rather than silently elaborate.
    verilog = """
module wrong_latency;
    wire clk = 1'b0;
    wire rst = 1'b0;
    wire in_valid = 1'b0;
    wire [31:0] a = 32'h0;
    wire [31:0] b = 32'h0;
    wire out_valid;
    wire a_gt_b;
    wire a_eq_b;
    wire a_lt_b;

    holoso_fcmp #(.WEXP(8), .WMAN(24), .STAGE_INPUT(0), .LATENCY(5)) u_cmp (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .a_sgnop(2'd0), .b_sgnop(2'd0), .a(a), .b(b),
        .out_valid(out_valid), .a_gt_b(a_gt_b), .a_eq_b(a_eq_b), .a_lt_b(a_lt_b)
    );
endmodule
"""
    result = _compile("wrong_latency", verilog, tmp_path)
    assert result.returncode != 0
    assert "_zkf_invalid_latency_mismatch" in result.stderr


def _integer_operators(ifmt: IntFormat) -> list[HardwareOperator]:
    return [
        IAddOperator.build(ifmt, IAddsOptions()),
        IAddOperator.build(ifmt, IAddsOptions(fast=True)),
        IDivOperator.build(ifmt, IDivsOptions()),
        IAbsOperator.build(ifmt, IAbssOptions()),
        IShftOperator.build(ifmt, IShftOptions()),
        IPopcntOperator.build(ifmt, IPopcntOptions()),
        *(IMulOperator.build(ifmt, IMulsOptions(stage_product=stage)) for stage in range(5)),
    ]


def _mixed_format_operators(ffmt: FloatFormat, ifmt: IntFormat) -> list[HardwareOperator]:
    return [
        FFromIntOperator.build(ffmt, ifmt, FFromIntOptions()),
        FFromIntOperator.build(
            ffmt, ifmt, FFromIntOptions(stage_input=1, stage_normalize=1, stage_pack=1, stage_output=1)
        ),
        FRintOperator.build(ffmt, ifmt, FRintOptions()),
        FRintOperator.build(ffmt, ifmt, FRintOptions(stage_input=2)),
        FMulILog2Operator.build(ffmt, ifmt, FMulILog2Options()),
        FMulILog2Operator.build(ffmt, ifmt, FMulILog2Options(stage_input=1, stage_decode=1)),
        FILog2Operator.build(ffmt, ifmt, FILog2Options()),
    ]


def _net(scalar_type: ScalarType) -> str:
    if isinstance(scalar_type, IntType):
        return f"signed [{scalar_type.width - 1}:0] "
    return f"[{scalar_type.width - 1}:0] " if scalar_type.is_wide else ""


def _pooled_probe(name: str, operators: list[HardwareOperator], single_modes: bool = False) -> str:
    """
    A module instantiating each operator through the ports, widths, parameters, mode and error ports it declares
    for itself -- so a port, sign sideband or parameter it declares and the RTL lacks fails right here. With
    `single_modes`, an operator that can be elaborated for one of its modes alone is instantiated once more per such
    elaboration.
    """
    lines = [f"module {name};", "    wire clk = 1'b0;", "    wire rst = 1'b0;", "    wire in_valid = 1'b0;"]
    elaborations = [
        (operator, params)
        for operator in operators
        for params in (operator.params, *(operator.single_mode_params.values() if single_modes else ()))
    ]
    for index, (operator, elaboration) in enumerate(elaborations):
        connections = []
        for position, operand in enumerate(operator.operand_ports):
            port, ty = operand.name, operand.scalar_type
            lines.append(f"    wire {_net(ty)}u{index}_{port} = {ty.width}'d0;")
            connections.append(f".{port}(u{index}_{port})")
            if operator.conditions_operand(position):
                connections.append(f".{port}_sgnop(2'd0)")
        for output in operator.output_ports:
            port, ty = output.name, output.scalar_type
            lines.append(f"    wire {_net(ty)}u{index}_{port};")
            connections.append(f".{port}(u{index}_{port})")
        if (mode_port := operator.mode_port) is not None:
            connections.append(f".{mode_port.name}({mode_port.width}'d0)")
        for port in operator.error_ports:
            lines.append(f"    wire u{index}_{port};")
            connections.append(f".{port}(u{index}_{port})")
        params = ", ".join(f".{pname}({value})" for pname, value in elaboration.items())
        # out_valid and the saturation sideband are deliberately left out: an omitted named port is unconnected.
        lines.append(
            f"    {operator.module_name} #({params}) u{index} "
            f"(.clk(clk), .rst(rst), .in_valid(in_valid), {', '.join(connections)});"
        )
    return "\n".join([*lines, "endmodule", ""])


@_requires_iverilog
@pytest.mark.parametrize("width", (2, 3, 24, 33, 44))
def test_integer_operators_elaborate_as_they_declare_themselves(width: int, tmp_path: Path) -> None:
    # A wrong latency instantiates the undefined _holoso_invalid_integer_latency; an odd width is mandatory because
    # that is where the divider's ceiling can slip. The adder is elaborated for each of its modes alone as well, a
    # `MODE` the module does not know instantiating the undefined _holoso_invalid_iadds_mode.
    name = f"int_probe_w{width}"
    _elaborate(name, _pooled_probe(name, _integer_operators(IntFormat(width)), single_modes=True), tmp_path)


@_requires_iverilog
@pytest.mark.parametrize("wexp,wman,wint", ((6, 18, 44), (8, 36, 24), (3, 4, 12), (6, 18, 17)))
def test_mixed_format_operators_elaborate_as_they_declare_themselves(
    wexp: int, wman: int, wint: int, tmp_path: Path
) -> None:
    # Several triples because the integer side is sized independently of the float one.
    name = f"mixed_probe_e{wexp}m{wman}i{wint}"
    operators = _mixed_format_operators(FloatFormat(wexp, wman), IntFormat(wint))
    _elaborate(name, _pooled_probe(name, operators), tmp_path)


@_requires_iverilog
@pytest.mark.parametrize("wexp,wman", ((6, 18), (8, 24), (3, 5), (8, 37)))
def test_fdivsqrt_wrapper_elaborates_as_it_declares_itself(wexp: int, wman: int, tmp_path: Path) -> None:
    # Both significand parities, which the kernel-level and cocotb coverage (even significands only) never reach: the
    # core's digit count, the reach of its decode stage and the frame of each single-operation build all depend on
    # WMAN's parity.
    name = f"fdivsqrt_probe_e{wexp}m{wman}"
    fmt = FloatFormat(wexp, wman)
    operators: list[HardwareOperator] = [
        FDivsqrtOperator.build(fmt, FDivsqrtOptions()),
        FDivsqrtOperator.build(fmt, FDivsqrtOptions(stage_decode=1)),
        FDivsqrtOperator.build(fmt, FDivsqrtOptions(stage_input=2, stage_decode=1, stage_pack=1, stage_output=1)),
    ]
    _elaborate(name, _pooled_probe(name, operators, single_modes=True), tmp_path)


@_requires_iverilog
def test_support_library_elaborates_with_the_keep_hooks_defined(tmp_path: Path) -> None:
    # A flow that defines the hooks compiles a text of the support library that no simulation here otherwise sees.
    name = "keep_hooks_probe"
    ffmt, ifmt = FloatFormat(8, 36), IntFormat(44)
    operators: list[HardwareOperator] = [
        IDivOperator.build(ifmt, IDivsOptions()),
        FDivsqrtOperator.build(ffmt, FDivsqrtOptions()),
        FMulILog2Operator.build(ffmt, ifmt, FMulILog2Options()),
    ]
    probe = _pooled_probe(name, operators, single_modes=True)
    _elaborate(name, "`define HOLOSO_ATTRIBUTE_KEEP (* syn_keep = 1 *)\n" + probe, tmp_path)
    # The one hook must reach the ZKF library's own keep sites too, which a body that cannot parse shows: it fails at
    # the integer divider's site and at each of ZKF's two.
    broken = _compile(name, "`define HOLOSO_ATTRIBUTE_KEEP (* syn_keep = *)\n" + probe, tmp_path)
    assert broken.returncode != 0 and broken.stderr.count("syntax error") >= 3, broken.stderr


@_requires_iverilog
def test_support_library_passes_the_rom_hook_to_the_lookup_tables(tmp_path: Path) -> None:
    # The microcode ROM carrying this hook is in the generated module, so its only sites in the support library are
    # ZKF's lookup tables: a body that cannot parse fails there only if the hook is passed on.
    name = "rom_hook_probe"
    fmt = FloatFormat(8, 36)
    operators: list[HardwareOperator] = [
        FExp2Operator.build(fmt, FExp2Options(), 18),
        FLog2Operator.build(fmt, FLog2Options(), 18),
    ]
    probe = _pooled_probe(name, operators)
    _elaborate(name, '`define HOLOSO_ATTRIBUTE_ROM (* rom_style = "block" *)\n' + probe, tmp_path)
    broken = _compile(name, "`define HOLOSO_ATTRIBUTE_ROM (* rom_style = *)\n" + probe, tmp_path)
    assert broken.returncode != 0 and "syntax error" in broken.stderr, broken.stderr


@_requires_iverilog
def test_integer_wrapper_rejects_wrong_latency(tmp_path: Path) -> None:
    # The negative twin of the probe above, so its silence means something.
    operator = IDivOperator.build(IntFormat(33), IDivsOptions())
    latency = operator.latencies[0]
    verilog = _pooled_probe("wrong_int_latency", [operator]).replace(f".LATENCY({latency})", f".LATENCY({latency + 1})")
    result = _compile("wrong_int_latency", verilog, tmp_path)
    assert result.returncode != 0
    assert "_holoso_invalid_integer_latency" in result.stderr


def _rotating(x: float) -> tuple[float, float]:
    return math.sin(x), math.cos(x)


def _vectoring(y: float, x: float) -> tuple[float, float]:
    return math.atan2(y, x), math.hypot(y, x)


def _rotating_a_vectored_angle(y: float, x: float) -> float:
    return math.cos(math.atan2(y, x))


def _rotating_beside_vectoring(a: float, y: float, x: float) -> tuple[float, float]:
    return math.sin(a), math.atan2(y, x)


@pytest.mark.parametrize(
    "kernel, instances, modes",
    [
        (_rotating, 1, [{"MODE": 0, "LATENCY_ROTATION": True, "LATENCY_VECTORING": False}]),
        (_vectoring, 1, [{"MODE": 1, "LATENCY_ROTATION": False, "LATENCY_VECTORING": True}]),
        (_rotating_a_vectored_angle, 1, [{"MODE": 2, "LATENCY_ROTATION": True, "LATENCY_VECTORING": True}]),
        (
            _rotating_beside_vectoring,
            2,
            [
                {"MODE": 0, "LATENCY_ROTATION": True, "LATENCY_VECTORING": False},
                {"MODE": 1, "LATENCY_ROTATION": False, "LATENCY_VECTORING": True},
            ],
        ),
    ],
    ids=lambda value: value.__name__.strip("_") if callable(value) else None,
)
def test_a_cordic_instance_is_elaborated_for_the_modes_its_firings_run(
    kernel: Callable[..., object], instances: int, modes: list[dict[str, int | bool]]
) -> None:
    # A rotation and a vectoring issued together take an instance each, so each is elaborated for its own mode alone;
    # an instance running both keeps the per-transaction core and both latencies.
    options = Options(
        OperatorOptions(fmul=FMulOptions(), fmul_ilog2=FMulILog2Options(), fcordic=FCordicOptions(instances=instances)),
        ffmt=FloatFormat(6, 18),
    )
    verilog = synthesize(kernel, options, name=f"cordic_elab_{kernel.__name__.strip('_')}").verilog_output.verilog
    elaborations = re.findall(r"holoso_fcordic #\(\n\s*(.*?)\n\) u_fcordic_\d+ \(", verilog)
    found = [
        {
            "MODE": int(re.findall(r"\.MODE\((\d)\)", params)[0]),
            "LATENCY_ROTATION": ".LATENCY_ROTATION(" in params,
            "LATENCY_VECTORING": ".LATENCY_VECTORING(" in params,
        }
        for params in elaborations
    ]
    assert sorted(found, key=lambda elaboration: elaboration["MODE"]) == modes


def _cordic_probe(name: str, mode: int, wrong: CordicMode) -> str:
    """The CORDIC wrapper elaborated with MODE `mode` and the latency parameter of mode `wrong` off by one."""
    operator = FCordicOperator.build(FloatFormat(6, 18), FCordicOptions(), 0)
    param = f"LATENCY_{wrong.name}"
    latency = operator.params[param]
    verilog = _pooled_probe(name, [operator]).replace(f".{param}({latency})", f".{param}({latency + 1})")
    return verilog.replace(".MODE(2)", f".MODE({mode})")


@_requires_iverilog
@pytest.mark.parametrize("mode", list(CordicMode), ids=lambda mode: mode.name)
def test_cordic_wrapper_fixed_to_one_mode_ignores_the_other_modes_latency(mode: CordicMode, tmp_path: Path) -> None:
    # Only a MODE that reaches zkf_cordic makes it ignore the other mode's latency parameter; one the wrapper dropped
    # would leave the core checking both, and no simulation tells the two apart.
    name = f"cordic_fixed_{mode.name.lower()}"
    other = CordicMode.VECTORING if mode is CordicMode.ROTATION else CordicMode.ROTATION
    _elaborate(name, _cordic_probe(name, int(mode), other), tmp_path)


@_requires_iverilog
def test_cordic_wrapper_running_both_modes_checks_both_latencies(tmp_path: Path) -> None:
    # The negative twin of the probe above, so its silence means something.
    result = _compile("cordic_both_wrong", _cordic_probe("cordic_both_wrong", 2, CordicMode.VECTORING), tmp_path)
    assert result.returncode != 0
    assert "_zkf_invalid_latency_mismatch" in result.stderr


@_requires_iverilog
def test_popcount_wrapper_rejects_wrong_result_width(tmp_path: Path) -> None:
    # The count port is narrower than the word it counts, and Verilog would accept a wrapper that disagreed about
    # how much narrower -- silently padding or truncating. The guard is what makes the width a checked claim, so it
    # needs its own negative twin.
    operator = IPopcntOperator.build(IntFormat(33), IPopcntOptions())
    width = operator.params["WY"]
    verilog = _pooled_probe("wrong_popcnt_width", [operator]).replace(f".WY({width})", f".WY({width + 1})")
    result = _compile("wrong_popcnt_width", verilog, tmp_path)
    assert result.returncode != 0
    assert "_holoso_invalid_ipopcnt_result_width" in result.stderr


@_requires_iverilog
@pytest.mark.parametrize(
    "operator",
    (
        FFromIntOperator.build(FloatFormat(6, 18), IntFormat(44), FFromIntOptions()),
        FRintOperator.build(FloatFormat(6, 18), IntFormat(44), FRintOptions()),
        FMulILog2Operator.build(FloatFormat(6, 18), IntFormat(44), FMulILog2Options()),
    ),
    ids=lambda operator: operator.name,
)
def test_mixed_format_wrapper_rejects_wrong_latency(operator: HardwareOperator, tmp_path: Path) -> None:
    # The negative twin on the conversion side, so the probe's silence means something.
    name = f"wrong_mixed_latency_{operator.name}"
    latency = operator.latencies[0]
    verilog = _pooled_probe(name, [operator]).replace(f".LATENCY({latency})", f".LATENCY({latency + 1})")
    result = _compile(name, verilog, tmp_path)
    assert result.returncode != 0
    assert "_zkf_invalid_latency_mismatch" in result.stderr


@_requires_iverilog
def test_small_kernel_elaborates(tmp_path: Path) -> None:
    def kernel(a: float, b: float) -> float:
        return (a - b) * 0.25 + a * b

    fmt = FloatFormat(8, 24)
    lir = build_lir(_run(kernel, _ops(fmt), fmt), "kernel")
    _elaborate("kernel", generate(lir).verilog, tmp_path)


@_requires_iverilog
def test_kernel_with_division_elaborates(tmp_path: Path) -> None:
    def blend(a: float, b: float, c: float) -> float:
        return a / b + c * 2.0

    fmt = FloatFormat(6, 18)
    lir = build_lir(_run(blend, _ops(fmt), fmt), "blend")
    _elaborate("blend", generate(lir).verilog, tmp_path)


@_requires_iverilog
def test_constant_only_module_elaborates(tmp_path: Path) -> None:
    # No inputs and an all-constant output leave the wide bank empty, so the register array must be omitted rather
    # than declared zero-length.
    def const_only() -> float:
        return 3.5

    fmt = FloatFormat(8, 24)
    lir = build_lir(_run(const_only, _ops(fmt), fmt), "const_only")
    _elaborate("const_only", generate(lir).verilog, tmp_path)


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
    verilog = generate(lir).verilog
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
    verilog = generate(lir).verilog
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
    verilog = generate(lir).verilog
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
    _elaborate("collide_state", generate(lir).verilog, tmp_path)


@_requires_iverilog
def test_ekf1_stateless_elaborates(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    fmt = FloatFormat(6, 18)
    lir = build_lir(_run(ekf1_stateless.update_x_P, _ops(fmt), fmt), "update_x_P")
    _elaborate("update_x_P", generate(lir).verilog, tmp_path)


@_requires_iverilog
def test_ekf1_stateful_elaborates(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateful

    fmt = FloatFormat(6, 18)
    filt = ekf1_stateful.Ekf1(
        x=[0.0, 0.0, 0.0], P_urt=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0], R_diag=[1.0, 1.0], Q_diag=np.array([1.0, 1.0, 1.0])
    )
    lir = build_lir(_run(filt.update, _ops(fmt), fmt), "ekf1_stateful")
    _elaborate("ekf1_stateful", generate(lir).verilog, tmp_path)


def _two_division_kernel(a: float, b: float, c: float, d: float) -> float:
    return a / b + c / d  # two divisions share one fdivsqrt instance and land in two distinct registers


def test_error_gate_ors_over_multiple_landing_registers() -> None:
    # An error-bearing operator (fdivsqrt) whose result lands in >=2 distinct registers reconstructs its commit window
    # as the OR, over those registers, of `uc_op_<reg> == <its source code>`. No bundled example produces a multi-term
    # err gate, and the numerical model does not simulate `err`, so cosim cannot reach it -- pin the reconstruction.
    options = Options(
        OperatorOptions(fadd=FAddOptions(), fdivsqrt=FDivsqrtOptions()),
        ffmt=FloatFormat(6, 18),
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
    from holoso._lir._ir import PortWrite, RegFileLayout, WideOutputWire, boundary_step
    from holoso._operators import FloatSignControl

    _FETCH_LAG = 2  # datapath lag matching the 3-stage control fetch: one less than fetch_stages

    fmt = FloatFormat(6, 18)
    op = PooledScheduledOp(
        primitive=FSortPrimitive(FSortOperator.build(fmt, FSortOptions())),
        instance=0,
        operands=[WideOperand(RegRef(0), FloatSignControl()), WideOperand(RegRef(1), FloatSignControl())],
        writes=[PortWrite(0, RegRef(2), None), PortWrite(1, RegRef(3), None)],
        issue_cycle=1,
    )
    lir = Lir(
        module_name="fsort_probe",
        instances=[op.inst],
        wide_consts=[],
        float_format=fmt,
        int_format=default_ifmt(fmt),
        fetch_lag=_FETCH_LAG,
        regfile=RegFileLayout(nreg=4),
        inputs=[WideInputLoad("a", RegRef(0), FloatType(fmt)), WideInputLoad("b", RegRef(1), FloatType(fmt))],
        outputs=[
            WideOutputWire("out_0", WideOperand(RegRef(2), FloatSignControl()), FloatType(fmt)),
            WideOutputWire("out_1", WideOperand(RegRef(3), FloatSignControl()), FloatType(fmt)),
        ],
        wide_state_slots=[],
        blocks=[LirBlock(0, [op], [], [], Jump(Exit()), boundary_step(op.commit_cycle, _FETCH_LAG))],
        bool_regfile=RegFileLayout(nreg=0),
        bool_state_slots=[],
    )
    verilog = generate(lir).verilog
    for q in (0, 1):
        assert f"_y{q}_q" not in verilog, "the per-port result register must not be emitted"
        assert re.search(rf"wire\s+\[WFLT-1:0\]\s+s_fsort_0_y{q}\s*;", verilog), "per-port combinational result wire"
        assert re.search(
            rf"regs\[\d+\] <= s_fsort_0_y{q}\b", verilog
        ), "the wide write must read the combinational output wire directly"
    assert ".min(" in verilog and ".max(" in verilog
    if shutil.which("iverilog") is None:
        pytest.skip("iverilog not installed")
    _elaborate("fsort_probe", verilog, tmp_path)


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
    assert bool_lir.regfile.nreg == 0
    bool_v = generate(bool_lir).verilog
    assert "NREG      =   0;" in bool_v
    assert "reg  [WREG-1:0] regs" not in bool_v and "[0:-1]" not in bool_v
    _elaborate("and_gate", bool_v, tmp_path)

    float_lir = build_lir(_run(_madd_only, _ops(FloatFormat(8, 24)), FloatFormat(8, 24)), "madd_only")
    assert float_lir.bool_regfile.nreg == 0
    float_v = generate(float_lir).verilog
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
    _elaborate(name, generate(lir).verilog, tmp_path)


_INT_OPTIONS = Options(
    OperatorOptions(
        fadd=FAddOptions(),
        fcmp=FCmpOptions(),
        fsort=FSortOptions(),  # a min alone leaves the max lane untapped
        imuls=IMulsOptions(),
        ffromint=FFromIntOptions(),
        frint=FRintOptions(),
    ),
    ffmt=FloatFormat(6, 18),
    wint_min=34,  # wider than the float, so a port sized at WFLT would silently lose its top bits
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
    found = re.search(rf"holoso_{mnemonic}\b.*?\n\);", verilog, re.S)
    assert found is not None, f"holoso_{mnemonic} is not instantiated"
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
    The sideband exists only on a float operand port, and the read mux feeding an integer one must be as wide as the
    register rather than as the float -- the silent half, which elaborates either way and drops the top bits.
    """
    verilog = _integer_result.verilog_output.verilog
    ffromint, frint, iadds = (_instantiation(verilog, name) for name in ("ffromint", "frint", "iadds"))
    assert "_sgnop(" not in ffromint  # integer operand, float result
    assert frint.count("_sgnop(") == 1 and ".a_sgnop(" in frint  # float operand, integer result
    assert "_sgnop(" not in iadds
    assert re.search(r"reg  \[WINT-1:0\] s_iadds_\w+_a;", verilog)
    assert re.search(r"wire \[WINT-1:0\] s_frint_\w+_y1;", verilog)
    assert re.search(r"reg  \[WFLT-1:0\] s_frint_\w+_a;", verilog)


def test_only_a_float_port_is_allocated_a_microcode_sign_field(_integer_result: SynthesisResult) -> None:
    """
    The literal field set for this known kernel: float operand ports only -- no integer port and no result, a result
    never being conditioned at its producer.
    """
    assert set(re.findall(r"\buc_\w+?sgn\b", _integer_result.verilog_output.verilog)) == {
        "uc_fadd_0_asgn",
        "uc_fadd_0_bsgn",
        "uc_fsort_0_asgn",
        "uc_fsort_0_bsgn",
        "uc_frint_0_asgn",
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
    verilog = generate(lir).verilog
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
        _offset_by_one, Options(OperatorOptions(fadd=FAddOptions()), ffmt=FloatFormat(6, 18), wint_min=33), name="Gap9"
    )
    assert "{{(WREG-WFLT){1'bx}}, s_fadd" in gapped.verilog_output.verilog
    flat = synthesize(
        _offset_by_one, Options(OperatorOptions(fadd=FAddOptions()), ffmt=FloatFormat(6, 18)), name="Gap0"
    )
    assert "1'bx" not in flat.verilog_output.verilog

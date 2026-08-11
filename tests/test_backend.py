"""Elaboration tests for the generated Verilog backend (structural correctness under Icarus)."""

import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from holoso import (
    BoolType,
    FAddOptions,
    FCmpOptions,
    FDivOptions,
    FFromIntOptions,
    FMulILog2Options,
    FMulOptions,
    FSortOptions,
    FToIntOptions,
    FloatFormat,
    IMulOptions,
    IntFormat,
    OperatorOptions,
    Options,
    UnsupportedConstruct,
    synthesize,
)
from holoso._operators import (
    FFromIntOperator,
    FMulILog2Operator,
    FSortOperator,
    FToIntOperator,
    IAbsOperator,
    IAddOperator,
    ICmpOperator,
    IDivOperator,
    IMulOperator,
    IShlOperator,
    IShrOperator,
    ISubOperator,
    OpConfig,
    PooledHardwareOperator,
)
from holoso._type import FloatType, IntType, ScalarType
from holoso._backend.verilog import generate
from holoso._backend.verilog._microcode import PORT_LETTERS, base_name, tapped_lanes
from holoso._eel import lower
from holoso._lir import BoolRegRef, Lir, RegRef, pooled_write_word
from holoso._mir import Mir, lower as lower_to_mir

from .hdl.hdl_float_oracle import HDL_DIR, sources
from ._modelref import DEFAULT_IFCONV_MAX_OPS, default_ifmt, build_lir, build_ops

requires_iverilog = pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")


def _ops(fmt: FloatFormat) -> OpConfig:
    return build_ops(
        Options(
            OperatorOptions(
                fadd=FAddOptions(),
                fmul=FMulOptions(),
                fdiv=FDivOptions(),
                fmul_ilog2=FMulILog2Options(),
                fcmp=FCmpOptions(),
            ),
            ffmt=fmt,
        )
    )


def _run(target: object, ops: OpConfig, fmt: FloatFormat) -> Mir:
    return lower_to_mir(lower(target).hir, ops, fmt, default_ifmt(fmt), DEFAULT_IFCONV_MAX_OPS)


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


def test_operator_instance_names_include_hardware_identity() -> None:
    def scale(a: float, b: float) -> float:
        return a * 4.0 + b * 8.0

    fmt = FloatFormat(6, 18)
    lir = build_lir(_run(scale, _ops(fmt), fmt), "scale")
    names = re.findall(r"\bholoso_fmul_ilog2\s+#\([^;]+?\)\s+u_([A-Za-z_][A-Za-z0-9_]*)\s+\(", generate(lir).verilog)

    assert len(names) == 1  # both exponents ride one pooled scaler
    assert all(re.fullmatch(r"fmul_ilog2_[0-9a-f]{8}_0", name) for name in names)
    assert all("stage_decode" not in name and "e6_m18" not in name for name in names)
    assert all(name == name.lower() for name in names)


@requires_iverilog
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
    # holoso_fcmp defaults LATENCY to 1 + STAGE_INPUT (the only correct value), so an instance need not specify it. An
    # explicitly wrong LATENCY must be caught by the zkf_cmp register-stage-count guard rather than silently elaborate.
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


def _integer_operators(ifmt: IntFormat) -> list[PooledHardwareOperator]:
    return [
        IAddOperator(ifmt),
        ISubOperator(ifmt),
        IDivOperator(ifmt),
        IAbsOperator(ifmt),
        IShlOperator(ifmt),
        IShrOperator(ifmt),
        ICmpOperator(ifmt),
        *(IMulOperator(ifmt, IMulOptions(stage_product=stage)) for stage in range(5)),
    ]


def _mixed_format_operators(ffmt: FloatFormat, ifmt: IntFormat) -> list[PooledHardwareOperator]:
    return [
        FFromIntOperator(ffmt, ifmt, FFromIntOptions()),
        FFromIntOperator(ffmt, ifmt, FFromIntOptions(stage_input=1, stage_normalize=1, stage_pack=1, stage_output=1)),
        FToIntOperator(ffmt, ifmt, FToIntOptions()),
        FToIntOperator(ffmt, ifmt, FToIntOptions(stage_input=2)),
        FMulILog2Operator(ffmt, ifmt, FMulILog2Options()),
        FMulILog2Operator(ffmt, ifmt, FMulILog2Options(stage_input=1, stage_decode=1)),
    ]


def _net(scalar_type: ScalarType) -> str:
    if isinstance(scalar_type, IntType):
        return f"signed [{scalar_type.width - 1}:0] "
    return f"[{scalar_type.width - 1}:0] " if scalar_type.is_wide else ""


def _pooled_probe(name: str, operators: list[PooledHardwareOperator]) -> str:
    """
    A module instantiating each operator through the ports, widths, parameters and immediates it declares for itself,
    all read off its signature -- so a declaration that drifted from the RTL fails right here. A sign-conditioning
    sideband exists on a float port and on no other, which is what makes the two conversion wrappers asymmetric.
    """
    lines = [f"module {name};", "    wire clk = 1'b0;", "    wire rst = 1'b0;", "    wire in_valid = 1'b0;"]
    for index, operator in enumerate(operators):
        signature = operator.signature
        connections = []
        for port, ty in zip(operator.operand_hdl_ports, signature.operand_types, strict=True):
            lines.append(f"    wire {_net(ty)}u{index}_{port} = {ty.width}'d0;")
            connections.append(f".{port}(u{index}_{port})")
            if isinstance(ty, FloatType):
                connections.append(f".{port}_sgnop(2'd0)")
        for port, ty in zip(operator.output_hdl_ports, signature.result_types, strict=True):
            lines.append(f"    wire {_net(ty)}u{index}_{port};")
            connections.append(f".{port}(u{index}_{port})")
            if isinstance(ty, FloatType):
                connections.append(f".{port}_sgnop(2'd0)")
        for immediate in operator.immediate_ports:
            connections.append(f".{immediate.name}({immediate.width}'d0)")
        for port in operator.error_ports:
            lines.append(f"    wire u{index}_{port};")
            connections.append(f".{port}(u{index}_{port})")
        params = ", ".join(f".{pname}({value})" for pname, value in operator.params.items())
        # out_valid and the saturation sideband are deliberately left out: an omitted named port is unconnected.
        lines.append(
            f"    {operator.module_name} #({params}) u{index} "
            f"(.clk(clk), .rst(rst), .in_valid(in_valid), {', '.join(connections)});"
        )
    return "\n".join([*lines, "endmodule", ""])


@requires_iverilog
@pytest.mark.parametrize("width", (2, 3, 24, 33, 44))
def test_integer_operators_elaborate_as_they_declare_themselves(width: int, tmp_path: Path) -> None:
    # A wrong latency instantiates the undefined _holoso_invalid_integer_latency; an odd width is mandatory because
    # that is where the divider's ceiling can slip.
    name = f"int_probe_w{width}"
    _elaborate(name, _pooled_probe(name, _integer_operators(IntFormat(width))), tmp_path)


@requires_iverilog
@pytest.mark.parametrize("wexp,wman,wint", ((6, 18, 44), (8, 36, 24), (3, 4, 12), (6, 18, 17)))
def test_mixed_format_operators_elaborate_as_they_declare_themselves(
    wexp: int, wman: int, wint: int, tmp_path: Path
) -> None:
    # Several triples because the integer side is sized independently of the float one.
    name = f"mixed_probe_e{wexp}m{wman}i{wint}"
    operators = _mixed_format_operators(FloatFormat(wexp, wman), IntFormat(wint))
    _elaborate(name, _pooled_probe(name, operators), tmp_path)


@requires_iverilog
def test_integer_wrapper_rejects_wrong_latency(tmp_path: Path) -> None:
    # The negative twin of the probe above, so its silence means something.
    operator = IDivOperator(IntFormat(33))
    verilog = _pooled_probe("wrong_int_latency", [operator]).replace(
        f".LATENCY({operator.latency})", f".LATENCY({operator.latency + 1})"
    )
    result = _compile("wrong_int_latency", verilog, tmp_path)
    assert result.returncode != 0
    assert "_holoso_invalid_integer_latency" in result.stderr


@requires_iverilog
@pytest.mark.parametrize(
    "operator",
    (
        FFromIntOperator(FloatFormat(6, 18), IntFormat(44), FFromIntOptions()),
        FToIntOperator(FloatFormat(6, 18), IntFormat(44), FToIntOptions()),
        FMulILog2Operator(FloatFormat(6, 18), IntFormat(44), FMulILog2Options()),
    ),
    ids=lambda operator: operator.mnemonic,
)
def test_mixed_format_wrapper_rejects_wrong_latency(operator: PooledHardwareOperator, tmp_path: Path) -> None:
    # The negative twin on the conversion side, so the probe's silence means something.
    name = f"wrong_mixed_latency_{operator.mnemonic}"
    verilog = _pooled_probe(name, [operator]).replace(
        f".LATENCY({operator.latency})", f".LATENCY({operator.latency + 1})"
    )
    result = _compile(name, verilog, tmp_path)
    assert result.returncode != 0
    assert "_zkf_invalid_latency_mismatch" in result.stderr


@requires_iverilog
def test_small_kernel_elaborates(tmp_path: Path) -> None:
    def kernel(a: float, b: float) -> float:
        return (a - b) * 0.25 + a * b

    fmt = FloatFormat(8, 24)
    lir = build_lir(_run(kernel, _ops(fmt), fmt), "kernel")
    _elaborate("kernel", generate(lir).verilog, tmp_path)


@requires_iverilog
def test_kernel_with_division_elaborates(tmp_path: Path) -> None:
    def blend(a: float, b: float, c: float) -> float:
        return a / b + c * 2.0

    fmt = FloatFormat(6, 18)
    lir = build_lir(_run(blend, _ops(fmt), fmt), "blend")
    _elaborate("blend", generate(lir).verilog, tmp_path)


@requires_iverilog
def test_constant_only_module_elaborates(tmp_path: Path) -> None:
    # No inputs and an all-constant output => zero registers; NREG must floor to >=1 so the regfile parameter
    # guard does not instantiate its error stub (BUG1 regression).
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


@requires_iverilog
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
    with pytest.raises(UnsupportedConstruct, match="duplicate port"):
        build_lir(_run(collide, _ops(fmt), fmt), "collide")


def test_kernel_without_outputs_is_rejected() -> None:
    def empty(x: float) -> tuple[()]:
        return ()

    fmt = FloatFormat(6, 18)
    with pytest.raises(UnsupportedConstruct, match="an empty aggregate cannot be returned"):
        _run(empty, _ops(fmt), fmt)


@requires_iverilog
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
            self.y = -self._p  # sign-flipped state boundary copy -> inline holoso_fsgnop() in the state install
            self._p = x
            return self.y

    fmt = FloatFormat(8, 24)
    lir = build_lir(_run(Collide().__call__, _ops(fmt), fmt), "collide_state")
    _elaborate("collide_state", generate(lir).verilog, tmp_path)


@requires_iverilog
def test_ekf1_stateless_elaborates(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    fmt = FloatFormat(6, 18)
    lir = build_lir(_run(ekf1_stateless.update_x_P, _ops(fmt), fmt), "update_x_P")
    _elaborate("update_x_P", generate(lir).verilog, tmp_path)


@requires_iverilog
def test_ekf1_stateful_elaborates(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateful

    fmt = FloatFormat(6, 18)
    filt = ekf1_stateful.Ekf1(
        x=[0.0, 0.0, 0.0], P_urt=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0], R_diag=[1.0, 1.0], Q_diag=np.array([1.0, 1.0, 1.0])
    )
    lir = build_lir(_run(filt.update, _ops(fmt), fmt), "ekf1_stateful")
    _elaborate("ekf1_stateful", generate(lir).verilog, tmp_path)


def test_both_bank_lane_write_commit_rides_the_commit_step() -> None:
    # A pooled lane commits on its commit step: the destination register's write opcode holds that lane's source code
    # at ROM step ``pooled_write_word(commit)`` == the commit step itself (valid on that executing step; one later would
    # land a wide result past the branch's boundary read, which has exactly one cycle of slack). Checked white-box
    # against the microcode tables of a kernel with both lane kinds.
    from holoso._backend.verilog._microcode import (
        OpWriteSource,
        build_microcode,
        f_op,
        read_codebook,
        tapped_lanes,
        write_codebook,
        write_events,
    )
    from holoso._operators import BoolInversion
    from ._modelref import branch_boundary_kernel, fcmp_staged_ops

    fmt = FloatFormat(6, 18)
    lir = build_lir(_run(branch_boundary_kernel, fcmp_staged_ops(fmt, 1), fmt), "lane_steps")
    events = write_events(lir)
    write_books = write_codebook(events)
    fields = build_microcode(lir, read_codebook(lir), write_books, events, tapped_lanes(lir))
    checked_bool = checked_wide = 0
    for op in lir.ops:
        for write in op.writes:
            is_wide = not isinstance(write.dst, BoolRegRef)
            if is_wide:
                invert = False
            else:
                assert isinstance(write.conditioner, BoolInversion)
                invert = write.conditioner.invert
            source = OpWriteSource(op.inst, write.port, invert)
            code = write_books[write.dst].code(source)
            assert (
                fields[f_op(write.dst)].values[pooled_write_word(op.commit_cycle)] == code
            ), "a pooled lane's write opcode must ride the commit step on both banks"
            if is_wide:
                checked_wide += 1
            else:
                checked_bool += 1
    assert checked_bool >= 1 and checked_wide >= 2  # the kernel has a comparison and several float results


def _two_division_kernel(a: float, b: float, c: float, d: float) -> float:
    return a / b + c / d  # two divisions share one fdiv instance and land in two distinct registers


def test_error_gate_ors_over_multiple_landing_registers() -> None:
    # An error-bearing operator (fdiv) whose result lands in >=2 distinct registers reconstructs its commit window as
    # the OR, over those registers, of ``uc_op_<reg> == <its source code>``. No bundled example produces a multi-term
    # err gate, and the numerical model does not simulate ``err``, so cosim cannot reach it -- pin the reconstruction.
    verilog = generate(
        build_lir(_run(_two_division_kernel, _ops(FloatFormat(6, 18)), FloatFormat(6, 18)), "two_div")
    ).verilog
    err = next(line.strip() for line in verilog.splitlines() if line.strip().startswith("assign err ="))
    assert err.count("uc_op_") >= 2 and " | " in err and "div0" in err, err


def test_wide_multi_output_operator_elaborates_with_per_port_lanes(tmp_path: Path) -> None:
    from holoso._lir import (
        Lir,
        LirBlock,
        OperatorInstance,
        PooledScheduledOp,
        PortWrite,
        RegFileLayout,
        Ret,
        WideInputLoad,
        WideOperand,
        WideOutputWire,
        boundary_step,
    )
    from holoso._lir._ir import BoolRegFileLayout
    from holoso._operators import FloatSignControl

    _FETCH_LAG = 2  # datapath lag matching the 3-stage control fetch: one less than fetch_stages

    fmt = FloatFormat(6, 18)
    inst = OperatorInstance(FSortOperator(fmt, FSortOptions()), 0)
    op = PooledScheduledOp(
        inst=inst,
        operands=[WideOperand(RegRef(0), FloatSignControl()), WideOperand(RegRef(1), FloatSignControl())],
        writes=[
            PortWrite(0, RegRef(2), FloatSignControl()),
            PortWrite(1, RegRef(3), FloatSignControl(negate=True)),
        ],
        issue_cycle=1,
        latency=inst.operator.latency,
        immediates=(),
    )
    lir = Lir(
        module_name="fsort_probe",
        instances=[inst],
        wide_consts=[],
        float_format=fmt,
        int_format=default_ifmt(fmt),
        fetch_lag=_FETCH_LAG,
        regfile=RegFileLayout(nreg=4, nrd=2, nwr=2, nload=2),
        inputs=[WideInputLoad("a", RegRef(0), FloatType(fmt)), WideInputLoad("b", RegRef(1), FloatType(fmt))],
        ops=[op],
        outputs=[
            WideOutputWire("out_0", WideOperand(RegRef(2), FloatSignControl()), FloatType(fmt)),
            WideOutputWire("out_1", WideOperand(RegRef(3), FloatSignControl()), FloatType(fmt)),
        ],
        wide_state_slots=[],
        blocks=[LirBlock(0, [op], [], [], [], Ret(), op.commit_cycle, boundary_step(op.commit_cycle, _FETCH_LAG))],
        block_base=[0],
        entry=0,
        last_pc=boundary_step(op.commit_cycle, _FETCH_LAG),
        min_initiation_interval=boundary_step(op.commit_cycle, _FETCH_LAG),
        bool_regfile=BoolRegFileLayout(nreg=0),
        bool_state_slots=[],
    )
    verilog = generate(lir).verilog
    for q in (0, 1):
        # Each per-port result is a combinational output wire (s_..._y{q}, no _q register) that drives the register
        # write directly.
        assert f"_y{q}_q" not in verilog, "the per-port result register must not be emitted"
        assert re.search(
            rf"wire\s+\[WFLT-1:0\]\s+s_fsort_\w+_0_y{q}\s*;", verilog
        ), "per-port combinational result wire"
        assert re.search(
            rf"regs\[\d+\] <= s_fsort_\w+_0_y{q}\b", verilog
        ), "the wide write must read the combinational output wire directly"
        assert re.search(rf"uc_fsort_\w+_0_y{q}sgn\b", verilog)
    assert ".min(" in verilog and ".max(" in verilog and ".min_sgnop(" in verilog and ".max_sgnop(" in verilog
    if shutil.which("iverilog") is None:
        pytest.skip("iverilog not installed")
    _elaborate("fsort_probe", verilog, tmp_path)


def _and_gate(a: bool, b: bool, /) -> bool:
    return a and b


def _madd_only(a: float, b: float, c: float) -> float:
    return a * b + c


@requires_iverilog
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


class SharedLiveOut:
    """
    Two slots ending the transaction holding one value. The read-modify-write pair frees ``a``'s home register
    mid-transaction and the allocator reuses it, so a boundary-installing slot's register also carries opcode writes --
    the shape the emitter used to refuse outright.
    """

    def __init__(self) -> None:
        self.a = 0.0
        self.b = 1.0

    def step(self, x: float) -> float:
        self.a = x + self.a
        self.a = x + self.a
        self.b = self.a
        return self.b


@requires_iverilog
def test_a_boundary_install_coexisting_with_opcode_writes_elaborates(tmp_path: Path) -> None:
    # The boundary install outranks the opcode arm on the same register, so this shape used to be refused outright.
    # It is safe because the install executes at present_step and every write event rides a strictly earlier step; the
    # kernel is cosimulated in test_cosim.py, so here only the premise and the elaboration are checked.
    from holoso._backend.verilog._microcode import write_events

    lir = build_lir(_run(SharedLiveOut().step, _ops(FloatFormat(6, 18)), FloatFormat(6, 18)), "shared_live_out")
    steps: dict[object, list[int]] = {}
    for event in write_events(lir):
        steps.setdefault(event.dst, []).append(event.step)
    coexisting = [
        slot
        for slot in lir.wide_state_slots
        if slot.needs_copy and lir.wide_state_install_is_boundary(slot) and slot.reg in steps
    ]
    assert coexisting, "the premise needs a boundary-installing slot whose own register also takes opcode writes"
    for slot in coexisting:
        assert max(steps[slot.reg]) < lir.present_step, f"{slot.name!r}: {sorted(steps[slot.reg])} vs present_step"
    _elaborate("shared_live_out", generate(lir).verilog, tmp_path)


_INT_OPTIONS = Options(
    OperatorOptions(
        fadd=FAddOptions(),
        fcmp=FCmpOptions(),
        fsort=FSortOptions(),  # a min alone leaves the max lane untapped
        imul=IMulOptions(),
        ffromint=FFromIntOptions(),
        ftoint=FToIntOptions(),
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


def _integer_lir() -> Lir:
    hir = lower(_IntegerKernel().step).hir
    ops = build_ops(_INT_OPTIONS)
    return build_lir(lower_to_mir(hir, ops, _INT_OPTIONS.ffmt, _INT_OPTIONS.ifmt, DEFAULT_IFCONV_MAX_OPS), "int_kernel")


@pytest.mark.whitebox
def test_an_integer_port_binds_no_sign_sideband_and_declares_its_own_width() -> None:
    """
    The sideband exists only on a float port, and the read mux feeding an integer one must be as wide as the
    register rather than as the float -- the silent half, which elaborates either way and drops the top bits.
    """
    verilog = generate(_integer_lir()).verilog
    ffromint, ftoint, iadds = (_instantiation(verilog, name) for name in ("ffromint", "ftoint", "iadds"))
    assert ".a_sgnop(" not in ffromint and ".y_sgnop(" in ffromint  # integer operand, float result
    assert ".a_sgnop(" in ftoint and ".y_sgnop(" not in ftoint  # float operand, integer result
    assert "_sgnop(" not in iadds
    assert re.search(r"reg  \[WINT-1:0\] s_iadds_\w+_a;", verilog)
    assert re.search(r"wire \[WINT-1:0\] s_ftoint_\w+_y0;", verilog)
    assert re.search(r"reg  \[WFLT-1:0\] s_ftoint_\w+_a;", verilog)


@pytest.mark.whitebox
def test_only_a_float_port_is_allocated_a_microcode_sign_field() -> None:
    """Expected from each instance's own signature, so a mixed-family operator cannot slip through by its name."""
    lir = _integer_lir()
    verilog = generate(lir).verilog
    tapped = tapped_lanes(lir)
    expected = set()
    for inst in lir.instances:
        base = base_name(inst)
        signature = inst.operator.signature
        expected |= {
            f"uc_{base}_{PORT_LETTERS[pos]}sgn"
            for pos, ty in enumerate(signature.operand_types)
            if isinstance(ty, FloatType)
        }
        expected |= {  # an untapped result drives nothing, so it is not allocated a field either
            f"uc_{base}_y{q}sgn"
            for q, ty in enumerate(signature.result_types)
            if isinstance(ty, FloatType) and (inst, q) in tapped
        }
    assert set(re.findall(r"\buc_\w+?sgn\b", verilog)) == expected
    assert any("ffromint" in name for name in expected), "the mixed-family conversions must be in this module"


@pytest.mark.whitebox
def test_an_integer_state_slot_resets_to_its_own_word() -> None:
    """
    Silent otherwise: the float codec answers a legal-looking literal for every integer, so the slot comes up
    holding the encoding of a float rather than its own reset, at the float's width rather than the register's.
    """
    lir = _integer_lir()
    verilog = generate(lir).verilog
    snapshots = [text.strip() for text in verilog.splitlines() if "reset snapshot" in text]
    assert any("<= 34'h3fffffffd;" in text for text in snapshots), snapshots  # -3 in two's complement at WREG
    # The boundary install taps the slot through its conditioner, the one place an integer tap meets that arm.
    (slot,) = [s for s in lir.wide_state_slots if s.needs_copy and lir.wide_state_install_is_boundary(s)]
    assert isinstance(slot.tap.source, RegRef)
    assert f"if (out_valid && out_ready) regs[{slot.reg.index}] <= regs[{slot.tap.source.index}];" in verilog


@pytest.mark.whitebox
def test_an_integer_port_declares_itself_signed() -> None:
    """A wider signed consumer zero-fills an unsigned port, so a negative integer arrives as a large positive one."""
    verilog = generate(_integer_lir()).verilog
    declared = {name: qualifiers for qualifiers, name in re.findall(r"wire (.*?)(\w+),$", verilog, re.M)}
    assert declared["in_a"] == "signed [33:0] " and declared["out_0"] == "signed [33:0] "
    assert declared["in_x"] == "[23:0] ", "a float is a bit pattern, not a signed number"


@pytest.mark.whitebox
@requires_iverilog
def test_an_integer_kernel_emits_rtl_that_elaborates(tmp_path: Path) -> None:
    """Every wide site keyed on float rather than on the port's own family, so none of this could be rendered."""
    _elaborate("int_kernel", generate(_integer_lir()).verilog, tmp_path)


def offset_by_one(x: float) -> float:
    return x + 1.0


def test_a_float_write_fills_the_wide_high_bits_with_dont_care() -> None:
    """The adopted bit-fill policy: don't-care high bits when WREG > WFLT, no fill machinery at all at gap 0."""
    gapped = synthesize(
        offset_by_one, Options(OperatorOptions(fadd=FAddOptions()), ffmt=FloatFormat(6, 18), wint_min=33), name="Gap9"
    )
    assert "{{(WREG-WFLT){1'bx}}, s_fadd" in gapped.verilog_output.verilog
    flat = synthesize(offset_by_one, Options(OperatorOptions(fadd=FAddOptions()), ffmt=FloatFormat(6, 18)), name="Gap0")
    assert "1'bx" not in flat.verilog_output.verilog

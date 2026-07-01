"""Elaboration tests for the generated Verilog backend (structural correctness under Icarus)."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from holoso import (
    BoolType,
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FloatFormat,
    FMulILog2OperatorFamily,
    FMulOperator,
    OpConfig,
    UnsupportedConstruct,
)
from holoso._backend.verilog import generate
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import BoolRegRef, RegRef, build, pooled_write_word
from holoso._mir import lower as lower_to_mir

from .hdl.hdl_float_oracle import HDL_DIR, sources

requires_iverilog = pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")


def _ops(fmt: FloatFormat) -> OpConfig:
    return OpConfig(
        FAddOperator(fmt), FMulOperator(fmt), FDivOperator(fmt), FMulILog2OperatorFamily(fmt), FCmpOperator(fmt)
    )


def _run(target, ops: OpConfig):  # type: ignore[no-untyped-def]
    return lower_to_mir(optimize(lower(target)), ops)


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
    def scale(a, b):  # type: ignore[no-untyped-def]
        return a * 4.0 + b * 8.0

    fmt = FloatFormat(6, 18)
    lir = build(_run(scale, _ops(fmt)), "scale", fetch_stages=3)
    names = re.findall(
        r"\bholoso_fmul_ilog2_const\s+#\([^;]+?\)\s+u_([A-Za-z_][A-Za-z0-9_]*)\s+\(", generate(lir).verilog
    )

    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"fmul_ilog2_const_[0-9a-f]{8}_0", name) for name in names)
    assert all("stage_decode" not in name and "e6_m18" not in name and "_k_" not in name for name in names)
    assert all(name == name.lower() for name in names)


@requires_iverilog
def test_comparisons_share_one_pooled_fcmp_instance() -> None:
    # Comparisons live in mutually-exclusive blocks and execute sequentially, so they share a single holoso_fcmp
    # (the one-instance-per-operator pooling convention), its operands riding the ordinary microcode read-mux
    # lanes -- not one instance per comparison.
    def kernel(x):  # type: ignore[no-untyped-def]
        if x > 1.0:
            y = x + 1.0
        elif x < -1.0:
            y = x - 1.0
        else:
            y = x
        return y

    verilog = generate(build(_run(kernel, _ops(FloatFormat(8, 24))), "two_cmp", fetch_stages=3)).verilog
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


@requires_iverilog
def test_small_kernel_elaborates(tmp_path: Path) -> None:
    def kernel(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    fmt = FloatFormat(8, 24)
    lir = build(_run(kernel, _ops(fmt)), "kernel", fetch_stages=3)
    _elaborate("kernel", generate(lir).verilog, tmp_path)


@requires_iverilog
def test_kernel_with_division_elaborates(tmp_path: Path) -> None:
    def blend(a, b, c):  # type: ignore[no-untyped-def]
        return a / b + c * 2.0

    fmt = FloatFormat(6, 18)
    lir = build(_run(blend, _ops(fmt)), "blend", fetch_stages=3)
    _elaborate("blend", generate(lir).verilog, tmp_path)


@requires_iverilog
def test_constant_only_module_elaborates(tmp_path: Path) -> None:
    # No inputs and an all-constant output => zero registers; NREG must floor to >=1 so the regfile parameter
    # guard does not instantiate its error stub (BUG1 regression).
    def const_only():  # type: ignore[no-untyped-def]
        return 3.5

    fmt = FloatFormat(8, 24)
    lir = build(_run(const_only, _ops(fmt)), "const_only", fetch_stages=3)
    _elaborate("const_only", generate(lir).verilog, tmp_path)


def test_boolean_output_port_is_one_bit_and_assigned() -> None:
    class Trigger:
        def __init__(self) -> None:
            self.high = 1.0
            self.low = -1.0
            self.y = False

        def __call__(self, x):  # type: ignore[no-untyped-def]
            if x > self.high:
                self.y = True
            elif x < self.low:
                self.y = False
            return self.y

    fmt = FloatFormat(8, 24)
    lir = build(_run(Trigger().__call__, _ops(fmt)), "bool_trigger", fetch_stages=3)
    (port,) = [port for port in lir.output_ports if port.name == "state_y"]
    assert isinstance(port.scalar_type, BoolType)
    assert port.width == 1
    verilog = generate(lir).verilog
    assert re.search(r"\boutput wire state_y\b", verilog)
    assert re.search(r"\bassign state_y = (?:1'b[01]|bregs\[\d+\]);", verilog)


def test_boolean_input_port_is_one_bit_and_loaded() -> None:
    def passthrough(flag: bool):  # type: ignore[no-untyped-def]
        return flag

    fmt = FloatFormat(8, 24)
    lir = build(_run(passthrough, _ops(fmt)), "bool_input", fetch_stages=3)
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
    lir = build(_run(Toggle().__call__, _ops(fmt)), "bool_toggle", fetch_stages=3)
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
    def collide(valid, ready):  # type: ignore[no-untyped-def]
        return valid + ready

    fmt = FloatFormat(6, 18)
    with pytest.raises(UnsupportedConstruct, match="duplicate port"):
        build(_run(collide, _ops(fmt)), "collide", fetch_stages=3)


def test_kernel_without_outputs_is_rejected() -> None:
    def empty(x):  # type: ignore[no-untyped-def]
        return ()

    fmt = FloatFormat(6, 18)
    with pytest.raises(UnsupportedConstruct, match="at least one output"):
        build(_run(empty, _ops(fmt)), "empty", fetch_stages=3)


@requires_iverilog
def test_state_port_name_does_not_collide_with_internal_sign_wire(tmp_path: Path) -> None:
    # A public attribute `y_d` becomes the port state_y_d; a sibling slot `y` whose boundary copy carries a folded sign
    # used to emit an internal wire also named state_y_d, producing a duplicate (multiply-driven) identifier. The
    # internal sign wire must live outside the state_<attr> port namespace, so the module elaborates cleanly.
    class Collide:
        def __init__(self) -> None:
            self.y = 0.0
            self.y_d = 0.0
            self._p = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            self.y_d = self._p
            self.y = -self._p  # sign-flipped boundary copy -> internal sign-conditioning wire for slot y
            self._p = x
            return self.y

    fmt = FloatFormat(8, 24)
    lir = build(_run(Collide().__call__, _ops(fmt)), "collide_state", fetch_stages=3)
    _elaborate("collide_state", generate(lir).verilog, tmp_path)


@requires_iverilog
def test_ekf1_stateless_elaborates(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    fmt = FloatFormat(6, 18)
    lir = build(_run(ekf1_stateless.update_x_P, _ops(fmt)), "update_x_P", fetch_stages=3)
    _elaborate("update_x_P", generate(lir).verilog, tmp_path)


@requires_iverilog
def test_ekf1_stateful_elaborates(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateful

    fmt = FloatFormat(6, 18)
    filt = ekf1_stateful.Ekf1(
        x=[0.0, 0.0, 0.0], P_urt=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0], R_diag=[1.0, 1.0], Q_diag=np.array([1.0, 1.0, 1.0])
    )
    lir = build(_run(filt.update, _ops(fmt)), "ekf1_stateful", fetch_stages=3)
    _elaborate("ekf1_stateful", generate(lir).verilog, tmp_path)


def test_both_bank_lane_write_enables_ride_the_commit_step() -> None:
    # A pooled lane's write-enable -- boolean OR wide -- sits at ROM step
    # ``pooled_write_word(commit)``, which is the commit step itself (the flag is valid on that executing step;
    # one later would land a wide result past the branch's boundary read, which has exactly one cycle of slack).
    # Checked white-box against the microcode tables of a kernel with both lane kinds.
    from holoso._backend.verilog._microcode import (
        base_name,
        build_microcode,
        f_wen,
        port_const_map,
        read_ports,
        write_target_lists,
    )
    from holoso._lir import BoolRegRef as LirBoolRegRef
    from ._modelref import branch_boundary_kernel, fcmp_staged_ops

    fmt = FloatFormat(6, 18)
    lir = build(_run(branch_boundary_kernel, fcmp_staged_ops(fmt, 1)), "lane_steps", fetch_stages=3)
    read_port = read_ports(lir)
    write_lists = write_target_lists(lir)
    fields = build_microcode(lir, read_port, port_const_map(lir, read_port), write_lists)
    checked_bool = checked_wide = 0
    for op in lir.ops:
        for write in op.writes:
            field = fields[f_wen(base_name(op.inst), write.port)]
            is_wide = not isinstance(write.dst, LirBoolRegRef)
            assert (
                field.values[pooled_write_word(op.commit_cycle)] == 1
            ), "a pooled lane's write-enable must ride the commit step on both banks"
            if is_wide:
                checked_wide += 1
            else:
                checked_bool += 1
    assert checked_bool >= 1 and checked_wide >= 2  # the kernel has a comparison and several float results


def test_wide_multi_output_operator_elaborates_with_per_port_lanes(tmp_path: Path) -> None:
    # No shipped operator has several WIDE outputs yet (fsort will), so the per-port wide lane machinery -- per-output
    # write-enable/address and sign-conditioner fields, each result driven combinationally from the operator into its
    # register write -- is exercised with a synthetic two-output operator on a hand-built
    # Lir, down to Icarus elaboration against a matching stub module.
    from dataclasses import dataclass
    from typing import ClassVar

    from holoso._lir import (
        FloatInputLoad,
        FloatOperand,
        FloatOutputWire,
        Lir,
        LirBlock,
        OperatorInstance,
        PooledScheduledOp,
        PortWrite,
        RegFileLayout,
        Ret,
        boundary_step,
    )
    from holoso._lir._ir import BoolRegFileLayout
    from holoso._operators import FloatHardwareOperator, FloatSignControl
    from holoso._type import ScalarSignature, FloatType as ScalarFloatType

    _FETCH_LAG = 2  # datapath lag matching the 3-stage control fetch: one less than fetch_stages

    @dataclass(frozen=True, slots=True)
    class _SortLike(FloatHardwareOperator):
        mnemonic: ClassVar[str] = "fsortlike"
        output_hdl_ports: ClassVar[list[str]] = ["min", "max"]

        @property
        def latency(self) -> int:
            return 1

        @property
        def signature(self) -> ScalarSignature:
            ty = ScalarFloatType(self.fmt)
            return ScalarSignature((ty, ty), (ty, ty))

        def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
            return f"sortlike({operands[0]},{operands[1]})"

        def hdl_params(self) -> dict[str, int]:
            return {}

        def evaluate(self, *operands, immediates=()):  # type: ignore[no-untyped-def]
            a, b = self._validated_operands(operands, 2)
            return (a, b)  # semantics are irrelevant here; only the lane structure is under test

    fmt = FloatFormat(6, 18)
    inst = OperatorInstance(_SortLike(fmt), 0)
    op = PooledScheduledOp(
        inst=inst,
        operands=[FloatOperand(RegRef(0)), FloatOperand(RegRef(1))],
        writes=[
            PortWrite(0, RegRef(2), FloatSignControl()),
            PortWrite(1, RegRef(3), FloatSignControl(negate=True)),
        ],
        issue_cycle=1,
        latency=1,
        immediates=(),
    )
    lir = Lir(
        module_name="sortlike_probe",
        instances=[inst],
        float_consts=[],
        float_format=fmt,
        fetch_lag=_FETCH_LAG,
        regfile=RegFileLayout(width=fmt.width, nreg=4, nrd=2, nwr=2, nload=2),
        inputs=[FloatInputLoad("a", RegRef(0)), FloatInputLoad("b", RegRef(1))],
        ops=[op],
        outputs=[FloatOutputWire("out_0", FloatOperand(RegRef(2))), FloatOutputWire("out_1", FloatOperand(RegRef(3)))],
        float_state_slots=[],
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
            rf"wire\s+\[W-1:0\]\s+s_fsortlike_\w+_0_y{q}\s*;", verilog
        ), "per-port combinational result wire"
        assert re.search(
            rf"regs\[\d+\] <= s_fsortlike_\w+_0_y{q}\b", verilog
        ), "the wide write must read the combinational output wire directly"
        assert re.search(rf"uc_wen_fsortlike_\w+_0_y{q}\b", verilog)
        assert re.search(rf"uc_fsortlike_\w+_0_y{q}sgn\b", verilog)
    assert ".min(" in verilog and ".max(" in verilog and ".min_sgnop(" in verilog and ".max_sgnop(" in verilog
    if shutil.which("iverilog") is None:
        pytest.skip("iverilog not installed")
    stub = """
module holoso_fsortlike#(parameter WEXP=6, parameter WMAN=18, parameter integer LATENCY=0) (
    input  wire clk, input wire rst, input wire in_valid,
    input  wire [1:0] a_sgnop, input wire [1:0] b_sgnop,
    input  wire [1:0] min_sgnop, input wire [1:0] max_sgnop,
    input  wire [WEXP+WMAN-1:0] a, input wire [WEXP+WMAN-1:0] b,
    output wire out_valid,
    output wire [WEXP+WMAN-1:0] min, output wire [WEXP+WMAN-1:0] max
);
    assign out_valid = 1'b0;
    assign min = a;
    assign max = b;
endmodule
"""
    top = tmp_path / "sortlike_probe.v"
    top.write_text(verilog)
    (tmp_path / "stub.v").write_text(stub)
    result = subprocess.run(
        ["iverilog", "-g2012", "-I", str(HDL_DIR), "-s", "sortlike_probe", "-o", str(tmp_path / "out")]
        + [str(top), str(tmp_path / "stub.v")]
        + [str(s) for s in sources()],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _and_gate(a: bool, b: bool, /):  # type: ignore[no-untyped-def]
    return a and b


def _madd_only(a, b, c):  # type: ignore[no-untyped-def]
    return a * b + c


@requires_iverilog
def test_unused_register_bank_is_omitted(tmp_path: Path) -> None:
    # A purely-boolean kernel uses no wide bank, and an arithmetic kernel with no booleans uses no boolean bank; the
    # unused bank must be omitted entirely rather than declared as a zero-length reg array (illegal Verilog).
    bool_lir = build(_run(_and_gate, _ops(FloatFormat(8, 24))), "and_gate", fetch_stages=3)
    assert bool_lir.regfile.nreg == 0
    bool_v = generate(bool_lir).verilog
    assert "reg  [W-1:0] regs" not in bool_v and "NREG" not in bool_v and "[0:-1]" not in bool_v
    _elaborate("and_gate", bool_v, tmp_path)

    float_lir = build(_run(_madd_only, _ops(FloatFormat(8, 24))), "madd_only", fetch_stages=3)
    assert float_lir.bool_regfile.nreg == 0
    float_v = generate(float_lir).verilog
    assert "bregs" not in float_v and "NBREG" not in float_v and "[0:-1]" not in float_v
    _elaborate("madd_only", float_v, tmp_path)

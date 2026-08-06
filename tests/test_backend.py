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
    FAddOptions,
    FCmpOptions,
    FDivOptions,
    FMulILog2Options,
    FMulOptions,
    FSortOptions,
    FloatFormat,
    IMulOptions,
    IntFormat,
    OperatorOptions,
    Options,
    UnsupportedConstruct,
)
from holoso._operators import (
    FSortOperator,
    IAbsOperator,
    IAddOperator,
    ICmpOperator,
    IDivOperator,
    IMulOperator,
    IShiftOperator,
    ISubOperator,
    IntHardwareOperator,
    OpConfig,
)
from holoso._backend.verilog import generate
from holoso._eel import lower
from holoso._hir import optimize
from holoso._lir import BoolRegRef, RegRef, pooled_write_word
from holoso._mir import Mir, lower as lower_to_mir

from .hdl.hdl_float_oracle import HDL_DIR, sources
from ._modelref import DEFAULT_IFCONV_MAX_OPS, DEFAULT_IFMT, build_lir, build_ops

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
    return lower_to_mir(optimize(lower(target).hir, DEFAULT_IFCONV_MAX_OPS), ops, fmt, DEFAULT_IFMT)


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


def _integer_operators(ifmt: IntFormat) -> list[IntHardwareOperator]:
    return [
        IAddOperator(ifmt),
        ISubOperator(ifmt),
        IDivOperator(ifmt),
        IAbsOperator(ifmt),
        IShiftOperator(ifmt),
        ICmpOperator(ifmt),
        *(IMulOperator(ifmt, IMulOptions(stage_product=stage)) for stage in range(5)),
    ]


def _integer_probe(name: str, operators: list[IntHardwareOperator]) -> str:
    """A module instantiating each operator through the port names and parameters it declares for itself."""
    lines = [f"module {name};", "    wire clk = 1'b0;", "    wire rst = 1'b0;", "    wire in_valid = 1'b0;"]
    for index, operator in enumerate(operators):
        high = operator.fmt.width - 1
        operands = operator.operand_hdl_ports
        results = [operator.output_hdl_ports[q] for q in range(len(operator.signature.result_types))]
        for port in operands:
            lines.append(f"    wire signed [{high}:0] u{index}_{port} = {operator.fmt.width}'d0;")
        for port, result_type in zip(results, operator.signature.result_types, strict=True):
            lines.append(f"    wire {f'signed [{high}:0] ' if result_type.is_wide else ''}u{index}_{port};")
        for port in operator.error_ports:
            lines.append(f"    wire u{index}_{port};")
        params = ", ".join(f".{pname}({value})" for pname, value in operator.params.items())
        # out_valid and the saturation sideband are deliberately left out: an omitted named port is unconnected.
        connections = ", ".join(f".{port}(u{index}_{port})" for port in operands + results)
        errors = "".join(f", .{port}(u{index}_{port})" for port in operator.error_ports)
        lines.append(
            f"    {operator.module_name} #({params}) u{index} "
            f"(.clk(clk), .rst(rst), .in_valid(in_valid), {connections}{errors});"
        )
    return "\n".join([*lines, "endmodule", ""])


@requires_iverilog
@pytest.mark.parametrize("width", (2, 3, 24, 33, 44))
def test_integer_operators_elaborate_as_they_declare_themselves(width: int, tmp_path: Path) -> None:
    # A wrong latency instantiates the undefined _holoso_invalid_integer_latency; an odd width is mandatory because
    # that is where the divider's ceiling can slip.
    name = f"int_probe_w{width}"
    _elaborate(name, _integer_probe(name, _integer_operators(IntFormat(width))), tmp_path)


@requires_iverilog
def test_integer_wrapper_rejects_wrong_latency(tmp_path: Path) -> None:
    # The negative twin of the probe above, so its silence means something.
    operator = IDivOperator(IntFormat(33))
    verilog = _integer_probe("wrong_int_latency", [operator]).replace(
        f".LATENCY({operator.latency})", f".LATENCY({operator.latency + 1})"
    )
    result = _compile("wrong_int_latency", verilog, tmp_path)
    assert result.returncode != 0
    assert "_holoso_invalid_integer_latency" in result.stderr


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
        int_format=DEFAULT_IFMT,
        fetch_lag=_FETCH_LAG,
        regfile=RegFileLayout(width=fmt.width, nreg=4, nrd=2, nwr=2, nload=2),
        inputs=[WideInputLoad("a", RegRef(0)), WideInputLoad("b", RegRef(1))],
        ops=[op],
        outputs=[
            WideOutputWire("out_0", WideOperand(RegRef(2), FloatSignControl())),
            WideOutputWire("out_1", WideOperand(RegRef(3), FloatSignControl())),
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
    # A purely-boolean kernel uses no wide bank, and an arithmetic kernel with no booleans uses no boolean bank; the
    # unused bank must be omitted entirely rather than declared as a zero-length reg array (illegal Verilog).
    bool_lir = build_lir(_run(_and_gate, _ops(FloatFormat(8, 24)), FloatFormat(8, 24)), "and_gate")
    assert bool_lir.regfile.nreg == 0
    bool_v = generate(bool_lir).verilog
    assert "reg  [WFLT-1:0] regs" not in bool_v and "NREG" not in bool_v and "[0:-1]" not in bool_v
    _elaborate("and_gate", bool_v, tmp_path)

    float_lir = build_lir(_run(_madd_only, _ops(FloatFormat(8, 24)), FloatFormat(8, 24)), "madd_only")
    assert float_lir.bool_regfile.nreg == 0
    float_v = generate(float_lir).verilog
    assert "bregs" not in float_v and "NBREG" not in float_v and "[0:-1]" not in float_v
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

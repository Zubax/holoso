"""Test-only driver that compiles a kernel and cosimulates the generated module against its bit-exact model."""

from collections.abc import Callable, Mapping

from holoso import FloatFormat, OpConfig
from holoso._backend.cocotb import generate as generate_testbench
from holoso._backend.numerical import generate as build_model
from holoso._backend.verilog import generate as generate_verilog
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import build
from holoso._mir import lower as lower_to_mir
from cocotb_tools.runner import get_runner

from ._modelref import default_ops
from .hdl.hdl_float_oracle import HDL_DIR, REPO_ROOT, build_args, sources


def run_cosim(
    sim: str,
    fn: Callable[..., object],
    fmt: FloatFormat,
    name: str,
    ops: OpConfig | None = None,
    vectors: list[Mapping[str, int]] | None = None,
) -> None:
    """
    Compile ``fn``, emit its Verilog and a self-checking cocotb bench, and run the bench on ``sim``.

    ``ops`` defaults to the minimum-latency configuration (no optional stages). ``vectors`` is an explicit input
    sequence (each maps an input-port name to its ZKF bits); when omitted the bench draws its own fixed-seed sweep.
    """
    ops = default_ops(fmt) if ops is None else ops
    lir = build(lower_to_mir(optimize(lower(fn)), ops), name)
    model = build_model(lir)
    # Generated sources live outside the cocotb build dir, which the runner wipes on clean=True.
    gen_dir = REPO_ROOT / "build" / "holoso_gen" / f"{name}_w{fmt.wexp}_{fmt.wman}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"synth_{name}_w{fmt.wexp}_{fmt.wman}"
    verilog_path = gen_dir / f"{name}.v"
    verilog_path.write_text(generate_verilog(lir).verilog)
    # The generated bench embeds the bit-exact model and checks the DUT's output bits exactly.
    test_module = f"test_{name}"
    (gen_dir / f"{test_module}.py").write_text(generate_testbench(model, vectors).testbench)

    runner = get_runner(sim)
    runner.build(
        sources=[verilog_path, *sources()],
        includes=[HDL_DIR],
        hdl_toplevel=name,
        build_args=build_args(sim),
        build_dir=str(build_dir),
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel=name,
        test_module=test_module,
        test_dir=str(gen_dir),
        build_dir=str(build_dir),
        results_xml=str(build_dir / "results.xml"),
    )

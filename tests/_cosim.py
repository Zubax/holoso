"""Test-only driver that cosimulates a synthesized module against its embedded bit-exact model."""

from collections.abc import Mapping, Sequence

from holoso import SynthesisResult
from holoso._backend.cocotb import generate as generate_testbench
from holoso._value import ScalarLike
from cocotb_tools.runner import get_runner

from .hdl.hdl_float_oracle import HDL_DIR, REPO_ROOT, build_args, sources


def run_cosim(sim: str, result: SynthesisResult, vectors: Sequence[Mapping[str, ScalarLike]] | None = None) -> None:
    """
    ``result`` is the caller's single public ``holoso.synthesize`` product; ``sim`` plus the module name (unique per
    machine across the suite) identify the build. ``vectors`` is an explicit input sequence (each maps an input-port
    name to its typed scalar value); when omitted the shipped bench replays its own fixed-seed sweep. There is no
    public way to hand vectors to the bench, so the explicit-vectors path re-renders it through the private backend.
    """
    name = result.module_name
    # Generated sources live outside the cocotb build dir, which the runner wipes on clean=True.
    gen_dir = REPO_ROOT / "build" / "holoso_gen" / sim / name
    gen_dir.mkdir(parents=True, exist_ok=True)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"synth_{name}"
    verilog_path = gen_dir / f"{name}.v"
    verilog_path.write_text(result.verilog_output.verilog)
    test_module = f"test_{name}"
    bench = result.cocotb_output if vectors is None else generate_testbench(result.numerical_model, vectors)
    (gen_dir / f"{test_module}.py").write_text(bench.testbench)

    runner = get_runner(sim)
    runner.build(
        sources=[verilog_path, *sources()],
        includes=[HDL_DIR],
        hdl_toplevel=name,
        build_args=build_args(sim),
        defines={"SIMULATION": 1},  # arm the ZKF cores' and CORDIC wrappers' $fatal protocol/over-issue checks
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

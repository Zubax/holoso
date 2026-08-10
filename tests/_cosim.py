"""Test-only driver that compiles a kernel and cosimulates the generated module against its bit-exact model."""

from collections.abc import Callable, Mapping, Sequence

from holoso import Options
from holoso._backend.cocotb import generate as generate_testbench
from holoso._backend.numerical import generate
from holoso._backend.verilog import generate as generate_verilog
from holoso._eel import lower
from holoso._value import ScalarLike
from ._modelref import build_lir, build_ops
from holoso._operators import OpConfig
from holoso._mir import lower as lower_to_mir
from cocotb_tools.runner import get_runner

from .hdl.hdl_float_oracle import HDL_DIR, REPO_ROOT, build_args, sources


def run_cosim(
    sim: str,
    fn: Callable[..., object],
    options: Options,
    name: str,
    ops: OpConfig | None = None,
    vectors: Sequence[Mapping[str, ScalarLike]] | None = None,
) -> None:
    """
    ``options`` names the formats and the default operator set; ``ops`` overrides the operator configuration alone.
    ``vectors`` is an explicit input sequence (each maps an input-port name to its typed scalar value); when omitted
    the bench draws its own fixed-seed sweep.
    """
    ops = build_ops(options) if ops is None else ops
    fmt, ifmt = options.ffmt, options.ifmt
    lir = build_lir(lower_to_mir(lower(fn).hir, ops, fmt, ifmt, options.ifconv_max_ops), name)
    model = generate(lir)
    # Generated sources live outside the cocotb build dir, which the runner wipes on clean=True.
    gen_dir = REPO_ROOT / "build" / "holoso_gen" / f"{name}_w{fmt.wexp}_{fmt.wman}_r{ifmt.width}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"synth_{name}_w{fmt.wexp}_{fmt.wman}_r{ifmt.width}"
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

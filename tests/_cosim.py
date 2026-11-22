"""Test-only driver that cosimulates a synthesized module against its embedded bit-exact model."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from holoso import SynthesisResult
from holoso._backend.cocotb import generate as generate_testbench
from holoso._value import ScalarLike
from cocotb_tools.runner import get_runner

from .hdl.hdl_float_oracle import REPO_ROOT, build_args


def run_cosim(sim: str, result: SynthesisResult, vectors: Sequence[Mapping[str, ScalarLike]] | None = None) -> None:
    """
    `result` is the caller's single public `holoso.synthesize` product; `sim` plus the module name (unique per
    machine across the suite) identify the build. `vectors` is an explicit input sequence (each maps an input-port
    name to its typed scalar value); when omitted the shipped bench replays its own fixed-seed sweep. There is no
    public way to hand vectors to the bench, so the explicit-vectors path re-renders it through the private backend.
    The kernel's file is the only source: it is self-contained.
    """
    name = result.module_name
    verilog_path = generated_dir(sim, name) / f"{name}.v"
    verilog_path.write_text(result.verilog_output.verilog)
    bench = result.cocotb_output if vectors is None else generate_testbench(result.numerical_model, vectors)
    run_bench(sim, name, [verilog_path], name, bench.testbench)


def generated_dir(sim: str, label: str) -> Path:
    """Where the sources of the build `label` are written: outside the cocotb build dir, which a clean build wipes."""
    path = REPO_ROOT / "build" / "holoso_gen" / sim / label
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_bench(sim: str, label: str, sources: Sequence[Path], toplevel: str, bench: str) -> None:
    """Simulate `toplevel` built from `sources` under the cocotb `bench` text; `label` is unique across the suite."""
    gen_dir = generated_dir(sim, label)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"synth_{label}"
    test_module = f"test_{label}"
    (gen_dir / f"{test_module}.py").write_text(bench)
    runner = get_runner(sim)
    runner.build(
        sources=list(sources),
        hdl_toplevel=toplevel,
        build_args=build_args(sim),
        defines={"SIMULATION": 1},  # arms the kernel's over-issue `$fatal` on every handshake operator instance
        build_dir=str(build_dir),
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel=toplevel,
        test_module=test_module,
        test_dir=str(gen_dir),
        build_dir=str(build_dir),
        results_xml=str(build_dir / "results.xml"),
    )


@dataclass(frozen=True, slots=True)
class RenderedInstance:
    """The private module one instance renders, as the kernel's file declares it: its name, ports and knob table."""

    module: str
    ports: frozenset[str]
    knobs: dict[str, str]


def rendered_instance(result: SynthesisResult, instance: str) -> RenderedInstance:
    verilog = result.verilog_output.verilog
    (module,) = re.findall(rf"^(_{result.module_name}_zkf_\w+_[0-9a-f]{{12}}) u_{instance} \(", verilog, re.MULTILINE)
    match = re.search(rf"((?:^//.*\n)*)^module {module}\(([^)]*)\);", verilog, re.MULTILINE)
    assert match is not None, module
    knobs = dict(re.findall(r"^//   (\w+): (.*)$", match.group(1), re.MULTILINE))
    return RenderedInstance(module, frozenset(port.strip() for port in match.group(2).split(",")), knobs)

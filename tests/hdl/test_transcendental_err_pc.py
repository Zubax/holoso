import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cocotb
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import (
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FExp2Operator,
    FloatFormat,
    FLog2Operator,
    FMulILog2OperatorFamily,
    FMulOperator,
    FSortOperator,
    OpConfig,
)
from holoso._backend.verilog import generate
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import build
from holoso._mir import lower as lower_to_mir

from .hdl_float_oracle import HDL_DIR, REPO_ROOT, SIMULATORS, build_args, drive_reset, sources, start_clock

FMT = FloatFormat(6, 18)


def _ops() -> OpConfig:
    return OpConfig(
        FAddOperator(FMT),
        FMulOperator(FMT),
        FDivOperator(FMT),
        FMulILog2OperatorFamily(FMT),
        FCmpOperator(FMT),
        fsort=FSortOperator(FMT),
        fexp2=FExp2Operator(FMT),
        flog2=FLog2Operator(FMT),
    )


def _sqrt(x: float) -> float:
    return math.sqrt(x)


def _hypot(y: float, x: float) -> float:
    return math.hypot(y, x)


@dataclass(frozen=True, slots=True)
class _Case:
    name: str
    fn: Callable[..., float]
    inputs: tuple[str, ...]
    vectors: tuple["_Vector", ...]


@dataclass(frozen=True, slots=True)
class _Vector:
    values: tuple[float, ...]
    err_latched: bool


CASES = (
    _Case(
        "sqrt",
        _sqrt,
        ("x",),
        (
            _Vector((0.0,), False),
            _Vector((-1.0,), True),
            _Vector((0.0,), False),
        ),
    ),
    _Case(
        "hypot",
        _hypot,
        ("y", "x"),
        (
            _Vector((0.0, 0.0), False),
            _Vector((float("inf"), 2.0), False),
        ),
    ),
)


async def _settle(dut: Any) -> None:
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


@cocotb.test()
async def err_pc_safe_transcendentals(dut: Any) -> None:
    config = json.loads(os.environ["HOLOSO_ERR_CASE"])

    await start_clock(dut)
    await drive_reset(dut)
    dut.out_ready.value = 1

    async def invoke(bits: list[int]) -> int:
        while int(dut.in_ready.value) != 1:
            await _settle(dut)
        for name, value in zip(config["inputs"], bits, strict=True):
            getattr(dut, f"in_{name}").value = value
        dut.in_valid.value = 1
        await _settle(dut)
        dut.in_valid.value = 0
        while int(dut.out_valid.value) != 1:
            await _settle(dut)
        latched = int(dut.err_pc.value)
        await _settle(dut)
        return latched

    for vector in config["vectors"]:
        err_pc = await invoke(vector["bits"])
        if vector["err_latched"]:
            assert err_pc != 0, vector["label"]
        else:
            assert err_pc == 0, vector["label"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_safe_transcendental_err_pc(sim: str, case: _Case) -> None:
    lir = build(lower_to_mir(optimize(lower(case.fn)), _ops()), f"err_{case.name}", fetch_stages=3)
    gen_dir = REPO_ROOT / "build" / "holoso_gen" / f"err_{case.name}_w{FMT.wexp}_{FMT.wman}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    verilog_path = gen_dir / f"err_{case.name}.v"
    verilog_path.write_text(generate(lir).verilog)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"err_{case.name}_w{FMT.wexp}_{FMT.wman}"
    vectors = [
        {
            "bits": [FMT.encode(value) for value in vector.values],
            "err_latched": vector.err_latched,
            "label": f"{case.name}{vector.values}",
        }
        for vector in case.vectors
    ]

    runner = get_runner(sim)
    runner.build(
        sources=[verilog_path, *sources()],
        includes=[HDL_DIR],
        hdl_toplevel=f"err_{case.name}",
        build_args=build_args(sim),
        build_dir=str(build_dir),
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel=f"err_{case.name}",
        test_module="tests.hdl.test_transcendental_err_pc",
        test_dir=str(REPO_ROOT),
        build_dir=str(build_dir),
        extra_env={"HOLOSO_ERR_CASE": json.dumps({"inputs": case.inputs, "vectors": vectors})},
        results_xml=str(build_dir / "results.xml"),
    )

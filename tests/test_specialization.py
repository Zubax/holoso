"""
Specialization beyond the shapes the shipped operators happen to have: the test-only operator of `_scaler`, with a
one-bit and a signed multi-bit immediate and two outputs, built through the MIR and the LIR as any kernel is. Its one
instance keeps an immediate exactly where the firings bound to it disagree on it, and a kept negative value reaches the
block as itself.
"""

import os
from collections.abc import Callable

import pytest

from holoso import FloatFormat, IntFormat
from holoso._backend.cocotb import generate as generate_testbench
from holoso._backend.numerical import generate as generate_model
from holoso._backend.verilog import generate
from holoso._lir import Lir
from holoso._mir import Mir
from holoso._mir._ir import MirBuilder
from holoso._operators import PooledPrimitive, identity_conditioner
from holoso._type import IntType

from ._cosim import generated_dir, run_bench
from ._modelref import build_lir
from ._scaler import OffsetPrimitive, ProductPrimitive, ScalerOperator
from .hdl.hdl_float_oracle import REPO_ROOT, SIMULATORS

_FMT, _IFMT = FloatFormat(6, 18), IntFormat(32)
_A, _B = 1_500_000_000, -7

type _Taps = Callable[[ScalerOperator], tuple[list[PooledPrimitive], list[PooledPrimitive]]]

# Each kernel: its taps over `a` and over `b`, the immediates its one instance keeps, and its outputs at (_A, _B), the
# products wrapping at 32 bits.
_KERNELS: dict[str, tuple[_Taps, set[str], list[int]]] = {
    "mode_agreed": (
        lambda op: (
            [OffsetPrimitive(op, False, 3), ProductPrimitive(op, False, 3)],
            [OffsetPrimitive(op, False, -5), ProductPrimitive(op, False, -5)],
        ),
        {"gain"},
        [1_500_000_003, 205_032_704, -12, 35],
    ),
    "gain_agreed": (
        lambda op: ([OffsetPrimitive(op, False, -5)], [OffsetPrimitive(op, True, -5), ProductPrimitive(op, True, -5)]),
        {"mode"},
        [1_499_999_995, -2, 35],
    ),
    "neither_agreed": (
        lambda op: ([OffsetPrimitive(op, True, 7)], [ProductPrimitive(op, False, -5)]),
        {"mode", "gain"},
        [1_499_999_993, 35],
    ),
    "all_agreed": (
        lambda op: ([OffsetPrimitive(op, False, -8), ProductPrimitive(op, False, -8)], []),
        set(),
        [1_499_999_992, 884_901_888],
    ),
}


def _lir(label: str, stages: int) -> Lir:
    over_a, over_b = _KERNELS[label][0](ScalerOperator.build(_IFMT, stage_input=stages, stage_output=stages))
    builder = MirBuilder(_FMT, _IFMT)
    builder.block()
    a = builder.input("a", IntType(_IFMT))
    b = builder.input("b", IntType(_IFMT))
    identity = identity_conditioner(IntType(_IFMT))
    taps = [*((a, primitive) for primitive in over_a), *((b, primitive) for primitive in over_b)]
    for index, (operand, primitive) in enumerate(taps):
        builder.output(f"out_{index}", builder.operation(primitive, [operand], [identity]), identity)
    builder.ret()
    mir: Mir = builder.finish()
    return build_lir(mir, f"scaler_{label}_s{stages}")


@pytest.mark.parametrize("stages", [0, 1])
@pytest.mark.parametrize("label", _KERNELS)
def test_an_instance_keeps_the_immediates_its_firings_disagree_on(label: str, stages: int) -> None:
    _, kept, outputs = _KERNELS[label]
    lir = _lir(label, stages)
    (block,) = lir.instances.values()
    assert {name for name in ("mode", "gain") if name in block.signature.members} == kept
    assert [int(value) for value in generate_model(lir).elaborate().run(_A, _B)] == outputs


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
@pytest.mark.parametrize("stages", [0, 1])
@pytest.mark.parametrize("label", _KERNELS)
def test_a_specialized_instance_runs_as_its_model_cosim(
    label: str, stages: int, sim: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The bench unpickles the model, which holds the test-only operator, so the bench must import this package.
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])))
    lir = _lir(label, stages)
    model = generate_model(lir)
    names = [port.name for port in model.inputs]
    rows = [(_A, _B), (-3, 2_000_000_000), (-(1 << 31), (1 << 31) - 1), (0, 1)]
    bench = generate_testbench(model, [dict(zip(names, row, strict=True)) for row in rows])
    path = generated_dir(sim, lir.module_name) / f"{lir.module_name}.v"
    path.write_text(generate(lir, {}).verilog)
    run_bench(sim, lir.module_name, [path], lir.module_name, bench.testbench)

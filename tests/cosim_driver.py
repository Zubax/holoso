"""Generic cocotb driver that replays a vector spec against a generated Holoso module and checks tolerances.

The vector-spec JSON path is passed via the ``HOLOSO_VECTORS`` environment variable.
"""

from __future__ import annotations

import json
import os

import cocotb
from cocotb.triggers import RisingEdge, Timer

from holoso.format import FloatFormat
from holoso.verify.tolerance import within
from holoso.verify.zkf_codec import decode

from hdl_float_oracle import drive_reset, start_clock


async def _settle(dut) -> None:
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


@cocotb.test()
async def cosim(dut) -> None:
    spec = json.loads(open(os.environ["HOLOSO_VECTORS"], encoding="utf-8").read())
    fmt = FloatFormat(spec["wexp"], spec["wman"])
    timeout = int(spec["timeout_cycles"])

    await start_clock(dut)
    await drive_reset(dut)
    dut.out_ready.value = 1

    for index, vec in enumerate(spec["vectors"]):
        while int(dut.in_ready.value) != 1:
            await _settle(dut)
        for name, bits in vec["in"].items():
            getattr(dut, f"in_{name}").value = int(bits)
        dut.in_valid.value = 1
        await _settle(dut)
        dut.in_valid.value = 0

        cycles = 0
        while int(dut.out_valid.value) != 1:
            await _settle(dut)
            cycles += 1
            assert cycles < timeout, f"vector {index}: timeout after {cycles} cycles waiting for out_valid"

        for name, expected in vec["exp"].items():
            got = decode(fmt, int(getattr(dut, name).value))
            assert within(got, expected, vec["rtol"], vec["atol"]), (
                f"vector {index} port {name}: got {got!r}, expected {expected!r} "
                f"(rtol={vec['rtol']:.3e}, atol={vec['atol']:.3e})"
            )
        await _settle(dut)  # accept the result (out_ready is held high) and return to idle

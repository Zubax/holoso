"""
Custom cocotb bench for ``test_gate_edge.py``: pin the rising edge of the ``transacting`` issue-gate qualifier.

The DUT is a cycle-0-leading kernel. ``transacting`` (the qualifier AND-ed into every operator's ``in_valid``) must be
LOW while the PC dwells idle at pc 0, stay LOW through the FETCH_LAG fetch-fill bubbles after the accept (the in-flight
``ucode[0]`` re-fetches ahead of the genuine step-0), and rise on EXACTLY the FETCH_LAG-th cycle after the accept -- the
cycle the genuine step-0 executes. A late rise drops step-0; an early rise fires a spurious issue in the fill window,
the cosim-invisible hazard the gate exists to stop once iterative operators land. The idle length k and FETCH_LAG come
from the environment so the test can sweep k and stay valid if FETCH_LAG becomes configurable.
"""

import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge


@cocotb.test()
async def transacting_edge(dut) -> None:  # type: ignore[no-untyped-def]
    fetch_lag = int(os.environ["HOLOSO_FETCH_LAG"])
    k = int(os.environ["HOLOSO_DWELL_K"])

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await FallingEdge(dut.clk)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.out_ready.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await FallingEdge(dut.clk)

    # Idle dwell: the PC holds at 0 awaiting in_valid; the gate must hold transacting low every idle cycle.
    for idle in range(k):
        assert int(dut.transacting.value) == 0, f"transacting high during the idle dwell (k={k}, idle cycle {idle})"
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)
    assert int(dut.transacting.value) == 0, f"transacting high on the accept cycle, before any issue (k={k})"

    # Accept: pulse in_valid for exactly one cycle, then run. transacting must stay low through the FETCH_LAG fill
    # bubbles and rise on exactly the FETCH_LAG-th cycle after the accept edge.
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    for cyc in range(1, fetch_lag + 1):
        await FallingEdge(dut.clk)
        got = int(dut.transacting.value)
        if cyc < fetch_lag:
            assert got == 0, f"transacting rose early at accept+{cyc}: spurious issue in the fill window (k={k})"
        else:
            assert got == 1, f"transacting did not rise at accept+{fetch_lag}: step-0 dropped (k={k})"
        if cyc < fetch_lag:
            await RisingEdge(dut.clk)


@cocotb.test()
async def state_inert_during_dwell(dut) -> None:  # type: ignore[no-untyped-def]
    # A cycle-0 constant install to a persistent-state slot sits on ``ucode[0]``. While the PC dwells idle at pc 0 the
    # held word re-fetches every cycle, so without the ``transacting`` gate on the install write-enable it would commit
    # to the state register before any transaction. Assert the slot keeps its reset value across the whole idle dwell.
    k = int(os.environ["HOLOSO_DWELL_K"])
    slot_idx = int(os.environ["HOLOSO_SLOT_IDX"])
    reset_bits = int(os.environ["HOLOSO_SLOT_RESET_BITS"])

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await FallingEdge(dut.clk)
    dut.rst.value = 1
    dut.in_valid.value = 0
    dut.out_ready.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await FallingEdge(dut.clk)

    for idle in range(k + 1):
        got = int(dut.regs[slot_idx].value)
        assert got == reset_bits, (
            f"state slot regs[{slot_idx}] changed during the idle dwell at cycle {idle}: got {got:#x}, "
            f"reset {reset_bits:#x} -- a const-install committed on the held ucode[0]"
        )
        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)

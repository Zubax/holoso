"""Yosys + nextpnr synthesis check for the Holoso support register file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from synth import yosys

REPO_ROOT = Path(__file__).resolve().parents[1]
HDL_DIR = REPO_ROOT / "hdl"
SUPPORT_RTL = HDL_DIR / "holoso_support.v"
TOP = "holoso_regfile_ooc"
TARGET_FREQ_MHZ = 100.0
ECP5_PACKAGE = os.environ.get("ECP5_PACKAGE", "CABGA381")
ECP5_SPEED_GRADE = os.environ.get("ECP5_SPEED_GRADE", "6")
ECP5_FABRIC_RESOURCES = (
    "TRELLIS_COMB",
    "TRELLIS_FF",
    "TRELLIS_IO",
    "TRELLIS_RAMW",
    "DP16KD",
    "MULT18X18D",
    "ALU54B",
)


@dataclass(frozen=True)
class RegfileConfig:
    w: int
    waddr: int
    nreg: int
    nrd: int
    nwr: int
    rwpass: int

    @property
    def wr_en_width(self) -> int:
        return self.nwr

    @property
    def wr_addr_width(self) -> int:
        return self.nwr * self.waddr

    @property
    def wr_data_width(self) -> int:
        return self.nwr * self.w

    @property
    def rd_addr_width(self) -> int:
        return self.nrd * self.waddr

    @property
    def rd_data_width(self) -> int:
        return self.nrd * self.w


REGFILE = RegfileConfig(w=44, waddr=6, nreg=8, nrd=8, nwr=8, rwpass=1)


def _bus_range(width: int) -> str:
    return f"[{width - 1}:0] "


def write_regfile_wrapper(path: Path, cfg: RegfileConfig) -> None:
    path.write_text(f"""`default_nettype none

module {TOP} (
    input  wire                         clk,
    input  wire {_bus_range(cfg.wr_en_width)}wr_en,
    input  wire {_bus_range(cfg.wr_addr_width)}wr_addr,
    input  wire {_bus_range(cfg.wr_data_width)}wr_data,
    input  wire {_bus_range(cfg.rd_addr_width)}rd_addr,
    output wire {_bus_range(cfg.rd_data_width)}rd_data
);
    // Measurement harness: register every DUT I/O so timing is constrained through real internal flops.
    {yosys.SYNTH_REG_ATTR}
    reg {_bus_range(cfg.wr_en_width)}r_wr_en;
    {yosys.SYNTH_REG_ATTR}
    reg {_bus_range(cfg.wr_addr_width)}r_wr_addr;
    {yosys.SYNTH_REG_ATTR}
    reg {_bus_range(cfg.wr_data_width)}r_wr_data;
    {yosys.SYNTH_REG_ATTR}
    reg {_bus_range(cfg.rd_addr_width)}r_rd_addr;

    wire {_bus_range(cfg.rd_data_width)}dut_rd_data;

    {yosys.SYNTH_REG_ATTR}
    reg {_bus_range(cfg.rd_data_width)}r_rd_data;

    assign rd_data = r_rd_data;

    holoso_regfile #(
        .W({cfg.w}),
        .WADDR({cfg.waddr}),
        .NRD({cfg.nrd}),
        .NWR({cfg.nwr}),
        .NREG({cfg.nreg}),
        .RWPASS({cfg.rwpass})
    ) dut (
        .clk(clk),
        .wr_en(r_wr_en),
        .wr_addr(r_wr_addr),
        .wr_data(r_wr_data),
        .rd_addr(r_rd_addr),
        .rd_data(dut_rd_data)
    );

    always @(posedge clk) begin
        r_wr_en   <= wr_en;
        r_wr_addr <= wr_addr;
        r_wr_data <= wr_data;
        r_rd_addr <= rd_addr;
        r_rd_data <= dut_rd_data;
    end
endmodule

`default_nettype wire
""")


def test_regfile_yosys_nextpnr_ecp5_ooc() -> None:
    build_dir = yosys.BUILD_ROOT / "regfile_yosys_ecp5_ooc"
    yosys.clean_build_dir(build_dir)

    wrapper = build_dir / "holoso_regfile_ooc.v"
    script = build_dir / "holoso_regfile_ooc.ys"
    netlist = build_dir / "holoso_regfile_ooc.json"
    routed_netlist = build_dir / "holoso_regfile_ooc_routed.json"
    report = build_dir / "holoso_regfile_ooc_nextpnr.json"
    yosys_log = build_dir / "yosys.log"
    nextpnr_log = build_dir / "nextpnr.log"

    write_regfile_wrapper(wrapper, REGFILE)
    yosys.write_synthesis_script(
        script,
        include_dirs=[HDL_DIR],
        sources=[SUPPORT_RTL, wrapper],
        top=TOP,
        commands=[
            f"synth_ecp5 -top {TOP} -noiopad -noabc9 -abc2 -dff -retime -run begin:check",
            "clean",
            f"hierarchy -check -top {TOP}",
            "stat",
            "check -noinit",
            "blackbox =A:whitebox",
            f"write_json {netlist}",
        ],
    )

    yosys.run_yosys(script, yosys_log)
    cells = yosys.read_cell_counts(netlist, TOP)
    assert cells.get("TRELLIS_IO", 0) == 0, f"Yosys inserted IO pads unexpectedly; see {yosys_log}"

    yosys.run_nextpnr_ecp5(
        [
            "--25k",
            "--package",
            ECP5_PACKAGE,
            "--speed",
            ECP5_SPEED_GRADE,
            "--freq",
            f"{TARGET_FREQ_MHZ:g}",
            "--timing-allow-fail",
            "--out-of-context",
            "--json",
            netlist,
            "--write",
            routed_netlist,
            "--report",
            report,
        ],
        nextpnr_log,
    )

    report_data = yosys.read_json(report)
    timings = yosys.clock_timings(report_data)
    assert timings, f"nextpnr did not report fmax; see {nextpnr_log}"
    print("\nSynthesis summary:")
    print(f"  fmax: {yosys.timing_summary(timings)}")
    print("  Fabric utilization:")
    for line in yosys.utilization_summary(report_data, ECP5_FABRIC_RESOURCES).splitlines():
        print(f"    {line}")
    failed = [item for item in timings if item.achieved_mhz < TARGET_FREQ_MHZ]
    assert not failed, (
        f"regfile missed {TARGET_FREQ_MHZ:g} MHz on ECP5-25k: {yosys.timing_summary(timings)}; " f"see {nextpnr_log}"
    )
    io_used = yosys.utilization_used(report_data, "TRELLIS_IO") or 0
    assert io_used == 0, f"nextpnr placed {io_used} TRELLIS_IO cells despite OOC/no-IO-pad flow; see {nextpnr_log}"

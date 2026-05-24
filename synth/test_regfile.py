"""Yosys + nextpnr synthesis check for the Holoso support register file."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
SYNTH_WORKERS = max(1, int(os.environ.get("SYNTH_WORKERS", str(os.cpu_count() or 1))))
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
    name: str
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

    @property
    def params(self) -> str:
        return f"W={self.w} WADDR={self.waddr} NREG={self.nreg} NRD={self.nrd} " f"NWR={self.nwr} RWPASS={self.rwpass}"


@dataclass(frozen=True)
class SynthesisResult:
    cfg: RegfileConfig
    timings: tuple[yosys.ClockTiming, ...]
    resources: tuple[yosys.ResourceUtilization, ...]
    yosys_log: Path
    nextpnr_log: Path

    @property
    def fmax_mhz(self) -> float:
        return min(item.achieved_mhz for item in self.timings)

    @property
    def resource_by_name(self) -> dict[str, yosys.ResourceUtilization]:
        return {item.name: item for item in self.resources}


REGFILES = (
    RegfileConfig(name="rwpass_1", w=44, waddr=6, nreg=8, nrd=8, nwr=8, rwpass=1),
    RegfileConfig(name="rwpass_0", w=44, waddr=6, nreg=8, nrd=8, nwr=8, rwpass=0),
)


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


def format_resource(item: yosys.ResourceUtilization | None) -> str:
    if item is None:
        return "n/a"
    if item.percent is None:
        return str(item.used)
    return f"{item.used}/{item.available} ({item.percent:.2f}%)"


def synthesize_regfile(cfg: RegfileConfig) -> SynthesisResult:
    build_dir = yosys.BUILD_ROOT / f"regfile_yosys_ecp5_ooc_{cfg.name}"
    yosys.clean_build_dir(build_dir)

    wrapper = build_dir / "holoso_regfile_ooc.v"
    script = build_dir / "holoso_regfile_ooc.ys"
    netlist = build_dir / "holoso_regfile_ooc.json"
    routed_netlist = build_dir / "holoso_regfile_ooc_routed.json"
    report = build_dir / "holoso_regfile_ooc_nextpnr.json"
    yosys_log = build_dir / "yosys.log"
    nextpnr_log = build_dir / "nextpnr.log"

    print(f"\n=== Synthesizing {cfg.name}: {cfg.params} ===", flush=True)
    print(f"  artifact dir: {build_dir}", flush=True)

    write_regfile_wrapper(wrapper, cfg)
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
    timings = tuple(yosys.clock_timings(report_data))
    assert timings, f"nextpnr did not report fmax; see {nextpnr_log}"
    resources = tuple(yosys.resource_utilization(report_data, ECP5_FABRIC_RESOURCES))

    print(f"\nSynthesis summary for {cfg.name}:")
    print(f"  Parameters: {cfg.params}")
    print(f"  fmax: {yosys.timing_summary(timings)}")
    print("  Fabric utilization:")
    for line in yosys.utilization_summary(report_data, ECP5_FABRIC_RESOURCES).splitlines():
        print(f"    {line}")

    return SynthesisResult(
        cfg=cfg,
        timings=timings,
        resources=resources,
        yosys_log=yosys_log,
        nextpnr_log=nextpnr_log,
    )


def print_comparison_report(results: tuple[SynthesisResult, ...]) -> None:
    headers = ["Config", "fmax MHz", "Slack ns", *ECP5_FABRIC_RESOURCES, "Logs"]
    rows = []
    for result in results:
        resources = result.resource_by_name
        worst_slack = min(item.slack_ns for item in result.timings)
        rows.append(
            [
                result.cfg.name,
                f"{result.fmax_mhz:.2f}",
                f"{worst_slack:.3f}",
                *[format_resource(resources.get(resource)) for resource in ECP5_FABRIC_RESOURCES],
                f"{result.yosys_log.parent.name}/",
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def fmt_row(row: list[str]) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(row, widths))

    print("\nRegfile synthesis comparison:")
    print(fmt_row(headers))
    print(fmt_row(["-" * width for width in widths]))
    for row in rows:
        print(fmt_row(row))


def test_regfile_yosys_nextpnr_ecp5_ooc() -> None:
    workers = min(len(REGFILES), SYNTH_WORKERS)
    print(f"\nRunning {len(REGFILES)} regfile synthesis configurations with {workers} workers.", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = tuple(executor.map(synthesize_regfile, REGFILES))
    print_comparison_report(results)

    failed = [result for result in results if any(item.achieved_mhz < TARGET_FREQ_MHZ for item in result.timings)]
    assert not failed, "\n".join(
        f"{result.cfg.name} missed {TARGET_FREQ_MHZ:g} MHz on ECP5-25k: "
        f"{yosys.timing_summary(result.timings)}; see {result.nextpnr_log}"
        for result in failed
    )

    io_failures = []
    for result in results:
        io_used = result.resource_by_name.get("TRELLIS_IO")
        if io_used is not None and io_used.used != 0:
            io_failures.append(result)

    assert not io_failures, "\n".join(
        f"{result.cfg.name} placed {result.resource_by_name['TRELLIS_IO'].used} TRELLIS_IO cells despite "
        f"OOC/no-IO-pad flow; see {result.nextpnr_log}"
        for result in io_failures
    )

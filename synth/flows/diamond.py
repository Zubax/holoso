import html
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from holoso import SynthesisResult

from .._detect import find_tool
from .._ooc import build_ooc_wrapper
from .._synth import CommandSpec, ResourceUse, SourceFile, SynthArtifact, SynthReport, assemble, run_logged
from . import Flow

_TCL = "run_diamond.tcl"
_LOG = "diamond.log"
_CLOCK_NET = "clk_c"  # Diamond names the net driven by the `clk` input port `clk_c`.


@dataclass(frozen=True, slots=True)
class DiamondEcp5Device:
    device: str = "LFE5U-25F-6BG381C"


@dataclass(frozen=True, slots=True)
class DiamondEcp5Flow(Flow):
    device: DiamondEcp5Device = field(default_factory=DiamondEcp5Device)
    target_frequency_MHz: float = 100.0
    options: dict[str, Any] = field(default_factory=dict)

    def available(self) -> bool:
        diamond = find_tool("diamond")
        if diamond is None:
            return False
        return (diamond.resolve().parent / "diamond_env").is_file() or find_tool("pnmainc") is not None

    def prepare(self, result: SynthesisResult) -> SynthArtifact:
        wrapper = build_ooc_wrapper(result)
        top = wrapper.top
        src = assemble(result, wrapper)
        verilog_paths = [sf.path for sf in src if sf.path.suffix == ".v"]

        lpf = SourceFile(Path(f"{top}.lpf"), _lpf(self.target_frequency_MHz))
        sty = SourceFile(Path(f"{top}.sty"), _strategy(self.target_frequency_MHz))
        ldf = SourceFile(Path(f"{top}.ldf"), _ldf(top, self.device.device, verilog_paths))
        tcl = SourceFile(Path(_TCL), _tcl(f"{top}.ldf"))
        commands = [CommandSpec(["pnmainc", _TCL])]

        def runner(directory: Path) -> SynthReport:
            run_logged(["bash", "-lc", _console_script(directory / _TCL)], directory / _LOG, cwd=directory)
            return _parse(self, directory)

        return SynthArtifact(
            flow="diamond-ecp5", top=top, files=[*src, lpf, sty, ldf, tcl], commands=commands, runner=runner
        )


def _lpf(freq_MHz: float) -> str:
    return (
        "BLOCK RESETPATHS ;\n"
        "BLOCK ASYNCPATHS ;\n"
        f'USE PRIMARY NET "{_CLOCK_NET}" ;\n'
        f'FREQUENCY NET "{_CLOCK_NET}" {freq_MHz:.6f} MHz ;\n'
    )


def _xml_attr(text: str) -> str:
    return html.escape(text, quote=True)


def _strategy(freq_MHz: float) -> str:
    properties = {
        "PROP_LST_CarryChain": "True",
        "PROP_LST_DSPStyle": "DSP",
        "PROP_LST_EdfFrequency": f"{freq_MHz:.0f}",
        "PROP_LST_IOInsertion": "True",
        "PROP_LST_OptimizeGoal": "Timing",
        "PROP_LST_PropagatConst": "True",
        "PROP_LST_ResourceShare": "False",
        "PROP_LST_UseLPF": "True",
        "PROP_MAP_RegRetiming": "True",
        "PROP_MAP_TimingDriven": "True",
        "PROP_PAR_EffortParDes": "5",
        "PROP_PAR_LowSkewClokNet": "True",  # Diamond property name spelling is intentional.
        "PROP_PAR_RoutePassParDes": "6",
        "PROP_PAR_RunParWithTrce": "True",
        "PROP_SYN_EdfFrequency": f"{freq_MHz:.0f}",
        "PROP_SYN_EdfInsertIO": "False",
        "PROP_SYN_UseLPF": "True",
    }
    # Some wide-argument kernels need an extra push to close timings.
    if int(os.getenv("HOLOSO_DIAMOND_HARD", "0").strip()) != 0:
        properties.update(
            {
                "PROP_PAR_PlcIterParDes": "6",
                "PROP_PAR_SaveBestRsltParDes": "1",
                "PROP_PAR_MultiSeedSortMode": "Worst Slack",
                "PROP_MAP_TimingDrivenPack": "True",
                "PROP_MAP_TimingDrivenNodeRep": "True",
                "PROP_PAR_RouteDlyRedParDes": "1",
                "PROP_PAR_RouteResOptParDes": "1",
            }
        )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!DOCTYPE strategy>",
        '<Strategy version="1.0" predefined="0" description="" label="HolosoOocMaxTiming">',
        *(
            f'    <Property name="{_xml_attr(name)}" value="{_xml_attr(value)}" time="0"/>'
            for name, value in properties.items()
        ),
        "</Strategy>",
    ]
    return "\n".join(lines) + "\n"


def _ldf(top: str, device: str, verilog_paths: list[Path]) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<BaliProject version="3.2" title="{_xml_attr(top)}" device="{_xml_attr(device)}" '
        'default_implementation="impl1">',
        "    <Options/>",
        '    <Implementation title="impl1" dir="impl1" description="impl1" synthesis="lse" '
        'default_strategy="Strategy1">',
        f'        <Options def_top="{_xml_attr(top)}">',
        f'            <Option name="top" value="{_xml_attr(top)}"/>',
        "        </Options>",
    ]
    for path in verilog_paths:
        rel = _xml_attr(path.as_posix())
        is_top = path.stem == top
        lines.append(f'        <Source name="{rel}" type="Verilog" type_short="Verilog">')
        lines.append(f'            <Options top_module="{_xml_attr(top)}"/>' if is_top else "            <Options/>")
        lines.append("        </Source>")
    lines += [
        f'        <Source name="{top}.lpf" type="Logic Preference" type_short="LPF">',
        "            <Options/>",
        "        </Source>",
        "    </Implementation>",
        f'    <Strategy name="Strategy1" file="{top}.sty"/>',
        "</BaliProject>",
    ]
    return "\n".join(lines) + "\n"


def _tcl(project_file: str) -> str:
    return (
        "proc fail {message} {\n"
        "    puts stderr $message\n"
        "    exit 1\n"
        "}\n"
        f'if {{[catch {{prj_project open "{project_file}"}} result]}} {{ fail $result }}\n'
        "if {[catch {prj_run PAR -impl impl1 -forceAll} result]} { catch {prj_project close} ; fail $result }\n"
        "if {[catch {prj_project close} result]} { fail $result }\n"
        "exit 0\n"
    )


def _console_script(tcl: Path) -> str:
    diamond = find_tool("diamond")
    if diamond is None:
        raise FileNotFoundError("diamond executable was not found on PATH, /opt, /usr, or /home")
    bindir = shlex.quote(str(diamond.resolve().parent))
    diamond_env = diamond.resolve().parent / "diamond_env"
    # diamond_env derives FOUNDRY/TCL_LIBRARY/LD_LIBRARY_PATH from ${bindir}, so it must be set before sourcing,
    # otherwise pnmainc cannot load libtcl8.5.so.
    source_env = f"source {shlex.quote(str(diamond_env))}" if diamond_env.is_file() else ":"
    return (
        "set -euo pipefail\n"
        f"bindir={bindir}\n"
        'export PATH="$bindir:$PATH"\n'
        "set +u\n"
        f"{source_env}\n"
        "set -u\n"
        'command -v pnmainc >/dev/null 2>&1 || { echo "error: pnmainc not found" >&2; exit 1; }\n'
        f"pnmainc < {shlex.quote(str(tcl))}\n"
    )


def _read(path: Path | None) -> str:
    return path.read_text(errors="replace") if path is not None and path.exists() else ""


def _resource(pattern: str, text: str, name: str) -> ResourceUse | None:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    return ResourceUse(name, int(match.group(1)), int(match.group(2)))


def _parse(flow: DiamondEcp5Flow, directory: Path) -> SynthReport:
    impl = directory / "impl1"
    twrs = sorted(p for p in impl.glob("*.twr") if not p.name.endswith("_lse.twr"))
    twr = _read(twrs[-1] if twrs else None)
    mrp = _read(next(iter(sorted(impl.glob("*.mrp"))), None))
    par = _read(next(iter(sorted(impl.glob("*.par"))), None))

    fmaxes = re.findall(r"(?:Report|Warning):\s+([0-9.]+)\s*MHz is the maximum frequency", twr, re.IGNORECASE)
    if not fmaxes:
        raise RuntimeError(f"Diamond TRACE report did not state a maximum frequency; see {impl}")
    fmax_MHz = min(float(value) for value in fmaxes)
    target = flow.target_frequency_MHz

    resources = {
        use.name: use
        for use in (
            _resource(r"Number of registers:\s+([0-9]+) out of ([0-9]+)", mrp, "Registers"),
            _resource(r"Number of LUT4s:\s+([0-9]+) out of ([0-9]+)", mrp, "LUT4"),
            _resource(r"SLICE\s+([0-9]+)/([0-9]+)", par, "SLICE"),
            _resource(r"PIO \(prelim\)\s+([0-9]+)/([0-9]+)", par, "PIO"),
        )
        if use is not None
    }

    return SynthReport(
        flow="diamond-ecp5",
        target_frequency_MHz=target,
        fmax_MHz=fmax_MHz,
        slack_ns=1000.0 / target - 1000.0 / fmax_MHz,
        resources=resources,
        artifact_dir=directory,
        logs=[directory / _LOG, *sorted(impl.glob("*.twr"))],
    )

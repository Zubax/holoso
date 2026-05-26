"""
Command-line entry point for the OOC synthesis-evaluation harness.
Usage::

    python -m synth <kernel.py> <entry> --wexp W --wman M --rtl PATH...

This will synthesize one Holoso-generated module across every available FPGA tool and report the achieved post-route
f_max and fabric usage.
Synthesis failure is recorded as a failure without stopping other tools.
"""

import argparse
import importlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from holoso import synthesize, FloatFormat, SynthesisResult
from holoso import FAddOp, FDivOp, FMulILog2GenericOp, FMulOp, OpConfig

from ._synth import BUILD_ROOT, SynthReport
from .flows import Flow
from .flows.diamond import DiamondEcp5Flow
from .flows.vivado import VivadoFlow
from .flows.yosys import YosysEcp5Flow

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"


@dataclass(frozen=True, slots=True)
class _Failure:
    tool: str
    directory: Path
    message: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m synth", description=__doc__)
    parser.add_argument("kernel", help="path to the Python file containing the kernel")
    parser.add_argument("entry", help="name of the function to synthesize")
    parser.add_argument("--wexp", type=int, default=6, help="float exponent bits")
    parser.add_argument("--wman", type=int, default=18, help="float significand bits")
    parser.add_argument(
        "--rtl",
        nargs="+",
        required=True,
        metavar="PATH",
        help="extra RTL the DUT needs: .v files or directories globbed for *.v",
    )
    parser.add_argument("--name", default=None, help="generated module name (default: the entry name)")
    parser.add_argument("--freq", type=float, default=100.0, metavar="MHz", help="target frequency in MHz")
    return parser.parse_args()


def _collect_rtl(specs: list[str]) -> list[Path]:
    rtl: list[Path] = []
    for spec in specs:
        path = Path(spec)
        rtl += sorted(path.glob("*.v")) if path.is_dir() else [path]
    return rtl


def _synthesize(kernel: Path, entry: str, fmt: FloatFormat, name: str) -> SynthesisResult:
    sys.path.insert(0, str(kernel.resolve().parent))
    module = importlib.import_module(kernel.stem)
    ops = OpConfig(fadd=FAddOp(fmt), fmul=FMulOp(fmt), fdiv=FDivOp(fmt), fmul_ilog2=FMulILog2GenericOp(fmt))
    return synthesize(getattr(module, entry), ops=ops, name=name)


def _select_flows(freq_MHz: float) -> tuple[list[Flow], list[str]]:
    # Yosys is the baseline and always runs; Diamond and Vivado run only when discoverable.
    flows: list[Flow] = [YosysEcp5Flow(target_frequency_MHz=freq_MHz)]
    skipped: list[str] = []
    for flow in (DiamondEcp5Flow(target_frequency_MHz=freq_MHz), VivadoFlow(target_frequency_MHz=freq_MHz)):
        if flow.available():
            flows.append(flow)
        else:
            skipped.append(type(flow).__name__)
    return flows, skipped


def _resources(report: SynthReport) -> list[str]:
    return [
        f"{use.name} {use.used}" + (f"/{use.available}" if use.available else "")
        for use in report.resources.values()
        if use.used
    ]


def _print_outcome(outcome: SynthReport | _Failure) -> None:
    if isinstance(outcome, _Failure):
        print(
            f"💥 {_BOLD}{_RED}{outcome.tool} FAILED{_RESET}: {outcome.message}; "
            f"logs in {_BOLD}{outcome.directory}{_RESET}"
        )
        return
    succ = outcome.fmax_MHz >= outcome.target_frequency_MHz
    color = _GREEN if succ else _RED
    emo = "✅" if succ else "❌"
    print(
        f"{emo} {_BOLD}{color}{outcome.flow}: f_max {outcome.fmax_MHz:.2f} MHz{_RESET} "
        f"(target {outcome.target_frequency_MHz:.0f}, slack {outcome.slack_ns:+.3f} ns)"
    )
    if resources := _resources(outcome):
        print(f"\t{_DIM}{'\n\t'.join(resources)}{_RESET}")


def main() -> int:
    args = _parse_args()
    fmt = FloatFormat(wexp=args.wexp, wman=args.wman)
    name = args.name or args.entry
    rtl = _collect_rtl(args.rtl)
    result = _synthesize(Path(args.kernel), args.entry, fmt, name)

    out_dir = BUILD_ROOT / name
    shutil.rmtree(out_dir, ignore_errors=True)

    flows, skipped = _select_flows(args.freq)
    if skipped:
        print(f"{_BOLD}{_YELLOW}Skipping unavailable tools:{_RESET} {', '.join(skipped)}{_RESET}")

    outcomes: list[SynthReport | _Failure] = []
    for flow in flows:
        directory = out_dir / type(flow).__name__
        print(
            f"🛠️ Synthesizing {_MAGENTA}{args.kernel}::{args.entry}{_RESET} as {_BOLD}{_MAGENTA}{name}{_RESET} "
            f"using {_BOLD}{_CYAN}{flow.__class__.__name__}{_RESET} in {_BOLD}{directory}{_RESET}..."
        )
        try:
            outcomes.append(flow.prepare(result, rtl).synthesize(directory))
        except Exception as exc:  # one tool's failure must not stop the others
            outcomes.append(_Failure(type(flow).__name__, directory, str(exc)))
        _print_outcome(outcomes[-1])

    succ = True
    for ou in outcomes:
        if isinstance(ou, _Failure) or ou.fmax_MHz < ou.target_frequency_MHz:
            succ = False
    return 0 if succ else 1


if __name__ == "__main__":
    sys.exit(main())

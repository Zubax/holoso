"""
Command-line entry point for the OOC synthesis-evaluation harness.
Usage::

    python -m synth <kernel.py> <entry> --wexp W --wman M --rtl PATH... --flow FLOW:freq=MHz

This will synthesize one Holoso-generated module across the requested FPGA tool flows and report the achieved post-route
f_max and fabric usage. Repeat ``--flow`` to run multiple flows, each with its own target frequency.
Synthesis failure is recorded as a failure without stopping other tools.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from holoso import synthesize, FloatFormat, SynthesisResult
from holoso import FAddOperator, FDivOperator, FMulILog2OperatorFamily, FMulOperator, OpConfig

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


@dataclass(frozen=True, slots=True)
class _FlowRequest:
    flow_id: str
    target_frequency_MHz: float


_FLOWS = {
    "yosys-ecp5": lambda request: YosysEcp5Flow(target_frequency_MHz=request.target_frequency_MHz),
    "diamond-ecp5": lambda request: DiamondEcp5Flow(target_frequency_MHz=request.target_frequency_MHz),
    "vivado": lambda request: VivadoFlow(target_frequency_MHz=request.target_frequency_MHz),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--flow",
        action="append",
        dest="flow_specs",
        required=True,
        metavar="FLOW:freq=MHz",
        help=f"synthesis flow to run; repeatable; supported flows: {', '.join(_FLOWS.keys())}",
    )
    args = parser.parse_args(argv)
    args.flow_requests = _parse_flow_requests(parser, args.flow_specs)
    return args


def _parse_flow_requests(parser: argparse.ArgumentParser, specs: list[str]) -> list[_FlowRequest]:
    requests: list[_FlowRequest] = []
    seen: set[str] = set()
    for spec in specs:
        request = _parse_flow_spec(parser, spec)
        if request.flow_id in seen:
            parser.error(f"flow {request.flow_id!r} was requested more than once")
        seen.add(request.flow_id)
        requests.append(request)
    return requests


def _parse_flow_spec(parser: argparse.ArgumentParser, spec: str) -> _FlowRequest:
    if ":" not in spec:
        parser.error(f"flow spec {spec!r} must be written as FLOW:freq=MHz")
    flow_id, raw_fields = (part.strip() for part in spec.split(":", 1))
    if flow_id not in _FLOWS.keys():
        parser.error(f"unknown flow {flow_id!r}; supported flows: {', '.join(_FLOWS.keys())}")
    if not raw_fields:
        parser.error(f"flow spec {spec!r} has no fields; expected freq=MHz")

    fields: dict[str, str] = {}
    for item in raw_fields.split(","):
        if "=" not in item:
            parser.error(f"flow field {item!r} in {spec!r} must be written as key=value")
        key, value = (part.strip() for part in item.split("=", 1))
        if key != "freq":
            parser.error(f"unknown flow field {key!r} in {spec!r}; supported field: freq")
        if key in fields:
            parser.error(f"flow field {key!r} is duplicated in {spec!r}")
        fields[key] = value

    raw_frequency = fields.get("freq")
    if raw_frequency is None:
        parser.error(f"flow spec {spec!r} is missing required field freq=MHz")
    try:
        target_frequency_MHz = float(raw_frequency)
    except ValueError:
        parser.error(f"flow spec {spec!r} has invalid frequency {raw_frequency!r}")
    if not math.isfinite(target_frequency_MHz) or target_frequency_MHz <= 0.0:
        parser.error(f"flow spec {spec!r} has invalid frequency {raw_frequency!r}")
    return _FlowRequest(flow_id=flow_id, target_frequency_MHz=target_frequency_MHz)


def _collect_rtl(specs: list[str]) -> list[Path]:
    rtl: list[Path] = []
    for spec in specs:
        path = Path(spec)
        rtl += sorted(path.glob("*.v")) if path.is_dir() else [path]
    return rtl


def _synthesize(kernel: Path, entry: str, fmt: FloatFormat, name: str) -> SynthesisResult:
    sys.path.insert(0, str(kernel.resolve().parent))
    module = importlib.import_module(kernel.stem)
    # TODO: specify pipeline stages per operator per flow via _FlowRequest; disable all stages by default.
    ops = OpConfig(
        fadd=FAddOperator(fmt, stage_decode=1),
        fmul=FMulOperator(fmt, stage_input=1),
        fdiv=FDivOperator(fmt),
        fmul_ilog2=FMulILog2OperatorFamily(fmt),
    )
    return synthesize(getattr(module, entry), ops=ops, name=name)


def _flow_from_request(request: _FlowRequest) -> Flow:
    try:
        return _FLOWS[request.flow_id](request)  # type: ignore
    except LookupError:
        raise AssertionError(f"unknown flow ID {request.flow_id!r}")


def _select_flows(requests: list[_FlowRequest]) -> tuple[list[Flow], list[str]]:
    flows: list[Flow] = []
    skipped: list[str] = []
    for request in requests:
        flow = _flow_from_request(request)
        if flow.available():
            flows.append(flow)
        else:
            skipped.append(request.flow_id)
    return flows, skipped


def _run_flow(flow: Flow, result: SynthesisResult, rtl: list[Path], directory: Path) -> SynthReport | _Failure:
    try:
        return flow.prepare(result, rtl).synthesize(directory)
    except Exception as exc:  # one tool's failure must not stop the others
        return _Failure(type(flow).__name__, directory, str(exc))


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
    flows, skipped = _select_flows(args.flow_requests)
    if skipped:
        print(f"{_BOLD}{_YELLOW}Skipping unavailable flows:{_RESET} {', '.join(skipped)}{_RESET}")
    if not flows:
        print(f"{_BOLD}{_RED}No requested synthesis flows are available; nothing to run.{_RESET}")
        return 10

    fmt = FloatFormat(wexp=args.wexp, wman=args.wman)
    name = args.name or args.entry
    rtl = _collect_rtl(args.rtl)
    result = _synthesize(Path(args.kernel), args.entry, fmt, name)

    out_dir = BUILD_ROOT / name
    shutil.rmtree(out_dir, ignore_errors=True)

    outcomes: list[SynthReport | _Failure] = []
    workers = max(2, (os.cpu_count() or 1) // 2)
    print(f"{_BOLD}{_CYAN}Running {len(flows)} synthesis flows with {workers} worker(s).{_RESET}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for flow in flows:
            directory = out_dir / type(flow).__name__
            print(
                f"🛠️ Synthesizing {_MAGENTA}{args.kernel}::{args.entry}{_RESET} as "
                f"{_BOLD}{_MAGENTA}{name}{_RESET} using {_BOLD}{_CYAN}{flow.__class__.__name__}{_RESET} "
                f"in {_BOLD}{directory}{_RESET}...",
                flush=True,
            )
            futures[executor.submit(_run_flow, flow, result, rtl, directory)] = flow

        for future in as_completed(futures):
            outcomes.append(future.result())
            _print_outcome(outcomes[-1])

    succ = True
    for ou in outcomes:
        if isinstance(ou, _Failure) or ou.fmax_MHz < ou.target_frequency_MHz:
            succ = False
    return 0 if succ else 1


if __name__ == "__main__":
    sys.exit(main())

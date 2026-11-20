"""
Command-line entry point for the OOC synthesis-evaluation harness.
Usage:

    python -m synth <kernel.py> <expression> --options OPTIONS --flow FLOW:freq=MHz[,device=CLASS][,grade=GRADE]

`OPTIONS` is a Python expression giving the `holoso.Options`, evaluated in the kernel module's namespace (with
`holoso` in scope) like the target expression, e.g.
`holoso.Options(operator=holoso.OperatorOptions(fadd=...), wmultiplier=18, ffmt=holoso.FloatFormat(6, 18))`.
Repeat `--flow` to run multiple flows, each with its own target frequency; each flow adds its own synthesis attributes.
Synthesis failure is recorded as a failure without stopping other tools.
"""

import argparse
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import holoso
from holoso import Options, synthesize

from ._synth import BUILD_ROOT, SynthReport, build_compiler_ooc_design
from .flows import DeviceClass, Flow, FlowId, SpeedGrade, make_flow


@dataclass(frozen=True, slots=True)
class _Failure:
    tool: str
    directory: Path
    message: str


@dataclass(frozen=True, slots=True)
class _FlowRequest:
    flow_id: FlowId
    target_frequency_MHz: float
    device_class: DeviceClass
    speed_grade: SpeedGrade


@dataclass(frozen=True, slots=True)
class _SelectedFlow:
    request: _FlowRequest
    flow: Flow


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m synth", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kernel", help="path to the Python file containing the kernel")
    parser.add_argument(
        "expression",
        help="Python expression evaluated in the kernel module to obtain the target;"
        " e.g., `Ekf1(Q_diag=np.array([...])).update`. Must be a function or a method.",
    )
    parser.add_argument(
        "--options",
        required=True,
        metavar="OPTIONS",
        help="Python expression evaluated in the kernel module, with `holoso` in scope, giving the holoso.Options",
    )
    parser.add_argument("--name", default=None, help="generated module name (automatic by default)")
    parser.add_argument(
        "--flow",
        action="append",
        dest="flow_specs",
        required=True,
        metavar="FLOW:freq=MHz[,device=CLASS][,grade=GRADE]",
        help=(
            f"synthesis flow to run; repeatable; supported flows: {', '.join(FlowId)}; "
            f"CLASS picks the device ({', '.join(DeviceClass)}; the larger one is for a design that outgrows the "
            f"default); GRADE picks the speed grade ({', '.join(SpeedGrade)})"
        ),
    )
    args = parser.parse_args(argv)
    args.flow_requests = _parse_flow_requests(parser, args.flow_specs)
    return args


def _parse_flow_requests(parser: argparse.ArgumentParser, specs: list[str]) -> list[_FlowRequest]:
    requests: list[_FlowRequest] = []
    seen: set[FlowId] = set()
    for spec in specs:
        request = _parse_flow_spec(parser, spec)
        if request.flow_id in seen:
            parser.error(f"flow {request.flow_id.value!r} was requested more than once")
        seen.add(request.flow_id)
        requests.append(request)
    return requests


def _parse_flow_spec(parser: argparse.ArgumentParser, spec: str) -> _FlowRequest:
    if ":" not in spec:
        parser.error(f"flow spec {spec!r} must be written as FLOW:freq=MHz[,device=CLASS][,grade=GRADE]")
    flow_id, raw_fields = (part.strip() for part in spec.split(":", 1))
    if flow_id not in FlowId:
        parser.error(f"unknown flow {flow_id!r}; supported flows: {', '.join(FlowId)}")
    if not raw_fields:
        parser.error(f"flow spec {spec!r} has no fields; expected freq=MHz")

    raw_frequency: str | None = None
    device_class: DeviceClass | None = None
    speed_grade: SpeedGrade | None = None
    for item in raw_fields.split(","):
        if "=" not in item:
            parser.error(f"flow field {item!r} in {spec!r} must be written as key=value")
        key, value = (part.strip() for part in item.split("=", 1))
        if key == "freq":
            if raw_frequency is not None:
                parser.error(f"flow field {key!r} is duplicated in {spec!r}")
            raw_frequency = value
            continue
        if key == "device":
            if device_class is not None:
                parser.error(f"flow field {key!r} is duplicated in {spec!r}")
            if value not in DeviceClass:
                parser.error(f"unknown device class {value!r} in {spec!r}; supported: {', '.join(DeviceClass)}")
            device_class = DeviceClass(value)
            continue
        if key == "grade":
            if speed_grade is not None:
                parser.error(f"flow field {key!r} is duplicated in {spec!r}")
            if value not in SpeedGrade:
                parser.error(f"unknown speed grade {value!r} in {spec!r}; supported: {', '.join(SpeedGrade)}")
            speed_grade = SpeedGrade(value)
            continue
        parser.error(f"unknown flow field {key!r} in {spec!r}")
    if raw_frequency is None:
        parser.error(f"flow spec {spec!r} is missing required field freq=MHz")
    try:
        target_frequency_MHz = float(raw_frequency)
    except ValueError:
        parser.error(f"flow spec {spec!r} has invalid frequency {raw_frequency!r}")
    if not math.isfinite(target_frequency_MHz) or target_frequency_MHz <= 0.0:
        parser.error(f"flow spec {spec!r} has invalid frequency {raw_frequency!r}")
    return _FlowRequest(
        flow_id=FlowId(flow_id),
        target_frequency_MHz=target_frequency_MHz,
        device_class=device_class or DeviceClass.DEFAULT,
        speed_grade=speed_grade or SpeedGrade.SLOWEST,
    )


def _namespace(kernel: Path) -> dict[str, Any]:
    """The kernel module's namespace, where the target and the options are evaluated, with `holoso` in scope."""
    sys.path.insert(0, str(kernel.resolve().parent))
    module = importlib.import_module(kernel.stem)
    return {"holoso": holoso, **vars(module)}


def _load_target(namespace: dict[str, Any], expression: str) -> object:
    """
    `expression` is evaluated in the kernel module's namespace, e.g. `Ekf1(Q_diag=np.array([...])).update`;
    for a bound method the `__self__` snapshot seeds the reset values.
    """
    return eval(expression, dict(namespace))


def _load_options(namespace: dict[str, Any], expression: str) -> Options:
    options = eval(expression, dict(namespace))
    if not isinstance(options, Options):
        raise TypeError(f"--options gives {type(options).__name__}, not holoso.Options")
    return options


def _select_flows(requests: list[_FlowRequest]) -> tuple[list[_SelectedFlow], list[FlowId]]:
    flows: list[_SelectedFlow] = []
    skipped: list[FlowId] = []
    for request in requests:
        flow: Flow = make_flow(request.flow_id, request.target_frequency_MHz, request.device_class, request.speed_grade)
        if flow.available():
            flows.append(_SelectedFlow(request, flow))
        else:
            skipped.append(request.flow_id)
    return flows, skipped


def _run_flow(
    flow: Flow,
    options: Options,
    target: Any,
    name: str,
    directory: Path,
) -> SynthReport | _Failure:
    try:
        result = synthesize(target, options, name=name)
        result.write(directory / "holoso_result")
        return flow.prepare(build_compiler_ooc_design(result)).synthesize(directory)
    except Exception as exc:  # one tool's failure must not stop the others
        return _Failure(type(flow).__name__, directory, str(exc))


def _print_outcome(outcome: SynthReport | _Failure) -> None:
    if isinstance(outcome, _Failure):
        print(f"💥 {outcome.tool} FAILED: {outcome.message}; logs in {outcome.directory}")
        return
    succ = outcome.fmax_MHz >= outcome.target_frequency_MHz
    emo = "✅" if succ else "❌"
    print(
        f"{emo} {outcome.flow}: f_max {outcome.fmax_MHz:.2f} MHz "
        f"(target {outcome.target_frequency_MHz:.0f}, slack {outcome.slack_ns:+.3f} ns)"
    )
    if resources := [
        f"{use.name} {use.used}" + (f"/{use.available}" if use.available else "")
        for use in outcome.resources.values()
        if use.used
    ]:
        print(f"\t{'\n\t'.join(resources)}")


def main() -> int:
    args = _parse_args()
    flows, skipped = _select_flows(args.flow_requests)
    if skipped:
        print(f"Skipping unavailable flows: {', '.join(skipped)}")
    if not flows:
        print("No requested synthesis flows are available; nothing to run.")
        return 10

    name = args.name or re.sub(r"\W+", "_", args.expression).strip("_")
    namespace = _namespace(Path(args.kernel))
    target = _load_target(namespace, args.expression)
    given = _load_options(namespace, args.options)

    out_dir = BUILD_ROOT / name
    shutil.rmtree(out_dir, ignore_errors=True)

    outcomes: list[SynthReport | _Failure] = []
    workers = max(2, (os.cpu_count() or 1) // 2)
    print(f"Running {len(flows)} synthesis flows with ≤{workers} worker(s).")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for selected in flows:
            flow = selected.flow
            directory = out_dir / type(flow).__name__
            options = flow.attributed(given)
            print(
                f"🛠️ Synthesizing {args.kernel}::{args.expression} as {name} "
                f"using {flow.__class__.__name__} in {directory}"
            )
            print("⚙ Operators:")
            for field in fields(options.operator):
                print(f"    {field.name:12}: {getattr(options.operator, field.name)}")
            print(flush=True)
            futures[executor.submit(_run_flow, flow, options, target, name, directory)] = flow

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

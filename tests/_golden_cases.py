"""
The golden-corpus catalogue: the single source of truth for what ``tests/golden/`` freezes, shared by
``tests/test_golden.py`` (the gate) and ``tools/refreeze_golden.py`` (the capture tool). Design:
``docs/decisions/freeze-design.md``.

A :class:`GoldenCase` fully identifies one frozen synthesis: the kernel factory (including the exact reset
variant -- the shipped and cosim-reset EKF are distinct cases), the float format, the complete operator
configuration, the module name, the fetch depth, and the register-allocation knobs. The inherited example
matrix derives mechanically from ``tests/_examples.SPECS`` (every spec at each of its declared formats), so a
spec or format added later fails the corpus bijection until deliberately refrozen. On top of it ride the
structural-only cases (no cosim driver), the compact format-sensitive probe, and one deeply staged operator
configuration.

A :class:`RejectionCase` pins one diagnostic: its kernel lives in the immutable module
``tests/golden/rejections/<case_id>.py`` (black-formatted from birth, never edited, so recorded line numbers
stay true by construction) and its expected public class, rendered message, location, and origin frames live
in ``tests/golden/diagnostics/<family>.jsonl``.
"""

import importlib.util
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import Any

import holoso
from holoso import (
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FloatFormat,
    FMulILog2OperatorFamily,
    FMulOperator,
    FSincosOperator,
    FSortOperator,
    HolosoError,
    OpConfig,
)
from holoso._backend.verilog import VerilogOutput, generate as generate_verilog
from holoso._frontend import lower as lower_frontend
from holoso._hir import optimize
from holoso._lir import ControlPort, DataPort, Lir, build
from holoso._mir import lower as lower_to_mir
from holoso._type import BoolType, FloatType

from ._examples import SPECS, ekf1_stateful, imu_frame_transform, polar
from ._hirdump import dump_hir
from ._modelref import default_ops, staged_ops

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_ROOT = REPO_ROOT / "tests" / "golden"

ABI_SCHEMA = "holoso-golden-abi/1"
DIAG_SCHEMA = "holoso-golden-diag/1"
INDEX_SCHEMA = "holoso-golden-index/1"
PROVENANCE_SCHEMA = "holoso-golden-provenance/1"

VERSION_PLACEHOLDER = "Holoso v<VERSION>"


def version_token() -> str:
    return f"Holoso v{holoso.__version__}"


def canonicalize_version(text: str) -> str:
    """
    The exact ``holoso.__version__`` header token replaced by a placeholder, so the frozen Verilog survives
    version bumps; the live header is separately asserted to carry the real version.
    """
    assert VERSION_PLACEHOLDER not in text
    return text.replace(version_token(), VERSION_PLACEHOLDER)


@dataclass(frozen=True, slots=True)
class RegallocKnobs:
    """
    The register-allocation tuning knobs pinned for every golden build, mirroring the shipped defaults (the
    ``getenv`` fallbacks in ``holoso._lir._regalloc``), so a developer's environment overrides cannot leak into
    the frozen corpus.
    """

    refine_maxiter: int = 5000
    reg_reuse_write_cap: int = 2
    reg_price: float = 2.0


DEFAULT_REGALLOC = RegallocKnobs()


@contextmanager
def pinned_regalloc(knobs: RegallocKnobs) -> Iterator[None]:
    import holoso._lir._regalloc as regalloc

    saved = (regalloc._REFINE_MAXITER, regalloc._REG_REUSE_WRITE_CAP, regalloc._REG_PRICE)
    regalloc._REFINE_MAXITER = knobs.refine_maxiter
    regalloc._REG_REUSE_WRITE_CAP = knobs.reg_reuse_write_cap
    regalloc._REG_PRICE = knobs.reg_price
    try:
        yield
    finally:
        regalloc._REFINE_MAXITER, regalloc._REG_REUSE_WRITE_CAP, regalloc._REG_PRICE = saved


@dataclass(frozen=True, slots=True)
class ScheduleMetrics:
    """The exact internal schedule figures frozen per case; the same shape doubles as a legacy ceiling row."""

    straight_line: bool
    nreg: int
    bnreg: int
    steering: int
    copies: int
    min_ii: int
    last_pc: int
    max_block_span: int


def measure_metrics(lir: Lir) -> ScheduleMetrics:
    straight = (
        len(lir.blocks) == 1
        and not lir.bool_state_slots
        and not any(b.inline_ops or b.copies or b.bool_writes for b in lir.blocks)
        and lir.bool_regfile.nreg == 0
    )
    read_fanin = sum(max(0, len(regs) - 1) for regs in lir.read_set_per_port.values())
    copies = sum(len(block.copies) + len(block.bool_writes) for block in lir.blocks)
    return ScheduleMetrics(
        straight_line=straight,
        nreg=lir.regfile.nreg,
        bnreg=lir.bool_regfile.nreg,
        steering=read_fanin + lir.write_select_fanin,
        copies=copies,
        min_ii=lir.min_initiation_interval,
        last_pc=lir.initiation_interval,
        max_block_span=max(block.term_offset for block in lir.blocks),
    )


@dataclass(frozen=True, slots=True)
class GoldenCase:
    case_id: str
    make_kernel: Callable[[], Callable[..., object]]
    fmt: FloatFormat
    make_ops: Callable[[FloatFormat], OpConfig]
    module_name: str
    fetch_stages: int = 3
    regalloc: RegallocKnobs = DEFAULT_REGALLOC
    # The pre-merge test_metrics non-regression ceiling (an upper bound per numeric field, classification
    # asserted equal), kept as a separate row per the freeze design; the exact figures live in the ABI capture.
    ceiling: ScheduleMetrics | None = None


@dataclass(frozen=True, slots=True)
class GoldenArtifacts:
    """Everything one case freezes, produced by a single build."""

    verilog: str  # version-canonicalized
    verilog_raw: str  # as emitted, carrying the real version header
    hir_dump: str  # pre-optimize frontend HIR
    abi_json: str
    metrics: ScheduleMetrics
    support_files: dict[str, str]  # version-canonicalized, case-independent


def _fmt_name(fmt: FloatFormat) -> str:
    return f"e{fmt.wexp}m{fmt.wman}"


def _abi_ports(lir: Lir) -> list[dict[str, Any]]:
    ports: list[dict[str, Any]] = []
    for port in lir.ports:
        entry: dict[str, Any] = {"direction": port.direction.value, "name": port.name, "width": port.width}
        match port:
            case ControlPort():
                entry["role"] = "diagnostic" if port.name == "err_pc" else "control"
            case DataPort(scalar_type=FloatType(fmt=fmt)):
                entry["role"] = "data"
                entry["scalar_kind"] = "float"
                entry["float_format"] = _fmt_name(fmt)
            case DataPort(scalar_type=BoolType()):
                entry["role"] = "data"
                entry["scalar_kind"] = "bool"
            case _:
                raise AssertionError(f"port {port!r} has no ABI spelling")
        ports.append(entry)
    return ports


def render_abi(case: GoldenCase, lir: Lir, metrics: ScheduleMetrics) -> str:
    ops = case.make_ops(case.fmt)
    max_ii = lir.min_initiation_interval if len(lir.blocks) == 1 else None
    manifest = {
        "schema": ABI_SCHEMA,
        "case": case.case_id,
        "module_name": case.module_name,
        "float_format": _fmt_name(case.fmt),
        "fetch_stages": case.fetch_stages,
        "regalloc": {
            "refine_maxiter": case.regalloc.refine_maxiter,
            "reg_reuse_write_cap": case.regalloc.reg_reuse_write_cap,
            "reg_price": case.regalloc.reg_price,
        },
        "ops": {field.name: _spell_operator(getattr(ops, field.name)) for field in fields(ops)},
        "initiation_interval": {"min": lir.min_initiation_interval, "max": max_ii},
        "ports": _abi_ports(lir),
        "metrics": {
            "straight_line": metrics.straight_line,
            "nreg": metrics.nreg,
            "bnreg": metrics.bnreg,
            "steering": metrics.steering,
            "copies": metrics.copies,
            "min_ii": metrics.min_ii,
            "last_pc": metrics.last_pc,
            "max_block_span": metrics.max_block_span,
        },
    }
    return json.dumps(manifest, indent=2, ensure_ascii=True) + "\n"


def _spell_operator(operator: object | None) -> str | None:
    if operator is None:
        return None
    return repr(operator)


def build_artifacts(case: GoldenCase) -> GoldenArtifacts:
    with pinned_regalloc(case.regalloc):
        kernel = case.make_kernel()
        pre_hir = lower_frontend(kernel)
        ops = case.make_ops(case.fmt)
        assert ops.float_format == case.fmt
        lir = build(lower_to_mir(optimize(pre_hir), ops), case.module_name, fetch_stages=case.fetch_stages)
        verilog_output: VerilogOutput = generate_verilog(lir)
        metrics = measure_metrics(lir)
    raw = verilog_output.verilog
    assert version_token() in raw, "the live Verilog header must carry the real holoso version"
    return GoldenArtifacts(
        verilog=canonicalize_version(raw),
        verilog_raw=raw,
        hir_dump=dump_hir(pre_hir),
        abi_json=render_abi(case, lir, metrics),
        metrics=metrics,
        support_files={name: canonicalize_version(text) for name, text in verilog_output.support_files.items()},
    )


def assert_within_ceiling(case: GoldenCase, got: ScheduleMetrics) -> None:
    ceiling = case.ceiling
    if ceiling is None:
        return
    assert got.straight_line == ceiling.straight_line, f"{case.case_id}: control-flow classification changed"
    for field in ("nreg", "bnreg", "steering", "copies", "min_ii", "last_pc", "max_block_span"):
        assert getattr(got, field) <= getattr(
            ceiling, field
        ), f"{case.case_id}: {field} regressed past the legacy ceiling {getattr(ceiling, field)} -> {getattr(got, field)}"


_F_WIDE = FloatFormat(8, 36)
_F_NARROW = FloatFormat(6, 18)
_F_BIN32 = FloatFormat(8, 24)
_F_CARRIER = FloatFormat(11, 53)  # the binary64 frontend constant-folding carrier, as a ZKF datapath

# The pre-merge test_metrics BASELINE, carried over verbatim as non-regression ceilings (every numeric field an
# upper bound, classification exact). All rows were measured at the wide e8m36 datapath under default_ops with
# the shipped regalloc knobs -- exactly how the matching golden cases build. The ekf1_stateful row was measured
# on the SHIPPED reset (default-constructed Ekf1), so it rides the shipped structural case, not the cosim one.
_CEILINGS: dict[str, ScheduleMetrics] = {
    "madd-e8m36": ScheduleMetrics(
        True, nreg=4, bnreg=0, steering=3, copies=0, min_ii=15, last_pc=15, max_block_span=15
    ),
    "poly3-e8m36": ScheduleMetrics(
        True, nreg=5, bnreg=0, steering=5, copies=0, min_ii=24, last_pc=24, max_block_span=24
    ),
    "signal_window-e8m36": ScheduleMetrics(
        False, nreg=4, bnreg=5, steering=8, copies=0, min_ii=10, last_pc=10, max_block_span=10
    ),
    "iir1_lpf-e8m36": ScheduleMetrics(
        False, nreg=3, bnreg=2, steering=2, copies=0, min_ii=16, last_pc=16, max_block_span=16
    ),
    "pid-e8m36": ScheduleMetrics(
        False, nreg=10, bnreg=2, steering=13, copies=1, min_ii=38, last_pc=71, max_block_span=32
    ),
    "schmitt_trigger-e8m36": ScheduleMetrics(
        False, nreg=1, bnreg=2, steering=2, copies=0, min_ii=7, last_pc=7, max_block_span=7
    ),
    "quadrature_encoder-e8m36": ScheduleMetrics(
        False, nreg=1, bnreg=7, steering=7, copies=0, min_ii=6, last_pc=6, max_block_span=6
    ),
    "phase_frequency_detector-e8m36": ScheduleMetrics(
        False, nreg=0, bnreg=5, steering=5, copies=0, min_ii=6, last_pc=6, max_block_span=6
    ),
    "latching_fault_register-e8m36": ScheduleMetrics(
        False, nreg=1, bnreg=6, steering=2, copies=0, min_ii=6, last_pc=6, max_block_span=6
    ),
    "majority_voter-e8m36": ScheduleMetrics(
        False, nreg=1, bnreg=21, steering=20, copies=0, min_ii=14, last_pc=19, max_block_span=12
    ),
    "recip_newton-e8m36": ScheduleMetrics(
        False, nreg=4, bnreg=1, steering=4, copies=2, min_ii=15, last_pc=32, max_block_span=16
    ),
    "remainder-e8m36": ScheduleMetrics(
        False, nreg=8, bnreg=4, steering=12, copies=2, min_ii=39, last_pc=58, max_block_span=17
    ),
    "octave_index-e8m36": ScheduleMetrics(
        False, nreg=3, bnreg=1, steering=6, copies=3, min_ii=16, last_pc=51, max_block_span=25
    ),
    "cordic_sincos-e8m36": ScheduleMetrics(
        False, nreg=7, bnreg=1, steering=53, copies=0, min_ii=105, last_pc=105, max_block_span=105
    ),
    "integrator-e8m36": ScheduleMetrics(
        True, nreg=5, bnreg=0, steering=4, copies=0, min_ii=17, last_pc=17, max_block_span=17
    ),
    "imu_frame_transform-e8m36": ScheduleMetrics(
        True, nreg=20, bnreg=0, steering=35, copies=0, min_ii=42, last_pc=42, max_block_span=42
    ),
    "ekf1_stateless-e8m36": ScheduleMetrics(
        True, nreg=41, bnreg=0, steering=100, copies=0, min_ii=126, last_pc=126, max_block_span=126
    ),
    "ekf1_stateful_shipped-e8m36": ScheduleMetrics(
        True, nreg=40, bnreg=0, steering=89, copies=0, min_ii=128, last_pc=128, max_block_span=128
    ),
}


def _fsc_ops(fmt: FloatFormat) -> OpConfig:
    return OpConfig(
        FAddOperator(fmt),
        FMulOperator(fmt),
        FDivOperator(fmt),
        FMulILog2OperatorFamily(fmt),
        FCmpOperator(fmt),
        fsort=FSortOperator(fmt),
        fsincos=FSincosOperator(fmt),
    )


def core_ops(fmt: FloatFormat) -> OpConfig:
    """Just the five required operators -- the format probe uses no transcendental, so it builds at any format."""
    return OpConfig(
        FAddOperator(fmt), FMulOperator(fmt), FDivOperator(fmt), FMulILog2OperatorFamily(fmt), FCmpOperator(fmt)
    )


def format_probe(x: float, y: float) -> float:
    """
    The compact format-sensitive probe: the 0.1 constant quantizes differently at every mantissa width, and the
    unspeculatable division keeps the diamond a real branch, so both the constant tables and the per-format
    operator latencies (hence the schedule) differ across the probed formats.
    """
    acc = x * 0.1 + y * 3.0
    if acc > y:
        acc = acc / (y * y + 0.125)
    return acc - x


def _inherited_cases() -> list[GoldenCase]:
    cases = []
    for spec in SPECS:
        for fmt in spec.formats:
            case_id = f"{spec.name}-{_fmt_name(fmt)}"
            cases.append(
                GoldenCase(
                    case_id=case_id,
                    make_kernel=spec.make_kernel,
                    fmt=fmt,
                    make_ops=default_ops,
                    module_name=spec.name,
                    ceiling=_CEILINGS.get(case_id),
                )
            )
    return cases


def _structural_cases() -> list[GoldenCase]:
    from finite_set_current_controller import FiniteSetCurrentController  # examples/ is on sys.path via _examples

    return [
        GoldenCase(
            case_id="finite_set_current_controller-e8m36",
            make_kernel=lambda: FiniteSetCurrentController().__call__,
            fmt=_F_WIDE,
            make_ops=_fsc_ops,
            module_name="finite_set_current_controller",
        ),
        GoldenCase(
            case_id="to_polar-e8m36",
            make_kernel=lambda: polar.to_polar,
            fmt=_F_WIDE,
            make_ops=default_ops,
            module_name="to_polar",
        ),
        GoldenCase(
            case_id="from_polar-e8m36",
            make_kernel=lambda: polar.from_polar,
            fmt=_F_WIDE,
            make_ops=default_ops,
            module_name="from_polar",
        ),
        GoldenCase(
            case_id="imu_frame_transform-e8m36",
            make_kernel=lambda: imu_frame_transform.transform,
            fmt=_F_WIDE,
            make_ops=default_ops,
            module_name="imu_frame_transform",
            ceiling=_CEILINGS["imu_frame_transform-e8m36"],
        ),
        GoldenCase(
            case_id="ekf1_stateful_shipped-e8m36",
            make_kernel=lambda: ekf1_stateful.Ekf1().update,
            fmt=_F_WIDE,
            make_ops=default_ops,
            module_name="ekf1_stateful_shipped",
            ceiling=_CEILINGS["ekf1_stateful_shipped-e8m36"],
        ),
    ]


def _probe_cases() -> list[GoldenCase]:
    probes = [
        GoldenCase(
            case_id=f"format_probe-{_fmt_name(fmt)}",
            make_kernel=lambda: format_probe,
            fmt=fmt,
            make_ops=core_ops,
            module_name="format_probe",
        )
        for fmt in (_F_NARROW, _F_BIN32, _F_WIDE, _F_CARRIER)
    ]
    staged = GoldenCase(
        case_id="pid-e6m18-staged",
        make_kernel=next(spec for spec in SPECS if spec.name == "pid").make_kernel,
        fmt=_F_NARROW,
        make_ops=staged_ops,
        module_name="pid_staged",
    )
    return [*probes, staged]


CASES: list[GoldenCase] = [*_inherited_cases(), *_structural_cases(), *_probe_cases()]
assert len({case.case_id for case in CASES}) == len(CASES)
assert {case.case_id for case in CASES} >= set(_CEILINGS), "a legacy ceiling row lost its golden case"


@dataclass(frozen=True, slots=True)
class RejectionCase:
    """
    One frozen diagnostic: the kernel lives in the immutable module ``tests/golden/rejections/<case_id>.py``;
    ``target`` extracts the synthesis target from that module (constructing the instance for a bound method).
    ``precedence`` documents a deliberately pinned competing-error resolution order, where one exists.
    """

    case_id: str
    family: str
    target: Callable[[ModuleType], Callable[..., object]]
    precedence: str | None = None


REJECTION_FMT = FloatFormat(6, 18)


def _kernel(module: ModuleType) -> Callable[..., object]:
    kernel = module.kernel
    assert callable(kernel)
    return kernel  # type: ignore[no-any-return]


REJECTIONS: list[RejectionCase] = [
    RejectionCase("trim_getattr", "trims", lambda m: m.Accumulator().step),
    RejectionCase("trim_array_mask", "trims", _kernel),
    RejectionCase("trim_zero_d_array", "trims", _kernel),
    RejectionCase("trim_isinstance", "trims", _kernel),
    RejectionCase("trim_enum_member_attribute", "trims", _kernel),
    RejectionCase("trim_str_method", "trims", _kernel),
    RejectionCase("trim_dataclass_post_init", "trims", _kernel),
    RejectionCase("trim_component_setattr", "trims", lambda m: m.Clamped().step),
    RejectionCase("trim_starred_target", "trims", _kernel),
    RejectionCase("trim_starred_display", "trims", _kernel),
    RejectionCase("b1_local_int_rebound_to_float", "b1", _kernel),
    RejectionCase("b1_local_float_rebound_to_bool", "b1", _kernel),
    RejectionCase("b1_state_int_slot_stored_float", "b1", lambda m: m.Counter().step),
    RejectionCase("b1_state_bool_slot_stored_float", "b1", lambda m: m.Latch().step),
    RejectionCase(
        "b1_competing_stores_preorder",
        "b1",
        _kernel,
        precedence="storage-schema obligations resolve in CFG preorder: the then-arm store on 'a' surfaces on "
        "every PYTHONHASHSEED, never the else-arm store on 'b'",
    ),
    RejectionCase("store_edge_inexact_int", "store_edge", _kernel),
    RejectionCase("store_edge_oversized_int", "store_edge", _kernel),
    RejectionCase("store_edge_state_inexact_int", "store_edge", lambda m: m.Total().step),
    RejectionCase("c2_bool_float_comparison", "analysis", _kernel),
    RejectionCase("f1_unroll_threshold", "analysis", _kernel),
    RejectionCase("d1_nested_none_annotation", "signature", _kernel),
    RejectionCase("d2_unresolvable_annotation", "signature", _kernel),
    RejectionCase("c3_inherited_unresolvable_annotation", "signature", _kernel),
    RejectionCase("c4_fake_dims_attribute", "signature", _kernel),
    RejectionCase("e2_fstring_raise", "build", _kernel),
    RejectionCase("legacy_recursion", "legacy", _kernel),
    RejectionCase("legacy_record_iteration", "legacy", _kernel),
    RejectionCase("legacy_stub_shape_mismatch", "legacy", _kernel),
    RejectionCase("legacy_beyond_carrier_constant", "legacy", _kernel),
    RejectionCase("legacy_power_chain", "legacy", _kernel),
    RejectionCase("legacy_never_returns", "legacy", _kernel),
    RejectionCase("legacy_shared_live_out", "legacy", lambda m: m.Shared().step),
]
assert len({rejection.case_id for rejection in REJECTIONS}) == len(REJECTIONS)


def rejection_module_path(case_id: str) -> Path:
    return GOLDEN_ROOT / "rejections" / f"{case_id}.py"


def _load_rejection_module(case_id: str) -> ModuleType:
    name = f"holoso_golden_rejection_{case_id}"
    loaded = sys.modules.get(name)
    if loaded is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, rejection_module_path(case_id))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _public_class(exc: HolosoError) -> type[HolosoError]:
    """The closest public ancestor -- the class a user catches -- of a possibly-internal rejection class."""
    mro: tuple[type[BaseException], ...] = type(exc).__mro__
    for cls in mro:
        exported: bool = getattr(holoso, cls.__name__, None) is cls
        if exported:
            assert issubclass(cls, HolosoError)
            return cls
    raise AssertionError(f"{type(exc).__name__} has no public ancestor in the holoso API")


def _relative_file(filename: str) -> str:
    return Path(filename).resolve().relative_to(REPO_ROOT).as_posix()


def capture_rejection(rejection: RejectionCase) -> dict[str, Any]:
    """
    Synthesize the corpus kernel and return its complete diagnostic row: the public class, the rendered
    ``str(exc)`` (which spells the primary function, line, column, and any ``in callee():`` context), the
    structured location with its source line text, and the semantic origin frames. File paths are recorded
    repo-relative, so the row is byte-stable across checkouts.
    """
    module = _load_rejection_module(rejection.case_id)
    kernel = rejection.target(module)
    try:
        holoso.synthesize(kernel, default_ops(REJECTION_FMT))
    except HolosoError as exc:
        location = getattr(exc, "location", None)
        origin = getattr(exc, "origin", None) or ()
        return {
            "schema": DIAG_SCHEMA,
            "case": rejection.case_id,
            "family": rejection.family,
            "class": _public_class(exc).__name__,
            "message": str(exc),
            "location": (
                None
                if location is None
                else {
                    "file": _relative_file(location.filename),
                    "line": location.lineno,
                    "column": location.col,
                    "source_line": location.line,
                }
            ),
            "origin": [
                {
                    "function": frame.function,
                    "file": _relative_file(frame.file),
                    "line": frame.line,
                    "column": frame.column,
                }
                for frame in origin
            ],
            "precedence": rejection.precedence,
        }
    raise AssertionError(f"{rejection.case_id}: the corpus kernel unexpectedly compiled")


def diagnostic_families() -> list[str]:
    return sorted({rejection.family for rejection in REJECTIONS})


def render_family_jsonl(family: str) -> str:
    ordered = sorted((r for r in REJECTIONS if r.family == family), key=lambda r: r.case_id)
    assert ordered, f"no rejection cases in family {family!r}"
    rows = [capture_rejection(rejection) for rejection in ordered]
    return "".join(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)


def expected_files() -> set[str]:
    """Every path (GOLDEN_ROOT-relative, posix) the corpus must contain -- the bijection reference."""
    files = {"index.json", "provenance.json", "README.md", "support/holoso_support.v"}
    for case in CASES:
        files |= {f"verilog/{case.case_id}.v", f"hir/{case.case_id}.txt", f"abi/{case.case_id}.json"}
    for rejection in REJECTIONS:
        files.add(f"rejections/{rejection.case_id}.py")
    files |= {f"diagnostics/{family}.jsonl" for family in diagnostic_families()}
    return files


def render_index() -> str:
    """The machine-readable corpus index, derived purely from the catalogue (no synthesis involved)."""
    index = {
        "schema": INDEX_SCHEMA,
        "cases": [
            {
                "id": case.case_id,
                "module_name": case.module_name,
                "float_format": _fmt_name(case.fmt),
                "fetch_stages": case.fetch_stages,
                "regalloc": {
                    "refine_maxiter": case.regalloc.refine_maxiter,
                    "reg_reuse_write_cap": case.regalloc.reg_reuse_write_cap,
                    "reg_price": case.regalloc.reg_price,
                },
                "has_ceiling": case.ceiling is not None,
                "artifacts": {
                    "verilog": f"verilog/{case.case_id}.v",
                    "hir": f"hir/{case.case_id}.txt",
                    "abi": f"abi/{case.case_id}.json",
                },
            }
            for case in sorted(CASES, key=lambda case: case.case_id)
        ],
        "rejections": [
            {
                "id": rejection.case_id,
                "family": rejection.family,
                "module": f"rejections/{rejection.case_id}.py",
                "diagnostics": f"diagnostics/{rejection.family}.jsonl",
            }
            for rejection in sorted(REJECTIONS, key=lambda rejection: rejection.case_id)
        ],
        "support": ["support/holoso_support.v"],
    }
    return json.dumps(index, indent=2, ensure_ascii=True) + "\n"

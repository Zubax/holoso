"""
The golden-corpus gate: every catalogued case must reproduce its frozen artifacts byte-for-byte, every
catalogued rejection must raise its pinned public class with the byte-exact rendered diagnostic, and the
declared catalogue must be a bijection with the on-disk corpus (stale extras fail, not just missing files).
Design: ``docs/decisions/freeze-design.md``; catalogue: ``tests/_golden_cases.py``.

A mismatch is either a defect or a deliberate change; there is no third option. For a deliberate change,
regenerate with ``tools/refreeze_golden.py --write`` and commit the golden diff together with the causing code
and DESIGN.md change in the same commit.
"""

import difflib
import json
from collections.abc import Callable
from functools import cache
from pathlib import Path

import pytest

from ._golden_cases import (
    CASES,
    DEFAULT_IFCONV_MAX_OPS,
    DEFAULT_REGALLOC,
    DIAG_SCHEMA,
    GOLDEN_ROOT,
    PROVENANCE_SCHEMA,
    REJECTIONS,
    REPO_ROOT,
    GoldenCase,
    RejectionCase,
    build_artifacts,
    assert_within_ceiling,
    capture_rejection,
    diagnostic_families,
    expected_files,
    render_index,
    version_token,
)

_REFREEZE_HINT = (
    "deliberate change -> tools/refreeze_golden.py --write; commit golden+code+DESIGN together in one commit"
)


@pytest.fixture(autouse=True)
def _pinned_codegen_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Pin every codegen tuning knob to its shipped default so the frozen corpus is reproducible regardless of the
    developer's environment (``HOLOSO_REGALLOC_EFFORT`` speed-ups, write-cap/price experiments, an
    ``HOLOSO_IFCONV_MAX_OPS`` budget override). The knobs are env-read-once at import, so the module attributes
    are patched to the named defaults; ``build_artifacts`` additionally pins the regalloc and ifconv knobs per
    case, but the fixture keeps any future direct measurement in this module honest as well.
    """
    import holoso._hir._if_convert as if_convert
    import holoso._lir._regalloc as regalloc

    monkeypatch.setattr(regalloc, "_REFINE_MAXITER", DEFAULT_REGALLOC.refine_maxiter)
    monkeypatch.setattr(regalloc, "_REG_REUSE_WRITE_CAP", DEFAULT_REGALLOC.reg_reuse_write_cap)
    monkeypatch.setattr(regalloc, "_REG_PRICE", DEFAULT_REGALLOC.reg_price)
    monkeypatch.setattr(if_convert, "_IFCONV_MAX_OPS", DEFAULT_IFCONV_MAX_OPS)


def _read_artifact(path: Path) -> str:
    assert path.is_file(), f"missing golden artifact {path.relative_to(REPO_ROOT)}; {_REFREEZE_HINT}"
    data = path.read_bytes()
    assert b"\r" not in data, f"{path.relative_to(REPO_ROOT)}: golden artifacts are LF-only"
    return data.decode("utf-8")


def _assert_matches(kind: str, case_id: str, got: str, path: Path) -> None:
    want = _read_artifact(path)
    if got == want:
        return
    diff = difflib.unified_diff(
        want.splitlines(), got.splitlines(), fromfile=f"golden/{path.name}", tofile="live", lineterm=""
    )
    head = "\n".join(list(diff)[:40])
    pytest.fail(f"{case_id}: {kind} differs from the frozen {path.relative_to(REPO_ROOT)}.\n{head}\n{_REFREEZE_HINT}")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_case_matches_golden(case: GoldenCase) -> None:
    artifacts = build_artifacts(case)
    assert version_token() in artifacts.verilog_raw, "the live Verilog header must carry the real holoso version"
    _assert_matches("Verilog", case.case_id, artifacts.verilog, GOLDEN_ROOT / "verilog" / f"{case.case_id}.v")
    _assert_matches("pre-optimize HIR", case.case_id, artifacts.hir_dump, GOLDEN_ROOT / "hir" / f"{case.case_id}.txt")
    _assert_matches("ABI manifest", case.case_id, artifacts.abi_json, GOLDEN_ROOT / "abi" / f"{case.case_id}.json")
    # The support map gated here is the one this very generation RETURNED, so a build handing out a corrupt or
    # incomplete library cannot pass on the strength of a separate, healthy generator call.
    assert set(artifacts.support_files) == {"holoso_support.v"}, f"{case.case_id}: unexpected support-file set"
    for name, text in artifacts.support_files.items():
        _assert_matches("support library", case.case_id, text, GOLDEN_ROOT / "support" / name)
    assert_within_ceiling(case, artifacts.metrics)


@cache
def _stored_diagnostics() -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for family in diagnostic_families():
        text = _read_artifact(GOLDEN_ROOT / "diagnostics" / f"{family}.jsonl")
        for line in text.splitlines():
            row = json.loads(line)
            assert row["schema"] == DIAG_SCHEMA
            case_id = row["case"]
            assert isinstance(case_id, str) and case_id not in rows
            rows[case_id] = row
    return rows


@pytest.mark.parametrize("rejection", REJECTIONS, ids=lambda rejection: rejection.case_id)
def test_rejection_matches_golden(rejection: RejectionCase) -> None:
    got = capture_rejection(rejection)
    want = _stored_diagnostics().get(rejection.case_id)
    assert want is not None, f"{rejection.case_id}: no frozen diagnostic row; {_REFREEZE_HINT}"
    if got != want:
        lines = [f"{rejection.case_id}: the rendered diagnostic differs from the frozen row."]
        for key in sorted(set(got) | set(want)):
            if got.get(key) != want.get(key):
                lines.append(f"  {key}: frozen {want.get(key)!r}")
                lines.append(f"  {' ' * len(key)}    live {got.get(key)!r}")
        lines.append(_REFREEZE_HINT)
        pytest.fail("\n".join(lines))


def test_hir_dump_spells_a_wide_integer_constant() -> None:
    # Folding admits integers up to 65536 bits while CPython's default int-to-decimal cap is 4300 digits, so the
    # corpus serializer must never spell a constant through the capped decimal conversion.
    from holoso._frontend import lower
    from holoso._hir import IntConst

    from ._hirdump import HIR_DUMP_SCHEMA, dump_hir

    class WideMask:
        def __init__(self) -> None:
            self.lfsr = 1

        def step(self, advance: bool) -> bool:
            if advance:
                self.lfsr = (self.lfsr << 1) & ((1 << 20000) - 1)
            return self.lfsr != 0

    hir = lower(WideMask().step)
    assert any(isinstance(node, IntConst) and node.value.bit_length() == 20000 for node in hir.nodes.values())
    text = dump_hir(hir)
    assert text.startswith(HIR_DUMP_SCHEMA)
    assert f"const int {(1 << 20000) - 1:#x}" in text


def _swap_tree() -> "Callable[[Path, Path], None]":
    import importlib.util

    spec = importlib.util.spec_from_file_location("refreeze_golden", REPO_ROOT / "tools" / "refreeze_golden.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.swap_tree  # type: ignore[no-any-return]


def _write_corpus(root: Path, marker: str) -> None:
    root.mkdir(parents=True)
    (root / "index.json").write_text(marker, encoding="utf-8")


def _inject_copy_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    def failing_copytree(src: object, dst: object, **kwargs: object) -> object:
        raise OSError("injected copy failure")

    monkeypatch.setattr(shutil, "copytree", failing_copytree)


def test_corpus_swap_recovers_an_interrupted_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Round-6 regression (Codex): a previous replacement interrupted between its two renames leaves the live
    # tree missing with the old corpus displaced to .old and the incoming copy staged at .new; the old cleanup
    # deleted BOTH recovery trees before the fresh copy landed, so a crash inside the retry lost every corpus
    # copy on disk. The retry must restore the displaced tree before deleting anything.
    swap_tree = _swap_tree()
    source, live = tmp_path / "generated", tmp_path / "golden"
    _write_corpus(source, "incoming")
    _write_corpus(live.parent / f"{live.name}.old", "previous")
    _write_corpus(live.parent / f"{live.name}.new", "staged-by-dead-run")

    with monkeypatch.context() as context:
        _inject_copy_failure(context)
        with pytest.raises(OSError, match="injected copy failure"):
            swap_tree(source, live)
    assert (live / "index.json").read_text(encoding="utf-8") == "previous", "the displaced corpus must be restored"

    swap_tree(source, live)
    assert (live / "index.json").read_text(encoding="utf-8") == "incoming"
    assert not (live.parent / f"{live.name}.old").exists() and not (live.parent / f"{live.name}.new").exists()


def test_corpus_swap_keeps_the_live_tree_until_the_copy_lands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    swap_tree = _swap_tree()
    source, live = tmp_path / "generated", tmp_path / "golden"
    _write_corpus(source, "incoming")
    _write_corpus(live, "current")

    with monkeypatch.context() as context:
        _inject_copy_failure(context)
        with pytest.raises(OSError, match="injected copy failure"):
            swap_tree(source, live)
    assert (live / "index.json").read_text(encoding="utf-8") == "current"

    swap_tree(source, live)
    assert (live / "index.json").read_text(encoding="utf-8") == "incoming"
    assert not (live.parent / f"{live.name}.old").exists() and not (live.parent / f"{live.name}.new").exists()


def test_corpus_bijection() -> None:
    expected = expected_files()
    found = {
        path.relative_to(GOLDEN_ROOT).as_posix()
        for path in GOLDEN_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    missing = sorted(expected - found)
    stale = sorted(found - expected)
    assert (
        not missing and not stale
    ), f"golden corpus does not match the declared catalogue; missing={missing} stale={stale}; {_REFREEZE_HINT}"
    _assert_matches("corpus index", "index.json", render_index(), GOLDEN_ROOT / "index.json")
    stored = _stored_diagnostics()
    declared = {rejection.case_id for rejection in REJECTIONS}
    assert set(stored) == declared, (
        f"diagnostic rows do not match the declared rejections; missing={sorted(declared - set(stored))} "
        f"stale={sorted(set(stored) - declared)}; {_REFREEZE_HINT}"
    )
    provenance = json.loads(_read_artifact(GOLDEN_ROOT / "provenance.json"))
    assert provenance["schema"] == PROVENANCE_SCHEMA
    assert "cpython" in provenance and "dependencies" in provenance and "container_digest" in provenance

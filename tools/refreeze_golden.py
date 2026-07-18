#!/usr/bin/env python3
"""
Regenerate and certify the golden corpus (tests/golden/). Design: docs/decisions/freeze-design.md.

The tool always generates the COMPLETE corpus into a temporary tree in a fresh interpreter, validates it fully
(bijection against the catalogue, black on the rejection modules, JSONL schema, UTF-8/LF), and shows a diff
summary against the committed tests/golden/. The committed corpus is replaced only under an explicit --write.
Goldens change only in the commit carrying the causing code+DESIGN change.

--check-determinism additionally certifies seed independence BEFORE anything is written: the full corpus is
regenerated under at least eight explicit PYTHONHASHSEED values, the cheap targeted witnesses run over seeds
0..63, and one fixed seed is repeated three times in fresh processes (non-hash process nondeterminism); every
artifact is byte-compared and the first mismatch aborts, naming the seed and the first differing file.

Usage:
    python tools/refreeze_golden.py                       # generate + validate + diff summary (no write)
    python tools/refreeze_golden.py --write               # ... and replace tests/golden/
    python tools/refreeze_golden.py --check-determinism   # seed matrix + validation, no write
    python tools/refreeze_golden.py --check-determinism --write

Exit status: 0 = success (corpus identical, or written); 1 = corpus differs and --write not given;
2 = validation or determinism failure (nothing written).
"""

import argparse
import concurrent.futures
import difflib
import filecmp
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NoReturn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FULL_SEEDS = [0, 1, 2, 3, 42, 31337, 7, 101]
REPEAT_SEED = 42
REPEAT_COUNT = 3
WITNESS_SEED_COUNT = 64
WITNESS_ENTRIES = [
    "dump_two_carried_hir",
    "emit_coalesce_conflict",
    "emit_competing_rejection",
    "emit_competing_fail",
    "emit_competing_bad_store",
    "emit_competing_state_store_violation",
    "emit_competing_reset_rejection",
    "emit_join_deferred_state_store",
    "emit_carried_obligation",
    "emit_unrolled_fanout_ports",
]

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if _COLOR else text


def _bold(text: str) -> str:
    return _c("1", text)


def _green(text: str) -> str:
    return _c("32", text)


def _red(text: str) -> str:
    return _c("31", text)


def _yellow(text: str) -> str:
    return _c("33", text)


def _dim(text: str) -> str:
    return _c("2", text)


def _fail(message: str) -> NoReturn:
    print(f"{_red('❌ ' + message)}", file=sys.stderr)
    raise SystemExit(2)


# ------------------------------------------------ emit mode ------------------------------------------------
# Runs in a FRESH interpreter with PYTHONHASHSEED fixed by the orchestrator; generates one complete corpus tree.


def emit_corpus(out_dir: Path) -> None:
    import logging

    logging.disable(logging.CRITICAL)
    from tests._golden_cases import (
        CASES,
        GOLDEN_ROOT,
        PROVENANCE_SCHEMA,
        REJECTIONS,
        build_artifacts,
        diagnostic_families,
        render_family_jsonl,
        render_index,
        rejection_module_path,
    )

    def write(rel: str, text: str) -> None:
        path = out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)

    support: dict[str, str] | None = None
    for case in CASES:
        artifacts = build_artifacts(case)
        write(f"verilog/{case.case_id}.v", artifacts.verilog)
        write(f"hir/{case.case_id}.txt", artifacts.hir_dump)
        write(f"abi/{case.case_id}.json", artifacts.abi_json)
        if support is None:
            support = artifacts.support_files
        assert support == artifacts.support_files, "the support library must be case-independent"
    assert support is not None and set(support) == {"holoso_support.v"}
    write("support/holoso_support.v", support["holoso_support.v"])

    for rejection in REJECTIONS:
        source = rejection_module_path(rejection.case_id)
        assert source.is_file(), f"missing rejection corpus module {source}; author it before refreezing"
        write(f"rejections/{rejection.case_id}.py", source.read_text(encoding="utf-8"))
    for family in diagnostic_families():
        write(f"diagnostics/{family}.jsonl", render_family_jsonl(family))

    readme = GOLDEN_ROOT / "README.md"
    assert readme.is_file(), f"missing {readme}; author the update protocol before refreezing"
    write("README.md", readme.read_text(encoding="utf-8"))
    write("index.json", render_index())
    write("provenance.json", _render_provenance(PROVENANCE_SCHEMA))


def _render_provenance(schema: str) -> str:
    versions = {name: importlib.metadata.version(name) for name in ("holoso", "zkf", "numpy", "scipy")}
    provenance = {
        "schema": schema,
        "cpython": platform.python_version(),
        "dependencies": versions,
        # Null until a pinned-container capture flow exists; captures today name no container.
        "container_digest": None,
    }
    return json.dumps(provenance, indent=2, ensure_ascii=True) + "\n"


def run_witnesses() -> None:
    """The cheap targeted determinism witnesses, concatenated to stdout with stable framing."""
    import logging

    logging.disable(logging.CRITICAL)
    import tests.test_determinism as witnesses

    for entry in WITNESS_ENTRIES:
        print(f"== {entry} ==")
        getattr(witnesses, entry)()
        print()


# --------------------------------------------- orchestration ---------------------------------------------


def _child_env(seed: int) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HOLOSO_")}
    env["PYTHONHASHSEED"] = str(seed)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_child(args: list[str], seed: int) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *args],
        capture_output=True,
        text=True,
        env=_child_env(seed),
        cwd=REPO,
        timeout=3600,
    )
    if proc.returncode != 0:
        _fail(f"child {' '.join(args)} under seed {seed} failed:\n{proc.stderr.strip()[-4000:]}")
    return proc


def _emit_tree(directory: Path, seed: int) -> Path:
    _run_child(["--emit", str(directory)], seed)
    return directory


def _tree_files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def _first_difference(reference: Path, other: Path) -> str | None:
    ref_files, other_files = _tree_files(reference), _tree_files(other)
    if ref_files != other_files:
        delta = sorted(set(ref_files) ^ set(other_files))
        return f"{delta[0]} (present in only one tree)"
    for rel in ref_files:
        if not filecmp.cmp(reference / rel, other / rel, shallow=False):
            return rel
    return None


def check_determinism(work: Path, seeds: list[int], jobs: int) -> Path:
    """Run the full seed matrix; abort on any mismatch; return the canonical (first-seed) tree."""
    assert len(seeds) >= 4, "the determinism matrix needs at least four full-corpus seeds"
    started = time.monotonic()
    print(_bold(f"🎲 Determinism matrix: full corpus under seeds {seeds},"))
    print(
        _bold(
            f"   {REPEAT_COUNT}× process repeat under seed {REPEAT_SEED}, "
            f"witnesses under seeds 0..{WITNESS_SEED_COUNT - 1}"
        )
    )

    emits: dict[str, tuple[Path, int]] = {f"seed {seed}": (work / f"seed_{seed}", seed) for seed in seeds}
    for index in range(REPEAT_COUNT):
        emits[f"seed {REPEAT_SEED} process repeat #{index + 1}"] = (work / f"repeat_{index}", REPEAT_SEED)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {label: pool.submit(_emit_tree, directory, seed) for label, (directory, seed) in emits.items()}
        for label, future in futures.items():
            future.result()
            print(f"  {_green('✅')} corpus tree generated: {label}")
        witness_futures = {seed: pool.submit(_run_child, ["--witnesses"], seed) for seed in range(WITNESS_SEED_COUNT)}
        witness_outputs = {seed: future.result().stdout for seed, future in witness_futures.items()}
    print(f"  {_green('✅')} witness sweep complete ({WITNESS_SEED_COUNT} seeds × {len(WITNESS_ENTRIES)} witnesses)")

    reference_label = f"seed {seeds[0]}"
    reference = emits[reference_label][0]
    for label, (directory, _) in emits.items():
        if label == reference_label:
            continue
        differing = _first_difference(reference, directory)
        if differing is not None:
            _fail(f"determinism violation: {label} vs {reference_label}: first differing artifact: {differing}")
    for seed in range(1, WITNESS_SEED_COUNT):
        if witness_outputs[seed] != witness_outputs[0]:
            diverged = "output framing"
            current = "preamble"
            for reference_line, other in zip(witness_outputs[0].splitlines(), witness_outputs[seed].splitlines()):
                if reference_line.startswith("== ") and reference_line.endswith(" =="):
                    current = reference_line.strip("= ")
                if reference_line != other:
                    diverged = current
                    break
            _fail(f"determinism violation: witness output differs under seed {seed} (first diverging: {diverged})")
    elapsed = time.monotonic() - started
    print(_green(f"🧊 Determinism certified in {elapsed:.0f} s: every artifact byte-identical across the matrix."))
    return reference


def _locate_black() -> list[str]:
    candidates = [
        [sys.executable, "-m", "black"],
        [str(REPO / ".nox" / "black" / "bin" / "black")],
    ]
    which = shutil.which("black")
    if which:
        candidates.append([which])
    for candidate in candidates:
        try:
            probe = subprocess.run([*candidate, "--version"], capture_output=True, text=True, timeout=60)
        except OSError:
            continue
        if probe.returncode == 0:
            return candidate
    _fail("black is unavailable (tried python -m black, .nox/black, PATH); it is required to validate the corpus")


def validate_tree(tree: Path) -> None:
    from tests._golden_cases import DIAG_SCHEMA, REJECTIONS, diagnostic_families, expected_files, render_index

    expected = expected_files()
    found = set(_tree_files(tree))
    missing, stale = sorted(expected - found), sorted(found - expected)
    if missing or stale:
        _fail(f"generated tree is not a bijection with the catalogue: missing={missing} stale={stale}")
    for rel in sorted(found):
        data = (tree / rel).read_bytes()
        if b"\r" in data:
            _fail(f"{rel}: golden artifacts must be LF-only")
        data.decode("utf-8")
    if (tree / "index.json").read_text(encoding="utf-8") != render_index():
        _fail("index.json does not match the catalogue rendering")
    by_family: dict[str, set[str]] = {}
    for family in diagnostic_families():
        jsonl = (tree / "diagnostics" / f"{family}.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in jsonl.splitlines()]
        assert rows == sorted(rows, key=lambda row: str(row["case"]))
        for row in rows:
            if row["schema"] != DIAG_SCHEMA or row["family"] != family:
                _fail(f"diagnostics/{family}.jsonl: malformed row for {row.get('case')}")
            required = {"schema", "case", "family", "class", "message", "location", "origin", "precedence"}
            if set(row) != required:
                _fail(f"diagnostics/{family}.jsonl: row keys {sorted(row)} != {sorted(required)}")
        by_family[family] = {str(row["case"]) for row in rows}
    declared = {rejection.case_id for rejection in REJECTIONS}
    recorded = {case for cases in by_family.values() for case in cases}
    if recorded != declared:
        _fail(f"diagnostic rows are not a bijection with the registry: {sorted(recorded ^ declared)}")
    black = _locate_black()
    proc = subprocess.run(
        # The temporary tree is outside the repo, so black cannot discover pyproject.toml on its own.
        [*black, "--check", "--config", str(REPO / "pyproject.toml"), str(tree / "rejections")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        _fail(f"rejection corpus modules are not black-clean:\n{proc.stdout}{proc.stderr}")
    print(_green(f"✅ Generated tree validated: {len(found)} files, bijection + JSONL schema + black + UTF-8/LF."))


def diff_summary(tree: Path, live: Path) -> bool:
    """Print the tree-vs-committed diff summary; return True when the trees differ."""
    new_files, old_files = set(_tree_files(tree)), set(_tree_files(live)) if live.is_dir() else set()
    added = sorted(new_files - old_files)
    removed = sorted(old_files - new_files)
    changed = sorted(rel for rel in new_files & old_files if not filecmp.cmp(tree / rel, live / rel, shallow=False))
    unchanged = len(new_files & old_files) - len(changed)
    print(_bold(f"📋 Diff vs {live.relative_to(REPO)}:"), end=" ")
    print(
        f"{_green(f'{unchanged} unchanged')}, {_yellow(f'{len(changed)} changed')}, "
        f"{_green(f'{len(added)} added')}, {_red(f'{len(removed)} removed')}"
    )
    for label, entries, paint in (("A", added, _green), ("D", removed, _red), ("M", changed, _yellow)):
        for rel in entries[:40]:
            print(f"  {paint(label)} {rel}")
        if len(entries) > 40:
            print(_dim(f"  … and {len(entries) - 40} more"))
    for rel in changed[:3]:
        try:
            old = (live / rel).read_text(encoding="utf-8").splitlines()
            new = (tree / rel).read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        head = list(difflib.unified_diff(old, new, fromfile=f"committed/{rel}", tofile=f"generated/{rel}", lineterm=""))
        if head:
            print(_dim("\n".join(head[:16])))
    return bool(added or removed or changed)


def replace_corpus(tree: Path, live: Path) -> None:
    from tests._golden_cases import REJECTIONS, rejection_module_path

    for rejection in REJECTIONS:  # the immutable inputs must have survived generation verbatim
        source = rejection_module_path(rejection.case_id)
        copied = tree / "rejections" / f"{rejection.case_id}.py"
        assert copied.is_file() and filecmp.cmp(source, copied, shallow=False), f"input drift on {source.name}"
    assert (tree / "README.md").is_file()
    # Copy aside, then swap by rename: the incoming tree is fully materialized next to the live one before the
    # live one is touched, so no failure window can lose both (the worst case leaves .new/.old for inspection).
    staged = live.parent / f"{live.name}.new"
    displaced = live.parent / f"{live.name}.old"
    for leftover in (staged, displaced):
        if leftover.is_dir():
            shutil.rmtree(leftover)
    shutil.copytree(tree, staged)
    if live.is_dir():
        live.rename(displaced)
    staged.rename(live)
    if displaced.is_dir():
        shutil.rmtree(displaced)
    print(_green(f"💾 Replaced {live.relative_to(REPO)}. Commit it together with the causing code+DESIGN change."))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="replace tests/golden/ with the validated tree")
    parser.add_argument("--check-determinism", action="store_true", help="run the seed matrix before anything else")
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated full-corpus seed list override")
    parser.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) * 2 // 3))
    parser.add_argument("--keep", action="store_true", help="keep the temporary trees for inspection")
    parser.add_argument("--emit", type=Path, default=None, help=argparse.SUPPRESS)  # internal: generate one tree
    parser.add_argument("--witnesses", action="store_true", help=argparse.SUPPRESS)  # internal: witness entries
    args = parser.parse_args()

    if args.emit is not None:
        emit_corpus(args.emit)
        return 0
    if args.witnesses:
        run_witnesses()
        return 0

    live = REPO / "tests" / "golden"
    work = Path(tempfile.mkdtemp(prefix="holoso-golden-"))
    print(_dim(f"working tree: {work}"))
    try:
        started = time.monotonic()
        if args.check_determinism:
            seeds = [int(seed) for seed in args.seeds.split(",")] if args.seeds else FULL_SEEDS
            tree = check_determinism(work, seeds, args.jobs)
        else:
            print(_bold("🛠️  Generating the corpus in a fresh interpreter (PYTHONHASHSEED=0)…"))
            tree = _emit_tree(work / "corpus", 0)
        print(_dim(f"generation finished in {time.monotonic() - started:.0f} s"))
        validate_tree(tree)
        differs = diff_summary(tree, live)
        if not differs:
            print(_green("🧊 The committed corpus is already identical. Nothing to do."))
            return 0
        if not args.write:
            print(_yellow("⚠️  Differences found; NOT written (pass --write to replace tests/golden/)."))
            return 1
        replace_corpus(tree, live)
        return 0
    finally:
        if args.keep:
            print(_dim(f"kept working tree: {work}"))
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

# Golden corpus

This directory freezes the complete observable output of the compiler for the catalogued cases: the emitted
Verilog (with the `holoso.__version__` header token canonicalized to `Holoso v<VERSION>`), the pre-optimize
frontend HIR dumps, the ABI manifests (ports, operator configuration, initiation interval, exact schedule
metrics), the shared support library, and the diagnostic corpus (immutable rejection kernels plus their
rendered rejections). The catalogue itself lives in `tests/_golden_cases.py`; the gate is
`tests/test_golden.py`; the design note is `docs/decisions/freeze-design.md`.

## Update protocol

Golden artifacts change only through `tools/refreeze_golden.py --write`, and only in the same commit as the
code and DESIGN.md change that causes the diff. A golden diff without a causing change is a defect. The tool
generates the complete corpus into a temporary tree in a fresh interpreter, validates it (catalogue bijection,
black on the rejection modules, JSONL schema, UTF-8/LF), shows the diff, and replaces this directory only
under the explicit `--write` flag. `--check-determinism` certifies seed independence first: the full corpus
under at least eight `PYTHONHASHSEED` values, the cheap targeted witnesses over seeds 0..63, and one seed
repeated in three fresh processes, all byte-compared.

Rejection modules under `rejections/` are append-only and immutable: once landed they are never edited, so the
line numbers recorded in `diagnostics/*.jsonl` stay true by construction. To retire a case, delete its module
and registry entry together; to change a kernel, add a new case instead.

`provenance.json` records the capturing CPython and key dependency versions. Its `container_digest` field is
null on local captures by design: CI fills it when a capture is made inside a pinned container, and there is
no way to carry a comment in JSON, so this file documents the convention instead.

All files are UTF-8 with LF line endings. Everything here is generated or immutable input; do not hand-edit.

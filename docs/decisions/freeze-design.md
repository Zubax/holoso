# Freeze design (S2.15), as amended by consult X3

X3 (Codex gpt-5.6-sol, ultra; transcript in the session scratchpad) reviewed campaign section F and returned
five positions, all CHANGE recommended, all adopted here. None weaken the maintainer's pre-authorized canonical
gate; several make it real. This document supersedes section F where they differ.

## 1. Diagnostics corpus: structured JSONL, not three-column text

Per rejection case: the public catch class, the complete public payload (rendered str(exc), location file/line/
column AND the location's source line text), the semantic origin frames (function names and positions, i.e. the
"in callee():" context), any selected advisories, and — where a kernel deliberately carries competing errors —
the pinned precedence. One JSON object per line, one file per case family, schema-versioned.

## 2. Rejection kernels: append-only DIRECTORY of immutable per-case modules

tests/golden/rejections/<case_id>.py, one kernel per file, black-formatted from birth, never edited after
landing (line numbers stay true by construction); a structured registry (the corpus index, point 6) maps
case_id -> module, expected class, and the JSONL row. This kills the monolithic-file line-shift hazard.

## 3. One GoldenCase catalogue; the latency/metrics example rows merge into it

A typed GoldenCase record: example factory INCLUDING the exact reset variant (the shipped and cosim-reset EKF
are DISTINCT cases — the (example, format) key alone is wrong), float format, complete operator configuration,
module name, fetch depth, regalloc knobs. test_golden.py parametrizes one item per case which lowers the
pre-optimize HIR, builds optimized HIR/MIR/LIR ONCE, and asserts Verilog bytes, the full ABI manifest, exact
schedule metrics, and any legacy ceiling — xdist sharing is automatic because one item runs on one worker.
test_latency_freeze.py and test_metrics.py keep only their non-overlapping specialty probes (chained-copy
behavior, allocator repeatability, determinism).

## 4. Determinism certification: a seed matrix in the refreeze tool, not a 3-seed smoke

tools/refreeze_golden.py --check-determinism generates complete corpus trees in fresh interpreters: the full
corpus under at least eight explicit seeds (0, 1, 2, 3, 42, 31337 plus two more), the cheap targeted witnesses
over seeds 0-63, and one fixed seed repeated three times in fresh processes (non-hash process nondeterminism);
byte-compares every artifact across trees and aborts BEFORE writing, naming the seed and first differing file.
CI's unset seed stays as ongoing sampling.

## 5. Load-bearing contracts section F lacked

- HIR dumps: a versioned COMPLETE serializer (blocks, operation order, terminators, ordered inputs, named
  outputs, state slots with resets, live-outs; explicit kind/operator spellings; floats as binary64 bit
  patterns) — never raw dataclass/enum repr. The test_determinism sorted-nodes dumper is not it.
- The canonical landing needs a real BLOCK-AND-VALUE alpha-canonicalizer (renumber() compacts block ids only),
  itself tested on deliberately permuted equivalent HIRs, and additionally the independent semantic gates
  (Python-reference differential and MIR-interpreter oracle) because cosim's numerical model derives from the
  same LIR it certifies.
- Exact metrics per case (nreg, bnreg, steering, copies, min_ii, last_pc, max_block_span) — the public
  initiation_interval loses last_pc for multi-block kernels; ceilings stay as separate non-regression rows.
- Corpus surface: add structural-only cases (no cosim driver) for finite_set_current_controller (the most
  composite compiling kernel), native vector-port polar, imu, and the shipped EKF reset.
- Format/config policy, deliberate not accidental: the inherited matrix (22 x e8m36, UARTs at e4m8, octave at
  e6m18+e8m36) plus a compact format-sensitive probe at e6m18/e8m24/e8m36/the frontend carrier format, plus one
  deeply staged operator configuration case. Formats and full op config are part of case identity.
- ABI manifest: ordered complete port list (direction, name, control/data role, scalar kind, width, float
  format), control and diagnostic ports included, module name, op config, public II and internal schedule.
- Text volatility: canonicalize the holoso.__version__ token in Verilog headers for comparison (live output
  separately asserted to carry the real version); freeze holoso_support.v as an artifact.
- Provenance: record CPython minor, dependency versions, and the CI container digest at capture; require
  UTF-8/LF. Raw INFO/DEBUG logs are excluded; semantic advisories follow the point-1 policy.
- Corpus index: machine-readable; test_golden asserts a BIJECTION between declared cases and on-disk artifacts
  (fail on stale extras, not just missing). The refreeze tool generates into a temporary tree, validates fully,
  shows the diff, and replaces only under an explicit write flag.

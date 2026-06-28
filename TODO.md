# TODO

## Scheduler

Multi-instance pooling: `resolve_pool` (`_schedule.py:91`) is hardcoded to budget 1. Downstream is already
N>1-ready (verified firsthand: budget 2 binds two `fmul` instances, II 21→20, model==interpreter on 300 vectors).
Needs (a) a demand/feedback signal — `resolve_pool(nodes, prior_schedule)` reading `busy_until` saturation — and
(b) a scored evaluation surface over `_layout_and_allocate` (II / mux fan-in / reg count). Revisit the reuse bound
in `OperatorInstance.__post_init__` (now `latency + boundary_step(0) + 1` since the read-latch removal); its deferred
re-entry-distance check is exactly what an II>1 instance on a back-edge loop would need.

Aggressive cross-block overlap (DEFERRED in DESIGN): cleanly separable — loosen `_issue_side_envelope`'s floor,
replicate commit-side control per mutually-exclusive successor arm in the emitter, exercise the single-writer
validator (`_microcode.py`). Promote write-control to a first-class edge/path event.

Loop-carried install pin is conservative: `install_issue_cycle` pins a computed-source phi-arm copy to
`work_makespan + 1`, but when the source is not the block's last work the install lands a terminator-cycle late
(recip_newton blk2: the `y_next → y` copy lands block-rel 17, could be 12 — read-first against the old `y` still holds —
costing +1 cycle per loop iteration). Pin the install to its actual source's landing instead of the block makespan;
re-freeze `test_latency_freeze`/`test_metrics`, and keep it correctness-sensitive (the read-first must hold). Surfaced
by the schedule-quality audit.

`_issue_side_envelope`'s hardcoded terminator `floor = 1` makes an empty overlapping branch block with a resident-only
condition pay 1 cycle vs the resident-only minimum of 0 (uart_tx blk1). Probably structural — every branch block has
term_offset ≥ 1 (the conditional PC-redirect settling through the fetch pipeline) while jump/`Ret` blocks reach 0 — so
confirm `floor = 1` is a genuine branch minimum before shedding; if sheddable, recover the cycle. Surfaced by the
schedule-quality audit (lower confidence than the install-pin item).

## Integer support adjacent

Front-end `_apply_binop` (`_lower.py`) dispatches arithmetic by AST syntax, not operand type; make it
type-dispatched like `_lower_compare`, and route int literals to a typed `IntConst`.

Scalar-family policy: `PortConditioner` is a closed `FloatSignControl | BoolInversion` union enforced on every
MIR port (`_operators.py:86`); add an int conditioner (likely identity/no-op) + a scalar-family table owning
conditioner/bank/coercion/reset/lowering hooks.

The oracles store wide values as `FloatValue` (`numerical.py`, `_mir/_interpret.py`); introduce a
`FloatValue | IntValue` wide-value union, a typed `lir.wide_consts` pool (constants are float-encoded across
microcode/emit/html/model today), and one shared scalar port codec (cocotb + model duplicate it).

Strength reduction is float-keyed (`cval: dict[ValueId, float]`); add a sibling int reduction + a typed-constant
cache when int lands.

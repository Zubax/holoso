# TODO

Front-end `_apply_binop` (`_lower.py`) dispatches arithmetic by AST syntax, not operand type; make it
type-dispatched like `_lower_compare`, and route int literals to a typed `IntConst`. (integer)

Scalar-family policy: `PortConditioner` is a closed `FloatSignControl | BoolInversion` union enforced on every
MIR port (`_operators.py:86`); add an int conditioner (likely identity/no-op) + a scalar-family table owning
conditioner/bank/coercion/reset/lowering hooks. (integer)

Both oracles store wide values as `FloatValue` (`numerical.py`, `_mir/_interpret.py`); introduce a
`FloatValue | IntValue` wide-value union, a typed `lir.wide_consts` pool (constants are float-encoded across
microcode/emit/html/model today), and one shared scalar port codec (cocotb + model duplicate it). (integer)

Strength reduction is float-keyed (`cval: dict[ValueId, float]`); add a sibling int reduction + a typed-constant
cache when int lands. (integer; a hook, not debt)

Multi-instance pooling: `resolve_pool` (`_schedule.py:91`) is hardcoded to budget 1. Downstream is already
N>1-ready (verified firsthand: budget 2 binds two `fmul` instances, II 21→20, model==interpreter on 300 vectors).
Needs (a) a demand/feedback signal — `resolve_pool(nodes, prior_schedule)` reading `busy_until` saturation — and
(b) a scored evaluation surface over `_layout_and_allocate` (II / mux fan-in / reg count). Revisit the `+2` II-bound
in `OperatorInstance.__post_init__`. (scheduler)

Aggressive cross-block overlap (DEFERRED in DESIGN): cleanly separable — loosen `_issue_side_envelope`'s floor,
replicate commit-side control per mutually-exclusive successor arm in the emitter, exercise the single-writer
validator (`_microcode.py`). Promote write-control to a first-class edge/path event. (scheduler)

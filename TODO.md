# TODO

## Integer support adjacent (the integer wiring milestone)

Scalar-family policy: `PortConditioner` is a closed `FloatSignControl | BoolInversion` union enforced on every
MIR port (`_operators.py:86`); add an int conditioner (likely identity/no-op) + a scalar-family table owning
conditioner/bank/coercion/reset/lowering hooks.

The oracles store wide values as `FloatValue` (`numerical.py`, `_mir/_interpret.py`); introduce a
`FloatValue | IntValue` wide-value union, a typed `lir.wide_consts` pool (constants are float-encoded across
microcode/emit/html/model today), and one shared scalar port codec (cocotb + model duplicate it).

Strength reduction is float-keyed (`cval: dict[ValueId, float]`); add a sibling int reduction + a typed-constant
cache when int lands.

Also queued for this milestone (see DESIGN.md "Integers"): collapse the `pow`/`np.power` spellings onto `**` with an
integer `np.power` overload, and give `np.sign` an integer lowering in place of its analyzer rejection.

## Known defects needing resolution

Bare `AssertionError` when two public state slots share a live-out. `assert RegRef(reg) not in write_books, "a
boundary-install slot must carry no opcode write sources"` (`_backend/verilog/_emit.py`) fires when two public state
slots end a transaction on the same live-out register -- e.g. two writes to `self.a` followed by `self.b = self.a`
(a single copy is fine). The front-end, HIR, MIR, and LIR all accept it; only Verilog emission trips. It fails
loudly with no output, but the message is addressed to a compiler developer, not the user, who deserves a located
`SynthesisError`.

## Deferred capability gaps (tracked in the FIR_PARITY_PENDING registry; stage 10 asserts the registry empty)

The analyzer has no aggregate story for W-typed state: tuple-valued attributes (the delay-line idiom
`self.window = (self.window[1], x)`) reject with "unsupported reset type". The W/D fixed point needs elementwise
per-leaf live-ins (the aggregate stages).

The emitter does not yet emit aggregate (tuple/list) returns or multi-leaf return places; such kernels get a
located EmissionRejection. The per-leaf decomposition of aggregate-valued Places (locals, state, the return place)
lands with the structural-spine stage. The differential harness covers scalar-returning kernels only until then.

Iteration/indexing over a static-LENGTH runtime-element aggregate (`for v in (x, y, x+y)`; `[x]*3` then indexed)
needs the aggregate layout to thread through named-local stores and the loop unroller -- same stage family.

The old front-end conservatively rejected a few valid corner kernels (arithmetic on an empty aggregate slice, an
empty-aggregate loop nested in a `while` demoting the outer counter, a comprehension target named `self`); re-triage
these against the FIR front-end when the aggregate stages land, and record the surviving ones here with FIR evidence.

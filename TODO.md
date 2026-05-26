# TODO/FIXME

## The generic/specialized duality is premature and currently pays negative rent

Every layer carries a generic base plus exactly one concrete float member: `Type`/`ScalarType` and the two `FloatType`s
(`_hir/_types.py`, `_type.py`); `Signature`/`ScalarSignature`; `Operator`/`HardwareOperator`/`FloatHardwareOperator`;
the `MirInput`/`MirConst`/`MirOperation`/`MirOutput` families each with a sole `MirFloat*` subclass; the LIR
`OperatorInstance`/`RegRef`/`ConstRef`/`Operand`/`ScheduledOp`/`InputLoad`/`OutputWire`/`RegFileLayout` families each
with a sole `Float*` subclass; and a `_DomainLowerer` ABC with a one-element `_domains` list (`_mir/_lower.py:63,84`).

Because float is the only member, the generic layer carries no behavior. It only erases the type that the float layer
immediately re-asserts, which shows up as pervasive re-narrowing: the `_operation()` assert helpers (`_build.py:37`,
`_schedule.py:27`, `_regalloc.py:29`), the `isinstance(inst.operator, FloatHardwareOperator)` recheck
(`_build.py:96`), and `_check_supported_domains` (`_build.py:77`) whose entire job is rejecting union members the
float-only frontend and single-domain lowerer cannot produce (it is dead in the current pipeline).

Accepted solution: Commit to the generality and push the typing through so code stops losing and re-recovering
the float type (carry `MirFloatOperation` lists after one validated narrowing rather than `MirOperation` + assert
at every use).

## ModuleInterface is not the single source of truth it claims to be

`interface_of` (`_build.py:252`) hardcodes the port list (`clk`, `rst`, `in_valid`, `in_ready`, `out_valid`,
`out_ready`, the `in_*` data ports, the outputs, `err_cyc`). The Verilog backend's `_emit_header`
(`verilog.py:141-168`) independently hardcodes the same names and widths, and `generate(lir)` (`verilog.py:121`) does
not take the interface at all. The authoritative port contract therefore lives in two places with nothing enforcing
agreement; a rename in one silently diverges from the other.

The consumers barely use the abstraction either: cocotb takes the full `ModuleInterface` only to read `module_name`,
and the HTML backend takes it but then re-lexes the emitted Verilog text with regexes (`html.py` `_module_header`,
which raises `RuntimeError` if its sibling backend's output fails to parse) to recover header structure it was already
handed in typed form.

Fix direction. Make the interface the source of the port list and feed it into the Verilog emitter (and the HTML
header), or trim `ModuleInterface` down to whatever is actually load-bearing. Eliminate the re-lexing of emitted
Verilog.

## LIR nodes are behavior-free, so every backend re-derives the same things

`RegRef`/`ConstRef`/`Operand` are pure data, so each backend independently re-implements source discrimination plus the
`r{i}`/`c{i}` stable label (`verilog.py:74-82`, `numerical.py:77-81`, `html.py:52-54`), per-cycle issue/commit grouping
(`verilog.py:_group_by_cycle` vs the HTML backend's inline walk), and liveness (`html.py:_liveness` reconstructs what
the register allocator already computed; the numerical model rebuilds its own write timeline). The `("r"/"c")` and
`("in"/"op")` tuple tags are stringly-typed discriminators whose two-armed `else` branches will silently misclassify a
third source kind (the bool/int domain item 1 is built for).

`FloatSignControl` (`_operators.py:24`) is the proof of the better pattern: it owns its semantics (`apply_float`),
encoding (`encoded`), and rendering (`decorate`), and is consequently the one thing no backend duplicates.

Fix direction. Give the ref/operand types the same treatment (a polymorphic stable label and an "is this a register"
answer on the type, or a structural match over a proper union), and add shared schedule-grouping and liveness helpers
in `_lir` so the backends consume them instead of re-deriving.

## HIR operators and HIR nodes are circularly dependent, papered over with lazy imports

`_ir.py` imports `Operator` from `_operators.py`, while `_operators.py` needs `Const`/`FloatConst` from `_ir.py`
because `fold_constants` constructs node objects. The cycle is worked around with a `TYPE_CHECKING` guard plus a
repeated `from ._ir import FloatConst` inside all six `fold_constants` bodies
(`_hir/_operators.py:53, 74, 95, 116, 136, 159`). Repeated in-method imports signal that something is in the wrong
module: the operator layer reaches up into the IR layer it is supposed to sit beneath.

Fix direction. Move `Const`/`FloatConst` into a lower module both `_ir` and `_operators` can import without a
cycle (it is closer to a value type than to the node graph).

## Domain validation is centralized in build(), but the passes trust-and-assert

The only real domain check is `_check_supported_domains` inside `build` (`_build.py:77`, using real `raise`). The
scheduler and allocator instead assume float via `assert isinstance` (`_schedule.py:27-30`, `_regalloc.py:29-32`),
which is stripped under `python -O`. That is fine while `build` is the sole entry, but `schedule_ops`/`resolve_pool`
are public enough that the tests already call them directly (`tests/test_schedule.py:26`); a pass invoked standalone on
non-float MIR would, optimized, narrow incorrectly with no diagnostic.

Fix direction. Move the cheap domain guard into the passes themselves, or make the narrowing structural (typed inputs)
rather than assertion-based. Folds into the resolution of item 1.

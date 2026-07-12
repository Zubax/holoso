# TODO

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

## Frontend subset limitations

A few valid kernels are conservatively rejected rather than compiled. None is a wrong answer; each is a located
rejection, and each is unusual enough that the precision has not been worth its cost yet.

Arithmetic on an empty aggregate (`-v[:0]`, `v[:0] + v[:0]`) is rejected. An empty aggregate carries no leaves, so the
leaf-type and shape checks cannot run -- which is exactly what must reject `-boolflags[:0]` (a boolean negation) and
`a[:0,:] + b[:0,:]` (a width mismatch), both of which CPython rejects. Distinguishing the valid empty-float case would
need an empty-but-typed aggregate in the value model. This is acceptable, no fix required at the moment.

An empty aggregate loop inside a `while` (`for i in range(2): pass` then `while c: for i in []: pass`) demotes the
outer counter `i`, so a later `v[i]` is rejected as a non-static index though `i` is still 1 at runtime. This is the
safe side of the scan/lowering demotion that keeps the state-write invariant sound; making it exact would require the
demotion to see that an empty aggregate rebinds nothing. This is acceptable, no fix required at the moment.

A comprehension target named `self` (`[self for self in [x]]`) is rejected as rebinding the instance parameter, though
a comprehension has its own scope and the name is a fresh local there. The self-rebinding guard should not apply to a
comprehension target. This is acceptable, no fix required at the moment.

## Known defects needing resolution

Closure free variables are not consulted, so a captured name resolves to a same-named module global (or to the
builtin). A kernel reading a closure variable folds in a module-level global sharing its name (`x * gain` takes a
module `gain`, not the freevar), and a captured rebinding of `range`/`len` is treated as the builtin (a freevar
`range` returning a one-element list still unrolls as `builtin range(2)`, giving two trips where Python runs one).
Both are silent. `_resolve_name` (`_lower.py:2635`) consults closure cells but is only reached when resolving a
callee; the value-position paths (`_lower_expr`, `_static_int`, `_static_float`, `_static_ndarray_value`) call
`_module_global` (`_lower.py:2611`) directly, and `_is_builtin_name` (`_lower.py:946`) checks `__globals__` only --
all skip `__closure__`. Absent a shadowing global the name is correctly rejected as unknown, which is why it has
hidden. Fix: route every name lookup, builtin detection included, through one resolver that checks `__closure__`
before `__globals__`, mirroring Python's LEGB order; it touches several static evaluators and wants its own tests.

`del` of a module global is not recognized as making the name local. `_collect_local_names` (`_lower.py:2580`) does
not handle `ast.Delete`, so a function that does `del GLOBAL` anywhere binds the name as local throughout in CPython
(a later read is `UnboundLocalError`), but Holoso still resolves it to the module global and folds its value in. The
collector also omits names bound by a nested `def`/`class`. Fix: add `ast.Delete` (and nested-definition targets) to
the local-name walk. A contrived case, but a silent divergence from Python's scoping.

Bare `AssertionError` when two public state slots share a live-out. `assert RegRef(reg) not in write_books, "a
boundary-install slot must carry no opcode write sources"` (`_emit.py:639`) fires when two public state slots end a
transaction on the same live-out register -- e.g. two writes to `self.a` followed by `self.b = self.a` (a single copy
is fine). The front-end, HIR, MIR, and LIR all accept it; only Verilog emission trips. It fails loudly with no output,
but the message is addressed to a compiler developer, not the user, who deserves a located `SynthesisError`.

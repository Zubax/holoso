# TODO

## Resolution: examples/finite_set_current_controller.py

The finite-set VSI current controller was lifted from a real project that knows nothing about Holoso, so it doubles
as a capability probe. Every gap below was verified against the frontend by compilation attempts; the verdicts
follow one rule: a feature that is easy to add is added, a hard feature that is easy to work around in the kernel is
worked around, and a hard feature that matters for typical control/DSP targets is added regardless. The example is
tweaked only where noted; everything else lands in the frontend.

ADD to the frontend, consolidated into four workpieces:

- R1, frozen records. One restricted record value kind serving all three uses the example makes of dataclasses:
  a runtime parameter (`kin: Kinematics` decomposes to field-path scalar ports `kin_pos`, `kin_vel`, ...), in-kernel
  construction (`CurrentControllerDecision(...)` -- structural and synthesized, admitting only plain generated
  frozen dataclasses with no user `__init__`/`__post_init__`, whose `object.__setattr__` could not inline anyway),
  and the return value (flattened to field-path output leaves). Needs recursive annotation conformance, field
  reads, joins, allocation traversal through record fields, and jaxtyping shapes on array-typed fields
  (`switch_balance: Float64[np.ndarray, "3 1"]`). Moderate cost; high value -- typed signal bundles are idiomatic
  in control code. Attribute-held frozen dataclass instances interact cleanly with hierarchical state: they are
  walked as components but seed nothing (generated methods do not desugar and field writes cannot exist), so they
  stay frozen-folded.

- R2, reductions: `np.mean`, `np.max`/`np.min` (with their alias and method spellings), and `np.sum` -- left folds
  over the leaves in the `_dot` style so FMA contraction stays reachable. Whole-array only (no axis/keepdims/dtype
  forms); integer `mean` promotes to float before accumulating (numpy semantics) while `sum`/`min`/`max` preserve
  the array's scalar family. Float min/max lower through `fsort` (which the example's operator set must
  configure); integer min/max keep the existing compare/select lowering. Easy once the first lands.

- R3, structural transforms: `reshape` in both spellings, `a.reshape(shape)` and `np.reshape(a, shape)`, with
  tuple shapes and the scalar `reshape(3)` form (the example uses `(2,1)`, `3`, and `(3,1)`), and the
  `dtype=float` keyword on `np.array`/`np.asarray` (needed to build a float vector from the bool switch tuple in
  `_balance_step`, and by `_zero_mean`). Neither is a trivial stub: `reshape` is a DERIVATION that conservatively
  tracks argument 0's allocation (the host MAY return a view) and needs the registry to accept a tuple shape
  argument -- today the `_array_call` sequence gate rejects it, and `derives` asserts a single argument, to be
  replaced by the invariant that the result tracks argument 0; the gate carve-out must be a per-stub opt-in
  sequence-argument position, not a global relaxation (the same opt-in serves `np.polyval`'s coefficient argument
  in R4). `dtype=` needs a keyword binder on the Conversion path and conditional copy semantics: a dtype-CHANGING
  `asarray` copies on the host, so it must mint a fresh allocation.

- R4, new library stubs: `np.polyval` (Horner left fold; ndarray coefficients, sequence
  coefficients following R3's argument-domain decision) and elementwise `np.abs` on arrays. The latter is not a
  plain registration: `np.abs` already resolves to the scalar intrinsic entries and the registry serves one entry
  per key, so the array form needs rank-sensitive per-key lifting (`np.abs`/`np.absolute`/builtin `abs` become
  array-capable; `math.fabs` stays scalar-only) -- a registry extension that would incidentally serve every scalar
  ufunc on arrays. `np.sum` folded into R2 above.

TWEAK the example (minimal edits, each spelled here):

- T1: the module-boundary `np.ndarray` annotations become fixed-shape jaxtyping annotations (import `Float64`;
  `i_ac`, `di_ac_dt`: `Float64[np.ndarray, "3"]`; `i_dq_ref`: `Float64[np.ndarray, "2"]`; and under R1 the
  OUTPUT-record field `CurrentControllerDecision.switch_balance: Float64[np.ndarray, "3 1"]` -- distinct from
  the persistent `self._switch_balance`, which needs no annotation). Required by design -- no dynamic shapes.
  Drop the example's stale `TODO FIXME: Currently unsupported` docstring line and the helper shape-TODO comments
  while at it.
  Helper annotations (`_dq0_to_ac`, `_zero_mean`) need no change: unrecognized annotations on inlined frames are
  deliberately unchecked. The state array's shape needs no annotation either (the reset snapshot carries it);
  the field-annotation comments inside the example are stale on this point.

- T2: the argmax tail of `_select_switch` -- `active_drive >= best - eps` builds a runtime boolean array,
  `np.flatnonzero(...)[0]` a runtime index, and `self._active_switch_candidates[active]` a runtime-indexed read
  of a static table: three value-model concepts (boolean arrays, data-dependent search, runtime aggregate
  indexing) serving one line. Rewrite preserving the exact tie rule (the EARLIEST candidate within tolerance of
  the global maximum, not the first strict maximum): pass 1 computes `best = float(np.max(active_drive))` (R2);
  pass 2 scans the static candidate list first-wins with `drive_i >= best - 1e-12 * max(abs(best), 1.0)`,
  selecting the switch triple through conditional rebinds. The scan shape compiles today (verified); `max` and
  `abs` on scalars resolve already.

- T3: `switch_balance=self._switch_balance.reshape((self._n_phases, 1))` returns a live view of persistent state
  once reshape is a derivation, and returning state aliases is refused (correctly -- hardware cannot honor the
  live handle Python would return). Rewrite: `np.array(self._switch_balance.reshape((self._n_phases, 1)))`;
  `np.array` copies, minting the fresh allocation the return gate demands.

Suggested order: R2 (unblocks T2), R3 (unblocks T3), R1 (the interface), R4 alongside R2/R3, then the example
tweaks and its catalogue registration.

## Frontend limitations

An empty array slice (`v[:0]`) is refused where it is taken rather than where it is used, so even `len(v[:0])` fails;
an empty sequence slice is fine. An empty array carries no leaves, so the leaf-type and shape checks cannot run --
which is what must reject `-boolflags[:0]` and `a[:0,:] + b[:0,:]`, both of which CPython rejects too. Accepting the
valid empty-float case needs an empty-but-typed array in the value model.

A counted `for` above the unroll threshold turns its target into a runtime integer, so a body that indexes an
aggregate with it (`v[i]`) is refused at the subscript, and one that converts it needs the conversion operator the
unrolled form folds away; forcing the unroll with `for i in list(range(...))` is the rewrite. Its body is a
data-dependent region, so installing a whole aggregate into a state attribute there is refused as it is in a
`while` -- storing elements is the rewrite. A range returned across a helper's own data-dependent region is refused
at the call, and an unconditional-exit body (`for i in range(n): break`) keeps the existing no-back-edge refusal.

A data-dependent loop carries the syntactic set of names its body assigns, fixed before the body is interpreted
because the header phis must exist first. A leaked `for` counter assigned only on a statically-dead path
(`if False: i = ...`), or by a loop that rebinds nothing (`for i in []`), is carried anyway and stops being a
compile-time integer, so a later `v[i]` is refused as a non-static index. Only the compile-time index is lost; the
loop computes correctly. Exactness needs a fold-aware carried set, which the loop setup cannot have without
interpreting the body first.

Re-installing a tensor derivation into the state attribute it came from (`self.P = self.P.T`) is refused by the
state-disjointness rule even though the slot is fully overwritten, because the derivation shares its source's storage.
The diagnostic names the fix (`np.array(self.P.T)`); lifting it needs the install check to see that the source and the
destination are the same slot.

A tuple swap of two state attributes (`self.a, self.b = self.b, self.a + x`) is refused because an unpack target must
be a plain name. It is the one refusal here with no one-line rewrite -- the swap needs a temporary.

Also refused, each naming the construct and each with a plain rewrite: a walrus that reads the name it binds
(`b = (a := a + x)`); a `list[...]` return annotation, in favour of `tuple[...]`, though a returned list literal stays
legal; a value that is an array in one arm and a sequence in the other, since aggregates join a branch only when every
arm agrees in kind; a comprehension `if` filter or a comprehension with more than one `for` clause; and a starred
unpack target (`first, *rest = v`).

`while True:` with a data-dependent `break` exhausts the graph expansion budget rather than residualizing: a literal
`True` header is decidable on every trip, so the loop unrolls until the budget stops it, while a runtime `break` opens
an exit lane without ever closing the fall path. The exit condition belongs in the header. This is deliberate --
every cheap detector for the shape misfires on the legitimate counter-spelled loop.

A state attribute's shape and type come from the reset snapshot, so a field annotation contradicting it
(`P: Float64[np.ndarray, "2 2"]` on an instance holding a 3x3) is documentation rather than a checked declaration.
Parameter and return annotations are checked, so the module boundary is judged while the state boundary is not.

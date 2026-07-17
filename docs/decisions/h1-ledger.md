# H1 ledger — the FIR_PARITY_PENDING triage

Measured at `dev` = 0c4afb6 on a scratch worktree with all skip decorators stripped (strip: 84 decorators = 83
single-line + 1 multi-line, 87 lines — correcting the earlier 86+1 census). Run: light-marker subset,
`.nox/tests/bin/python`, `-n 5`. Outcome: 47 pass / 35 fail / 2 not-run (cosim-marked). Every failing test was
re-run individually and classified below; per-test logs live in the session scratchpad. Fates land at their
named steps; a skip keeps the token in a relabeled reason only until its step lands; the S2.14 exit grep must
find nothing.

Non-decorator token sites (resolved at S2.14): `tests/_examples.py:674-684` (registry + `parity_marks` +
9 consumers), `test_matrix.py:1544-1550` (vacuous emptiness assertion), `tests/_fuzz.py:875/:1018/:1570`
(tuple-lane prose), `_synth_targets.py:97`, `test_cosim_examples.py:21` / `test_synth_examples.py:39`
docstrings, `test_latency_freeze.py:39` comment, `test_metrics.py:232-235` (mislabeled imu skip), TODO.md:32.

## ENABLE now (46) — pass on enablement, feature retained

test_backend.py: ekf1_stateful_elaborates. test_report.py: report_reveals_boolean_operators_and_casts.
test_frontend_aggregates.py: tuple_build_and_index, list_slice, list_is_identity_on_an_aggregate,
static_shape_queries_in_index_range_and_branch_positions, len_follows_python_and_accepts_a_ragged_list,
shape_queries_still_reach_arrays_through_the_same_spellings, for_loop_iterates_an_aggregate,
comprehension_target_shadows_a_same_named_module_constant, iteration_and_shape_queries_reach_every_aggregate_spelling,
indexing_a_sequence_of_arrays_yields_an_array. test_install_landing.py: resident_register_source_install_is_inline_class,
state_read_sourced_install_is_inline_class. test_synth.py: infinity_constants_are_allowed, synthesize_ekf1_stateless.
test_frontend_state.py: numpy_array_state_decomposes_like_a_list, jaxtyping_array_field_lowers_and_is_validated,
numpy_integer_array_values_coerce_to_real, state_write_under_an_aggregate_for_is_not_dropped_by_a_stale_counter,
an_untouched_state_attribute_is_not_resurrected_by_an_unrelated_branch,
a_scan_never_folds_a_shape_query_against_an_environment_lowering_will_not_have,
an_integer_vector_state_reset_keeps_exact_per_element_slots, a_loop_carries_only_attributes_that_are_really_state,
a_shape_query_reads_the_reset_value_not_the_state_decomposition, a_write_is_validated_only_where_lowering_reaches_it,
a_nested_reset_sequence_is_shaped_like_the_aggregate_it_denotes, a_ragged_or_empty_reset_sequence_still_has_a_length,
indexing_a_reset_sequence_of_arrays_yields_an_array, a_scan_must_not_fold_a_branch_on_a_counter_the_loop_body_rebinds,
a_scan_demotes_the_aggregate_target_before_discovering_body_rebinds. test_frontend_calls.py: inlined_global_function,
inlined_global_with_star_args, abs_accepts_a_star_unpacked_argument. test_language_features.py:
property_over_written_state_recomputes_each_read. test_overlap_behavior.py: multi_output_mixed_io_metadata_and_values.
test_public_api_behavior.py: logical_port_is_the_single_public_port_type, mixed_list_io_metadata_and_values.
test_verify.py: equal_temperament_default_sweep_has_no_log2_sidebands, tuple_unpacking_matches_python_reference,
model_matches_reference_ekf1_stateful, model_unrolled_cordic_sin_cos, merged_state_slots_preserve_behaviour,
aliased_slot_with_phi_live_in_builds. test_schedule.py: constant_pool_is_canonically_nonnegative,
stateful_slot_register_gaps_are_reused. (test_determinism.py: verilog_is_byte_identical_across_hash_seeds —
already enabled at S2.1.)

## ENABLE pending trial-push cosim verdict (2)

test_cosim.py: cosim_ekf1_stateless, cosim_ekf1_stateful — cosim-marked, not runnable locally; stripped in the
S2.2 trial push, adjudicated by the throwaway branch's cosim_examples/cosim CI jobs.

## PASS but trim-owned (1) — RELABEL:S2.10

test_frontend_aggregates.py: starred_and_nested_unpacking_route_values — exercises T9 starred assignment
targets; at S2.10 the starless nested-unpacking half survives as a positive test and the starred half becomes
the trim's located-rejection test.

## FAIL classifications (35)

FIX-EXPECTATION — strict return-contract family (re-annotate to the honest container): vector_scalar_broadcast,
flatten_collapses_nesting, unary_plus_and_minus_apply_elementwise_to_aggregates,
numpy_asarray_is_identity_on_an_aggregate (all → `Float64[np.ndarray, "2"]`), tuple_is_identity_on_an_aggregate
(→ `tuple[float, float]`), test_frontend_calls::call_dispatch_is_by_identity_not_spelling (asarray kernel →
ndarray annotation).

FIX-EXPECTATION — stale `match=` re-pins (new exact texts, all verified located at the user line):
indexing_a_scalar_is_rejected + divergent_loop_counter_as_static_index_is_rejected →
`subscript of a runtime value is not supported yet`; flatten_on_a_scalar_is_rejected +
a_shape_query_cannot_slip_past_a_rejection_the_stub_makes → `attribute access on a runtime value`;
numpy_only_shape_queries_are_rejected_on_a_sequence_however_it_is_spelled + a_sequence_stays_a_sequence_through_a_subscript
+ a_shape_query_on_a_nested_reset_sequence_element_is_rejected → `list method '<name>' is not supported (lists
are immutable values here); rebind with + instead`; list_comprehension_unrolls_and_scopes_its_target →
`name 'k' is not defined`; comprehension_yields_a_python_list_not_an_array → `arithmetic on an aggregate value`;
comprehension_scoping_follows_python_exactly → `local 'y' may be unbound here (Python would raise)`;
multi_axis_indexing_validates_its_axes_against_the_shape → `array index 99 is out of range for axis 1 of size 2`
(+ empty-slice multi-axis sub-case inverts to positive); an_empty_aggregate_makes_no_check_vacuous →
`arithmetic on a bool requires an explicit conversion` / `elementwise arithmetic on mismatched shapes (0, 3) and
(0, 2) (only a scalar broadcasts)`; non_operator_numpy_call_stays_unsupported → `call to 'sum' is not supported
in a kernel`; vector_state_slot_name_collision_is_rejected + state_slot_names_only_collide_among_the_attributes_lowering_keeps
→ `state slot name collision on 'v_0' between distinct component attributes`;
only_a_write_lowering_reaches_is_validated → `state attribute 'never_initialized' does not exist on the
component at compile time (assign it in __init__)` (body only; `:0:0` prefix is E3, S2.5);
comprehension_rejections → `positional containers of arities 0 and 1 merge here` +
`only a plain name is supported as a comprehension target` (+ walrus sub-case inverts to positive).

FIX-EXPECTATION — legitimate-behavior inversions (verified against DESIGN and the Python reference):
shape_queries_are_rejected_outside_a_static_position (shape queries fold to constants in value positions);
empty_slice_has_no_shape_but_still_has_a_length (empty slices carry layout; matmul sub-case re-pins
`matmul dimension mismatch: the inner dimensions disagree`); state_write_only_on_a_folded_away_shape_branch_is_not_state
and a_scan_never_rejects_an_arm_lowering_folds_away (folded-away-write attributes are frozen constants: no
slot, no port; model verified equal to Python); test_matrix::unsupported_operator_diagnostic_names_the_operator
(`bool | bool` legitimately lowers — bitor sub-case inverts; modulo-on-arrays re-pins
`operator '%' is not supported on arrays`).

RELABEL (4): raise_on_a_statically_taken_path_is_a_located_synthesis_error → S2.11 (double-blocked E2+E1-lite);
a_raise_message_may_interpolate_a_shape_and_a_counter → S2.3 (E2);
library_stub_error_is_attributed_to_the_call_site → S2.11 (E1);
test_matrix::library_shape_rejection_is_attributed_to_the_user_call_site → S2.11 (E1 anchor; its scalar-`.T`
sub-case expectation is obsolete — scalar attribute reads refuse up front, rewrite at S2.11).

CONVERT (1): test_verify::model_matches_reference_aggregates → T10 at S2.10 (sole blocker is the starred
element in a list display).

GAP (1): a_scalar_takes_an_empty_tuple_key_as_identity_like_a_numpy_scalar — numpy rank-0 parity `x[()]` is
unimplemented; ruled into S2.8's rank-0 doctrine: scalars take no subscript at all, `()` included; DESIGN.md
states it there and the test becomes a located-rejection case.

FIX-EXPECTATION at S2.10 (1): model_bool_cast_of_underflowing_constant_is_false → H3 fastmath ruling: assert
the True-fold (documented behavior) when the fastmath section lands.

BUG (0). Three diagnostic-polish notes folded into S2.5: the list-attribute refusal reuses mutator wording
("rebind with + instead") for numpy-isms like `.ndim`/`.T`; the dynamic comprehension filter surfaces as the
accumulator arity-merge instead of naming the filter rule (left as-is, truthful); `table[runtime_index]` says
"subscript of a runtime value" where the runtime thing is the index (reword).

## H5 hollow-test repairs (land with the S2.2 enablement commit)

test_fir_analyze.py:209 (assert the state leaf's entry fact, not the parameter's); test_fir_builder.py:58
(assert store order/final value, not a count); test_frontend_control.py:47 (point caplog at the live logger);
test_frontend_aggregates.py:115 (X1 find: `match="unpack"` matches the test's own qualified name — pin the real
diagnostic `expression Starred is not supported`... which is also T9/T10-adjacent; re-point after the trim
wording lands, interim: pin the current real message).

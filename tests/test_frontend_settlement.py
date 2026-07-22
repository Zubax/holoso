"""
Frontend tests: the SETTLEMENT verifier, exercised by mutating tables the producer built.

M6 moved the block order, the state slot table and the return plan into the definitive resolution, and emission
executes all three without re-deriving anything. That made the producer's word final: a missing or reordered
block, a wrong reset or slot name, a wrong return row would have reached the emitter unchallenged, with the
golden corpus and the behavioural suite as the only backstop. This module is the other half -- every test here
injects one defect into a table that verified clean and requires the verifier to name it, asserting on the
DIAGNOSTIC FRAGMENT its own check emits so that one over-eager check cannot stand in for a dozen dead ones.

The claim each mutation makes is about INDEPENDENCE, not merely about raising. A verifier that recomputed
`executable_rpo`, re-sorted the recorded store order or re-walked the exit layout would agree with the producer
by construction and pass every mutation below while proving nothing -- which is exactly how the routing
verifier's first state check came to be dead code. So the block order is checked against a reachability closure
and the defining property of a postorder, the slot table against the `PyStoreAttr` ops and the reset snapshots,
and the return rows against the contract walked in the OPPOSITE direction to the producer's own walk.
"""

import dataclasses
from collections.abc import Callable, Mapping

import numpy as np
import pytest

from holoso._frontend._fir._analyze import Analyzer, ResidualUnit
from holoso._frontend._fir._fact import RecordField, TupleIndex
from holoso._frontend._fir._ir import BlockId, StateLeaf
from holoso._frontend._fir._settle import (
    FoldedCell,
    ReturnsLeaves,
    ReturnsScalar,
    SettledReturn,
    StateCell,
    StateSlot,
    verify_settlement,
)
from holoso._frontend._fir._value import SemType
from holoso._hir import FloatConst


def _analyzed(fn: Callable[..., object]) -> ResidualUnit:
    return Analyzer(fn).fixpoint()


def _verify(
    result: ResidualUnit,
    *,
    order: list[BlockId] | None = None,
    stores: list[StateLeaf] | None = None,
    slots: Mapping[StateLeaf, list[StateCell]] | None = None,
    settled: SettledReturn | None = None,
) -> None:
    assert result.settled_return is not None
    verify_settlement(
        result.unit,
        result.executable_edges,
        result.binding_facts,
        result.block_in[result.unit.exit].facts,
        result.state_resets,
        result.runtime_state,
        result.state_livein,
        result.store_order if stores is None else stores,
        result.emission_order if order is None else order,
        result.state_slots if slots is None else slots,
        result.settled_return if settled is None else settled,
    )


def _caught(result: ResidualUnit, fragment: str, **mutation: object) -> None:
    _verify(result)  # the unmutated settlement must pass, or the mutant proves nothing
    with pytest.raises(AssertionError) as raised:
        _verify(result, **mutation)  # type: ignore[arg-type]
    assert fragment in str(raised.value), f"expected {fragment!r} in the diagnostic, got: {raised.value}"


def _branching(x: float, y: float) -> float:
    # A diamond, so the emission order has a genuine choice of shape to get wrong; a straight-line kernel would
    # make the postorder property hold vacuously.
    total = x
    if x > y:
        total = total + y
    else:
        total = total - y
    return total * 2.0


class _TwoLeaves:
    """A scalar leaf and an aggregate one, stored in a fixed source order that IS the state-port ABI."""

    def __init__(self) -> None:
        self.gain = 0.5
        self.window = [1.0, 2.0, 3.0]

    def step(self, x: float) -> float:
        self.gain = self.gain * 0.99
        self.window = [x, self.window[0], self.window[1]]
        return self.gain * self.window[2]


class _PartialFold:
    """Cell 0 is written back unchanged and folds; cell 1 takes the input and is carried."""

    def __init__(self) -> None:
        self.a = [1.0 + 2**-30, 0.0]

    def step(self, x: float) -> float:
        self.a = [self.a[0], x]
        return 10.0 if self.a[0] > 1.0 else 20.0


def _leaf(result: ResidualUnit, name: str) -> StateLeaf:
    return next(leaf for leaf in result.state_slots if leaf.path[-1] == name)


def _registers(result: ResidualUnit, name: str) -> list[StateSlot]:
    """A leaf's cells, all of which must be carried registers for a name mutation to have anything to perturb."""
    cells = result.state_slots[_leaf(result, name)]
    assert all(isinstance(cell, StateSlot) for cell in cells), f"'{name}' has folded cells; pick a carried leaf"
    return [cell for cell in cells if isinstance(cell, StateSlot)]


# ---------------------------------------- the block order ----------------------------------------


def test_a_block_missing_from_the_emission_order_is_caught() -> None:
    result = _analyzed(_branching)
    _caught(result, "the emission order omits", order=result.emission_order[:-1])


def test_a_block_the_emission_order_visits_twice_is_caught() -> None:
    result = _analyzed(_branching)
    _caught(result, "more than once", order=[*result.emission_order, result.emission_order[-1]])


def test_a_block_no_executable_path_reaches_is_caught() -> None:
    # Surplus is the silent-absence archetype inverted: emission would walk a block the analysis never resolved,
    # and every table it subscripts there is missing an entry rather than holding a wrong one.
    result = _analyzed(_branching)
    spare = BlockId(max(block_id.index for block_id in result.unit.blocks) + 1)
    _caught(result, "blocks no executable path reaches", order=[*result.emission_order, spare])


def test_an_emission_order_that_does_not_start_at_the_entry_is_caught() -> None:
    # Emission positions the builder at the first block and builds every input port there, so a rotated order
    # attaches the module's ports to whatever block happens to lead.
    result = _analyzed(_branching)
    rotated = [*result.emission_order[1:], result.emission_order[0]]
    _caught(result, "does not start at the entry block", order=rotated)


def test_an_emission_order_missing_the_canonical_exit_is_caught() -> None:
    # The never-returns decision, read from the other side: `settle_block_order` refuses a unit whose exit no
    # path reaches, and a producer that stopped refusing would hand over exactly this -- an order that walks
    # real blocks and never reaches the one where the outputs and state slots are built.
    result = _analyzed(_branching)
    without_exit = [block_id for block_id in result.emission_order if block_id != result.unit.exit]
    _caught(result, "the canonical exit is not in the emission order", order=without_exit)


def test_an_emission_order_that_is_not_a_postorder_is_caught() -> None:
    # The set is untouched here, so every set-shaped arm passes: only the ORDER is wrong. Emission seals a
    # block's phis once its predecessors are emitted, so a block placed ahead of all of them is reached with no
    # arm for the edge that defines its values.
    result = _analyzed(_branching)
    reversed_tail = [result.emission_order[0], *reversed(result.emission_order[1:])]
    assert set(reversed_tail) == set(result.emission_order)
    _caught(result, "so it is not a postorder", order=reversed_tail)


# ---------------------------------------- the state slot table ----------------------------------------


def test_a_state_leaf_missing_from_the_slot_table_is_caught() -> None:
    result = _analyzed(_TwoLeaves().step)
    dropped = {leaf: slots for leaf, slots in result.state_slots.items() if leaf.path[-1] != "gain"}
    _caught(result, "where the store order names", slots=dropped)


def test_a_reordered_store_order_is_caught() -> None:
    # The STORE ORDER is the state-port ABI -- emission reads the port order from it and takes only names and
    # resets from the slot table -- and nothing about a swapped pair is malformed: both leaves exist, every name
    # and reset is right, and the design still compiles. It just publishes its ports in the wrong order, a silent
    # ABI break for every consumer of the generated module.
    #
    # Measured, and the reason this test targets the store order rather than the slot dict: reversing the SLOT
    # TABLE's key order emits byte-identical HIR for all 36 corpus kernels, because emission never reads that
    # order. A verifier that checked the slot table's order instead would have been dead code with respect to
    # the very ABI it claimed to protect.
    result = _analyzed(_TwoLeaves().step)
    swapped = list(reversed(result.store_order))
    assert set(swapped) == set(result.store_order) and len(swapped) > 1
    _caught(result, "it IS the state-port order", stores=swapped)


def test_a_state_leaf_settled_with_the_wrong_cell_count_is_caught() -> None:
    result = _analyzed(_TwoLeaves().step)
    leaf = _leaf(result, "window")
    truncated = dict(result.state_slots)
    truncated[leaf] = result.state_slots[leaf][:-1]
    _caught(result, "over a 3-cell reset", slots=truncated)


def test_a_folded_cell_promoted_to_a_register_is_caught() -> None:
    # The Task-2 rule read from the verifier's side. A cell the fixed point folded has no register and no port;
    # a table that hands it one publishes the reset NARROWED to the carrier beside reads that use it exactly, and
    # the two disagree wherever the reset is inexact. The name is what gives it away -- a folded cell claims none,
    # so re-materialising one lands outside the canonical coordinates the snapshot spells.
    result = _analyzed(_PartialFold().step)
    leaf = _leaf(result, "a")
    cells = result.state_slots[leaf]
    assert isinstance(cells[0], FoldedCell), "cell 0 must fold, or this mutation removes nothing"
    assert isinstance(cells[1], StateSlot), "cell 1 must be carried, or the leaf is not partially folded"
    promoted = dict(result.state_slots)
    promoted[leaf] = [StateSlot("a_0", cells[0].reset), cells[1]]  # the canonical name, so only the SPLIT is wrong
    _caught(result, "folded it to a constant", slots=promoted)


def test_a_carried_cell_settled_as_folded_is_caught() -> None:
    # The other direction, and the worse one: a cell the design genuinely carries, settled as folded, loses its
    # register entirely and reads its reset on every transaction -- state silently dropped between transactions.
    result = _analyzed(_PartialFold().step)
    leaf = _leaf(result, "a")
    cells = result.state_slots[leaf]
    assert isinstance(cells[1], StateSlot)
    dropped = dict(result.state_slots)
    dropped[leaf] = [cells[0], FoldedCell(cells[1].reset)]
    _caught(result, "the fixed point carries it", slots=dropped)


def test_a_state_cell_settled_with_the_wrong_reset_is_caught() -> None:
    # The reset snapshot is the only OUTSIDE evidence about a slot: the plan's own constants are derived from it,
    # so a verifier reading them back would agree with any wrong answer the producer settled on.
    result = _analyzed(_TwoLeaves().step)
    leaf = _leaf(result, "window")
    wrong = dict(result.state_slots)
    original = result.state_slots[leaf]
    wrong[leaf] = [dataclasses.replace(original[0], reset=FloatConst(99.0)), *original[1:]]
    _caught(result, "but its snapshot cell holds", slots=wrong)


def test_a_state_cell_reset_taken_from_the_wrong_cell_is_caught() -> None:
    # The ordinal mix-up specifically: every constant in the table is a genuine reset of this very leaf, just
    # attached to the wrong cell. A check that only asked "is this one of the leaf's resets" would pass it, and
    # the emitted design would reset each cell of the window to its neighbour's value.
    result = _analyzed(_TwoLeaves().step)
    original = _registers(result, "window")
    rotated = dict(result.state_slots)
    rotated[_leaf(result, "window")] = [
        dataclasses.replace(slot, reset=original[(index + 1) % len(original)].reset)
        for index, slot in enumerate(original)
    ]
    assert {slot.reset for slot in rotated[_leaf(result, "window")]} == {slot.reset for slot in original}
    _caught(result, "but its snapshot cell holds", slots=rotated)


def test_a_state_cell_named_out_of_canonical_order_is_caught() -> None:
    # Names and resets are produced by two separate walks, so a permutation of one alone renames the ports
    # without touching a single constant. Cross-linking the name's cell coordinates back to the snapshot is what
    # sees it; deriving the name from the same ordinal the reset came from could not.
    result = _analyzed(_TwoLeaves().step)
    original = _registers(result, "window")
    renamed = dict(result.state_slots)
    # Cells 1 and 2 only: the first carried cell anchors the component prefix, which no outside source fixes, so
    # moving it would be caught by the coarser arm that asks whether the table names this attribute at all.
    renamed[_leaf(result, "window")] = [
        original[0],
        dataclasses.replace(original[1], name=original[2].name),
        dataclasses.replace(original[2], name=original[1].name),
    ]
    _caught(result, "is named", slots=renamed)


def test_a_slot_named_after_the_wrong_attribute_is_caught() -> None:
    # The attribute a slot publishes is re-derived from the STORE, not from the leaf record the producer keyed
    # its table by, so a table that names the right cells of the wrong attribute disagrees with the graph.
    result = _analyzed(_TwoLeaves().step)
    renamed = dict(result.state_slots)
    renamed[_leaf(result, "gain")] = [dataclasses.replace(_registers(result, "gain")[0], name="offset")]
    _caught(result, "does not name the attribute 'gain'", slots=renamed)


def test_two_state_leaves_sharing_a_slot_name_are_caught() -> None:
    # The producer refuses this collision while building; the verifier owes the same answer over a table it did
    # not build, because a merged name silently aliases two components' state onto one register.
    result = _analyzed(_TwoLeaves().step)
    aliased = dict(result.state_slots)
    aliased[_leaf(result, "gain")] = [
        dataclasses.replace(_registers(result, "gain")[0], name=_registers(result, "window")[0].name)
    ]
    _caught(result, "is shared by", slots=aliased)


# ---------------------------------------- the settled return ----------------------------------------


def _pair(x: float, flag: bool) -> tuple[float, bool]:
    return x * 2.0, not flag


def test_a_dropped_return_row_is_caught() -> None:
    result = _analyzed(_pair)
    plan = result.settled_return
    assert plan is not None and isinstance(plan.plan, ReturnsLeaves)
    mutated = dataclasses.replace(plan, plan=ReturnsLeaves(plan.plan.rows[:-1]))
    _caught(result, "disagree with the contract walked leaf by leaf", settled=mutated)


def test_a_return_row_carrying_the_wrong_kind_is_caught() -> None:
    # The kind on a row is what types the output PORT, so a bool leaf published as a float is an ABI divergence
    # with nothing malformed anywhere: the row count is right, the paths are right, only the type is wrong.
    result = _analyzed(_pair)
    plan = result.settled_return
    assert plan is not None and isinstance(plan.plan, ReturnsLeaves)
    rows = list(plan.plan.rows)
    assert rows[1][1] is SemType.BOOL
    rows[1] = (rows[1][0], SemType.FLOAT)
    _caught(result, "disagree with the contract", settled=dataclasses.replace(plan, plan=ReturnsLeaves(rows)))


def test_return_rows_in_the_wrong_order_are_caught() -> None:
    # Port names come from the rows' paths and port ORDER from the rows themselves, so a swapped pair renames
    # both outputs. The row set is unchanged, which is what a set-shaped comparison would have missed.
    result = _analyzed(_pair)
    plan = result.settled_return
    assert plan is not None and isinstance(plan.plan, ReturnsLeaves)
    swapped = list(reversed(plan.plan.rows))
    _caught(result, "disagree with the contract", settled=dataclasses.replace(plan, plan=ReturnsLeaves(swapped)))


def test_a_return_row_naming_the_wrong_leaf_path_is_caught() -> None:
    result = _analyzed(_pair)
    plan = result.settled_return
    assert plan is not None and isinstance(plan.plan, ReturnsLeaves)
    rows = list(plan.plan.rows)
    rows[0] = ((RecordField("bogus"),), rows[0][1])
    _caught(result, "disagree with the contract", settled=dataclasses.replace(plan, plan=ReturnsLeaves(rows)))


def test_a_return_plan_of_the_wrong_variant_is_caught() -> None:
    result = _analyzed(_pair)
    plan = result.settled_return
    assert plan is not None
    mutated = dataclasses.replace(plan, plan=ReturnsScalar(SemType.FLOAT))
    _caught(result, "settled as ReturnsScalar", settled=mutated)


def test_a_scalar_return_settled_with_the_wrong_kind_is_caught() -> None:
    def scalar(x: float) -> float:
        return x + 1.0

    result = _analyzed(scalar)
    plan = result.settled_return
    assert plan is not None and isinstance(plan.plan, ReturnsScalar)
    mutated = dataclasses.replace(plan, plan=ReturnsScalar(SemType.INT))
    _caught(result, "the return plan declares int", settled=mutated)


def test_an_array_return_row_count_is_checked_against_the_declared_shape() -> None:
    # An array contract fixes its own arity, unlike a list or a variadic tuple, so the contract-side walk can
    # disagree about the row COUNT rather than only about a row's content.
    from jaxtyping import Float64

    def rotate(x: float, y: float) -> Float64[np.ndarray, "2 2"]:
        return np.array([[x, y], [y, x]])

    result = _analyzed(rotate)
    plan = result.settled_return
    assert plan is not None and isinstance(plan.plan, ReturnsLeaves)
    assert len(plan.plan.rows) == 4
    mutated = dataclasses.replace(plan, plan=ReturnsLeaves([*plan.plan.rows, plan.plan.rows[-1]]))
    _caught(result, "disagree with the contract", settled=mutated)


def test_a_nested_return_row_of_the_wrong_depth_is_caught() -> None:
    # The recursive arm of the contract walk: a row whose path names a real position at the wrong NESTING DEPTH
    # is well-formed, keeps the row count, and still renames the port. Only walking the contract down to each
    # leaf sees it -- comparing row counts, or paths as a set of final segments, would not.
    def nested(x: float) -> tuple[float, tuple[float, float]]:
        return x, (x + 1.0, x + 2.0)

    result = _analyzed(nested)
    plan = result.settled_return
    assert plan is not None and isinstance(plan.plan, ReturnsLeaves)
    rows = list(plan.plan.rows)
    assert rows[1][0] == (TupleIndex(1), TupleIndex(0))
    rows[1] = ((TupleIndex(1),), rows[1][1])
    _caught(result, "disagree with the contract", settled=dataclasses.replace(plan, plan=ReturnsLeaves(rows)))

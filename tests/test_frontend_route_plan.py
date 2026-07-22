"""
Frontend tests: the route-plan VERIFIER, exercised by mutating plans the producer built.

Every test here injects one defect into a valid plan and requires the verifier to name it. That is the whole
claim these make -- a verifier nobody has watched reject anything is indistinguishable from `pass` -- so each
asserts on the DIAGNOSTIC FRAGMENT its own check emits rather than merely on "something raised". Without that,
one over-eager check could satisfy all six while five of them are dead.

The mutants are the six shapes the campaign named: a missing plan, a surplus plan, a zero-row plan where rows
are required, a wrong disposition, a wrong source place, and an illegal transfer. The wrong-disposition one is
the silent-absence archetype the whole restructure exists to stop -- a recorded `NoCell` where a copy belongs --
and it is the one a source-availability check was MEASURED not to catch at all.

What none of this reaches is an in-range WRONG permutation: it passes every structural check there is. The
behavioural witnesses in `test_frontend_routing.py` carry that, and this module does not replace them.
"""

from collections.abc import Callable, Iterator

import pytest

from holoso._frontend._fir._analyze import Analyzer, ResidualUnit
from holoso._frontend._fir._ir import Local, Op, executable_rpo
from holoso._frontend._fir._plan import (
    CellAction,
    CellRef,
    CellTransfer,
    ConstantCell,
    CopyCell,
    NoCell,
    PlanSite,
    RoutePlan,
    verify_route_plans,
)


def _analyzed(fn: Callable[..., object]) -> ResidualUnit:
    return Analyzer(fn).fixpoint()


def _verify(result: ResidualUnit, plans: dict[PlanSite, RoutePlan]) -> None:
    verify_route_plans(
        result.unit,
        result.executable_edges,
        {block_id: env.facts for block_id, env in result.block_in.items()},
        {block_id: env.schemas for block_id, env in result.block_in.items()},
        result.binding_facts,
        result.call_plans,
        result.construction_schemas,
        result.state_resets,
        result.runtime_state,
        plans,
    )


def _sites(result: ResidualUnit) -> Iterator[tuple[PlanSite, Op]]:
    for block_id in executable_rpo(result.unit.entry, result.executable_edges):
        for index, op in enumerate(result.unit.blocks[block_id].ops):
            yield PlanSite(block_id, index), op


def _first_copy(result: ResidualUnit) -> tuple[PlanSite, int, CopyCell]:
    """The first plan row that copies a cell, in walk order -- the anchor every source mutation perturbs."""
    for site, _ in _sites(result):
        plan = result.route_plans.get(site)
        if plan is None:
            continue
        for ordinal, action in enumerate(plan.actions):
            if isinstance(action, CopyCell):
                return site, ordinal, action
    raise AssertionError("the kernel routes no cells at all, so it cannot anchor a mutation")


def _replace(plan: RoutePlan, ordinal: int, action: CellAction) -> RoutePlan:
    rows = list(plan.actions)
    rows[ordinal] = action
    return RoutePlan(plan.target, tuple(rows))


def _pairs(x: float, y: float) -> float:
    # Two same-width aggregates so a source place can be swapped for one that is genuinely in range and
    # available at that point: a mutation the availability check cannot see, which is the point of having it.
    left = (x, y)
    right = (y, x)
    return left[0] + right[1] * 10.0


def _mutation_is_caught(result: ResidualUnit, plans: dict[PlanSite, RoutePlan], fragment: str) -> None:
    _verify(result, result.route_plans)  # the unmutated plan must pass, or the mutant proves nothing
    with pytest.raises(AssertionError) as raised:
        _verify(result, plans)
    assert fragment in str(raised.value), f"expected {fragment!r} in the diagnostic, got: {raised.value}"


def test_a_missing_plan_is_caught() -> None:
    result = _analyzed(_pairs)
    site, _, _ = _first_copy(result)
    mutated = {key: value for key, value in result.route_plans.items() if key != site}
    _mutation_is_caught(result, mutated, "routes but has no plan")


def test_a_surplus_plan_is_caught() -> None:
    result = _analyzed(_pairs)
    spare = next(site for site, op in _sites(result) if site not in result.route_plans)
    mutated = dict(result.route_plans)
    mutated[spare] = RoutePlan(Local(result.unit.params[0]), (NoCell(),))
    _mutation_is_caught(result, mutated, "a plan exists where the op routes nothing")


def test_a_zero_row_plan_where_rows_are_required_is_caught() -> None:
    result = _analyzed(_pairs)
    site, _, _ = _first_copy(result)
    mutated = dict(result.route_plans)
    mutated[site] = RoutePlan(result.route_plans[site].target, ())
    _mutation_is_caught(result, mutated, "plan has 0 row(s)")


def test_a_silently_absent_row_is_caught() -> None:
    # THE archetype: a copy quietly downgraded to "this site defines nothing here". It is well-formed, in
    # range, and consistent with every availability check -- only deriving the DISPOSITION independently from
    # the target fact's leaves catches it.
    result = _analyzed(_pairs)
    site, ordinal, _ = _first_copy(result)
    mutated = dict(result.route_plans)
    mutated[site] = _replace(result.route_plans[site], ordinal, NoCell())
    _mutation_is_caught(result, mutated, "expected COPY")


def test_a_wrong_source_place_is_caught() -> None:
    result = _analyzed(_pairs)
    site, ordinal, copy = _first_copy(result)
    other = next(
        action.source.place
        for _, plan in result.route_plans.items()
        for action in plan.actions
        if isinstance(action, CopyCell) and action.source.place != copy.source.place
    )
    mutated = dict(result.route_plans)
    mutated[site] = _replace(
        result.route_plans[site], ordinal, CopyCell(CellRef(other, copy.source.ordinal), copy.transfer)
    )
    _mutation_is_caught(result, mutated, "copies from")


def test_an_illegal_transfer_is_caught() -> None:
    # J6 in miniature: the promotion a copy applies is a recorded row, so a row claiming a promotion its source
    # and target kinds do not license must fail rather than be quietly executed.
    result = _analyzed(_pairs)
    site, ordinal, copy = _first_copy(result)
    assert copy.transfer is CellTransfer.IDENTITY
    mutated = dict(result.route_plans)
    mutated[site] = _replace(result.route_plans[site], ordinal, CopyCell(copy.source, CellTransfer.BOOL_TO_FLOAT))
    _mutation_is_caught(result, mutated, "declares BOOL_TO_FLOAT")


def test_a_wrong_constant_value_is_caught() -> None:
    # The target-side image is checked exactly, not merely for a legal kind: a constant row carrying the right
    # kind and the wrong number is the one shape a kind check alone would pass.
    def constants(x: float) -> tuple[float, float]:
        return x, 2.5

    result = _analyzed(constants)
    site, ordinal = next(
        (site, ordinal)
        for site, _ in _sites(result)
        if (plan := result.route_plans.get(site)) is not None
        for ordinal, action in enumerate(plan.actions)
        if isinstance(action, ConstantCell)
    )
    original = result.route_plans[site].actions[ordinal]
    assert isinstance(original, ConstantCell)
    from holoso._frontend._fir._value import StaticFloat

    mutated = dict(result.route_plans)
    mutated[site] = _replace(result.route_plans[site], ordinal, ConstantCell(StaticFloat(99.0), original.kind))
    _mutation_is_caught(result, mutated, "not the target-side image")


def test_an_under_promoted_state_leaf_is_caught() -> None:
    # The one mutant here that perturbs W rather than the plan, because W is the plan's other premise: a routing
    # row over a state leaf is sound only if the leaf is promoted OR its snapshot survives the canonical exit.
    # Dropping a genuine accumulator from W leaves every plan row untouched and well-formed, so no structural
    # check can see it -- yet emission would then route the reset constant where the carried value belongs and
    # ship a design that silently loses its state. Only re-deriving W's own premise from the exit facts catches
    # it. A verifier that took W on trust would report nothing at all here, which is how it read before.
    class Accumulator:
        def __init__(self) -> None:
            self.acc = 0.0

        def step(self, x: float) -> float:
            self.acc = self.acc + x
            return self.acc

    result = _analyzed(Accumulator().step)
    assert result.runtime_state, "the accumulator must promote, or the mutation below removes nothing"
    starved = {leaf for leaf in result.runtime_state if leaf.path[-1] != "acc"}
    _verify(result, result.route_plans)  # unmutated: passes
    with pytest.raises(AssertionError) as raised:
        verify_route_plans(
            result.unit,
            result.executable_edges,
            {block_id: env.facts for block_id, env in result.block_in.items()},
            {block_id: env.schemas for block_id, env in result.block_in.items()},
            result.binding_facts,
            result.call_plans,
            result.construction_schemas,
            result.state_resets,
            starved,
            result.route_plans,
        )
    assert "moves off its reset snapshot" in str(raised.value)

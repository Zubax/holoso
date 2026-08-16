"""
Refusal is SURVIVOR-BASED: an expression that names no number is refused only once it has OUTLIVED optimization.

The fold itself only signals -- the passes that fold speculatively catch the signal and leave the operation as it
stands -- because unrolling and inlining SUBSTITUTE values and so manufacture expressions the kernel never wrote
(`for w in [1.0, 0.0]: if w > 0.0: x / w` becomes `x / 0.0`). Refusing at the fold convicts the compiler of its own
transformation; refusing the survivors convicts only what is left after every deletion has run, which is the program.

The charter's license makes this sound: a refusal is a liberty the compiler takes, never a guarantee, so a MISSED
refusal is not a defect while a WRONG answer always is.

The divisions are exercised in `test_arithmetic_behavior`, which owns that operator's behaviour. What is here is
what those do not reach: a dead constant expression, a guard only HIR can prove, and the two survivor shapes that are
not divisions at all.
"""

import math
from collections.abc import Callable

import pytest

import holoso
from holoso import Options, FloatFormat
from holoso._errors import SynthesisError

from ._modelref import default_options

_FMT = FloatFormat(8, 23)


def _ops() -> Options:
    return default_options(_FMT)


def _sim(fn: Callable[..., object], name: str) -> holoso.NumericalSimulator:
    return holoso.synthesize(fn, _ops(), name=name).numerical_model.elaborate()


def _deleted_as_a_dead_constant(x: float) -> float:
    # Both operands are constants, so the fold settles it immediately -- and refusing there would convict an
    # expression before any pass could observe that nothing reads it.
    unused = 1.0 / 0.0  # noqa: F841
    return x + 1.0


def _excluded_by_a_guard_only_hir_can_prove(x: float) -> float:
    # The front end cannot see that `gate` is zero -- it is an operation over a parameter -- while HIR proves it from
    # `x*0 == 0` and resolves the guard. Which pass excludes the arm is not what decides whether the kernel builds.
    # The reciprocal has a CONSTANT numerator on purpose: a fold names a quotient only where it knows both operands, so
    # an unknown one would go unconvicted wherever it sat and the witness would say nothing about enterability.
    gate = x * 0.0
    acc = x
    if gate > 0.0:
        acc = 1.0 / gate
    return acc


def test_a_dead_constant_expression_is_not_refused() -> None:
    # 4.0 rather than the kernel's own answer: Python evaluates the statement the optimizer deletes and raises there.
    assert float(_sim(_deleted_as_a_dead_constant, "dead_constant").run(3.0)[0]) == 4.0


def test_an_arm_only_hir_proves_dead_is_not_refused() -> None:
    assert float(_sim(_excluded_by_a_guard_only_hir_can_prove, "hir_proved_guard").run(3.0)[0]) == (
        _excluded_by_a_guard_only_hir_can_prove(3.0)
    )


def _a_constant_a_further_round_deletes(c: bool, x: float) -> float:
    if c:
        flag = False
        if (x * 0.0) > 1.0:
            flag = True
    else:
        flag = c
        if (x * 0.0) > 1.0:
            flag = True
    bias = 1e-40 if flag else 1.0  # degrades in this format, and no path selects it
    return x + bias


def test_a_constant_a_further_round_deletes_is_not_judged() -> None:
    # The guards are all decidable, so `flag` is always false and the degrading constant is dead -- but only after
    # the round that threading and pruning open up. Judging a graph the optimizer has not finished with refuses a
    # kernel over a value its own next round erases.
    sim = _sim(_a_constant_a_further_round_deletes, "further_round")
    for c in (False, True):
        assert float(sim.run(c, 2.0)[0]) == 3.0


def _a_constant_only_dead_code_removal_uncovers(c: bool, x: float) -> float:
    if c:
        flag = False
        if (x * 0.0) > 1.0:
            flag = True
        dead = x + 7.0
        zero = dead * 0.0  # nothing reads `dead` once this absorbs it, and until it goes the merges cannot thread
    else:
        flag = c
        zero = 0.0
    bias = 1e-40 if flag else 1.0
    return x + bias + zero


def test_a_constant_only_dead_code_removal_uncovers_is_not_judged() -> None:
    # Dead-code removal is what opens the round that erases the degrading constant, so it cannot be the last pass.
    sim = _sim(_a_constant_only_dead_code_removal_uncovers, "dce_uncovered")
    for c in (False, True):
        assert float(sim.run(c, 2.0)[0]) == 3.0


def _surviving_domain_fault(x: float) -> float:
    return math.sqrt(-1.0) + x


def _surviving_indeterminate_form(x: float) -> float:
    return (1e300 * 1e300) * 0.0 + x  # an infinity times zero is an indeterminate form, not a zero


def _erased_indeterminate(x: float) -> float:
    huge = 1e300 * 1e300
    return x + (huge - huge) * 0.0  # inf - inf names no number, and the absorbing zero erases it


def test_the_judgement_follows_optimization_and_never_precedes_it() -> None:
    # Judging first would convict what optimization then erases. The order used to be structural -- the sweep was the
    # last statement of optimize -- and is now the boundary's to keep, so it needs a witness of its own.
    assert float(_sim(_erased_indeterminate, "erased_indeterminate").run(3.0)[0]) == 3.0


@pytest.mark.parametrize(
    "kernel", [_surviving_domain_fault, _surviving_indeterminate_form], ids=lambda fn: fn.__name__[1:]
)
def test_an_expression_that_survives_optimization_is_refused(kernel: Callable[..., float]) -> None:
    # Nothing deletes these: each feeds the returned sum, so each is the program's and is refused as such.
    with pytest.raises(SynthesisError, match="names no number"):
        _sim(kernel, kernel.__name__.lstrip("_"))

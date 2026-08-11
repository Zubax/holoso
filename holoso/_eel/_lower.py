"""The full Eel pipeline, desugar -> partial evaluation -> emission, behind one call."""

import inspect
import logging
import types
from dataclasses import dataclass

from .._errors import UnsupportedConstruct
from .._hir import Hir
from ._desugar import desugar
from ._emit import emit
from ._pe import partial_evaluate
from ._print import print_eel

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FrontendOutput:
    """
    The lowered HIR plus the canonical Eel text of every front-end pass, earliest first: the raw program the
    desugarer produced, then the residual one emission consumed. Keeping them is what makes a front-end decision
    reviewable after the fact, since neither Eel program survives into HIR.
    """

    hir: Hir
    passes: list[str]


def resolve_target(target: object) -> tuple[types.FunctionType, object | None]:
    """A bound method contributes its receiver, bound as a frozen snapshot root by the partial evaluator."""
    if inspect.ismethod(target):
        fn = target.__func__
        if not isinstance(fn, types.FunctionType):
            raise UnsupportedConstruct(f"the target {target!r} is not a plain function or a bound method")
        return fn, target.__self__
    if not isinstance(target, types.FunctionType):
        raise UnsupportedConstruct(f"the target {target!r} is not a plain function")
    return target, None


def lower(target: object, unroll_max_trips: int) -> FrontendOutput:
    fn, instance = resolve_target(target)
    eel = desugar(fn)
    desugared = print_eel(eel, locations=True)
    _logger.debug("%s: desugared Eel:\n%s", fn.__qualname__, desugared)
    residual = partial_evaluate(eel, fn, instance, unroll_max_trips)
    refined = print_eel(residual, locations=True)
    _logger.debug("%s: residual Eel:\n%s", fn.__qualname__, refined)
    hir = emit(residual)
    _logger.info(
        "%s: lowered to HIR: %d block(s), %d node(s), %d output(s)",
        fn.__qualname__,
        len(hir.blocks),
        len(hir.nodes),
        len(hir.outputs),
    )
    return FrontendOutput(hir=hir, passes=[desugared, refined])

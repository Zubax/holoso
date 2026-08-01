"""
The shadow entry point: runs the deepest landed Eel stage next to the primary frontend, with zero effect on the
primary output. Deleted at cutover. A bound method's receiver binds as a frozen snapshot root, so a stateless
method lowers fully while a state-writing one stops at its first attribute store until M8.
"""

import inspect
import logging
import types
from dataclasses import dataclass
from enum import Enum

from .._errors import SynthesisError
from ._desugar import desugar
from ._emit import emit
from ._pe import partial_evaluate
from ._print import print_eel

_logger = logging.getLogger(__name__)


class ShadowStage(Enum):
    LOWERED = "lowered"
    DESUGARED = "desugared"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ShadowReport:
    """``text`` is the canonical Eel of the deepest reached seam: residual when LOWERED, desugared otherwise."""

    stage: ShadowStage
    error: SynthesisError | None
    text: str | None


def probe(target: object) -> ShadowReport:
    fn = target.__func__ if inspect.ismethod(target) else target
    if not isinstance(fn, types.FunctionType):
        error = SynthesisError(f"the target {target!r} is not a plain function or a bound method")
        return ShadowReport(ShadowStage.REJECTED, error, None)
    try:
        eel = desugar(fn)
    except SynthesisError as error:
        return ShadowReport(ShadowStage.REJECTED, error, None)
    text = print_eel(eel, locations=True)
    _logger.debug("Eel shadow desugared text:\n%s", text)
    try:
        residual = partial_evaluate(eel, fn, target.__self__ if inspect.ismethod(target) else None)
        emit(residual)
    except SynthesisError as error:
        return ShadowReport(ShadowStage.DESUGARED, error, text)
    return ShadowReport(ShadowStage.LOWERED, None, print_eel(residual, locations=True))


def run_shadow(target: object) -> None:
    try:
        report = probe(target)
        if report.text is None:
            _logger.info("Eel shadow: %s: %s", report.stage.value, report.error)
        else:
            name = getattr(target, "__name__", target)
            suffix = "" if report.error is None else f" ({report.error})"
            _logger.info("Eel shadow: %s %r: %d lines%s", report.stage.value, name, report.text.count("\n"), suffix)
            _logger.debug("Eel shadow text:\n%s", report.text)
    except Exception:  # the single sanctioned broad except: the shadow must never affect the primary result
        _logger.exception("Eel shadow failed outside the SynthesisError contract")

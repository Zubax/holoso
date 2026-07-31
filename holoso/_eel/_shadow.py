"""
The shadow entry point: runs the deepest landed Eel stage next to the primary frontend, with zero effect on the
primary output. Deleted at cutover.
"""

import inspect
import logging
import types
from dataclasses import dataclass
from enum import Enum

from .._errors import SynthesisError
from ._desugar import desugar
from ._print import print_eel

_logger = logging.getLogger(__name__)


class ShadowStage(Enum):
    DESUGARED = "desugared"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ShadowReport:
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
    return ShadowReport(ShadowStage.DESUGARED, None, print_eel(eel, locations=True))


def run_shadow(target: object) -> None:
    try:
        report = probe(target)
        if report.error is not None or report.text is None:
            _logger.info("Eel shadow: %s: %s", report.stage.value, report.error)
        else:
            name = getattr(target, "__name__", target)
            _logger.info("Eel shadow: %s %r: %d lines", report.stage.value, name, report.text.count("\n"))
            _logger.debug("Eel shadow text:\n%s", report.text)
    except Exception:  # the single sanctioned broad except: the shadow must never affect the primary result
        _logger.exception("Eel shadow failed outside the SynthesisError contract")

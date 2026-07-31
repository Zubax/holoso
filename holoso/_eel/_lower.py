"""The full Eel pipeline, desugar -> partial evaluation -> emission, behind one call."""

import inspect
import logging
import types

from .._errors import SynthesisError, UnsupportedConstruct
from .._hir import Hir
from ._desugar import desugar
from ._emit import emit
from ._pe import partial_evaluate
from ._print import print_eel

_logger = logging.getLogger(__name__)


def resolve_target(target: object) -> types.FunctionType:
    if inspect.ismethod(target):
        raise UnsupportedConstruct("bound methods (persistent state) are not supported yet")
    if not isinstance(target, types.FunctionType):
        raise SynthesisError(f"the target {target!r} is not a plain function")
    return target


def lower(target: object) -> Hir:
    fn = resolve_target(target)
    eel = desugar(fn)
    _logger.debug("%s: desugared Eel:\n%s", fn.__qualname__, print_eel(eel, locations=True))
    residual = partial_evaluate(eel, fn)
    _logger.debug("%s: residual Eel:\n%s", fn.__qualname__, print_eel(residual, locations=True))
    hir = emit(residual)
    _logger.info(
        "%s: lowered to HIR: %d block(s), %d node(s), %d output(s)",
        fn.__qualname__,
        len(hir.blocks),
        len(hir.nodes),
        len(hir.outputs),
    )
    return hir

"""
The fold-rule registry: the ONE door through which the analyzer evaluates anything concretely.

Thirteen review rounds established that every concrete-evaluation hazard family (live-object reads outside the
state machinery, folds on type-unfaithful reconstructions, snapshot layout observation, compile-time side effects
and unbounded cost) enters through the same mechanism -- executing host Python over carriers the domain cannot
vouch for -- under an open-ended variety of spellings. The registry closes the mechanism: a callable evaluates
concretely only through a FoldRule row that declares WHAT may cross (per-position admission classes) and HOW MUCH
it may cost (a bound checked before evaluation), with a located rejection as the default for everything else.
Rule implementations may delegate to the host exactly where the carriers round-trip bijectively (the scalar
precedent of ``_opsem``); structural transfers live in the analyzer and never come through here.

Admission classes, applied per argument position before any evaluation:

- SCALAR: a Known bool/int/float/str of either provenance, bounded in magnitude for integers (a value method's
  padding width sizes its allocation).
- DATA: any Known or all-Known aggregate whose materialization contains no object reference and no record (a
  record's reconstruction is not type-faithful; a reference is live state), with oversized static ranges refused
  anywhere in the tree.
- CLASSINFO: the isinstance classinfo position -- object references to plain types admitted, resolved member by
  member by the analyzer before the rule runs.
- INERT_TYPE: a reference to one of the dtype-ish builtin types (float/int/bool and the numpy scalar types),
  which carry no live state.

The registry is data, not prose: reviews of new admissions happen row by row (the round-12 pattern), and a
single greppable invariant -- the harness is the only caller of host evaluation -- replaces spelling-by-spelling
hunts.
"""

import enum
from collections.abc import Callable
from dataclasses import dataclass


class Admission(enum.Enum):
    SCALAR = enum.auto()
    DATA = enum.auto()
    CLASSINFO = enum.auto()
    INERT_TYPE = enum.auto()


@dataclass(frozen=True, slots=True)
class FoldRule:
    """
    One admitted concrete-evaluation form. ``targets`` resolve by identity (an unhashable shadow of a builtin
    name must miss cleanly, so containment is scanned, never hashed). ``positions`` is the admission class per
    positional argument; ``rest`` admits any further positionals and every keyword value. ``bound`` is the
    integer-magnitude budget applied to SCALAR int arguments (a rule whose work or result scales with an
    argument keeps the shared 2**20 default; casts that merely read the value may lift it).
    """

    name: str
    targets: tuple[object, ...]
    positions: tuple[Admission, ...]
    rest: Admission | None
    bound: int | None = 1 << 20

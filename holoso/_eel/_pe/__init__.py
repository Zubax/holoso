"""
Sublayer 2: Eel -> residual Eel. The partial evaluator (PE) owns binding time, scalar types, static control, and the
module-boundary typing; see the driver in `_interpret`, the value domain and expansion budget in `_values`,
and the sole HIR-facing operator tables in `_ops`.
"""

from ._interpret import partial_evaluate as partial_evaluate

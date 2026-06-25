"""
Shared auxiliary entities used across the IR layers.
"""

import enum

type ValueId = int
"""An SSA value identifier, unique within one IR graph."""

type BlockId = int
"""A basic-block identifier, unique within one control-flow graph."""


class RelationalOp(enum.Enum):
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    EQ = "eq"
    NE = "ne"

    def apply(self, left: float, right: float) -> bool:
        """The single definition of each relation's truth function; every other comparison path routes through here."""
        match self:
            case RelationalOp.LT:
                return left < right
            case RelationalOp.LE:
                return left <= right
            case RelationalOp.GT:
                return left > right
            case RelationalOp.GE:
                return left >= right
            case RelationalOp.EQ:
                return left == right
            case RelationalOp.NE:
                return left != right

    def holds(self, ordering: int) -> bool:
        """Apply the relation to a three-way comparison result (-1/0/+1) -- the bit-exact model's comparison path."""
        return self.apply(ordering, 0)

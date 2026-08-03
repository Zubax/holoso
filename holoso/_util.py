"""
Shared auxiliary entities used across the IR layers.
"""

type ValueId = int
"""An SSA value identifier, unique within one IR graph."""

type BlockId = int
"""A basic-block identifier, unique within one control-flow graph."""

"""The single scalar floating-point format used throughout a synthesized module."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FloatFormat:
    """
    A Zubax Kulibin float (ZKF) format: ``wexp`` exponent bits and ``wman`` significand bits.

    ``wman`` counts the significand *including* the hidden leading bit, matching the ``WMAN`` convention of
    ``holoso_support.v``. The total port width is ``wexp + wman`` (a sign bit, ``wexp`` exponent bits, and
    ``wman - 1`` stored significand bits).
    """

    wexp: int
    wman: int

    def __post_init__(self) -> None:
        if self.wexp < 2:
            raise ValueError(f"wexp must be >= 2, got {self.wexp}")
        if self.wman < 4:
            raise ValueError(f"wman must be >= 4, got {self.wman}")

    @property
    def width(self) -> int:
        """Total bit width of one scalar (``WFULL = wexp + wman``)."""
        return self.wexp + self.wman

    def __str__(self) -> str:
        return f"ZKF(wexp={self.wexp}, wman={self.wman}, width={self.width})"

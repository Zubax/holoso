"""Exception hierarchy for Holoso synthesis, with optional source locations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """A location in the user's Python source, used to make synthesis errors actionable."""

    filename: str
    lineno: int
    col: int
    line: str | None = None

    def __str__(self) -> str:
        where = f"{self.filename}:{self.lineno}:{self.col + 1}"
        if self.line is None:
            return where
        return f"{where}\n    {self.line.strip()}"


class HolosoError(Exception): ...


class SynthesisError(HolosoError):
    def __init__(self, message: str, location: SourceLocation | None = None) -> None:
        self.message = message
        self.location = location
        super().__init__(message if location is None else f"{message}\n  at {location}")


class UnsupportedConstruct(SynthesisError):
    """The input uses a Python construct that Holoso cannot (yet) synthesize."""


class MissingIntrinsic(SynthesisError):
    """The input calls a numeric operator that has no Holoso implementation yet."""


class SourceUnavailable(SynthesisError):
    """The target's source code could not be retrieved for analysis."""

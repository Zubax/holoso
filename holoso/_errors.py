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
        where = f"{self.filename}:{self.lineno}:{self.col}"  # 0-based column, matching the rendered rejection prefix
        if self.line is None:
            return where
        return f"{where}\n    {self.line.strip()}"


class HolosoError(Exception): ...


class SynthesisError(HolosoError):
    def __init__(self, message: str) -> None:
        self.message = message
        self.location: SourceLocation | None = None
        super().__init__(message)


class UnsupportedConstruct(SynthesisError):
    """The input uses a Python construct that Holoso cannot (yet) synthesize."""


class UnsupportedLibraryFunction(SynthesisError):
    """The input calls a recognized math/numpy library function that Holoso does not implement yet."""


class SourceUnavailable(SynthesisError):
    """The target's source code could not be retrieved for analysis."""

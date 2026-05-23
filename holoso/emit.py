"""A tiny indentation-aware text builder for emitting Verilog."""

from __future__ import annotations


class VerilogWriter:
    """Accumulates 4-space-indented lines. Use :meth:`line` for content and :meth:`push`/:meth:`pop` for nesting."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._depth = 0

    def line(self, text: str = "") -> None:
        self._lines.append(("    " * self._depth + text) if text else "")

    def lines(self, *texts: str) -> None:
        for text in texts:
            self.line(text)

    def push(self) -> None:
        self._depth += 1

    def pop(self) -> None:
        assert self._depth > 0
        self._depth -= 1

    def render(self) -> str:
        return "\n".join(self._lines) + "\n"

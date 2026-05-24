"""Access to the shared ``holoso_support`` HDL that generated modules instantiate."""

from __future__ import annotations

from functools import cache
from importlib import resources

# Packaged as ``holoso/hdl/`` data (see pyproject ``package-data``) and read via importlib.resources, so it resolves
# from an installed wheel -- not just an editable checkout. Same idiom as the report's html.css/html.js.
_HDL = resources.files(__package__).joinpath("hdl")


@cache
def support_verilog() -> str:
    """Return the contents of ``holoso_support.v`` (the operator wrappers and register file)."""
    return _HDL.joinpath("holoso_support.v").read_text(encoding="utf-8")


@cache
def support_header() -> str:
    """Return the contents of ``holoso_support.vh`` (lane-select and sign-op macros)."""
    return _HDL.joinpath("holoso_support.vh").read_text(encoding="utf-8")

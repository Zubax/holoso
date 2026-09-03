"""Port naming shared by HIR emission and the test harnesses."""

import itertools
import re
import unicodedata

from .._errors import UnsupportedConstruct

_GREEK = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "ς": "sigma",
}
_TRANSLITERATION = {ord(letter): word for letter, word in _GREEK.items()}
_TRANSLITERATION |= {ord(letter.upper()): word.capitalize() for letter, word in _GREEK.items()}

_VERILOG_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def spelled(name: str) -> str:
    """
    A Python identifier in Verilog's alphabet as far as it goes: `σ_i` -> `sigma_i`, `Δt` -> `Deltat`, `Σ` ->
    `Sigma`. An unaccented Greek letter becomes its English word with no separator inserted, and both sigmas give
    `sigma`, so two names may land on one; the duplicate checks over ports and state slots catch that either way.
    The normalization is CPython's own, so a name spelled in an explicit `name=` reads as the parser would read it.
    """
    return unicodedata.normalize("NFKC", name).translate(_TRANSLITERATION)


def hardware_name(name: str) -> str:
    """`spelled`, demanding the result be a name Verilog can carry -- what a synthesized identifier must be."""
    verilog = spelled(name)
    if not _VERILOG_IDENTIFIER.match(verilog):
        stray = "".join(dict.fromkeys(re.findall(r"[^A-Za-z0-9_]", verilog)))
        raise UnsupportedConstruct(
            f"the name {name!r} becomes a synthesized identifier, and Verilog spells {stray!r} no way at all; "
            "rename it in ASCII (unaccented Greek letters are spelled out for you)"
        )
    return verilog


def port_name(path: tuple[int | str, ...]) -> str:
    """Map a returned leaf path to its output-port name, e.g. `(0, "x")` -> `out_0_x`."""
    return "out" + "".join(f"_{_component(key)}" for key in path)


def slot_name(path: tuple[str | int, ...]) -> str:
    """The flattened slot-register name of a state leaf path, e.g. `("x", 0, 1)` -> `x_0_1`."""
    assert path and isinstance(path[0], str)
    return "_".join(_component(key) for key in path)


def state_port_name(path: tuple[str | int, ...]) -> str:
    """The observability port of a public state leaf, e.g. `("x", 0)` -> `state_x_0`."""
    return "state_" + slot_name(path)


def public_slot(path: tuple[str | int, ...]) -> bool:
    """A public path gets a `state_...` port; one with any underscore-prefixed component stays register-only."""
    assert path and isinstance(path[0], str)
    return not any(isinstance(key, str) and key.startswith("_") for key in path)


def indexed_names(base: str, shape: tuple[int, ...]) -> list[str]:
    """
    The row-major per-element names of a shaped base name: `()` -> `[base]`, `(2,)` -> `[base_0, base_1]`,
    `(2, 2)` -> `[base_0_0, base_0_1, base_1_0, base_1_1]`. The one naming convention shared by decomposed
    state slots and decomposed array-parameter input ports.
    """
    return [base + "".join(f"_{i}" for i in index) for index in itertools.product(*(range(dim) for dim in shape))]


def _component(key: str | int) -> str:
    return hardware_name(key) if isinstance(key, str) else str(key)

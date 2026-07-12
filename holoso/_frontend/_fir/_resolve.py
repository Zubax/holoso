"""
Name resolution from the live function object. The code object is authoritative: CPython's own compiler already
classified every name (``co_varnames``/``co_cellvars`` are locals, ``co_freevars`` are closure captures), so lexical
classification cannot drift from Python's -- including ``del NAME`` making a name local throughout, and PEP 709
comprehension targets appearing as plain function locals (their isolation is the builder's AST-level responsibility).
Non-local values follow Python's lookup order: closure cell, then module global, then builtin; the values are read at
resolve time, which is compile time -- the snapshot doctrine that already governs globals and the instance state.
"""

import ast
import builtins as _builtins_module
import types
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Local:
    """A function-scope local slot: a parameter, an assigned or deleted name, or a PEP 709 comprehension target."""

    name: str


class _ValueCarrier:
    """
    Equality/hash for the value-carrying resolutions, keyed on the name and the VALUE'S IDENTITY: the generated
    value-based forms would call the payload's own ``==`` (an ndarray coefficient table poisons enclosing
    comparisons) and are partial under hashing. Same doctrine as ObjectRef in the value domain.
    """

    __slots__ = ()

    name: str
    value: object

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self.name == other.name and self.value is other.value

    def __hash__(self) -> int:
        return hash((type(self), self.name, id(self.value)))


@dataclass(frozen=True, slots=True, eq=False)
class Free(_ValueCarrier):
    """A closure capture; ``value`` is the cell's content at resolve time."""

    name: str
    value: object


@dataclass(frozen=True, slots=True, eq=False)
class Global(_ValueCarrier):
    """A module-global binding; ``value`` is the module's binding at resolve time."""

    name: str
    value: object


@dataclass(frozen=True, slots=True, eq=False)
class Builtin(_ValueCarrier):
    name: str
    value: object


@dataclass(frozen=True, slots=True)
class Missing:
    """No binding anywhere Python would look: reading this name raises NameError in Python."""

    name: str


type Resolution = Local | Free | Global | Builtin | Missing


class UnboundCell(Exception):
    """A closure cell exists but holds no value yet (Python raises NameError on the read)."""


def _mangling_class(qualname: str) -> str | None:
    # The nearest lexically enclosing class, recovered from the qualname: every function nesting level appends
    # "name.<locals>", so walking right to left, a part not directly followed by "<locals>" is a class. Mangling
    # applies transitively (a function nested inside a method still mangles), hence nearest-class, not parent-only.
    parts = qualname.split(".")[:-1]
    index = len(parts) - 1
    while index >= 0:
        if parts[index] == "<locals>":
            index -= 2
        else:
            return parts[index]
    return None


def _mangle(name: str, klass: str | None) -> str:
    """CPython private-name mangling: the code object spells ``__NAME`` as ``_Klass__NAME`` inside a class."""
    if klass is None or not name.startswith("__") or name.endswith("__"):
        return name
    stripped = klass.lstrip("_")
    return f"_{stripped}{name}" if stripped else name


def comprehension_only_targets(fndef: ast.FunctionDef) -> frozenset[str]:
    """
    Names that are comprehension targets and bound NOWHERE else in the function. Under PEP 709 the code object lists
    such a target among ``co_varnames`` although outside the comprehension the name still resolves per the enclosing
    scopes (``[x for BOUND in range(BOUND)]`` reads a module ``BOUND`` for the outermost iterable) -- so these names
    are subtracted from the function-local set, and their in-comprehension binding is the builder's own scoped frame.
    A name that is ALSO bound elsewhere stays a plain local; the builder's frame overlay handles its shadowing.
    """
    targets: set[str] = set()
    others: set[str] = set()
    declared: set[str] = set()

    def stored_names(node: ast.AST) -> list[str]:
        # Only Store/Del-context Names bind: in ``a[i] = x`` the target Subscript READS both a and i.
        return [n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))]

    def scan(node: ast.AST) -> None:
        match node:
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(name=name):
                # A nested scope's internal bindings are not bindings of this function; only its name is.
                # (Kernels reject nested def/class outright, so enclosing-scope subtleties of their decorators,
                # defaults, and bases need no modeling here; lambdas are expressions, so their defaults are.)
                others.add(name)
                return
            case ast.Lambda(args=lambda_args):
                for default in [*lambda_args.defaults, *lambda_args.kw_defaults]:
                    if default is not None:
                        scan(default)
                return
            case ast.comprehension(target=target):
                targets.update(stored_names(target))
                for child in ast.iter_child_nodes(node):
                    if child is not target:
                        scan(child)
                return
            case ast.Global(names=names) | ast.Nonlocal(names=names):
                declared.update(names)
            case ast.Name(id=name, ctx=ast.Store() | ast.Del()):
                others.add(name)
            case ast.ExceptHandler(name=str(name)) | ast.MatchAs(name=str(name)) | ast.MatchStar(name=str(name)):
                others.add(name)
            case ast.MatchMapping(rest=str(name)):
                others.add(name)
            case ast.Import(names=aliases) | ast.ImportFrom(names=aliases):
                others.update(alias.asname or alias.name.partition(".")[0] for alias in aliases)
        for child in ast.iter_child_nodes(node):
            scan(child)

    # Only the body binds function-locally: the decorators, defaults, and annotations of fndef itself evaluate in
    # the ENCLOSING scope, so a walrus there must not count. A store declared global/nonlocal is not a local binding
    # either, yet its comprehension-target namesake still occupies a co_varnames slot -- hence the subtraction.
    for statement in fndef.body:
        scan(statement)
    args = fndef.args
    for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
        if arg is not None:
            others.add(arg.arg)
    return frozenset(targets - (others - declared))


class NameResolver:
    """
    Per-function name classification and non-local value lookup. Locals are fixed by the code object minus the
    PEP 709 comprehension-only targets (see :func:`comprehension_only_targets`); non-local values are read live at
    each :meth:`resolve` call so a lookup made while lowering sees exactly what Python would see at that moment of
    the compile.
    """

    def __init__(self, fn: object, comprehension_only: frozenset[str] = frozenset()) -> None:
        assert isinstance(fn, types.FunctionType), fn
        code = fn.__code__
        self._klass = _mangling_class(code.co_qualname)  # co_qualname is lexical truth; fn.__qualname__ is mutable
        # metadata that decorators (functools.wraps) routinely overwrite with the wrapped function's spelling
        carved = frozenset(_mangle(name, self._klass) for name in comprehension_only)
        self._locals = (frozenset(code.co_varnames) | frozenset(code.co_cellvars)) - carved
        self._cells = dict(zip(code.co_freevars, fn.__closure__ or ()))
        assert len(self._cells) == len(code.co_freevars), "closure cell count must match co_freevars"
        self._globals = fn.__globals__
        raw_builtins = getattr(fn, "__builtins__", None)
        if raw_builtins is None:
            raw_builtins = vars(_builtins_module)
        elif isinstance(raw_builtins, types.ModuleType):
            raw_builtins = vars(raw_builtins)
        assert isinstance(raw_builtins, Mapping)
        self._builtins: Mapping[str, object] = raw_builtins  # a live reference, not a copy: lookups read like Python

    @property
    def local_names(self) -> frozenset[str]:
        return self._locals

    def runtime_spelling(self, name: str) -> str:
        """
        The code-object spelling of an AST identifier: private names mangle inside a class, and that applies to
        ATTRIBUTES as well as locals (``self.__state`` stores under ``_Klass__state``). Idempotent on names that
        are already mangled.
        """
        return _mangle(name, self._klass)

    def resolve(self, name: str) -> Resolution:
        """
        The queried name is the AST spelling; the returned Resolution carries the runtime spelling (private names
        mangle inside a class, and the code object plus every runtime lookup use the mangled form).
        """
        name = _mangle(name, self._klass)
        if name in self._locals:
            return Local(name)
        cell = self._cells.get(name)
        if cell is not None:
            try:
                return Free(name, cell.cell_contents)
            except ValueError as exc:
                raise UnboundCell(name) from exc
        if name in self._globals:
            return Global(name, self._globals[name])
        if name in self._builtins:
            return Builtin(name, self._builtins[name])
        return Missing(name)

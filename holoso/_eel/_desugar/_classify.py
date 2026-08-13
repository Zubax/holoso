"""
Name classification: CPython's own, taken from the function's code object and corrected for the one distortion
PEP 709 introduces — comprehension targets land in `co_varnames` without creating a function-level binding.

A name occurrence not bound by an active comprehension rename is local iff it is in
`co_varnames ∪ co_cellvars` minus the comp-target-only names: names whose every binding site anywhere in the
function is a comprehension target. The binding-site scan is the ONE analysis that descends into asserts
(their contents are whitelist-exempt but still shape the code object: an assert-resident walrus is a real
binding, and an assert-resident capturer moves a local into `co_cellvars`). The scan is rooted at the body —
the root function's default, decorator, and annotation expressions execute in the enclosing scope — and at a
nested function/lambda/class boundary it scans only the enclosing-scope expressions (defaults, decorators,
bases), reapplying the boundary recursively. Generator-expression targets are their own scope and record
nothing, while a walrus inside one still binds this scope. Annotations are never scanned: CPython does not
evaluate them in function bodies, and a walrus inside one is a syntax error.

A name bound only inside an assert classifies local, so a later read is a definite-assignment rejection at the
partial evaluator — matching `-O` CPython and never silently folding a same-named module global.
"""

import ast
import types
from dataclasses import dataclass
from enum import Enum


class NameKind(Enum):
    LOCAL = "local"
    FREE = "free"
    GLOBAL = "global"


def mangles(name: str) -> bool:
    """CPython's private-name rule: two or more leading underscores and not two or more trailing."""
    return name.startswith("__") and not name.endswith("__")


@dataclass(frozen=True, slots=True)
class Classifier:
    local_names: frozenset[str]
    free_names: frozenset[str]

    def classify(self, name: str) -> NameKind:
        if name in self.local_names:
            return NameKind.LOCAL
        if name in self.free_names:
            return NameKind.FREE
        return NameKind.GLOBAL


def build_classifier(fn: types.FunctionType, fndef: ast.FunctionDef) -> Classifier:
    scan = _BindingScan()
    for stmt in fndef.body:
        scan.visit(stmt)
    args = fndef.args
    params = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    if args.vararg is not None:
        params.add(args.vararg.arg)
    if args.kwarg is not None:
        params.add(args.kwarg.arg)
    code = fn.__code__
    code_names = set(code.co_varnames) | set(code.co_cellvars)
    comp_target_only = {
        _code_name(name, fn.__qualname__, code_names) for name in scan.comp_targets - scan.bindings - params
    }
    local_names = code_names - comp_target_only
    # Mangling-eligible params appear mangled in the code object; the transformer rejects them with a located
    # diagnostic, which this consistency assert must not preempt.
    assert {p for p in params if not mangles(p)} <= local_names
    return Classifier(frozenset(local_names), frozenset(code.co_freevars))


def _code_name(name: str, qualname: str, code_names: set[str]) -> str:
    """
    The code object's rendition of a scanned source name: mangling-eligible names inside a class body are
    stored mangled (an assert-resident comprehension target `__x` in class C appears as `_C__x`), and the
    subtraction must remove the stored form. The mangled candidate is used only when the plain form is absent
    and the candidate is present, so a plain function's coincidental `_g__x` local can never be captured.
    """
    if name in code_names or not mangles(name):
        return name
    parts = [part for part in qualname.split(".") if part != "<locals>"]
    if len(parts) >= 2:
        mangled = f"_{parts[-2].lstrip('_')}{name}"
        if parts[-2].lstrip("_") and mangled in code_names:
            return mangled
    return name


class _BindingScan:
    def __init__(self) -> None:
        self.bindings: set[str] = set()
        self.comp_targets: set[str] = set()

    def visit(self, node: ast.AST) -> None:
        match node:
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                self.bindings.add(node.name)
                for expr in node.decorator_list:
                    self.visit(expr)
                self._visit_defaults(node.args)
            case ast.Lambda():
                self._visit_defaults(node.args)
            case ast.ClassDef():
                self.bindings.add(node.name)
                for expr in (*node.decorator_list, *node.bases):
                    self.visit(expr)
                for keyword in node.keywords:
                    self.visit(keyword.value)
            case ast.ListComp() | ast.SetComp() | ast.DictComp() | ast.GeneratorExp():
                own_scope = isinstance(node, ast.GeneratorExp)
                for gen in node.generators:
                    if not own_scope:
                        self.comp_targets |= _leaf_names(gen.target)
                    self.visit(gen.iter)
                    for cond in gen.ifs:
                        self.visit(cond)
                for child in ast.iter_child_nodes(node):
                    if not isinstance(child, ast.comprehension):
                        self.visit(child)
            case ast.NamedExpr():
                if isinstance(node.target, ast.Name):
                    self.bindings.add(node.target.id)
                self.visit(node.value)
            case ast.Assign():
                for target in node.targets:
                    self.bindings |= _leaf_names(target)
                    self.visit(target)
                self.visit(node.value)
            case ast.AnnAssign():  # the annotation is deliberately not visited
                self.bindings |= _leaf_names(node.target)
                self.visit(node.target)
                if node.value is not None:
                    self.visit(node.value)
            case ast.AugAssign():
                self.bindings |= _leaf_names(node.target)
                self.visit(node.target)
                self.visit(node.value)
            case ast.For() | ast.AsyncFor():
                self.bindings |= _leaf_names(node.target)
                self.visit(node.target)
                self.visit(node.iter)
                for stmt in (*node.body, *node.orelse):
                    self.visit(stmt)
            case _:
                for child in ast.iter_child_nodes(node):
                    self.visit(child)

    def _visit_defaults(self, args: ast.arguments) -> None:
        for default in (*args.defaults, *args.kw_defaults):
            if default is not None:
                self.visit(default)


def _leaf_names(target: ast.expr) -> set[str]:
    match target:
        case ast.Name(id=name):
            return {name}
        case ast.Starred(value=value):
            return _leaf_names(value)
        case ast.Tuple(elts=elts) | ast.List(elts=elts):
            names: set[str] = set()
            for elt in elts:
                names |= _leaf_names(elt)
            return names
        case _:
            return set()

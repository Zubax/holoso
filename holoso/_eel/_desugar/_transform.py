"""
The ANF transformer: whitelisted CPython AST -> Eel, materializing evaluation order.

Every non-atomic subexpression is hoisted to a fresh temp in CPython evaluation order; operands are atoms.
Conditionally-evaluated positions (conditional-expression arms, comparison-chain suffixes) become real guarded
branches assigning a result temp — never eager hoists, so a loop or non-speculatable operation in an untaken
arm cannot execute. ``and``/``or`` stay eager binary gates per the ratified data model.

Atom embedding is sound because constants are immutable, aggregate contents are only read through hoisted
non-atomic expressions and written by store statements, and the one intra-statement rebinder — the walrus — is
rejected whenever the same expression region also reads its target name (the augmented-assignment target
counts as a read: ``x += (x := 2)`` reads the old x in CPython).
"""

import ast
import inspect
import types
from typing import NoReturn

from ..._errors import UnsupportedConstruct
from .._ir import *
from ._classify import Classifier, NameKind, build_classifier, mangles
from ._source import SourceUnit, load

_BIN_OPS: dict[type[ast.operator], BinaryOp] = {
    ast.Add: BinaryOp.ADD,
    ast.Sub: BinaryOp.SUB,
    ast.Mult: BinaryOp.MUL,
    ast.Div: BinaryOp.DIV,
    ast.FloorDiv: BinaryOp.FLOORDIV,
    ast.Mod: BinaryOp.MOD,
    ast.Pow: BinaryOp.POW,
    ast.MatMult: BinaryOp.MATMUL,
    ast.LShift: BinaryOp.LSHIFT,
    ast.RShift: BinaryOp.RSHIFT,
    ast.BitAnd: BinaryOp.BITAND,
    ast.BitOr: BinaryOp.BITOR,
    ast.BitXor: BinaryOp.BITXOR,
}

_CMP_OPS: dict[type[ast.cmpop], CompareOp] = {
    ast.Lt: CompareOp.LT,
    ast.LtE: CompareOp.LE,
    ast.Gt: CompareOp.GT,
    ast.GtE: CompareOp.GE,
    ast.Eq: CompareOp.EQ,
    ast.NotEq: CompareOp.NE,
}

_UNARY_OPS: dict[type[ast.unaryop], UnaryOp] = {
    ast.USub: UnaryOp.NEG,
    ast.UAdd: UnaryOp.POS,
    ast.Not: UnaryOp.NOT,
    ast.Invert: UnaryOp.INVERT,
}


def desugar(fn: types.FunctionType) -> EelFunction:
    unit = load(fn)
    if getattr(fn, "__wrapped__", None) is not None:
        # inspect resolves __wrapped__ chains, so the retrieved source is the INNER function's while the live
        # callable is the wrapper: compiling it would silently drop the wrapper's behavior.
        raise UnsupportedConstruct(
            "the target is wrapped by a decorator; pass the undecorated function", unit.location(unit.fndef)
        )
    if fn.__code__.co_flags & (inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR):
        # A yield can hide inside an ignored assert; the code object is the truth about the function's kind.
        raise UnsupportedConstruct("generators are not supported", unit.location(unit.fndef))
    return _Transformer(unit, build_classifier(fn, unit.fndef)).function()


class _Transformer:
    def __init__(self, unit: SourceUnit, classifier: Classifier) -> None:
        self._unit = unit
        self._classifier = classifier
        self._temp_count = 0
        self._comp_count = 0
        self._comp_renames: list[tuple[str, str]] = []

    def function(self) -> EelFunction:
        fndef = self._unit.fndef
        if fndef.type_params:
            self._reject(fndef.type_params[0], "generic type parameters are not supported")
        return EelFunction(
            origin=self._origin(fndef),
            name=fndef.name,
            params=self._params(fndef.args),
            body=self._suite(fndef.body),
        )

    def _origin(self, node: ast.AST) -> Origin:
        return Origin(self._unit.location(node))

    def _reject(self, node: ast.AST, message: str) -> NoReturn:
        location = self._unit.location(node)
        assert location is not None
        raise UnsupportedConstruct(message, location)

    def _params(self, args: ast.arguments) -> tuple[Param, ...]:
        if args.vararg is not None:
            self._reject(args.vararg, "*args parameters are not supported")
        if args.kwarg is not None:
            self._reject(args.kwarg, "**kwargs parameters are not supported")
        params: list[Param] = []
        for arg, kind in (
            *((a, ParamKind.POSITIONAL_ONLY) for a in args.posonlyargs),
            *((a, ParamKind.POSITIONAL_OR_KEYWORD) for a in args.args),
            *((a, ParamKind.KEYWORD_ONLY) for a in args.kwonlyargs),
        ):
            if mangles(arg.arg):
                self._reject(arg, f"the private name {arg.arg!r} is subject to name mangling; rename it")
            params.append(Param(self._origin(arg), arg.arg, kind))
        return tuple(params)

    def _suite(self, stmts: list[ast.stmt]) -> Block:
        sink: list[Stmt] = []
        for stmt in stmts:
            self._statement(stmt, sink)
        return tuple(sink)

    def _statement(self, node: ast.stmt, sink: list[Stmt]) -> None:
        match node:
            case ast.Expr(value=ast.Constant(value=str())):
                pass  # a docstring or stray string constant: a no-op in CPython
            case ast.Expr(value=ast.Constant(value=v)) if v is Ellipsis:
                self._reject(node, "an `...` statement has no effect (a stub body cannot be synthesized)")
            case ast.Expr(value=ast.Yield() | ast.YieldFrom()):
                self._reject(node, "generators are not supported")
            case ast.Expr():
                self._reject(node, "expression statement result is unused; bind it to a name or remove it")
            case ast.Assert() | ast.Pass():
                pass
            case ast.Assign():
                self._assign(node, sink)
            case ast.AnnAssign():
                self._ann_assign(node, sink)
            case ast.AugAssign():
                self._aug_assign(node, sink)
            case ast.If():
                cond = self._atom(node.test, sink)
                self._check_region([node.test])
                sink.append(If(self._origin(node), cond, self._suite(node.body), self._suite(node.orelse)))
            case ast.While():
                self._while(node, sink)
            case ast.For():
                self._for(node, sink)
            case ast.Return():
                self._return(node, sink)
            case ast.Raise():
                self._raise(node, sink)
            case ast.Break():
                sink.append(Break(self._origin(node)))
            case ast.Continue():
                sink.append(Continue(self._origin(node)))
            case _:
                self._reject(node, f"unsupported statement: {type(node).__name__}")

    def _assign(self, node: ast.Assign, sink: list[Stmt]) -> None:
        # Region checks run AFTER the structural transform throughout, so a banned construct (the outer shape)
        # rejects before a walrus collision inside it; a raised rejection discards the whole desugar output.
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = self._local_bind(node.targets[0])
            sink.append(Assign(self._origin(node), target, self._expr(node.value, sink)))
            self._check_region([node.value, *node.targets])
            return
        value = self._atom(node.value, sink)
        if len(node.targets) > 1 and isinstance(value, LocalRef):
            # An earlier target may rebind the very name the RHS atom reads (`t, second = out = t`); the temp
            # pins the single evaluation CPython guarantees. A single target cannot race its own value.
            value = self._to_temp(node.value, value, sink)
        for tgt in node.targets:  # bound left to right, to the same value, as in CPython
            self._bind_target(tgt, value, sink)
        self._check_region([node.value, *node.targets])

    def _ann_assign(self, node: ast.AnnAssign, sink: list[Stmt]) -> None:
        if node.value is None:
            if not isinstance(node.target, ast.Name):
                # CPython evaluates a non-name annotated target expression even without a value.
                self._reject(node, "a bare annotation on a non-name target is not supported")
            return  # a bare name annotation has no runtime effect in a function body
        if isinstance(node.target, ast.Name):
            target = self._local_bind(node.target)
            sink.append(Assign(self._origin(node), target, self._expr(node.value, sink)))
        else:
            self._bind_target(node.target, self._atom(node.value, sink), sink)
        self._check_region([node.value, node.target])

    def _aug_assign(self, node: ast.AugAssign, sink: list[Stmt]) -> None:
        op = _BIN_OPS.get(type(node.op))
        assert op is not None, "every augmentable ast operator is mapped"
        aug_target = node.target.id if isinstance(node.target, ast.Name) else None
        match node.target:
            case ast.Name():
                target = self._local_bind(node.target)
                sink.append(AugAssign(self._origin(node), target=target, op=op, value=self._atom(node.value, sink)))
            case ast.Attribute() | ast.Subscript():
                root, path = self._store_path(node.target, sink)
                value = self._atom(node.value, sink)
                sink.append(AugStore(self._origin(node), root=root, path=path, op=op, value=value))
            case _:
                self._reject(node.target, "unsupported augmented-assignment target")
        self._check_region([node.target, node.value], aug_target=aug_target)

    def _while(self, node: ast.While, sink: list[Stmt]) -> None:
        if node.orelse:
            self._reject(node, "`while ... else` is not supported")
        header: list[Stmt] = []
        cond = self._atom(node.test, header)
        self._check_region([node.test], in_while_test=True)
        sink.append(While(self._origin(node), tuple(header), cond, self._suite(node.body)))

    def _for(self, node: ast.For, sink: list[Stmt]) -> None:
        if node.orelse:
            self._reject(node, "`for ... else` is not supported")
        if not isinstance(node.target, ast.Name):
            self._reject(node.target, "for-loop target unpacking is not supported; unpack inside the body")
        target = self._local_bind(node.target)
        iterable = self._atom(node.iter, sink)  # evaluated once, before the loop, as in CPython
        self._check_region([node.iter])
        sink.append(For(self._origin(node), target, iterable, self._suite(node.body)))

    def _return(self, node: ast.Return, sink: list[Stmt]) -> None:
        if node.value is None or (isinstance(node.value, ast.Constant) and node.value.value is None):
            sink.append(Return(self._origin(node), None))
            return
        sink.append(Return(self._origin(node), self._atom(node.value, sink)))
        self._check_region([node.value])

    def _raise(self, node: ast.Raise, sink: list[Stmt]) -> None:
        if node.cause is not None:
            self._reject(node, "`raise ... from` is not supported")
        exc = node.exc
        if exc is None:
            self._reject(node, "bare `raise` is not supported")
        if not (isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name)):
            self._reject(exc, "`raise` requires `ExceptionName(...)` with an optional literal message")
        if mangles(exc.func.id):
            self._reject(exc.func, f"the private name {exc.func.id!r} is subject to name mangling; rename it")
        if exc.keywords or len(exc.args) > 1:
            self._reject(exc, "`raise` supports at most one positional message argument")
        parts: list[str | Atom] = []
        if exc.args:
            parts = self._message_parts(exc.args[0], sink)
        sink.append(Raise(self._origin(node), exc.func.id, tuple(parts)))
        self._check_region([exc])

    def _message_parts(self, message: ast.expr, sink: list[Stmt]) -> list[str | Atom]:
        match message:
            case ast.Constant(value=str(text)):
                return [text]
            case ast.JoinedStr(values=values):
                parts: list[str | Atom] = []
                for value in values:
                    match value:
                        case ast.Constant(value=str(text)):
                            parts.append(text)
                        case ast.FormattedValue(value=ast.Constant(value=str(text)), conversion=-1, format_spec=None):
                            parts.append(text)  # a literal in a field is still a literal string part
                        case ast.FormattedValue(conversion=-1, format_spec=None):
                            parts.append(self._atom(value.value, sink))
                        case ast.FormattedValue():
                            self._reject(value, "f-string conversions and format specs are not supported in raise")
                        case _:
                            self._reject(value, "unsupported f-string content")
                return parts
            case _:
                self._reject(message, "the raise message must be a string literal or f-string")

    def _bind_target(self, target: ast.expr, value: Atom, sink: list[Stmt]) -> None:
        match target:
            case ast.Name():
                sink.append(Assign(self._origin(target), self._local_bind(target), value))
            case ast.Tuple(elts=elts) | ast.List(elts=elts):
                bindings: list[Binding] = []
                for elt in elts:
                    if isinstance(elt, ast.Starred):
                        self._reject(elt, "starred assignment targets are not supported")
                    if not isinstance(elt, ast.Name):
                        self._reject(elt, "unpack targets must be plain names")
                    bindings.append(self._local_bind(elt))
                sink.append(Unpack(self._origin(target), tuple(bindings), value))
            case ast.Attribute() | ast.Subscript():
                root, path = self._store_path(target, sink)
                sink.append(Store(self._origin(target), root=root, path=path, value=value))
            case ast.Starred():
                self._reject(target, "starred assignment targets are not supported")
            case _:
                self._reject(target, f"unsupported assignment target: {type(target).__name__}")

    def _store_path(self, target: ast.expr, sink: list[Stmt]) -> tuple[LocalRef | EnvRead, tuple[Selector, ...]]:
        match target:
            case ast.Name(id=name):
                if mangles(name):
                    self._reject(target, f"the private name {name!r} is subject to name mangling; rename it")
                origin = self._origin(target)
                match self._resolve(name):
                    case NameKind.LOCAL:
                        return LocalRef(origin, self._renamed(name)), ()
                    case NameKind.FREE:
                        return EnvRead(origin, name, free=True), ()
                    case NameKind.GLOBAL:
                        return EnvRead(origin, name, free=False), ()
            case ast.Attribute(value=base, attr=attr):
                if mangles(attr):
                    self._reject(target, f"the private attribute {attr!r} is subject to name mangling; rename it")
                root, path = self._store_path(base, sink)
                return root, (*path, AttrSel(attr))
            case ast.Subscript(value=base, slice=index):
                if isinstance(index, ast.Slice) or (
                    isinstance(index, ast.Tuple) and any(isinstance(e, ast.Slice) for e in index.elts)
                ):
                    self._reject(index, "slice assignment is not supported; rebind the whole value instead")
                root, path = self._store_path(base, sink)
                if isinstance(index, ast.Tuple):
                    return root, (*path, *(IndexSel(self._atom(e, sink)) for e in index.elts))
                return root, (*path, IndexSel(self._atom(index, sink)))
            case _:
                self._reject(target, "a store target must be a plain name followed by attributes and indexes")

    def _local_bind(self, node: ast.Name) -> LocalBind:
        if mangles(node.id):
            self._reject(node, f"the private name {node.id!r} is subject to name mangling; rename it")
        if self._resolve(node.id) is not NameKind.LOCAL:
            self._reject(node, f"assignment to the non-local name {node.id!r} is not supported")
        return LocalBind(self._origin(node), self._renamed(node.id))

    def _renamed(self, name: str) -> str:
        for source, fresh in reversed(self._comp_renames):
            if source == name:
                return fresh
        return name

    def _resolve(self, name: str) -> NameKind:
        if any(source == name for source, _ in self._comp_renames):
            return NameKind.LOCAL
        return self._classifier.classify(name)

    def _check_region(self, exprs: list[ast.expr], aug_target: str | None = None, in_while_test: bool = False) -> None:
        """
        One expression region — a simple statement's expressions, an if/while test, or a for iterable — may not
        both walrus-bind and read the same name: the hoisted walrus assignment would make later atom reads see
        the new value where CPython sequences some reads before it. Conservatively rejects both directions.
        """
        walruses = [n for e in exprs for n in ast.walk(e) if isinstance(n, ast.NamedExpr)]
        if not walruses:
            return
        targets = {w.target.id for w in walruses if isinstance(w.target, ast.Name)}
        reads = {n.id for e in exprs for n in ast.walk(e) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        if aug_target is not None:
            reads.add(aug_target)  # CPython reads the augmented target's old value before the RHS
        collisions = targets & reads
        if collisions:
            witness = next(w for w in walruses if isinstance(w.target, ast.Name) and w.target.id in collisions)
            advice = "restructure as `while True:` with a guarded break" if in_while_test else "split the statement"
            self._reject(witness, f"a name assigned by `:=` may not also be read in the same statement; {advice}")

    def _reject_walrus_in(self, nodes: list[ast.expr], where: str) -> None:
        for node in nodes:
            for sub in ast.walk(node):
                if isinstance(sub, ast.NamedExpr):
                    self._reject(sub, f"`:=` inside {where} is not supported")

    def _atom(self, node: ast.expr, sink: list[Stmt]) -> Atom:
        expr = self._expr(node, sink)
        if isinstance(expr, (TempRef, LocalRef, Const)):
            return expr
        return self._to_temp(node, expr, sink)

    def _to_temp(self, node: ast.AST, expr: Expr, sink: list[Stmt]) -> TempRef:
        origin = self._origin(node)
        index = self._fresh_temp()
        sink.append(Assign(origin, TempBind(origin, index), expr))
        return TempRef(origin, index)

    def _fresh_temp(self) -> int:
        index = self._temp_count
        self._temp_count += 1
        return index

    def _expr(self, node: ast.expr, sink: list[Stmt]) -> Expr:
        match node:
            case ast.Constant():
                return self._constant(node)
            case ast.Name(id=name):
                return self._name(node, name)
            case ast.NamedExpr():
                assert isinstance(node.target, ast.Name), "the walrus target is always a plain name"
                target = self._local_bind(node.target)
                value = self._atom(node.value, sink)
                sink.append(Assign(self._origin(node), target, value))
                return value  # the walrus expression's value, never a re-read of the target name
            case ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=int() | float() as v)) if not isinstance(
                v, bool
            ):
                return Const(self._origin(node), -v)
            case ast.UnaryOp(op=ast.UAdd(), operand=ast.Constant(value=int() | float() as v)) if not isinstance(
                v, bool
            ):
                return Const(self._origin(node), +v)
            case ast.UnaryOp():
                op = _UNARY_OPS.get(type(node.op))
                assert op is not None, "every ast unary operator is mapped"
                return Unary(self._origin(node), op, self._atom(node.operand, sink))
            case ast.BinOp():
                bop = _BIN_OPS.get(type(node.op))
                assert bop is not None, "every ast binary operator is mapped"
                left = self._atom(node.left, sink)
                return Binary(self._origin(node), bop, left, self._atom(node.right, sink))
            case ast.BoolOp():
                return self._bool_op(node, sink)
            case ast.Compare():
                return self._compare(node, sink)
            case ast.IfExp():
                return self._if_exp(node, sink)
            case ast.Call():
                return self._call(node, sink)
            case ast.Attribute(value=base, attr=attr):
                if mangles(attr):
                    self._reject(node, f"the private attribute {attr!r} is subject to name mangling; rename it")
                return AttrRead(self._origin(node), self._atom(base, sink), attr)
            case ast.Subscript():
                return self._subscript(node, sink)
            case ast.Tuple(elts=elts):
                return TupleExpr(self._origin(node), self._display_items(elts, sink))
            case ast.List(elts=elts):
                return ListExpr(self._origin(node), self._display_items(elts, sink))
            case ast.ListComp():
                return self._comprehension(node, sink)
            case ast.SetComp() | ast.DictComp():
                self._reject(node, "set/dict comprehensions are not supported")
            case ast.GeneratorExp():
                self._reject(node, "generator expressions are not supported")
            case ast.Dict():
                self._reject(node, "dictionary displays are not supported")
            case ast.Set():
                self._reject(node, "set displays are not supported")
            case ast.Lambda():
                self._reject(node, "lambdas are not supported")
            case ast.JoinedStr():
                self._reject(node, "f-strings are only supported as raise messages")
            case ast.Await():
                self._reject(node, "`await` is not supported")
            case ast.Yield() | ast.YieldFrom():
                self._reject(node, "generators are not supported")
            case ast.Starred():
                self._reject(node, "a starred expression is only supported as a call argument or display element")
            case _:
                self._reject(node, f"unsupported expression: {type(node).__name__}")

    def _constant(self, node: ast.Constant) -> Const:
        value = node.value
        if isinstance(value, (bool, int, float)):
            return Const(self._origin(node), value)
        if isinstance(value, str):
            self._reject(node, "string literals are only supported as raise messages")
        if value is None:
            self._reject(node, "`None` is only supported as a bare return value")
        self._reject(node, f"unsupported constant: {value!r}")

    def _name(self, node: ast.Name, name: str) -> Expr:
        if mangles(name):
            self._reject(node, f"the private name {name!r} is subject to name mangling; rename it")
        origin = self._origin(node)
        match self._resolve(name):
            case NameKind.LOCAL:
                return LocalRef(origin, self._renamed(name))
            case NameKind.FREE:
                return EnvRead(origin, name, free=True)
            case NameKind.GLOBAL:
                return EnvRead(origin, name, free=False)

    def _bool_op(self, node: ast.BoolOp, sink: list[Stmt]) -> Expr:
        self._reject_walrus_in(node.values[1:], "an `and`/`or` operand after the first")
        op = BinaryOp.AND if isinstance(node.op, ast.And) else BinaryOp.OR
        result = self._atom(node.values[0], sink)
        for i, operand in enumerate(node.values[1:]):  # eager gates, nested left-associated
            gate = Binary(self._origin(node), op, result, self._atom(operand, sink))
            if i + 2 == len(node.values):
                return gate
            result = self._to_temp(node, gate, sink)
        raise AssertionError("ast guarantees at least two operands")

    def _compare(self, node: ast.Compare, sink: list[Stmt]) -> Expr:
        for op in node.ops:
            if type(op) not in _CMP_OPS:
                spelled = {ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in"}
                name = spelled.get(type(op), type(op).__name__)
                self._reject(node, f"the comparison operator `{name}` is not supported")
        self._reject_walrus_in(node.comparators[1:], "a comparison-chain comparator after the first")
        left = self._atom(node.left, sink)
        return self._chain(node, left, list(node.ops), list(node.comparators), sink)

    def _chain(
        self, node: ast.Compare, left: Atom, ops: list[ast.cmpop], comparators: list[ast.expr], sink: list[Stmt]
    ) -> Expr:
        """
        A chain short-circuits as in CPython: the shared operand is evaluated once, and each later comparator
        evaluates only if the running result is still true — as a guarded branch assigning a result temp.
        """
        right = self._atom(comparators[0], sink)
        comparison = Compare(self._origin(node), _CMP_OPS[type(ops[0])], left, right)
        if len(comparators) == 1:
            return comparison
        origin = self._origin(node)
        cond = self._to_temp(node, comparison, sink)
        result = self._fresh_temp()
        then_sink: list[Stmt] = []
        rest = self._chain(node, right, ops[1:], comparators[1:], then_sink)
        then_sink.append(Assign(origin, TempBind(origin, result), rest))
        orelse = (Assign(origin, TempBind(origin, result), Const(origin, False)),)
        sink.append(If(origin, cond, tuple(then_sink), orelse))
        return TempRef(origin, result)

    def _if_exp(self, node: ast.IfExp, sink: list[Stmt]) -> Expr:
        self._reject_walrus_in([node.body, node.orelse], "a conditional-expression arm")
        origin = self._origin(node)
        cond = self._atom(node.test, sink)
        result = self._fresh_temp()
        then_sink: list[Stmt] = []
        then_sink.append(Assign(origin, TempBind(origin, result), self._expr(node.body, then_sink)))
        else_sink: list[Stmt] = []
        else_sink.append(Assign(origin, TempBind(origin, result), self._expr(node.orelse, else_sink)))
        sink.append(If(origin, cond, tuple(then_sink), tuple(else_sink)))
        return TempRef(origin, result)

    def _call(self, node: ast.Call, sink: list[Stmt]) -> Expr:
        callee = self._atom(node.func, sink)
        args: list[PosArg | StarArg | KwArg] = []
        for arg in node.args:  # positional and starred evaluate first, in source order, as in CPython
            if isinstance(arg, ast.Starred):
                args.append(StarArg(self._atom(arg.value, sink)))
            else:
                args.append(PosArg(self._atom(arg, sink)))
        for keyword in node.keywords:  # keyword values evaluate after, in source order
            if keyword.arg is None:
                self._reject(keyword.value, "`**` call arguments are not supported")
            args.append(KwArg(keyword.arg, self._atom(keyword.value, sink)))
        return Call(self._origin(node), callee, tuple(args))

    def _subscript(self, node: ast.Subscript, sink: list[Stmt]) -> Expr:
        base = self._atom(node.value, sink)
        match node.slice:
            case ast.Slice(step=step) if step is not None:
                self._reject(step, "slice steps are not supported")
            case ast.Slice(lower=lower, upper=upper):
                lo = self._atom(lower, sink) if lower is not None else None
                hi = self._atom(upper, sink) if upper is not None else None
                return SliceRead(self._origin(node), base, lo, hi)
            case ast.Tuple(elts=elts):
                axes: list[Atom | SliceSel] = []
                for elt in elts:  # one selector per axis, bounds hoisted in source order as CPython evaluates
                    match elt:
                        case ast.Slice(step=step) if step is not None:
                            self._reject(step, "slice steps are not supported")
                        case ast.Slice(lower=lower, upper=upper):
                            lo = self._atom(lower, sink) if lower is not None else None
                            hi = self._atom(upper, sink) if upper is not None else None
                            axes.append(SliceSel(lo, hi))
                        case _:
                            axes.append(self._atom(elt, sink))
                return MultiIndexRead(self._origin(node), base, tuple(axes))
            case _:
                return IndexRead(self._origin(node), base, self._atom(node.slice, sink))

    def _display_items(self, elts: list[ast.expr], sink: list[Stmt]) -> tuple[Atom | StarArg, ...]:
        items: list[Atom | StarArg] = []
        for elt in elts:
            if isinstance(elt, ast.Starred):
                items.append(StarArg(self._atom(elt.value, sink)))
            else:
                items.append(self._atom(elt, sink))
        return tuple(items)

    def _comprehension(self, node: ast.ListComp, sink: list[Stmt]) -> Expr:
        self._reject_walrus_in([node], "a comprehension")
        if len(node.generators) != 1:
            self._reject(node, "comprehensions with multiple `for` clauses are not supported")
        gen = node.generators[0]
        if gen.ifs:
            self._reject(gen.ifs[0], "comprehension `if` filters are not supported")
        if gen.is_async:
            self._reject(node, "async comprehensions are not supported")
        if not isinstance(gen.target, ast.Name):
            self._reject(gen.target, "comprehension target unpacking is not supported")
        if mangles(gen.target.id):
            self._reject(gen.target, f"the private name {gen.target.id!r} is subject to name mangling; rename it")
        iterable = self._atom(gen.iter, sink)  # the enclosing naming context: the target is not yet in scope
        fresh = f"{gen.target.id}${self._comp_count}"
        self._comp_count += 1
        self._comp_renames.append((gen.target.id, fresh))
        try:
            body_sink: list[Stmt] = []
            element = self._atom(node.elt, body_sink)
        finally:
            self._comp_renames.pop()
        return Comp(self._origin(node), fresh, iterable, tuple(body_sink), element)

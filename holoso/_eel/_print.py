"""
The canonical printed Eel text: print-only (never parsed back) and deterministic, so a difference between two
runs means a transform changed. Locations off by default, on for logs and for the written artifacts.
"""

import dataclasses
import math

from ._ir import *
from ._names import access_path

_INDENT = " " * 4


def print_eel(fn: EelFunction, *, locations: bool = False) -> str:
    lines: list[str] = [_header(fn)]
    for decl in fn.slots:
        lines.append(f"{_INDENT}state {_slot(decl.slot)}: {decl.stype.value} reset {_const(decl.reset)}")
    for out in fn.outputs:
        lines.append(f"{_INDENT}output {access_path(out.path)}: {out.stype.value}")
    _block(fn.body, 1, lines, locations)
    return "\n".join(lines) + "\n"


def _header(fn: EelFunction) -> str:
    parts: list[str] = []
    kinds = [param.kind for param in fn.params]
    for i, param in enumerate(fn.params):
        if param.kind is ParamKind.KEYWORD_ONLY and (i == 0 or kinds[i - 1] is not ParamKind.KEYWORD_ONLY):
            parts.append("*")
        parts.append(param.name if param.stype is None else f"{param.name}: {param.stype.value}")
        if param.kind is ParamKind.POSITIONAL_ONLY and (
            i + 1 == len(fn.params) or kinds[i + 1] is not ParamKind.POSITIONAL_ONLY
        ):
            parts.append("/")
    return f"fn {fn.name}({', '.join(parts)}):"


def _slot(slot: SlotPath) -> str:
    assert slot and isinstance(slot[0], str)
    return access_path(slot)[1:]


def _block(body: Block, depth: int, lines: list[str], locations: bool) -> None:
    pad = _INDENT * depth
    for stmt in body:
        _statement(stmt, depth, lines, locations)
    if not body:
        lines.append(pad + "pass")


def _statement(stmt: Stmt, depth: int, lines: list[str], locations: bool) -> None:
    pad = _INDENT * depth

    def put(text: str, origin: Origin) -> None:
        lines.append(pad + text + (_loc_suffix(origin) if locations else ""))

    match stmt:
        case Assign(origin=origin, target=target, value=value, stype=stype):
            annotation = "" if stype is None else f": {stype.value}"
            if isinstance(value, Comp):
                put(f"{_binding(target)}{annotation} = comp for {value.target} in {_atom(value.iterable)}:", origin)
                if value.body:
                    _block(value.body, depth + 1, lines, locations)
                lines.append(_INDENT * (depth + 1) + f"yield {_atom(value.element)}")
            else:
                put(f"{_binding(target)}{annotation} = {_expr(value)}", origin)
        case Unpack(origin=origin, targets=targets, value=value):
            rendered = ", ".join(_binding(t) for t in targets) + ("," if len(targets) == 1 else "")
            put(f"{rendered or '()'} = {_atom(value)}", origin)
        case Store(origin=origin, root=root, path=path, value=value):
            put(f"{_store_path(root, path)} = {_atom(value)}", origin)
        case AugMark(origin=origin, mark=mark):
            put(f"mark !{mark}", origin)
        case AugStore(origin=origin, root=root, path=path, op=op, value=value, mark=mark):
            put(f"{_store_path(root, path)} {op.value}= {_atom(value)} !{mark}", origin)
        case AugAssign(origin=origin, target=target, op=op, value=value):
            put(f"{target.name} {op.value}= {_atom(value)}", origin)
        case If(origin=origin, cond=cond, then=then, orelse=orelse):
            put(f"if {_atom(cond)}:", origin)
            _block(then, depth + 1, lines, locations)
            if orelse:
                lines.append(pad + "else:")
                _block(orelse, depth + 1, lines, locations)
        case While(origin=origin, header=header, cond=cond, body=body) | ResidualWhile(
            origin=origin, header=header, cond=cond, body=body
        ):
            for phi in stmt.phis if isinstance(stmt, ResidualWhile) else ():
                put(f"phi %{phi.index}: {phi.stype.value} = {_atom(phi.entry)} -> {_atom(phi.back)}", phi.origin)
            if header:
                put("while:", origin)
                _block(header, depth + 1, lines, locations)
                lines.append(pad + f"do {_atom(cond)}:")
            else:
                put(f"while {_atom(cond)}:", origin)
            _block(body, depth + 1, lines, locations)
        case For(origin=origin, target=target, iterable=iterable, body=body):
            put(f"for {_binding(target)} in {_atom(iterable)}:", origin)
            _block(body, depth + 1, lines, locations)
        case Return(origin=origin, value=value):
            put("return" if value is None else f"return {_atom(value)}", origin)
        case ResidualReturn(origin=origin, values=values):
            put("return" if not values else f"return {', '.join(_atom(v) for v in values)}", origin)
        case SlotWrite(origin=origin, slot=slot, value=value):
            put(f"slot {_slot(slot)} = {_atom(value)}", origin)
        case Break(origin=origin) | ResidualBreak(origin=origin):
            put("break", origin)
        case Continue(origin=origin) | ResidualContinue(origin=origin):
            put("continue", origin)
        case ResidualFrame(origin=origin, rows=rows, body=body):
            for row in rows:
                put(f"result %{row.index}: {row.stype.value}", row.origin)
            put("frame:", origin)
            _block(body, depth + 1, lines, locations)
        case ResidualFrameReturn(origin=origin, values=values):
            # Spelled apart from a kernel return: this one converges at the frame exit and commits no
            # outputs or slots, and the printed form must not read the same for both.
            put("exit" if not values else f"exit {', '.join(_atom(v) for v in values)}", origin)
        case Raise(origin=origin, exc_type=exc_type, parts=parts):
            rendered = " ".join(repr(p) if isinstance(p, str) else _atom(p) for p in parts)
            put(f"raise {exc_type}" + (f" {rendered}" if rendered else ""), origin)
        case _:
            raise AssertionError(stmt)


def _expr(expr: Expr) -> str:
    match expr:
        case TempRef() | LocalRef() | Const():
            return _atom(expr)
        case Unary(op=op, operand=operand):
            rendered = _atom(operand)
            if op is UnaryOp.NOT:
                return f"{op.value} {rendered}"
            if rendered.startswith("-"):  # a folded negative literal: `--2.5` would read as a typo
                rendered = f"({rendered})"
            return f"{op.value}{rendered}"
        case Binary(op=op, left=left, right=right) | Compare(op=op, left=left, right=right):
            return f"{_atom(left)} {op.value} {_atom(right)}"
        case IsNone(operand=operand, negated=negated):
            return f"{_atom(operand)} {'is not' if negated else 'is'} None"
        case Call(callee=callee, args=args):
            return f"call {_atom(callee)}({', '.join(_argument(a) for a in args)})"
        case AttrRead(base=base, attr=attr):
            return f"{_atom(base)}.{attr}"
        case IndexRead(base=base, index=index):
            return f"{_atom(base)}[{_axis(index)}]"
        case MultiIndexRead(base=base, axes=axes):
            return f"{_atom(base)}[{', '.join(_axis(a) for a in axes)}]"
        case TupleExpr(items=items):
            if len(items) == 1:
                return f"({_item(items[0])},)"
            return f"({', '.join(_item(i) for i in items)})"
        case ListExpr(items=items):
            return f"[{', '.join(_item(i) for i in items)}]"
        case EnvRead(name=name, free=free):
            return f"{'free' if free else 'env'} {name}"
        case IntrinsicCall(operator=operator, args=args):
            return f"intrinsic {_operator(operator)}({', '.join(_atom(a) for a in args)})"
        case SlotRead(slot=slot):
            return f"slot {_slot(slot)}"
        case Comp():
            raise AssertionError("a comprehension prints only as the value of an assignment")
        case _:
            raise AssertionError(expr)


def _operator(operator: object) -> str:
    """
    The mnemonic plus any operator parameters (`fmul_pow2<3>`), duck-typed so the printer stays HIR-free;
    stable across operator-class internals, unlike the dataclass repr.
    """
    mnemonic = getattr(operator, "mnemonic")
    assert isinstance(mnemonic, str) and dataclasses.is_dataclass(operator) and not isinstance(operator, type)
    parts = [repr(getattr(operator, field.name)) for field in dataclasses.fields(operator)]
    return mnemonic + (f"<{','.join(parts)}>" if parts else "")


def _item(item: Atom | StarArg) -> str:
    if isinstance(item, StarArg):
        return f"*{_atom(item.value)}"
    return _atom(item)


def _axis(axis: Atom | SliceSel) -> str:
    if isinstance(axis, SliceSel):
        return f"{'' if axis.lo is None else _atom(axis.lo)}:{'' if axis.hi is None else _atom(axis.hi)}"
    return _atom(axis)


def _argument(arg: Argument) -> str:
    if isinstance(arg, KwArg):
        return f"{arg.name}={_atom(arg.value)}"
    return _item(arg)


def _atom(atom: Atom) -> str:
    match atom:
        case TempRef(index=index):
            return f"%{index}"
        case LocalRef(name=name):
            return name
        case Const(value=value):
            return _const(value)


def _const(value: bool | int | float) -> str:
    if isinstance(value, float):
        # A non-finite value would print as `inf`/`nan` — indistinguishable from a local of that name.
        return repr(value) if math.isfinite(value) else f"float('{value!r}')"
    if not isinstance(value, bool) and value.bit_length() > 64:
        # Decimal conversion of huge integers hits CPython's digit cap; hex conversion is unlimited.
        return hex(value)
    return repr(value)


def _binding(binding: Binding) -> str:
    match binding:
        case TempBind(index=index):
            return f"%{index}"
        case LocalBind(name=name):
            return name


def _store_path(root: LocalRef | EnvRead, path: tuple[Selector, ...]) -> str:
    text = _expr(root)
    for selector in path:
        match selector:
            case AttrSel(name=name):
                text += f".{name}"
            case IndexSel(index=index):
                text += f"[{_atom(index)}]"
    return text


def _loc_suffix(origin: Origin) -> str:
    """Every call site the expansion passed through, outermost first, then the line itself."""
    hops = [f"{frame.site.brief} via {frame.callee}" for frame in origin.frames]
    return "  # " + ", ".join([*hops, origin.location.brief])

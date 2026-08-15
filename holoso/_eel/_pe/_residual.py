"""
Statement-tree utilities of the partial evaluator: the syntactic assigned-name scan feeding loop carries and
the state seed, origin re-chaining for inlined callees, and the residual liveness sweep.
"""

import dataclasses

from ..._errors import SourceLocation
from .._ir import *


def rechained(node: object, frames: tuple[CallFrame, ...]) -> object:
    """
    An inlined callee's tree with every origin carrying the frame chain, so residual nodes and rejections
    alike re-attribute; shape-generic over the frozen node dataclasses so IR growth needs no maintenance here.
    """
    if isinstance(node, Origin):
        return Origin(node.location, frames)
    if isinstance(node, SourceLocation):
        return node
    if isinstance(node, tuple):
        return tuple(rechained(item, frames) for item in node)
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return dataclasses.replace(
            node, **{field.name: rechained(getattr(node, field.name), frames) for field in dataclasses.fields(node)}
        )
    return node


def assigned_names(
    stmts: tuple[Stmt, ...],
) -> tuple[set[str], set[str], dict[tuple[str, tuple[str, ...]], Origin]]:
    """
    The syntactic (rebound, store-rooted, attr-chain-stored) sets: the loop-carry over-approximation for
    residualization and the raw material of the assumed-state seed. Rebinds and stores are kept apart
    because a store mutates its root in place rather than rebinding it, so a non-aggregate store root is no
    carry at all -- the store step itself owns the precise rejection. Attr-chain stores keep the FIRST
    write per (root, chain) pair for the pinning diagnostic; the caller resolves roots (a method's receiver
    at seeding, the live bindings at loop residualization). Comprehension bodies bind only temps and their
    renamed targets never collide with user locals, so the uniform walk stays exact for names.
    """
    rebound: set[str] = set()
    stored: set[str] = set()
    attrs: dict[tuple[str, tuple[str, ...]], Origin] = {}
    for stmt in stmts:
        parts: tuple[tuple[Stmt, ...], ...] = ()
        match stmt:
            case Assign(target=LocalBind(name=name)) | AugAssign(target=LocalBind(name=name)):
                rebound.add(name)
            case Unpack(targets=targets):
                rebound |= {target.name for target in targets if isinstance(target, LocalBind)}
            case Store(root=LocalRef(name=name)) | AugStore(root=LocalRef(name=name)):
                chain: list[str] = []
                for selector in stmt.path:
                    if not isinstance(selector, AttrSel):
                        break
                    chain.append(selector.name)
                if chain:
                    attrs.setdefault((name, tuple(chain)), stmt.origin)
                else:
                    stored.add(name)
            case If(then=then, orelse=orelse):
                parts = (then, orelse)
            case While(header=header, body=body):
                parts = (header, body)
            case For(target=target, body=body):
                rebound.add(target.name)
                parts = (body,)
            case _:
                pass
        for part in parts:
            inner_rebound, inner_stored, inner_attrs = assigned_names(part)
            rebound |= inner_rebound
            stored |= inner_stored
            for pair, origin in inner_attrs.items():
                attrs.setdefault(pair, origin)
    return rebound, stored, attrs


def drop_return_rows(stmts: list[Stmt] | tuple[Stmt, ...], keep: list[bool]) -> list[Stmt]:
    """Every return site carries one atom per pre-elision output row; drop the elided positions everywhere."""
    rebuilt: list[Stmt] = []
    for stmt in stmts:
        match stmt:
            case ResidualReturn(origin=origin, values=values):
                rebuilt.append(ResidualReturn(origin, tuple(v for v, kept in zip(values, keep, strict=True) if kept)))
            case If(origin=origin, cond=cond, then=then, orelse=orelse):
                rebuilt.append(
                    If(origin, cond, tuple(drop_return_rows(then, keep)), tuple(drop_return_rows(orelse, keep)))
                )
            case ResidualWhile(header=header, body=body):
                rebuilt.append(
                    dataclasses.replace(
                        stmt, header=tuple(drop_return_rows(header, keep)), body=tuple(drop_return_rows(body, keep))
                    )
                )
            case _:
                rebuilt.append(stmt)
    return rebuilt


def _terminated(stmts: tuple[Stmt, ...]) -> bool:
    """Whether every path through the suffix ends in a return site."""
    if not stmts:
        return False
    match stmts[-1]:
        case ResidualReturn():
            return True
        case If(then=then, orelse=orelse):
            return _terminated(then) and _terminated(orelse)
        case _:
            return False


def prune(stmts: list[Stmt] | tuple[Stmt, ...], live_after: set[int] | None = None) -> tuple[list[Stmt], set[int]]:
    """
    One reverse-liveness sweep; returns the surviving statements and the temp reads they need live-in.
    A branch whose arms both empty is dropped whole, de-rooting its condition so the condition's producer can
    disappear too. An arm is swept against the post-branch live set, so a join temp it assigns stays exactly
    when something after the branch reads it; an arm terminated by its own return sites sweeps against the
    EMPTY set, since nothing after the branch runs on that path.
    """
    kept_reversed: list[Stmt] = []
    live = set(live_after) if live_after is not None else set()
    for stmt in reversed(stmts):
        match stmt:
            case Assign(target=TempBind(index=index), value=value):
                if index not in live:
                    continue
                live.discard(index)
                live |= reads(value)
                kept_reversed.append(stmt)
            case If(origin=origin, cond=cond, then=then, orelse=orelse):
                then_kept, then_live = prune(then, set() if _terminated(then) else live)
                else_kept, else_live = prune(orelse, set() if _terminated(orelse) else live)
                if not then_kept and not else_kept:
                    continue
                live = then_live | else_live | reads(cond)
                kept_reversed.append(If(origin, cond, tuple(then_kept), tuple(else_kept)))
            case ResidualWhile() | ResidualFrame():
                # A residual loop is never dead (termination is observable), and a frame always contains
                # one; the interior is kept whole and HIR's own sweeps clean any dead interior operations.
                defined, read = _loop_temps((stmt,))
                live -= defined
                live |= read - defined
                kept_reversed.append(stmt)
            case SlotWrite(value=value):
                live |= reads(value)
                kept_reversed.append(stmt)
            case ResidualReturn(values=values):
                for atom in values:
                    live |= reads(atom)
                kept_reversed.append(stmt)
            case _:
                raise AssertionError(stmt)
    return list(reversed(kept_reversed)), live


def _loop_temps(stmts: tuple[Stmt, ...]) -> tuple[set[int], set[int]]:
    """The (defined, read) temp-index sets over a residual subtree, uncut by liveness."""
    defined: set[int] = set()
    read: set[int] = set()
    for stmt in stmts:
        match stmt:
            case Assign(target=TempBind(index=index), value=value):
                defined.add(index)
                read |= reads(value)
            case If(cond=cond, then=then, orelse=orelse):
                read |= reads(cond)
                for arm in (then, orelse):
                    inner_defined, inner_read = _loop_temps(arm)
                    defined |= inner_defined
                    read |= inner_read
            case ResidualWhile(phis=phis, header=header, cond=cond, body=body):
                defined |= {phi.index for phi in phis}
                for phi in phis:
                    read |= reads(phi.entry) | reads(phi.back)
                read |= reads(cond)
                for part in (header, body):
                    inner_defined, inner_read = _loop_temps(part)
                    defined |= inner_defined
                    read |= inner_read
            case SlotWrite(value=value):
                read |= reads(value)
            case ResidualReturn(values=values) | ResidualFrameReturn(values=values):
                for atom in values:
                    read |= reads(atom)
            case ResidualBreak() | ResidualContinue():
                pass
            case ResidualFrame(rows=rows, body=body):
                defined |= {row.index for row in rows}
                inner_defined, inner_read = _loop_temps(body)
                defined |= inner_defined
                read |= inner_read
            case _:
                raise AssertionError(stmt)
    return defined, read


def reads(expr: Expr) -> set[int]:
    match expr:
        case TempRef(index=index):
            return {index}
        case IntrinsicCall(args=args):
            found: set[int] = set()
            for arg in args:
                if isinstance(arg, TempRef):
                    found.add(arg.index)
            return found
        case SlotRead():
            return set()
        case LocalRef() | Const():
            return set()
        case _:
            raise AssertionError(expr)

"""
Sublayer 3: residual Eel -> HIR, mechanically. Every semantic decision was the partial evaluator's: operations
arrive as ``IntrinsicCall`` nodes carrying their HIR operator, every assignment is typed, and control is
structured branches only. The emitter keeps one environment of binding -> value id, forks it per branch arm,
and at the join keeps the intersection, phi-ing entries whose ids differ -- names bound on one side only were
already dropped (and any later read rejected) by the partial evaluator's definite-assignment rule. Parameters
become input ports in declaration order whether read or not: the module interface mirrors the signature.

Every ``ResidualReturn`` is one return SITE: an arm that returns simply never jumps to its local join. All
sites meet in a single exit block -- one phi per output row and per slot leaf when there are several -- so
``Ret`` stays the sole function exit and each ``StateSlot.live_out`` is the value at that exit. The phi types
agree across sites because the partial evaluator conformed every site against the one annotation-fixed table.
"""

from dataclasses import dataclass

from ..._hir import BoolConst, BoolType, Const as HirConst, FloatConst, FloatType, Hir, HirBuilder, IntConst, IntType
from ..._hir import Operator, Type
from .._ir import *
from .._names import port_name, public_slot, slot_name, state_port_name

_TYPES: dict[ScalarType, Type] = {
    ScalarType.BOOL: BoolType(),
    ScalarType.INT: IntType(),
    ScalarType.FLOAT: FloatType(),
}

type _Env = dict[str | int, int]


@dataclass(frozen=True, slots=True)
class _Site:
    block: int
    outputs: tuple[int, ...]
    slots: dict[str, int]


def emit(fn: EelFunction) -> Hir:
    builder = HirBuilder()
    builder.position_at(builder.block())
    env: _Env = {}
    for param in fn.params:
        assert param.stype is not None
        env[param.name] = builder.input(param.name, _TYPES[param.stype])
    sites: list[_Site] = []
    terminated = _block(builder, fn, fn.body, env, sites)
    assert terminated, "the residual body ends in a return site on every path"
    assert sites
    if len(sites) == 1:
        (site,) = sites
        builder.position_at(site.block)
        _finish(builder, fn, list(site.outputs), site.slots)
    else:
        exit_block = builder.block()
        for site in sites:
            builder.position_at(site.block)
            builder.jump(exit_block)
        builder.position_at(exit_block)
        outputs = [
            _meet(builder, [(site.block, site.outputs[row]) for site in sites]) for row in range(len(fn.outputs))
        ]
        slots = {name: _meet(builder, [(site.block, site.slots[name]) for site in sites]) for name in sites[0].slots}
        _finish(builder, fn, outputs, slots)
    return builder.finish()


def _meet(builder: HirBuilder, arms: list[tuple[int, int]]) -> int:
    first = arms[0][1]
    if all(vid == first for _, vid in arms):
        return first
    merged_type = builder.type_of(first)
    assert all(builder.type_of(vid) == merged_type for _, vid in arms), "the sites share one annotation-fixed table"
    return builder.phi(merged_type, arms)


def _finish(builder: HirBuilder, fn: EelFunction, outputs: list[int], slots: dict[str, int]) -> None:
    for decl, value_id in zip(fn.outputs, outputs, strict=True):
        assert builder.type_of(value_id) == _TYPES[decl.stype]
        builder.output(port_name(decl.path), value_id)
    for slot in fn.slots:
        name = slot_name(slot.slot)
        live_out = slots[name]
        assert builder.type_of(live_out) == _TYPES[slot.stype]
        builder.state_slot(name, _reset(slot), live_out)
        if public_slot(slot.slot):
            builder.output(state_port_name(slot.slot), live_out)
    builder.ret()


def _reset(slot: SlotDecl) -> HirConst:
    match slot.stype:
        case ScalarType.BOOL:
            assert isinstance(slot.reset, bool)
            return BoolConst(slot.reset)
        case ScalarType.INT:
            assert isinstance(slot.reset, int) and not isinstance(slot.reset, bool)
            return IntConst(slot.reset)
        case ScalarType.FLOAT:
            assert isinstance(slot.reset, float)
            return FloatConst(slot.reset)


def _block(builder: HirBuilder, fn: EelFunction, stmts: tuple[Stmt, ...], env: _Env, sites: list[_Site]) -> bool:
    pending_slots: dict[str, int] = {}
    for stmt in stmts:
        match stmt:
            case Assign(target=TempBind(index=index), value=value, stype=stype):
                assert stype is not None
                env[index] = _value(builder, value, env, stype)
            case If(cond=cond, then=then, orelse=orelse):
                if _branch(builder, fn, cond, then, orelse, env, sites):
                    return True
            case ResidualWhile():
                _loop(builder, fn, stmt, env, sites)
            case SlotWrite(slot=slot, value=value):
                pending_slots[slot_name(slot)] = _atom(builder, value, env)
            case ResidualReturn(values=values):
                assert set(pending_slots) == {slot_name(slot.slot) for slot in fn.slots}
                outputs = tuple(_atom(builder, atom, env) for atom in values)
                sites.append(_Site(builder.current_block, outputs, pending_slots))
                return True
            case _:
                raise AssertionError(stmt)
    assert not pending_slots, "slot commits immediately precede their return site"
    return False


def _branch(
    builder: HirBuilder,
    fn: EelFunction,
    cond: Atom,
    then: tuple[Stmt, ...],
    orelse: tuple[Stmt, ...],
    env: _Env,
    sites: list[_Site],
) -> bool:
    then_block = builder.block()
    else_block = builder.block()
    builder.branch(_atom(builder, cond, env), then_block, else_block)

    builder.position_at(then_block)
    then_env = dict(env)
    then_done = _block(builder, fn, then, then_env, sites)
    then_exit = builder.current_block

    builder.position_at(else_block)
    else_env = dict(env)
    else_done = _block(builder, fn, orelse, else_env, sites)
    else_exit = builder.current_block

    if then_done and else_done:
        return True
    if then_done or else_done:
        surviving_exit, surviving_env = (else_exit, else_env) if then_done else (then_exit, then_env)
        builder.position_at(surviving_exit)
        env.clear()
        env.update(surviving_env)
        return False

    join_block = builder.block()
    builder.position_at(then_exit)
    builder.jump(join_block)
    builder.position_at(else_exit)
    builder.jump(join_block)
    builder.position_at(join_block)

    env.clear()
    for key in sorted(set(then_env) & set(else_env), key=lambda k: (isinstance(k, str), str(k))):
        a, b = then_env[key], else_env[key]
        if a == b:
            env[key] = a
            continue
        merged_type = builder.type_of(a)
        assert merged_type == builder.type_of(b), f"the arms bind {key!r} to differing types"
        env[key] = builder.phi(merged_type, [(then_exit, a), (else_exit, b)])
    return False


def _loop(builder: HirBuilder, fn: EelFunction, stmt: ResidualWhile, env: _Env, sites: list[_Site]) -> None:
    """
    The genuine back-edge loop: phis and the latch target live in the loop-entry block; the header (which may
    itself branch and re-join) runs before every test, so the exit block continues with the environment as of
    header end -- header-defined values dominate the exit, body-defined ones are never read after the loop.
    """
    entry_ids = [_atom(builder, phi.entry, env) for phi in stmt.phis]
    pre_exit = builder.current_block
    loop_entry = builder.block()
    builder.jump(loop_entry)
    builder.position_at(loop_entry)
    phi_ids: list[int] = []
    for phi, entry_id in zip(stmt.phis, entry_ids, strict=True):
        phi_id = builder.open_phi(_TYPES[phi.stype], (pre_exit, entry_id))
        env[phi.index] = phi_id
        phi_ids.append(phi_id)
    terminated = _block(builder, fn, stmt.header, env, sites)
    assert not terminated, "a residual loop interior holds no return site"
    body_block = builder.block()
    exit_block = builder.block()
    builder.branch(_atom(builder, stmt.cond, env), body_block, exit_block)
    post_header = dict(env)

    builder.position_at(body_block)
    terminated = _block(builder, fn, stmt.body, env, sites)
    assert not terminated, "the loop body keeps a surviving latch path"
    back_ids = [_atom(builder, phi.back, env) for phi in stmt.phis]
    latch_exit = builder.current_block
    builder.jump(loop_entry)
    for phi_id, entry_id, back_id in zip(phi_ids, entry_ids, back_ids, strict=True):
        builder.set_phi_arms(phi_id, [(pre_exit, entry_id), (latch_exit, back_id)])

    builder.position_at(exit_block)
    # Only header-end bindings dominate the exit; a body binding leaking past it would be phi-ed by a later
    # arm join against a twin copy of this loop, referencing a value a zero-trip path never computes.
    env.clear()
    env.update(post_header)


def _value(builder: HirBuilder, value: Expr, env: _Env, stype: ScalarType) -> int:
    match value:
        case IntrinsicCall(operator=operator, args=args):
            assert isinstance(operator, Operator)
            value_id = builder.operation(operator, [_atom(builder, arg, env) for arg in args])
        case SlotRead(slot=slot):
            value_id = builder.state_read(slot_name(slot), _TYPES[stype])
        case TempRef() | LocalRef() | Const():
            value_id = _atom(builder, value, env)
        case _:
            raise AssertionError(value)
    assert builder.type_of(value_id) == _TYPES[stype]
    return value_id


def _atom(builder: HirBuilder, atom: Atom, env: _Env) -> int:
    match atom:
        case TempRef(index=index):
            return env[index]
        case LocalRef(name=name):
            return env[name]
        case Const(value=value):
            if type(value) is bool:
                return builder.bool_const(value)
            if type(value) is int:
                return builder.int_const(value)
            assert type(value) is float
            return builder.float_const(value)

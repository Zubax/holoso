"""
Sublayer 3: residual Eel -> HIR, mechanically. Every semantic decision was the partial evaluator's: operations
arrive as `IntrinsicCall` nodes carrying their HIR operator, every assignment is typed, and control is
structured: branches, data-dependent loops with their carried phis, and the loop and inlined-frame exits. The emitter
keeps one environment of binding -> value id, forks it per branch arm,
and at the join keeps the intersection, phi-ing entries whose ids differ -- names bound on one side only were
already dropped (and any later read rejected) by the partial evaluator's definite-assignment rule. Parameters
become input ports in declaration order whether read or not: the module interface mirrors the signature.

Every `ResidualReturn` is one return SITE: an arm that returns simply never jumps to its local join. All
sites meet in a single exit block -- one phi per output row and per slot leaf when there are several -- so
`Ret` stays the sole function exit and each `StateSlot.live_out` is the value at that exit. The phi types
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


@dataclass(slots=True)
class _LoopSites:
    breaks: list[tuple[int, _Env]]
    continues: list[tuple[int, _Env]]


@dataclass(slots=True)
class _FrameSites:
    returns: list[tuple[int, tuple[int, ...]]]


@dataclass(frozen=True, slots=True)
class _Emit:
    """The invariants of one emission: the builder, the function, and the collectors every exit reports to."""

    builder: HirBuilder
    fn: EelFunction
    sites: list[_Site]
    loops: list[_LoopSites]
    frames: list[_FrameSites]


def emit(fn: EelFunction) -> Hir:
    builder = HirBuilder()
    builder.position_at(builder.block())
    env: _Env = {}
    for param in fn.params:
        assert param.stype is not None
        env[param.name] = builder.input(param.name, _TYPES[param.stype])
    sites: list[_Site] = []
    terminated = _block(_Emit(builder, fn, sites, [], []), fn.body, env)
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
            _meet(builder, [(site.block, site.outputs[row]) for site in sites], decl.path)
            for row, decl in enumerate(fn.outputs)
        ]
        slots = {
            name: _meet(builder, [(site.block, site.slots[name]) for site in sites], name) for name in sites[0].slots
        }
        _finish(builder, fn, outputs, slots)
    return builder.finish()


def _key_order(key: str | int) -> tuple[bool, str]:
    return isinstance(key, str), str(key)


def _meet(builder: HirBuilder, arms: list[tuple[int, int]], what: object) -> int:
    """One phi per differing value, none when the arms already agree; the types were fixed by the PE."""
    first = arms[0][1]
    if all(vid == first for _, vid in arms):
        return first
    merged_type = builder.type_of(first)
    assert all(builder.type_of(vid) == merged_type for _, vid in arms), f"the lanes joined at {what!r} disagree on type"
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


def _block(em: _Emit, stmts: tuple[Stmt, ...], env: _Env) -> bool:
    builder = em.builder
    pending_slots: dict[str, int] = {}
    for stmt in stmts:
        match stmt:
            case Assign(target=TempBind(index=index), value=value, stype=stype):
                assert stype is not None
                env[index] = _value(builder, value, env, stype)
            case If(cond=cond, then=then, orelse=orelse):
                if _branch(em, cond, then, orelse, env):
                    return True
            case ResidualWhile():
                _loop(em, stmt, env)
            case SlotWrite(slot=slot, value=value):
                pending_slots[slot_name(slot)] = _atom(builder, value, env)
            case ResidualReturn(values=values):
                assert set(pending_slots) == {slot_name(slot.slot) for slot in em.fn.slots}
                outputs = tuple(_atom(builder, atom, env) for atom in values)
                em.sites.append(_Site(builder.current_block, outputs, pending_slots))
                return True
            case ResidualBreak():
                assert em.loops, "a break terminator lies inside a residual loop body"
                em.loops[-1].breaks.append((builder.current_block, dict(env)))
                return True
            case ResidualContinue():
                assert em.loops, "a continue terminator lies inside a residual loop body"
                em.loops[-1].continues.append((builder.current_block, dict(env)))
                return True
            case ResidualFrame():
                _frame(em, stmt, env)
            case ResidualFrameReturn(values=values):
                assert em.frames, "a frame return lies inside a residual frame body"
                em.frames[-1].returns.append(
                    (builder.current_block, tuple(_atom(builder, atom, env) for atom in values))
                )
                return True
            case _:
                raise AssertionError(stmt)
    assert not pending_slots, "slot commits immediately precede their return site"
    return False


def _branch(em: _Emit, cond: Atom, then: tuple[Stmt, ...], orelse: tuple[Stmt, ...], env: _Env) -> bool:
    builder = em.builder
    then_block = builder.block()
    else_block = builder.block()
    builder.branch(_atom(builder, cond, env), then_block, else_block)

    builder.position_at(then_block)
    then_env = dict(env)
    then_done = _block(em, then, then_env)
    then_exit = builder.current_block

    builder.position_at(else_block)
    else_env = dict(env)
    else_done = _block(em, orelse, else_env)
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
    for key in sorted(set(then_env) & set(else_env), key=_key_order):
        arms = [(then_exit, then_env[key]), (else_exit, else_env[key])]
        env[key] = _meet(builder, arms, key)
    return False


def _loop(em: _Emit, stmt: ResidualWhile, env: _Env) -> None:
    """
    The genuine back-edge loop: phis live in the loop-entry block, whose predecessors are the preheader plus
    every back edge (the latch fall-through and each continue site, resolving the common back atom in its own
    environment). Break sites and the condition-false edge meet at the exit block over the header-end keys:
    every path through the body carries those forward, so a break site binds each of them, and the ones whose
    value ids differ are exactly the partial evaluator's exit join temps.
    """
    builder = em.builder
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
    terminated = _block(em, stmt.header, env)
    assert not terminated, "a residual loop header holds no exit"
    body_block = builder.block()
    exit_block = builder.block()
    header_exit = builder.current_block
    builder.branch(_atom(builder, stmt.cond, env), body_block, exit_block)
    post_header = dict(env)

    mine = _LoopSites([], [])
    em.loops.append(mine)
    builder.position_at(body_block)
    terminated = _block(em, stmt.body, env)
    em.loops.pop()
    back_sites: list[tuple[int, list[int]]] = []
    if terminated:
        assert mine.continues, "a body with no back edge was rejected by the partial evaluator"
    else:
        back_sites.append((builder.current_block, [_atom(builder, phi.back, env) for phi in stmt.phis]))
        builder.jump(loop_entry)
    for block, site_env in mine.continues:
        builder.position_at(block)
        back_sites.append((block, [_atom(builder, phi.back, site_env) for phi in stmt.phis]))
        builder.jump(loop_entry)
    for position, (phi_id, entry_id) in enumerate(zip(phi_ids, entry_ids, strict=True)):
        arms = [(pre_exit, entry_id)] + [(block, back_ids[position]) for block, back_ids in back_sites]
        builder.set_phi_arms(phi_id, arms)

    exit_preds: list[tuple[int, _Env]] = [(header_exit, post_header)]
    for block, site_env in mine.breaks:
        builder.position_at(block)
        builder.jump(exit_block)
        exit_preds.append((block, site_env))
    builder.position_at(exit_block)
    env.clear()
    for key in sorted(post_header, key=_key_order):
        arms = [(block, pred_env[key]) for block, pred_env in exit_preds]
        env[key] = _meet(builder, arms, key)


def _frame(em: _Emit, stmt: ResidualFrame, env: _Env) -> None:
    """
    The callee region whose return sites converge at one frame-exit block: each site jumps there and each
    row meets across the sites' carried atoms -- the root-site meet relocated. The environment continues
    from the frame entry plus the row bindings; callee-internal temps die with the region.
    """
    builder = em.builder
    entry_env = dict(env)
    mine = _FrameSites([])
    em.frames.append(mine)
    terminated = _block(em, stmt.body, env)
    em.frames.pop()
    assert terminated, "a frame body ends in a frame return on every path"
    assert mine.returns, "a frame holds at least one return site"
    exit_block = builder.block()
    for block, _ in mine.returns:
        builder.position_at(block)
        builder.jump(exit_block)
    builder.position_at(exit_block)
    env.clear()
    env.update(entry_env)
    for column, row in enumerate(stmt.rows):
        arms = [(block, values[column]) for block, values in mine.returns]
        met = _meet(builder, arms, row.index)
        assert builder.type_of(met) == _TYPES[row.stype]
        env[row.index] = met


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

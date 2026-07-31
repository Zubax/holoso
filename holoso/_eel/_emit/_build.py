"""
Sublayer 3: residual Eel -> HIR, mechanically. Every semantic decision was the partial evaluator's: operations
arrive as ``IntrinsicCall`` nodes carrying their HIR operator, every assignment is typed, and control is
structured branches only. The emitter keeps one environment of binding -> value id, forks it per branch arm,
and at the join keeps the intersection, phi-ing entries whose ids differ -- names bound on one side only were
already dropped (and any later read rejected) by the partial evaluator's definite-assignment rule. Parameters
become input ports in declaration order whether read or not: the module interface mirrors the signature.
"""

from ..._hir import BoolType, FloatType, Hir, HirBuilder, IntType, Operator, Type
from .._ir import *
from .._names import port_name

_TYPES: dict[ScalarType, Type] = {
    ScalarType.BOOL: BoolType(),
    ScalarType.INT: IntType(),
    ScalarType.FLOAT: FloatType(),
}

type _Env = dict[str | int, int]


def emit(fn: EelFunction) -> Hir:
    assert not fn.slots, "state slots land at M8"
    builder = HirBuilder()
    builder.position_at(builder.block())
    env: _Env = {}
    for param in fn.params:
        assert param.stype is not None
        env[param.name] = builder.input(param.name, _TYPES[param.stype])
    returned = _block(builder, fn, fn.body, env)
    assert returned, "the residual body always ends in a return"
    return builder.finish()


def _block(builder: HirBuilder, fn: EelFunction, stmts: tuple[Stmt, ...], env: _Env) -> bool:
    for stmt in stmts:
        match stmt:
            case Assign(target=TempBind(index=index), value=value, stype=stype):
                assert stype is not None
                env[index] = _value(builder, value, env, stype)
            case If(cond=cond, then=then, orelse=orelse):
                _branch(builder, fn, cond, then, orelse, env)
            case ResidualReturn(values=values):
                for decl, atom in zip(fn.outputs, values, strict=True):
                    value_id = _atom(builder, atom, env)
                    assert builder.type_of(value_id) == _TYPES[decl.stype]
                    builder.output(port_name(decl.path), value_id)
                builder.ret()
                return True
            case _:
                raise AssertionError(stmt)
    return False


def _branch(
    builder: HirBuilder,
    fn: EelFunction,
    cond: Atom,
    then: tuple[Stmt, ...],
    orelse: tuple[Stmt, ...],
    env: _Env,
) -> None:
    then_block = builder.block()
    else_block = builder.block()
    builder.branch(_atom(builder, cond, env), then_block, else_block)

    builder.position_at(then_block)
    then_env = dict(env)
    returned = _block(builder, fn, then, then_env)
    assert not returned
    then_exit = builder.current_block

    builder.position_at(else_block)
    else_env = dict(env)
    returned = _block(builder, fn, orelse, else_env)
    assert not returned
    else_exit = builder.current_block

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


def _value(builder: HirBuilder, value: Expr, env: _Env, stype: ScalarType) -> int:
    match value:
        case IntrinsicCall(operator=operator, args=args):
            assert isinstance(operator, Operator)
            value_id = builder.operation(operator, [_atom(builder, arg, env) for arg in args])
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

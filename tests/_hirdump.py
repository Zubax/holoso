"""
Complete, versioned HIR serializer for the golden corpus and the determinism witnesses.

Every field of :class:`holoso._hir.Hir` is spelled explicitly -- node kinds, operator mnemonics with their
parameters, types, terminators, phi arms, ordered inputs, named outputs, and state slots with their resets and
live-outs. Nothing falls back to a raw dataclass or enum repr, so the dump is stable against incidental repr
changes and any future field silently missing from the serializer trips an assertion instead of vanishing.
Floats are rendered as binary64 bit patterns (the authoritative spelling; the decimal repr rides along for
humans). Bump :data:`HIR_DUMP_SCHEMA` whenever the rendering changes shape.
"""

import struct
from dataclasses import fields
from typing import assert_never

from holoso._hir import (
    BoolConst,
    Branch,
    Const,
    FloatConst,
    FloatMulPow2,
    FloatRelational,
    Hir,
    InPort,
    IntConst,
    IntRelational,
    Jump,
    Node,
    Operation,
    Phi,
    Ret,
    StateRead,
    Terminator,
)
from holoso._hir import BoolType, FloatType, IntType
from holoso._hir._ir import Block, OutputPort, StateSlot
from holoso._hir._types import Type

HIR_DUMP_SCHEMA = "holoso-hir-dump/2"

# Completeness guard: a field added to any serialized dataclass must be spelled here (and the schema bumped),
# never silently dropped from the dump.
_SERIALIZED_FIELDS: dict[type, set[str]] = {
    Hir: {"nodes", "blocks", "input_ids", "outputs", "state_slots"},
    Block: {"id", "phis", "operations", "terminator"},
    OutputPort: {"name", "value"},
    StateSlot: {"name", "reset_value", "live_out"},
    InPort: {"name", "type"},
    Operation: {"operator", "operands"},
    StateRead: {"slot", "type"},
    Phi: {"type", "arms"},
    Jump: {"target"},
    Branch: {"cond", "if_true", "if_false"},
    Ret: set(),
    FloatConst: {"value"},
    BoolConst: {"value"},
    IntConst: {"value"},
    FloatMulPow2: {"k"},
    FloatRelational: {"op"},
    IntRelational: {"op"},
    FloatType: set(),
    BoolType: set(),
    IntType: set(),
}
for _cls, _expected in _SERIALIZED_FIELDS.items():
    assert {f.name for f in fields(_cls)} == _expected, f"{_cls.__name__} grew fields the HIR dump does not spell"


def _spell_type(ty: Type) -> str:
    match ty:
        case FloatType():
            return "float"
        case BoolType():
            return "bool"
        case IntType():
            return "int"
        case _:
            raise AssertionError(f"type {type(ty).__name__} has no dump spelling")


def _spell_float(value: float) -> str:
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    return f"0x{bits:016x} ({value!r})"


def _spell_int(value: int) -> str:
    # Folding admits integers up to 65536 bits, far past CPython's default int-to-decimal digit cap; wide values
    # render in hex, which the cap does not govern.
    return str(value) if value.bit_length() <= 64 else f"{value:#x}"


def _spell_const(const: Const) -> str:
    match const:
        case FloatConst(value=value):
            return f"const float {_spell_float(value)}"
        case BoolConst(value=value):
            return f"const bool {'true' if value else 'false'}"
        case IntConst(value=value):
            return f"const int {_spell_int(value)}"
        case _:
            raise AssertionError(f"constant {type(const).__name__} has no dump spelling")


def _spell_operator(operation: Operation) -> str:
    operator = operation.operator
    match operator:
        case FloatMulPow2(k=k):
            return f"{operator.mnemonic}[k={k}]"
        case FloatRelational(op=rel) | IntRelational(op=rel):
            return f"{operator.mnemonic}[{rel.value}]"
        case _:
            assert not fields(operator), f"operator {operator.mnemonic} carries parameters the dump does not spell"
            return operator.mnemonic


def _spell_node(node: Node) -> str:
    match node:
        case InPort(name=name, type=ty):
            return f"input {name}: {_spell_type(ty)}"
        case Const():
            return _spell_const(node)
        case Operation(operands=operands):
            args = ", ".join(f"v{operand}" for operand in operands)
            return f"op {_spell_operator(node)}({args}): {_spell_type(node.type)}"
        case StateRead(slot=slot, type=ty):
            return f"state_read {slot}: {_spell_type(ty)}"
        case Phi(type=ty, arms=arms):
            rendered = ", ".join(f"b{pred}: v{value}" for pred, value in arms)
            return f"phi [{rendered}]: {_spell_type(ty)}"
        case _:
            assert_never(node)


def _spell_terminator(terminator: Terminator) -> str:
    match terminator:
        case Jump(target=target):
            return f"jump b{target}"
        case Branch(cond=cond, if_true=if_true, if_false=if_false):
            return f"branch v{cond} ? b{if_true} : b{if_false}"
        case Ret():
            return "ret"
        case _:
            assert_never(terminator)


def _spell_block(block: Block) -> list[str]:
    return [
        f"block b{block.id}:",
        f"  phis: {' '.join(f'v{vid}' for vid in block.phis)}".rstrip(),
        f"  ops: {' '.join(f'v{vid}' for vid in block.operations)}".rstrip(),
        f"  term: {_spell_terminator(block.terminator)}",
    ]


def dump_hir(hir: Hir) -> str:
    """The complete canonical text of one HIR, opened by the schema-version header."""
    assert len(hir.nodes) == len(set(hir.nodes)) and all(isinstance(vid, int) for vid in hir.nodes)
    lines = [HIR_DUMP_SCHEMA]
    lines.append(f"inputs: {' '.join(f'v{vid}' for vid in hir.input_ids)}".rstrip())
    lines.append("nodes:")
    lines += [f"  v{vid} = {_spell_node(hir.nodes[vid])}" for vid in sorted(hir.nodes)]
    lines.append("blocks:")
    for block in hir.blocks:
        lines += [f"  {line}" for line in _spell_block(block)]
    lines.append("outputs:")
    lines += [f"  {port.name} = v{port.value}" for port in hir.outputs]
    lines.append("state_slots:")
    lines += [f"  {slot.name} = v{slot.live_out} reset {_spell_const(slot.reset_value)}" for slot in hir.state_slots]
    return "\n".join(lines) + "\n"

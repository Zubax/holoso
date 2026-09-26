"""
Register allocation of one bank, jointly with the orientation of every commutative firing (which read port each
operand takes) and, for a pooled class realized on several instances, with the instance each firing (one activation
of a pooled operator) binds, by one deterministic incremental annealer over an explicit interference graph.

Register sharing is decided entirely by the interference graph the caller supplies (built in `._liveness` from
per-block hardware-frame residence), so one `color` routine colors a straight-line block or a whole control-flow graph,
and either bank; of the timeline this module sees only each firing's busy window, which decides where it may be rebound.

The objective is the steering of the sparse register file the backend emits (its read and write multiplexers), plus a
price per register:

    sum over ports (|sources| - 1) + sum over registers (|writers| - 1) + register_price * open registers

A read port is one `(operator, instance, operand position)`; its sources, the arms of its read mux, are the registers
and the constant-pool words it reads (a constant is an arm like a register, an immovable pseudo-register). A register's
writers are the output lanes `(operator, instance, output port)` landing in it and the fixed producers the emitter
places beside them, keyed exactly as the emitter's write codebook keys them: an inline result by its operator and
resolved operands (two results of one expression over the same registers are one arm), a residual phi arm by the operand
it moves, and the handshake-gated input load and slot install each by itself. Those keys depend on the assignment, so a
value move re-keys every producer reading it. An open register is one above the pinned registers that holds at least one
value.

Orientation belongs to the same objective because a commutative firing may read its operands either way round:
swapping moves each operand from one port's mux to the other's and permutes the firing's output taps through the
operator's `swap_output_permutation`, a pure relabeling at zero latency (Chen & Cong, ASP-DAC 2004). Binding
belongs to it for the same reason: the instance a firing runs on decides which ports read its operands and which
lanes write its results, and the scheduler's first-free choice is only the seed. A firing may move to any instance of
its class where the other firings of its block leave its busy window free, so neither the instance count nor the
latency changes.

The search starts from the seed, a greedy allocation guided by port affinity, refined by simulated annealing over value
moves, instance moves, pair swaps and orientation flips with incremental cost deltas, then a first-improvement descent
over the complete move set, so the result is a local optimum of that neighborhood and never worse than the seed. Pinned
values (the i-th input port on register i, where the input load writes it; state live-ins and coalesced live-outs on
their slot registers) are fixed by the caller; everything else may reuse any occupied register the interference graph
allows except the reserved ones (a non-coalesced slot register belongs to its copy-back machinery). There is no
spilling: a value opens a new register when no other fits. The random number generator is `random.Random(0)` and every
iteration order is hash-independent, so the result is a function of the problem alone.
"""

import logging
import math
import os
import random
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

from .._operators import InlineHardwareOperator, PooledHardwareOperator, PortConditioner
from .._util import ValueId
from ._ir import BoolOperand, WideConstRef
from ._sources import InlineWriteSource, MoveWriteSource, OperandTemplate, WideOperandTemplate

_logger = logging.getLogger(__name__)

_TEMPERATURE_START = 2.0
_TEMPERATURE_END = 0.05

# The shares of proposals drawn as instance moves, pair swaps and orientation flips, each only while such decisions
# exist; the rest are value moves. Set by hand when each move kind was added and confirmed flat by a sweep over the
# example kernels, summing the allocator's objective: halving or doubling the flip share, or the two binding shares,
# moves the sum by under one percent, comparable to the half-percent spread between random seeds, while dropping all
# three, leaving flips and rebinds to the descent, costs about 2.5 percent. Not exposed through `RegallocTuning`: unlike
# `effort` and `register_price` they trade nothing a user can reason about.
# DO NOT DELETE the environment overrides: they are used for allocator tuning experiments.
_BIND_SHARE = float(os.getenv("HOLOSO_REGALLOC_BIND_SHARE", "0.25"))
_SWAP_SHARE = float(os.getenv("HOLOSO_REGALLOC_SWAP_SHARE", "0.10"))
_FLIP_SHARE = float(os.getenv("HOLOSO_REGALLOC_FLIP_SHARE", "0.12"))

_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class RegallocTuning:
    """
    `effort` is the search budget in proposals per decision (a movable value's register, a commutative firing's
    orientation, a bindable firing's instance); 0 keeps the greedy seed and its descent. `register_price` is what one
    freed register is worth in mux arms: the allocator minimizes `steering + register_price * registers`.
    """

    effort: int
    register_price: float


type _Source = ValueId | WideConstRef


class FixedProducer(ABC):
    """
    A writer of a register beside the pooled output lanes, keyed under an assignment as the emitter's write select
    keys it: `holes` are the values of this bank whose registers the key depends on, `substitute` renames them
    (to a coalescing class leader), `resolve` yields the key. A handshake-gated arm has no holes and is its own key.
    """

    @property
    @abstractmethod
    def holes(self) -> list[ValueId]: ...

    @abstractmethod
    def substitute(self, fn: Callable[[ValueId], ValueId]) -> "FixedProducer": ...

    @abstractmethod
    def resolve(self, register_of: Callable[[ValueId], int]) -> "_WriterKey": ...


class _StaticWriter(FixedProducer):
    """
    A writer keyed by itself, with no holes: a handshake-gated arm, which the emitter counts beside the opcode arms
    whatever operand it moves, or an early slot install, alone on its reserved register.
    """

    @property
    def holes(self) -> list[ValueId]:
        return []

    def substitute(self, fn: Callable[[ValueId], ValueId]) -> "_StaticWriter":
        return self

    def resolve(self, register_of: Callable[[ValueId], int]) -> "_StaticWriter":
        return self


@dataclass(frozen=True, slots=True)
class InputWriter(_StaticWriter):
    """The load of an input port into its register."""

    value: ValueId


@dataclass(frozen=True, slots=True)
class SlotWriter(_StaticWriter):
    """A state slot's install of a live-out that ends up in another register, into the slot register."""

    slot: str


@dataclass(frozen=True, slots=True)
class InlineWriter(FixedProducer):
    """An inline operator's combinational result; its boolean operands are resolved, that bank being colored first."""

    operator: InlineHardwareOperator
    operands: tuple[WideOperandTemplate | BoolOperand, ...]
    conditioner: PortConditioner

    @property
    def holes(self) -> list[ValueId]:
        return [t.hole for t in self.operands if isinstance(t, WideOperandTemplate) and t.hole is not None]

    def substitute(self, fn: Callable[[ValueId], ValueId]) -> "InlineWriter":
        operands = tuple(t.substitute(fn) if isinstance(t, WideOperandTemplate) else t for t in self.operands)
        return InlineWriter(self.operator, operands, self.conditioner)

    def resolve(self, register_of: Callable[[ValueId], int]) -> InlineWriteSource:
        operands = tuple(t.resolve(register_of) if isinstance(t, WideOperandTemplate) else t for t in self.operands)
        return InlineWriteSource(self.operator, operands, self.conditioner)


@dataclass(frozen=True, slots=True)
class MoveWriter(FixedProducer):
    """A residual (non-coalesced) phi arm's install copy."""

    source: OperandTemplate

    @property
    def holes(self) -> list[ValueId]:
        return [] if self.source.hole is None else [self.source.hole]

    def substitute(self, fn: Callable[[ValueId], ValueId]) -> "MoveWriter":
        return MoveWriter(self.source.substitute(fn))

    def resolve(self, register_of: Callable[[ValueId], int]) -> MoveWriteSource:
        return MoveWriteSource(self.source.resolve(register_of))


type _WriterKey = _StaticWriter | InlineWriteSource | MoveWriteSource


@dataclass(frozen=True, slots=True)
class Firing:
    """
    One pooled firing as the bank's objective sees it: the leader the build keys the orientation and the binding by,
    the operator, the block and block-local issue cycle, the instance the scheduler bound, its operand sources in source
    order (a value of this bank or a constant-pool word), and the tapped output ports landing in this bank with the
    value each writes. A firing tapping nothing into this
    bank (a comparator's boolean taps) still reads its wide operands.
    """

    leader: ValueId
    operator: PooledHardwareOperator
    block: int
    issue: int
    seed_instance: int
    reads: list[_Source]
    writes: list[tuple[int, ValueId]]

    @property
    def window(self) -> range:
        return range(self.issue, self.issue + self.operator.initiation_interval)


@dataclass(frozen=True, slots=True)
class ColoringProblem:
    """
    `movable` are the values to place (in a stable order); `pinned` fixes inputs and state live-ins to their
    registers; `interferes` is the symmetric adjacency over every value of the bank; `fixed_producers` names each
    value's writers other than pooled lanes, every hole a value of this bank; `reserved` are the registers no movable
    value may join; `fresh_start` is the first register index above the pinned registers; `firings` are the bank's
    pooled firings, whose reads and writes are keyed by the values here (a coalesced class appears as its leader, so a
    merged register is read and written by every member's port and lane); `instances` is the realized instance count
    per operator, the bound a firing may be rebound within. The boolean
    bank passes no firings, only its residual arms and slot installs, so its objective is nearly the register count.
    """

    movable: list[ValueId]
    pinned: dict[ValueId, int]
    interferes: dict[ValueId, set[ValueId]]
    fixed_producers: dict[ValueId, list[FixedProducer]]
    reserved: frozenset[int]
    fresh_start: int
    firings: list[Firing]
    instances: dict[PooledHardwareOperator, int]
    tuning: RegallocTuning


@dataclass(frozen=True, slots=True)
class Coloring:
    """
    The allocation: every value's register (labels compacted above `fresh_start` by first use), the register
    count, per firing leader whether the build reads its operands the other way round and the instance it binds
    (labels canonical by first use in block, cycle, leader order), and the objective's two steering terms as
    allocated.
    """

    assign: dict[ValueId, int]
    nreg: int
    swap: dict[ValueId, bool]
    instance: dict[ValueId, int]
    read_arms: int
    write_arms: int


# Named tuples, not dataclasses: these are the counter keys of the annealer's hot loop.
class _Port(NamedTuple):
    operator: PooledHardwareOperator
    instance: int
    position: int


class _Lane(NamedTuple):
    operator: PooledHardwareOperator
    instance: int
    port: int


class _InstanceSlot(NamedTuple):
    """One instance of a pooled class within one block: what a firing occupies for its busy window."""

    block: int
    operator: PooledHardwareOperator
    instance: int


type _Writer = _Lane | _WriterKey
type _ReadSource = int | WideConstRef


@dataclass(frozen=True, slots=True)
class _Snapshot:
    assign: dict[ValueId, int]
    flip: list[bool]
    instance: list[int]


class _State:
    """
    The allocation under search with its incremental objective: per port the sources it reaches, per register its
    writers (both with multiplicity, so a source leaves a mux only when its last read or write moves away), the
    register membership, and per instance the cycles its firings occupy. A fixed producer's writer key is its
    resolution under the current assignment, so a value move re-keys the producers reading it (`dependents`) as
    well as those writing it. Every move goes through the counters, so its cost delta is exact and the inverse move
    is its undo.
    """

    def __init__(self, problem: ColoringProblem, start: _Snapshot) -> None:
        self.problem = problem
        self.price = problem.tuning.register_price
        self.flip = list(start.flip)
        self.instance = list(start.instance)
        self.occupancy: dict[tuple[_InstanceSlot, int], int] = {}  # (slot, cycle) -> the firing busy there
        self.assign: dict[ValueId, int] = {}
        self.members: dict[int, set[ValueId]] = {}
        self.port_sources: dict[_Port, Counter[_ReadSource]] = {}
        self.reg_writers: dict[int, Counter[_Writer]] = {}
        self.writer_regs: dict[_Writer, Counter[int]] = {}
        self.read_arms = 0
        self.write_arms = 0
        self.open = 0
        self.readers: dict[ValueId, list[tuple[int, int]]] = {}  # value -> (firing index, operand position)
        self.producers: dict[ValueId, list[tuple[int, int]]] = {}  # value -> (firing index, output port)
        self.dependents: dict[ValueId, list[tuple[ValueId, FixedProducer]]] = {}  # hole -> (written value, producer)
        for i, firing in enumerate(problem.firings):
            for pos, source in enumerate(firing.reads):
                if not isinstance(source, WideConstRef):
                    self.readers.setdefault(source, []).append((i, pos))
            for port, value in firing.writes:
                self.producers.setdefault(value, []).append((i, port))
        for vid in sorted(problem.fixed_producers):
            producers = problem.fixed_producers[vid]
            assert len(set(producers)) == len(producers), f"value {vid} lists a producer twice"
            for producer in producers:
                for hole in producer.holes:
                    assert hole in problem.interferes, f"producer of {vid} depends on {hole}, not of this bank"
                    self.dependents.setdefault(hole, []).append((vid, producer))
        for vid, reg in start.assign.items():
            self._join(vid, reg)
        for vid in start.assign:
            self._produce(vid, 1)
        for i in range(len(problem.firings)):
            self._attach(i, 1)

    @property
    def cost(self) -> float:
        return self.read_arms + self.write_arms + self.price * self.open

    def snapshot(self) -> _Snapshot:
        return _Snapshot(dict(self.assign), list(self.flip), list(self.instance))

    def _read(self, port: _Port, source: _ReadSource, delta: int) -> None:
        sources = self.port_sources.setdefault(port, Counter())
        before = len(sources)
        count = sources[source] + delta
        assert count >= 0
        if count:
            sources[source] = count
        else:
            del sources[source]
        self.read_arms += max(0, len(sources) - 1) - max(0, before - 1)

    def _write(self, reg: int, writer: _Writer, delta: int) -> None:
        writers = self.reg_writers.setdefault(reg, Counter())
        before = len(writers)
        count = writers[writer] + delta
        assert count >= 0
        regs = self.writer_regs.setdefault(writer, Counter())
        if count:
            writers[writer] = count
            regs[reg] += delta
        else:
            del writers[writer]
            del regs[reg]
            if not regs:
                del self.writer_regs[writer]
        self.write_arms += max(0, len(writers) - 1) - max(0, before - 1)

    def _port(self, i: int, pos: int) -> _Port:
        firing = self.problem.firings[i]
        if self.flip[i]:
            pos = 1 - pos  # commutative firings have arity 2 (checked by `color`)
        return _Port(firing.operator, self.instance[i], pos)

    def _lane(self, i: int, port: int) -> _Lane:
        firing = self.problem.firings[i]
        if self.flip[i]:
            permutation = firing.operator.swap_output_permutation
            assert permutation is not None
            port = permutation[port]
        return _Lane(firing.operator, self.instance[i], port)

    def _slot(self, i: int, instance: int) -> _InstanceSlot:
        firing = self.problem.firings[i]
        return _InstanceSlot(firing.block, firing.operator, instance)

    def _read_source(self, source: _Source) -> _ReadSource:
        return source if isinstance(source, WideConstRef) else self.assign[source]

    def _attach(self, i: int, delta: int) -> None:
        firing = self.problem.firings[i]
        for pos, source in enumerate(firing.reads):
            self._read(self._port(i, pos), self._read_source(source), delta)
        for port, value in firing.writes:
            self._write(self.assign[value], self._lane(i, port), delta)
        slot = self._slot(i, self.instance[i])
        for cycle in firing.window:
            if delta > 0:
                assert (slot, cycle) not in self.occupancy, "two firings of one class busy on one instance"
                self.occupancy[(slot, cycle)] = i
            else:
                del self.occupancy[(slot, cycle)]

    def _join(self, vid: ValueId, reg: int) -> None:
        self.assign[vid] = reg
        members = self.members.setdefault(reg, set())
        if not members and reg >= self.problem.fresh_start:
            self.open += 1
        members.add(vid)

    def _leave(self, vid: ValueId) -> None:
        reg = self.assign.pop(vid)
        members = self.members[reg]
        members.discard(vid)
        if not members and reg >= self.problem.fresh_start:
            self.open -= 1

    def _resolve(self, producer: FixedProducer) -> _Writer:
        return producer.resolve(self.assign.__getitem__)

    def _produce(self, vid: ValueId, delta: int) -> None:
        for producer in self.problem.fixed_producers.get(vid, ()):
            self._write(self.assign[vid], self._resolve(producer), delta)

    def _rekeyed(self, vid: ValueId) -> list[tuple[ValueId, FixedProducer]]:
        """The producers whose key changes when `vid` moves: those it writes with and those reading it, once each."""
        own = [(vid, producer) for producer in self.problem.fixed_producers.get(vid, ())]
        return list(dict.fromkeys([*own, *self.dependents.get(vid, ())]))

    def _writers_of(self, vid: ValueId) -> list[_Writer]:
        lanes: list[_Writer] = [self._lane(i, port) for i, port in self.producers.get(vid, ())]
        return [*(self._resolve(p) for p in self.problem.fixed_producers.get(vid, ())), *lanes]

    def _touch(self, vid: ValueId, delta: int) -> None:
        """Count (or discount) every endpoint keyed by `vid`'s register: reads, lanes, the producers it re-keys."""
        reg = self.assign[vid]
        for i, pos in self.readers.get(vid, ()):
            self._read(self._port(i, pos), reg, delta)
        for i, port in self.producers.get(vid, ()):
            self._write(reg, self._lane(i, port), delta)
        for dst, producer in self._rekeyed(vid):
            self._write(self.assign[dst], self._resolve(producer), delta)

    def move_value(self, vid: ValueId, reg: int) -> None:
        self._touch(vid, -1)
        self._leave(vid)
        self._join(vid, reg)
        self._touch(vid, 1)

    def flip_firing(self, i: int) -> None:
        self._attach(i, -1)
        self.flip[i] = not self.flip[i]
        self._attach(i, 1)

    def instance_free(self, i: int, instance: int) -> bool:
        firing = self.problem.firings[i]
        slot = self._slot(i, instance)
        return all(self.occupancy.get((slot, cycle), i) == i for cycle in firing.window)

    def move_instance(self, i: int, instance: int) -> None:
        self._attach(i, -1)
        self.instance[i] = instance
        self._attach(i, 1)

    def swap_instances(self, i: int, j: int) -> bool:
        """Exchange two same-class firings' instances if both windows are free after the exchange; else no change."""
        assert self.problem.firings[i].operator == self.problem.firings[j].operator
        self._attach(i, -1)
        self._attach(j, -1)
        legal = self.instance_free(i, self.instance[j]) and self.instance_free(j, self.instance[i])
        if legal:
            self.instance[i], self.instance[j] = self.instance[j], self.instance[i]
        self._attach(i, 1)
        self._attach(j, 1)
        return legal

    def fits(self, vid: ValueId, reg: int) -> bool:
        if reg in self.problem.reserved:
            return False
        members = self.members.get(reg, set())
        assert members or reg >= self.problem.fresh_start
        return self.problem.interferes[vid].isdisjoint(members)

    def fresh(self) -> int:
        reg = self.problem.fresh_start
        while self.members.get(reg):
            reg += 1
        return reg

    def occupied(self) -> list[int]:
        return sorted(reg for reg, members in self.members.items() if members)

    def propose_register(self, vid: ValueId, rng: random.Random) -> int | None:
        """
        A target register for `vid` drawn from its affinities -- the registers its ports already reach, the ones its
        writers already drive -- or any occupied register, or a fresh one; None when the draw is infeasible (which
        consumes budget like any other proposal).
        """
        draw = rng.random()
        candidates: list[int]
        if draw < 0.45:
            candidates = [
                source
                for i, pos in self.readers.get(vid, ())
                for source in self.port_sources.get(self._port(i, pos), ())
                if isinstance(source, int)
            ]
        elif draw < 0.65:
            candidates = [reg for writer in self._writers_of(vid) for reg in self.writer_regs.get(writer, ())]
        elif draw < 0.9:
            candidates = [reg for reg, members in self.members.items() if members]
        else:
            candidates = [self.fresh()]
        old = self.assign[vid]
        candidates = [reg for reg in candidates if reg != old]
        if not candidates:
            return None
        reg = candidates[rng.randrange(len(candidates))]
        return reg if self.fits(vid, reg) else None


def _check_swappable(operator: PooledHardwareOperator) -> None:
    """A flip exchanges the two operands with their conditioners, which only a symmetric pair of ports can carry."""
    assert operator.signature.arity == 2, operator.mnemonic
    assert operator.conditions_operand(0) == operator.conditions_operand(1), operator.mnemonic
    assert (0 in operator.unconditioned_operands) == (1 in operator.unconditioned_operands), operator.mnemonic


type _SeedWriter = _Lane | FixedProducer  # a producer stands for itself before any register is assigned


def _seed_incidence(
    problem: ColoringProblem,
) -> tuple[dict[ValueId, set[_Port]], dict[ValueId, frozenset[_SeedWriter]]]:
    """Per value, the ports reading it and the writers driving it at the scheduler's binding and source orientation."""
    ports: dict[ValueId, set[_Port]] = {vid: set() for vid in problem.interferes}
    writers: dict[ValueId, set[_SeedWriter]] = {
        vid: set(problem.fixed_producers.get(vid, ())) for vid in problem.interferes
    }
    for firing in problem.firings:
        for pos, source in enumerate(firing.reads):
            if not isinstance(source, WideConstRef):
                ports[source].add(_Port(firing.operator, firing.seed_instance, pos))
        for port, value in firing.writes:
            writers[value].add(_Lane(firing.operator, firing.seed_instance, port))
    return ports, {vid: frozenset(w) for vid, w in writers.items()}


def _greedy(
    problem: ColoringProblem, ports_of: dict[ValueId, set[_Port]], writers_of: dict[ValueId, frozenset[_SeedWriter]]
) -> dict[ValueId, int]:
    """
    Port-affinity-biased graph coloring: each value takes the admissible register of least marginal mux growth, a
    fresh register being the fallback (a fixed producer counts as one writer of its own here; the exact,
    assignment-dependent keys are computed once the seed is placed).
    """
    assign: dict[ValueId, int] = {}
    reg_ports: dict[int, set[_Port]] = {}
    reg_writers: dict[int, set[_SeedWriter]] = {}
    reg_members: dict[int, set[ValueId]] = {}
    reached: set[_Port] = set()

    def place(vid: ValueId, reg: int) -> None:
        assign[vid] = reg
        ports = reg_ports.setdefault(reg, set())
        for port in ports_of[vid]:
            ports.add(port)
            reached.add(port)
        reg_writers.setdefault(reg, set()).update(writers_of[vid])
        reg_members.setdefault(reg, set()).add(vid)

    def marginal_cost(vid: ValueId, reg: int) -> int:
        ports: frozenset[_Port] | set[_Port] = reg_ports.get(reg, frozenset())
        writers: frozenset[_SeedWriter] | set[_SeedWriter] = reg_writers.get(reg, frozenset())
        read = sum(1 for port in ports_of[vid] if port not in ports and port in reached)
        merged = writers | writers_of[vid]
        return read + max(0, len(merged) - 1) - max(0, len(writers) - 1)

    def admissible(vid: ValueId, reg: int) -> bool:
        if reg in problem.reserved or reg not in reg_members:
            return False
        return problem.interferes[vid].isdisjoint(reg_members[reg])

    for vid, reg in sorted(problem.pinned.items(), key=lambda item: (item[1], item[0])):
        place(vid, reg)
    next_reg = problem.fresh_start
    for vid in problem.movable:
        best_reg = next_reg
        best_key = (marginal_cost(vid, next_reg), 1, next_reg)
        for reg in range(next_reg):
            if not admissible(vid, reg):
                continue
            key = (marginal_cost(vid, reg), 0, reg)
            if key < best_key:
                best_key, best_reg = key, reg
        if best_reg == next_reg:
            next_reg += 1
        place(vid, best_reg)
    return assign


@dataclass(frozen=True, slots=True)
class _Decisions:
    """What the search may change."""

    movable: list[ValueId]
    flippable: list[int]
    bindable: list[int]
    swappable: list[list[int]]  # the classes with at least two bindable firings

    @classmethod
    def of(cls, problem: ColoringProblem) -> "_Decisions":
        flippable = [i for i, firing in enumerate(problem.firings) if firing.operator.is_commutative]
        bindable = [i for i, firing in enumerate(problem.firings) if problem.instances[firing.operator] > 1]
        by_class: dict[PooledHardwareOperator, list[int]] = {}
        for i in bindable:
            by_class.setdefault(problem.firings[i].operator, []).append(i)
        return cls(problem.movable, flippable, bindable, [members for members in by_class.values() if len(members) > 1])

    @property
    def count(self) -> int:
        return len(self.movable) + len(self.flippable) + len(self.bindable)


def _unswap(state: _State, i: int, j: int) -> None:
    restored = state.swap_instances(i, j)
    assert restored


def _anneal(state: _State, rng: random.Random, proposals: int, decisions: _Decisions) -> _Snapshot:
    """
    Simulated annealing over the incremental state: each proposal is a value move drawn from the value's affinities,
    an instance move, a pair swap within a class, or an orientation flip, accepted by the Metropolis rule under
    geometric cooling; an infeasible draw spends its proposal. Returns the best allocation seen.
    """
    problem = state.problem
    best_cost = state.cost
    best = state.snapshot()
    bind_share = _BIND_SHARE if decisions.bindable else 0.0
    swap_share = _SWAP_SHARE if decisions.swappable else 0.0
    flip_share = _FLIP_SHARE if decisions.flippable else 0.0
    for step in range(proposals):
        temperature = _TEMPERATURE_START * (_TEMPERATURE_END / _TEMPERATURE_START) ** (step / proposals)
        before = state.cost
        undo: Callable[[], None]
        draw = rng.random()
        if draw < bind_share:
            i = decisions.bindable[rng.randrange(len(decisions.bindable))]
            target = rng.randrange(problem.instances[problem.firings[i].operator])
            if target == state.instance[i] or not state.instance_free(i, target):
                continue
            undo = partial(state.move_instance, i, state.instance[i])
            state.move_instance(i, target)
        elif draw < bind_share + swap_share:
            members = decisions.swappable[rng.randrange(len(decisions.swappable))]
            i, j = rng.sample(members, 2)
            if state.instance[i] == state.instance[j] or not state.swap_instances(i, j):
                continue
            undo = partial(_unswap, state, i, j)
        elif draw < bind_share + swap_share + flip_share:
            i = decisions.flippable[rng.randrange(len(decisions.flippable))]
            state.flip_firing(i)
            undo = partial(state.flip_firing, i)
        else:
            if not decisions.movable:
                continue
            vid = decisions.movable[rng.randrange(len(decisions.movable))]
            reg = state.propose_register(vid, rng)
            if reg is None:
                continue
            undo = partial(state.move_value, vid, state.assign[vid])
            state.move_value(vid, reg)
        delta = state.cost - before
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            if state.cost < best_cost - _EPSILON:
                best_cost = state.cost
                best = state.snapshot()
        else:
            undo()
    return best


def _descend(state: _State, decisions: _Decisions) -> int:
    """
    First-improvement descent over the complete move set -- every movable value against every occupied non-reserved
    register and one fresh register, every bindable firing against every free instance of its class, every legal pair
    swap, every flip -- until no move improves, so the result is a local optimum of the annealer's neighborhood.
    Returns the number of sweeps.
    """
    problem = state.problem
    sweeps = 0
    improved = True
    while improved:
        improved = False
        sweeps += 1
        for vid in decisions.movable:
            old = state.assign[vid]
            best_reg, best_cost = None, state.cost
            for reg in [*state.occupied(), state.fresh()]:
                if reg == old or not state.fits(vid, reg):
                    continue
                state.move_value(vid, reg)
                if state.cost < best_cost - _EPSILON:
                    best_reg, best_cost = reg, state.cost
                state.move_value(vid, old)
            if best_reg is not None:
                state.move_value(vid, best_reg)
                improved = True
        for i in decisions.bindable:
            old = state.instance[i]
            best_instance, best_cost = None, state.cost
            for instance in range(problem.instances[problem.firings[i].operator]):
                if instance == old or not state.instance_free(i, instance):
                    continue
                state.move_instance(i, instance)
                if state.cost < best_cost - _EPSILON:
                    best_instance, best_cost = instance, state.cost
                state.move_instance(i, old)
            if best_instance is not None:
                state.move_instance(i, best_instance)
                improved = True
        for members in decisions.swappable:
            for a, i in enumerate(members):
                for j in members[a + 1 :]:
                    if state.instance[i] == state.instance[j]:
                        continue
                    before = state.cost
                    if not state.swap_instances(i, j):
                        continue
                    if state.cost < before - _EPSILON:
                        improved = True
                    else:
                        _unswap(state, i, j)
        for i in decisions.flippable:
            before = state.cost
            state.flip_firing(i)
            if state.cost < before - _EPSILON:
                improved = True
            else:
                state.flip_firing(i)
    return sweeps


def _compact(problem: ColoringProblem, assign: dict[ValueId, int]) -> tuple[dict[ValueId, int], int]:
    """Relabel the open registers by first use (pinned values first, then the movable order); pins keep their index."""
    relabel: dict[int, int] = {}
    next_reg = problem.fresh_start
    compacted: dict[ValueId, int] = {}
    for vid in [*sorted(problem.pinned), *problem.movable]:
        reg = assign[vid]
        if reg < problem.fresh_start:
            compacted[vid] = reg
            continue
        if reg not in relabel:
            relabel[reg] = next_reg
            next_reg += 1
        compacted[vid] = relabel[reg]
    return compacted, max(next_reg, problem.fresh_start)


def _canonical_instances(problem: ColoringProblem, instance: list[int]) -> dict[ValueId, int]:
    """
    Per firing leader, its instance relabeled per class by first use in block, cycle, leader order, so the labels
    are dense and every labeled instance is used.
    """
    labels: dict[tuple[PooledHardwareOperator, int], int] = {}
    used: Counter[PooledHardwareOperator] = Counter()
    order = sorted(range(len(problem.firings)), key=lambda i: (problem.firings[i].block, problem.firings[i].issue, i))
    for i in order:
        key = (problem.firings[i].operator, instance[i])
        if key not in labels:
            labels[key] = used[key[0]]
            used[key[0]] += 1
    assert all(used[operator] <= count for operator, count in problem.instances.items())
    return {problem.firings[i].leader: labels[(problem.firings[i].operator, instance[i])] for i in order}


def color(problem: ColoringProblem) -> Coloring:
    """Allocate one bank."""
    for operator in {firing.operator for firing in problem.firings if firing.operator.is_commutative}:
        _check_swappable(operator)
    ports_of, writers_of = _seed_incidence(problem)
    seed = _Snapshot(
        _greedy(problem, ports_of, writers_of),
        [False] * len(problem.firings),
        [firing.seed_instance for firing in problem.firings],
    )
    state = _State(problem, seed)
    seed_cost = state.cost
    decisions = _Decisions.of(problem)
    proposals = problem.tuning.effort * decisions.count
    if proposals > 0:
        state = _State(problem, _anneal(state, random.Random(0), proposals, decisions))
    sweeps = _descend(state, decisions)
    assert state.cost <= seed_cost + _EPSILON
    # Window disjointness per instance is asserted by `_attach` on every move.
    assert all(state.instance_free(i, state.instance[i]) for i in range(len(problem.firings)))
    assert all(state.assign[vid] not in problem.reserved for vid in problem.movable)
    compacted, nreg = _compact(problem, state.assign)
    _logger.info(
        "Register allocation: values=%d pinned=%d firings=%d commutative=%d bindable=%d proposals=%d; objective "
        "%g -> %g (read arms %d, write arms %d, open registers %d) after %d descent sweeps",
        len(problem.interferes),
        len(problem.pinned),
        len(problem.firings),
        len(decisions.flippable),
        len(decisions.bindable),
        proposals,
        seed_cost,
        state.cost,
        state.read_arms,
        state.write_arms,
        state.open,
        sweeps,
    )
    swap = {firing.leader: state.flip[i] for i, firing in enumerate(problem.firings)}
    bound = _canonical_instances(problem, state.instance)
    return Coloring(compacted, nreg, swap, bound, state.read_arms, state.write_arms)

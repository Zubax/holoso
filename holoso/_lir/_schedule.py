"""
Software-pipelined (zero-bubble) list scheduling over selected MIR.

The hardware operators' latencies are static and data-independent, so the whole schedule is computed here at compile
time and the backend controller is just a cycle counter replaying it. The scheduling unit is the FIRING: operations
sharing one block, primitive, operands, and operand conditioners while tapping distinct output ports fuse into a
single firing (a multi-output module computes all its results at once), so each member value gets the same issue
cycle and bound instance. Pooled primitives contend for their operator's physical instances through per-instance busy
windows; inline primitives are independent gates.

Every firing issues from cycle 0, reclaiming each block's first control word: inputs and other block-resident operands
load into the register array before that word reaches the datapath, so a consumer reads them from the first cycle.
The entry block is no different -- the sequencer dwells at pc 0 while awaiting `in_valid` and re-fetches `ucode[0]`
each idle cycle, but the backend gates every operator's `in_valid` with `transacting` (see `_emit`), so that
re-fetched word commits nothing and a cycle-0 issue is safe regardless of operator kind.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from .._util import ValueId
from .._mir import MirNode, MirOperation
from .._operators import Drive, HardwareOperator, PooledPrimitive, PortConditioner, Primitive
from ._ir import READ_FIRST_EDGE, OperatorInstance, read_cycle

_logger = logging.getLogger(__name__)

# A pooled firing's fusion identity: the drive (the operator and its immediates), operand values, and their conditioners
# -- everything the module activation consumes. Primitives, hence output ports, and inversions are excluded (members
# differ there).
type _FiringKey = tuple[Drive, tuple[ValueId, ...], tuple[PortConditioner, ...]]


@dataclass(frozen=True, slots=True)
class Schedule:
    """
    The scheduler's output: per-value issue cycle, per pooled firing leader the bound instance and the members, and
    the makespan. Members of one firing share an issue cycle, an instance, and a leader (the smallest member id); the
    LIR build collapses each leader group into one scheduled op with one write per member. The binding is the
    first-free one and only a SEED, the binding the register allocator starts from: it rebinds firings among the
    realized instances.
    """

    issue_cycle: dict[ValueId, int]
    inst_of: dict[ValueId, OperatorInstance]  # pooled firing leader -> the instance it binds
    firings: dict[ValueId, list[ValueId]]  # pooled firing leader -> its members
    makespan: int  # max commit cycle (issue_cycle + latency), or 0 if there are no ops
    # Per scheduled value, its primitive's latency, so the commit cycle (issue + latency) has a single owner here rather
    # than being recomputed by every consumer of the schedule.
    latency: dict[ValueId, int]

    def commit_cycle(self, vid: ValueId) -> int:
        return self.issue_cycle[vid] + self.latency[vid]


def _op(nodes: dict[ValueId, MirNode], vid: ValueId) -> MirOperation:
    node = nodes[vid]
    assert isinstance(node, MirOperation)
    return node


def _scheduled_operands(nodes: dict[ValueId, MirNode], vid: ValueId, schedulable: set[ValueId]) -> list[ValueId]:
    """Operand values scheduled alongside `vid` (same block); all other operands are resident at block start."""
    return [operand for operand in _op(nodes, vid).operands if operand in schedulable]


def _port(nodes: dict[ValueId, MirNode], vid: ValueId) -> str:
    primitive = _op(nodes, vid).primitive
    assert isinstance(primitive, PooledPrimitive)
    return primitive.output_name


def _fuse_block_firings(nodes: dict[ValueId, MirNode], schedulable: set[ValueId]) -> dict[ValueId, list[ValueId]]:
    """
    Group one block's pooled operations into firings: leader (smallest member id) -> members.
    Operations fuse when they share the drive, operands, and operand conditioners while tapping DISTINCT output ports
    -- one module firing computes them all. Two taps of the same port (e.g. a flag and its inversion) do not fuse:
    each output-port lane writes once per firing, so they become separate firings serialized by instance
    contention. Inline operations enter the result directly as singleton firings -- they never group.
    """
    firings: dict[ValueId, list[ValueId]] = {}
    by_key: dict[_FiringKey, list[ValueId]] = {}
    for vid in sorted(schedulable):
        node = _op(nodes, vid)
        if not isinstance(node.primitive, PooledPrimitive):
            firings[vid] = [vid]
            continue
        key: _FiringKey = (node.primitive.drive, node.operands, node.operand_conditioners)
        by_key.setdefault(key, []).append(vid)
    for members in by_key.values():
        open_groups: list[tuple[set[str], list[ValueId]]] = []  # (ports taken, members) per firing being assembled
        for vid in members:  # ascending id order keeps the grouping deterministic
            port = _port(nodes, vid)
            group = next((g for g in open_groups if port not in g[0]), None)
            if group is None:
                group = (set(), [])
                open_groups.append(group)
            group[0].add(port)
            group[1].append(vid)
        for _ports, group_members in open_groups:
            firings[min(group_members)] = group_members
    return firings


def _occupancy(primitive: Primitive) -> int:
    if isinstance(primitive, PooledPrimitive):
        return max(primitive.latency, primitive.drive.timing.initiation_interval)
    return 1


def _critical_path(
    nodes: dict[ValueId, MirNode], op_ids: list[ValueId], schedulable: set[ValueId]
) -> dict[ValueId, int]:
    """Priority height: longest latency-weighted path to a sink, counting the read-first edge per dependence."""
    consumers: dict[ValueId, list[ValueId]] = {vid: [] for vid in op_ids}
    for vid in op_ids:
        for operand in _scheduled_operands(nodes, vid, schedulable):
            consumers[operand].append(vid)
    # Sinks first, then each operation once all its consumers have a height.
    waiting = {vid: len(consumers[vid]) for vid in op_ids}
    settled = [vid for vid in op_ids if not waiting[vid]]
    height: dict[ValueId, int] = {}
    while settled:
        vid = settled.pop()
        height[vid] = _op(nodes, vid).primitive.latency + max(
            (READ_FIRST_EDGE + height[c] for c in consumers[vid]),
            default=0,
        )
        for operand in _scheduled_operands(nodes, vid, schedulable):
            waiting[operand] -= 1
            if not waiting[operand]:
                settled.append(operand)
    assert len(height) == len(op_ids), "a block's dependences are acyclic"
    return height


def schedule_ops(
    nodes: dict[ValueId, MirNode],
    schedulable: set[ValueId],
    fetch_lag: int,
    livein_landing: Mapping[ValueId, int],
) -> Schedule:
    """
    Place every firing of `schedulable` (one block's operations, across both register banks) on the earliest cycle
    its operands are ready and -- for a pooled firing -- a free instance exists. A single dependency-aware pass spans
    wide and boolean-result operations, so cross-bank chains (a value feeding a comparison feeding a cast) schedule
    correctly without a barrier. Operands outside `schedulable` are block live-ins; under per-block draining each is
    resident at the block start (a prior block's drained result, a state read, an input, or a phi), but a predecessor
    result spilled past an OVERLAPPED boundary lands mid-block instead -- `livein_landing` carries its block-local
    landing cycle, and a consumer's operand read must not precede it. A pooled instance accepts its next firing once
    the previous firing's `initiation_interval` has elapsed, and is free again by the time any successor frame begins
    (see `HardwareOperator`). The operator's `instances` option is a CAP, not a count: the schedule binds only the
    copies it can use.
    """
    op_ids = sorted(schedulable)
    if not op_ids:
        return Schedule(
            issue_cycle={},
            inst_of={},
            firings={},
            makespan=0,
            latency={},
        )
    schedulable_set = set(op_ids)

    firings = _fuse_block_firings(nodes, schedulable_set)
    height = _critical_path(nodes, op_ids, schedulable_set)
    issue_cycle: dict[ValueId, int] = {}
    inst_of: dict[ValueId, OperatorInstance] = {}
    busy_until: dict[tuple[HardwareOperator, int], int] = {}  # instance slot -> first cycle it is free again

    def commit_cycle(vid: ValueId) -> int:
        return issue_cycle[vid] + _op(nodes, vid).primitive.latency

    def is_ready(leader: ValueId, cycle: int) -> bool:
        # A consumer may issue only `READ_FIRST_EDGE` cycles after a same-block producer commits (see `read_cycle`).
        # Every other operand -- a state read, an input, a phi, or a result drained in from a prior block -- is resident
        # at the block start (constants are static nets with no read constraint), so nothing else delays the firing.
        # Members of one firing share operands and conditioners, so the leader's readiness is the whole firing's.
        for operand in _op(nodes, leader).operands:
            if operand in schedulable_set:
                if operand not in issue_cycle:
                    return False
                if cycle < commit_cycle(operand) + READ_FIRST_EDGE:
                    return False
            elif operand in livein_landing:
                # A predecessor result spilled past an overlapped boundary lands mid-block; the consumer's operand read
                # must not precede its block-local landing cycle.
                if read_cycle(cycle, fetch_lag) < livein_landing[operand]:
                    return False
        return True

    unscheduled = set(firings)
    # A cycle that issues nothing waits on a producer's commit, an instance's busy window or a spilled-in landing, so
    # serializing every firing behind all of them bounds the last issue.
    bound = max(livein_landing.values(), default=0) + sum(
        _occupancy(_op(nodes, leader).primitive) + READ_FIRST_EDGE for leader in firings
    )
    for cycle in range(bound + 1):
        ready = sorted(
            (leader for leader in unscheduled if is_ready(leader, cycle)),
            key=lambda leader: (-max(height[member] for member in firings[leader]), leader),
        )
        for leader in ready:
            primitive = _op(nodes, leader).primitive
            if isinstance(primitive, PooledPrimitive):
                # A pooled firing binds the first instance of its operator whose busy window has elapsed; none free ->
                # next cycle.
                operator = primitive.operator
                slot = next((s for s in range(operator.instances) if busy_until.get((operator, s), 0) <= cycle), None)
                if slot is None:
                    continue
                busy_until[(operator, slot)] = cycle + primitive.drive.timing.initiation_interval
                inst_of[leader] = OperatorInstance(operator, slot)
            for member in firings[leader]:
                issue_cycle[member] = cycle
            unscheduled.discard(leader)
        if not unscheduled:
            break
    else:
        assert False, "a firing waited past every wait the block adds up to"

    pooled_firings = {leader: members for leader, members in firings.items() if leader in inst_of}
    latency = {vid: _op(nodes, vid).primitive.latency for vid in op_ids}
    makespan = max((commit_cycle(vid) for vid in op_ids), default=0)
    _logger.debug(
        "Schedule: %d firings (%d answering several results), makespan %d, %d live-ins in flight",
        len(firings),
        sum(len(members) > 1 for members in pooled_firings.values()),
        makespan,
        len(livein_landing),
    )
    return Schedule(
        issue_cycle=issue_cycle,
        inst_of=inst_of,
        firings=pooled_firings,
        makespan=makespan,
        latency=latency,
    )

"""
Software-pipelined (zero-bubble) list scheduling over selected MIR.

The hardware operators' latencies are static and data-independent, so the whole schedule is computed here at compile
time and the backend controller is just a cycle counter replaying it. The scheduling unit is the FIRING: operations
sharing one block, operator, operands, and operand conditioners while tapping distinct output ports fuse into a
single firing (a multi-output module computes all its results at once), so each member value gets the same issue
cycle and bound instance. Pooled operators contend for physical instances through per-instance busy windows (an
instance accepts a new firing every ``initiation_interval`` cycles); inline operators are independent gates.

An operation issues from cycle 0, reclaiming a block's first control word (inputs and other block-resident operands
load into the register array before that word reaches the datapath, so a consumer reads them from the first cycle).
Two constraints raise an op's earliest issue to cycle 1:
  - a POOLED (instance-backed) op presents its read address one step early (``rci = issue - 1`` in the emitter), so
    issuing at cycle 0 would place the read-address word before the block -- a hard hardware constraint;
  - an ENTRY-block op producing a persistent-state live-out is dwell-guarded off the first control word as defense-in-
    depth: the sequencer holds pc 0 during the accept wait and re-fires ``ucode[0]`` each idle cycle, and re-firing a
    cycle-0 STATE write would corrupt the carried state. That write cannot occur today (an inline producer writes a
    temporary, never the state register; see ``_assert_entry_dwell_safe``), so this floor is cost-free on every
    current kernel, but it keeps the producer off cycle 0 cheaply since the dwell is invisible to cosim and the model.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .._util import ValueId
from .._mir import MirNode, MirOperation
from .._operators import HardwareOperator, PooledHardwareOperator, PortConditioner
from ._ir import OperatorInstance, dependency_edge, landing_cycle, operand_read_cycle, pooled_wide_read_cycle

# A pooled firing's fusion identity: the operator, its operand values, and their conditioners -- everything the
# module activation consumes. Output ports and output conditioners are deliberately excluded (members differ there).
type _FiringKey = tuple[PooledHardwareOperator, tuple[ValueId, ...], tuple[PortConditioner, ...]]

# The largest commit-to-issue dependency edge any (producer, consumer) pair can require: the result landing (fetch lag +
# read-first edge) against the pooled read latch, derived from the same helpers as dependency_edge. Exact only while
# producer->pooled remains the maximal pair; it merely pads a generously-slack progress cap, so a future larger pair
# costs nothing worse than a later no-progress diagnosis.
_MAX_DEPENDENCY_EDGE = landing_cycle(0) - pooled_wide_read_cycle(0)


@dataclass(frozen=True, slots=True)
class Schedule:
    """
    The scheduler's output: per-value issue cycle, the bound instance and firing leader for pooled values, the full
    instance set, and the makespan. Members of one firing share an issue cycle, an instance, and a leader (the
    smallest member id); the LIR build collapses each leader group into one scheduled op with one write per member.
    """

    issue_cycle: dict[ValueId, int]
    inst_of: dict[ValueId, OperatorInstance]
    firings: dict[ValueId, list[ValueId]]  # pooled firing leader -> its members, sorted by output port
    instances: list[OperatorInstance]
    makespan: int  # max commit cycle (issue_cycle + latency), or 0 if there are no ops
    # Per scheduled value, its operator latency, so the commit cycle (issue + latency) has a single owner here rather
    # than being recomputed by every consumer of the schedule.
    latency: dict[ValueId, int]
    # Per pooled instance slot, the first cycle it is free again (last firing's issue + initiation_interval). An
    # overlapping successor inherits this residue as its ``entry_busy`` so a firing bound to the same physical slot
    # waits out the predecessor's in-flight activation instead of double-driving it across the overlapped boundary.
    busy_until: dict[tuple[PooledHardwareOperator, int], int]

    def commit_cycle(self, vid: ValueId) -> int:
        return self.issue_cycle[vid] + self.latency[vid]


def _op(nodes: dict[ValueId, MirNode], vid: ValueId) -> MirOperation:
    node = nodes[vid]
    assert isinstance(node, MirOperation)
    return node


def _operator_operands(nodes: dict[ValueId, MirNode], vid: ValueId, schedulable: set[ValueId]) -> list[ValueId]:
    """Operand values scheduled alongside ``vid`` (same block); all other operands are resident at block start."""
    return [operand for operand in _op(nodes, vid).operands if operand in schedulable]


def resolve_pool(nodes: dict[ValueId, MirNode]) -> dict[type[HardwareOperator], int]:
    """
    The per-class instance budget over the FULL node table: at least one of every pooled operator class present in
    the graph, whichever bank its taps land in (a comparator whose every tap is boolean still needs its instance).

    The budget is applied per distinct hardware operator, so ``fmul_ilog2_const`` gets the requested number of
    instances for each distinct ``K``. Only pooled (module-backed) operators are budgeted; inline operators carry no
    physical instance.
    """
    pool: dict[type[HardwareOperator], int] = {}
    for node in nodes.values():
        if isinstance(node, MirOperation) and isinstance(node.operator, PooledHardwareOperator):
            requested = 1  # TODO: we can add heuristics for determining how many operator instances to use.
            pool[type(node.operator)] = max(1, requested)
    return pool


def fuse_block_firings(nodes: dict[ValueId, MirNode], schedulable: set[ValueId]) -> dict[ValueId, list[ValueId]]:
    """
    Group one block's pooled operations into firings: leader (smallest member id) -> members sorted by output port.
    Operations fuse when they share the operator, operands, and operand conditioners while tapping DISTINCT output
    ports -- one module firing computes them all. Two taps of the same port (e.g. a flag and its inversion) do not
    fuse: each output-port lane writes once per firing, so they become separate firings serialized by instance
    contention. Inline operations enter the result directly as singleton firings -- they never group.
    """
    firings: dict[ValueId, list[ValueId]] = {}
    by_key: dict[_FiringKey, list[ValueId]] = {}
    for vid in sorted(schedulable):
        node = _op(nodes, vid)
        if not isinstance(node.operator, PooledHardwareOperator):
            firings[vid] = [vid]
            continue
        key: _FiringKey = (node.operator, tuple(node.operands), tuple(node.operand_conditioners))
        by_key.setdefault(key, []).append(vid)
    for members in by_key.values():
        open_groups: list[tuple[set[int], list[ValueId]]] = []  # (ports taken, members) per firing being assembled
        for vid in members:  # ascending id order keeps the grouping deterministic
            port = _op(nodes, vid).output_port
            group = next((g for g in open_groups if port not in g[0]), None)
            if group is None:
                group = (set(), [])
                open_groups.append(group)
            group[0].add(port)
            group[1].append(vid)
        for _ports, group_members in open_groups:
            group_members.sort(key=lambda member: _op(nodes, member).output_port)
            firings[min(group_members)] = group_members
    return firings


def _critical_path(
    nodes: dict[ValueId, MirNode], op_ids: list[ValueId], schedulable: set[ValueId]
) -> dict[ValueId, int]:
    """Priority height: longest latency-weighted path to a sink, counting the per-pair dependency edge per edge."""
    consumers: dict[ValueId, list[ValueId]] = {vid: [] for vid in op_ids}
    for vid in op_ids:
        for operand in _operator_operands(nodes, vid, schedulable):
            consumers[operand].append(vid)
    height: dict[ValueId, int] = {}
    for vid in sorted(op_ids, reverse=True):  # consumers have larger IDs; process them first
        node = _op(nodes, vid)
        height[vid] = node.operator.latency + max(
            (
                dependency_edge(node.operator, node.output_port, _op(nodes, c).operator) + height[c]
                for c in consumers[vid]
            ),
            default=0,
        )
    return height


def schedule_ops(
    nodes: dict[ValueId, MirNode],
    pool: Mapping[type[HardwareOperator], int],
    schedulable: set[ValueId],
    entry_busy: Mapping[tuple[PooledHardwareOperator, int], int] | None = None,
    livein_landing: Mapping[ValueId, int] | None = None,
    dwell_guarded: frozenset[ValueId] = frozenset(),
) -> Schedule:
    """
    Place every firing of ``schedulable`` (one block's operations, across both register banks) on the earliest cycle
    its operands are ready and -- for a pooled firing -- a free instance exists. A single dependency-aware pass spans
    float and boolean-result operations, so cross-bank chains (a value feeding a comparison feeding a cast) schedule
    correctly without a barrier. Operands outside ``schedulable`` are block live-ins; under per-block draining each is
    resident at the block start (a prior block's drained result, a state read, an input, or a phi), but a predecessor
    result spilled past an OVERLAPPED boundary lands mid-block instead -- ``livein_landing`` carries its block-local
    landing cycle, and a consumer's operand read must not precede it. A pooled instance accepts a new firing every
    ``initiation_interval`` cycles; ``entry_busy`` seeds each instance's busy window with the residue inherited from an
    overlapping predecessor (empty under draining, where an instance is necessarily idle by the boundary for every
    operator whose initiation interval stays within ``OperatorInstance.__post_init__``'s bound). Both carries are empty
    for a fully-drained block, leaving the schedule identical to an isolated per-block pass.
    """
    entry_busy = entry_busy or {}
    livein_landing = livein_landing or {}
    op_ids = sorted(schedulable)
    if not op_ids:
        return Schedule(
            issue_cycle={}, inst_of={}, firings={}, instances=[], makespan=0, latency={}, busy_until=dict(entry_busy)
        )
    schedulable_set = set(op_ids)

    firings = fuse_block_firings(nodes, schedulable_set)
    height = _critical_path(nodes, op_ids, schedulable_set)
    issue_cycle: dict[ValueId, int] = {}
    inst_count: dict[PooledHardwareOperator, int] = {}
    slot_of: dict[ValueId, tuple[PooledHardwareOperator, int]] = {}  # per firing leader
    # instance slot -> first cycle it is free again, seeded with the busy residue inherited from overlapping
    # predecessors
    busy_until: dict[tuple[PooledHardwareOperator, int], int] = dict(entry_busy)

    def commit_cycle(vid: ValueId) -> int:
        return issue_cycle[vid] + _op(nodes, vid).operator.latency

    def is_ready(leader: ValueId, cycle: int) -> bool:
        # A consumer may issue only ``dependency_edge`` cycles after a same-block operator producer commits -- the
        # edge derives from the producer's result landing and the consumer's read mechanism (see _ir). Every
        # other operand -- a state read, an input, a phi, or a result drained in from a prior block -- is resident at
        # the block start (constants are immediates with no read constraint), so the per-firing cycle-1 floor below is
        # all that delays it. Members of one firing share operands and conditioners, so the leader's readiness is the
        # firing's, and any member being state-live-out dwell-guards the whole firing.
        consumer = _op(nodes, leader).operator
        if cycle < 1 and (
            isinstance(consumer, PooledHardwareOperator) or any(member in dwell_guarded for member in firings[leader])
        ):
            return False
        for operand in _op(nodes, leader).operands:
            if operand in schedulable_set:
                if operand not in issue_cycle:
                    return False
                producer = _op(nodes, operand)
                if cycle < commit_cycle(operand) + dependency_edge(producer.operator, producer.output_port, consumer):
                    return False
            elif operand in livein_landing:
                # A predecessor result spilled past an overlapped boundary lands mid-block; the consumer's operand read
                # must not precede its block-local landing cycle (the read mechanism dispatches per consumer class).
                if operand_read_cycle(consumer, cycle) < livein_landing[operand]:
                    return False
        return True

    unscheduled = set(firings)
    # The progress cap charges each firing the larger of its latency and its busy window: N firings contending for
    # one instance at initiation interval K legitimately need ~N*K cycles before the last one issues.
    cap = (
        sum(max(_op(nodes, vid).operator.latency, _op(nodes, vid).operator.initiation_interval) for vid in op_ids)
        + _MAX_DEPENDENCY_EDGE * len(op_ids)
        + 64
    )
    cycle = 0
    while unscheduled:
        if cycle > cap:
            raise RuntimeError("scheduler made no progress")
        ready = sorted(
            (leader for leader in unscheduled if is_ready(leader, cycle)),
            key=lambda leader: (-max(height[member] for member in firings[leader]), leader),
        )
        for leader in ready:
            operator = _op(nodes, leader).operator
            if isinstance(operator, PooledHardwareOperator):
                # A pooled firing binds the first instance slot whose busy window has elapsed; none free -> next cycle.
                slot = next((s for s in range(pool[type(operator)]) if busy_until.get((operator, s), 0) <= cycle), None)
                if slot is None:
                    continue
                busy_until[(operator, slot)] = cycle + operator.initiation_interval
                inst_count[operator] = max(inst_count.get(operator, 0), slot + 1)
                slot_of[leader] = (operator, slot)
            for member in firings[leader]:
                issue_cycle[member] = cycle
            unscheduled.discard(leader)
        cycle += 1

    inst_of, instances = _bind_instances(inst_count, slot_of, firings)
    pooled_firings = {leader: members for leader, members in firings.items() if leader in slot_of}
    latency = {vid: _op(nodes, vid).operator.latency for vid in op_ids}
    makespan = max((commit_cycle(vid) for vid in op_ids), default=0)
    return Schedule(
        issue_cycle=issue_cycle,
        inst_of=inst_of,
        firings=pooled_firings,
        instances=instances,
        makespan=makespan,
        latency=latency,
        busy_until=busy_until,
    )


def _bind_instances(
    inst_count: dict[PooledHardwareOperator, int],
    slot_of: dict[ValueId, tuple[PooledHardwareOperator, int]],
    firings: dict[ValueId, list[ValueId]],
) -> tuple[dict[ValueId, OperatorInstance], list[OperatorInstance]]:
    """
    Bind every member of each pooled firing to its physical instance. Instance indices are local to one concrete
    hardware operator value.
    """
    inst_of = {
        member: OperatorInstance(operator, slot)
        for leader, (operator, slot) in slot_of.items()
        for member in firings[leader]
    }
    instances = [OperatorInstance(operator, slot) for operator in inst_count for slot in range(inst_count[operator])]
    return inst_of, instances

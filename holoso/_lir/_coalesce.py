"""Per-bank phi-arm coalescing and quotient register coloring for the LIR builder."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .._mir import MirPhi
from .._operators import PortConditioner
from .._util import ValueId
from ._ir import ReadPort
from ._regalloc import ColoringProblem, Producer, color, find_coloring_conflict
from ._build_base import ColorObjective


@dataclass(frozen=True, slots=True)
class _PhiCoalescing:
    leader: dict[ValueId, ValueId]  # value -> class leader (a value never merged maps to itself, implicitly)
    coalesced: frozenset[tuple[int, ValueId]]  # (pred, phi) arms that share the merged register (no install copy)


def coalescable_arms(
    phi_nodes: Mapping[ValueId, MirPhi], values: set[ValueId], identity: PortConditioner
) -> dict[ValueId, list[tuple[int, ValueId]]]:
    """
    Per phi, the register-backed, identity-conditioner arms eligible to coalesce (so the arm flows into the merged
    register with no install copy). An arm that ANOTHER arm of the same phi reads under a non-identity conditioner is
    excluded: its residual copy would become a same-step self-conditioned copy (``r <= -r`` / ``b <= ~b``) into the
    merged register, which the install-free oracle cannot see and which the final interference rightly flags. Both banks
    differ only in their identity conditioner (:class:`FloatSignControl` vs :class:`BoolInversion`).
    """
    candidates: dict[ValueId, list[tuple[int, ValueId]]] = {}
    for vid, phi in phi_nodes.items():
        conditioned = {arm for _p, arm, cond in phi.arms if arm in values and cond != identity}
        candidates[vid] = [
            (pred, arm)
            for pred, arm, conditioner in phi.arms
            if arm in values and conditioner == identity and arm not in conditioned
        ]
    return candidates


def _coalesce_phis(
    phi_nodes: Mapping[ValueId, MirPhi],
    phi_order: list[ValueId],
    candidate_arms: dict[ValueId, list[tuple[int, ValueId]]],
    oracle: dict[ValueId, set[ValueId]],
    pinned: dict[ValueId, int],
    reserved_regs: set[int],
    forbidden: set[tuple[int, ValueId]],
) -> _PhiCoalescing:
    """
    Union-find phi-arm coalescing for one bank. Each phi result and its register-backed, identity-conditioner arms
    (``candidate_arms``) merge into one congruence class whenever the merge introduces no interference -- judged on
    ``oracle``, the install-free interference graph, since a coalesced class carries no install copy. The arms and the
    phi then share a register and the install copy vanishes. A class carries at most one pinned register; a class may
    not land on a ``reserved`` register -- the NON-coalesced state-slot registers, whose copy-back/early-install
    machinery owns them. A COALESCED slot's register is NOT reserved: it is seeded by the slot live-out's pin, so the
    phi live-out and the slot live-in (its "unchanged" arm) merge onto it for an in-place commit. ``phi_order`` is the
    deterministic processing order (block reverse-postorder, then value id); arms are processed in their phi-arm order.

    ``oracle`` is only an OVER-APPROXIMATION of coalescability: it omits the residual (non-coalesced) arms' install
    writes, so it can admit a merge the final install-aware interference rejects (a coalesced phi whose residual
    sibling arm's install lands in a register a class member is still live in). The caller (:func:`coalesce_and_color`)
    corrects this by rebuilding the final interference and re-running with the offending arms in ``forbidden`` -- the
    ``(pred, phi)`` arms this call must skip, never admitting them into a class. The fixpoint converges because
    forbidding only ever grows.
    """
    parent: dict[ValueId, ValueId] = {}
    members: dict[ValueId, set[ValueId]] = {}
    pin: dict[ValueId, int] = {}

    def find(x: ValueId) -> ValueId:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def ensure(x: ValueId) -> None:
        if x not in parent:
            parent[x] = x
            members[x] = {x}
            if x in pinned:
                pin[x] = pinned[x]

    for phi_vid in phi_order:
        ensure(phi_vid)
        for pred, arm in candidate_arms.get(phi_vid, []):
            if (pred, phi_vid) in forbidden:
                continue  # a prior fixpoint round found this merge unsound under the final interference
            ensure(arm)
            la, lb = find(phi_vid), find(arm)
            if la == lb:
                continue
            pa, pb = pin.get(la), pin.get(lb)
            if pa is not None and pb is not None and pa != pb:
                continue  # the two classes are pinned to different registers
            # Equal pins (pa == pb) merge consistently onto that register; they arise only on a COALESCED slot register
            # (the slot's live-in and its in-place live-out share its slot pin) and SHOULD merge -- that merge is the
            # in-place commit. The other non-reserved pins are the input lanes, each a distinct register, so two
            # distinct non-reserved classes never share a pin. The guard below still rejects the NON-coalesced slot
            # registers.
            merged_pin = pa if pa is not None else pb
            if merged_pin is not None and merged_pin in reserved_regs:
                continue  # a class touching a reserved (non-coalesced state-slot) register may not absorb a phi
            if any(not oracle[m].isdisjoint(members[lb]) for m in members[la]):
                continue  # some member of one class interferes with some member of the other
            lo, hi = (la, lb) if la < lb else (lb, la)  # leader = lowest value id, for determinism
            parent[hi] = lo
            members[lo] |= members.pop(hi)
            if merged_pin is not None:
                pin[lo] = merged_pin
            pin.pop(hi, None)

    leader = {v: find(v) for v in parent}
    coalesced = frozenset(
        (pred, phi_vid)
        for phi_vid in phi_nodes
        for pred, arm in candidate_arms.get(phi_vid, [])
        if leader.get(arm, arm) == leader.get(phi_vid, phi_vid)
    )
    return _PhiCoalescing(leader, coalesced)


def _color_quotient(
    leader: dict[ValueId, ValueId],
    pinned: dict[ValueId, int],
    interferes: dict[ValueId, set[ValueId]],
    objective: ColorObjective,
) -> tuple[dict[ValueId, int], int, int | None]:
    """
    Color the per-value interference graph after collapsing each coalescing class to its leader, then expand the
    leader's color back onto every member. The quotient unions each class's consumer ports and producers so the
    steering objective stays exact (a coalesced register really is read/written by every member's port/producer). A
    class with a pinned member pins its leader. Reduces to the plain per-value coloring when ``leader`` is the identity
    (every value its own singleton class, e.g. a kernel with no coalescable phi arms). The third return is the register
    of an interfering co-assignment under the FULL (residual-install) interference, or None when the coloring is sound
    -- a conflict can only come from the pins, which the caller resolves by backing the offending slot out of
    coalescing.
    """

    def lead(v: ValueId) -> ValueId:
        return leader.get(v, v)

    leaders = sorted({lead(v) for v in interferes})
    q_pinned: dict[ValueId, int] = {}
    for vid, reg in pinned.items():
        head = lead(vid)
        assert q_pinned.setdefault(head, reg) == reg, f"coalescing class {head} spans two pinned registers"
    q_interferes: dict[ValueId, set[ValueId]] = {head: set() for head in leaders}
    q_ports: dict[ValueId, set[ReadPort]] = {head: set() for head in leaders}
    q_producers: dict[ValueId, set[Producer]] = {head: set() for head in leaders}
    for vid in sorted(interferes):
        head = lead(vid)
        q_ports[head] |= objective.consumer_ports.get(vid, set())
        q_producers[head] |= objective.producer_key[vid]
        for other in interferes[vid]:
            head_other = lead(other)
            if head_other != head:
                q_interferes[head].add(head_other)
                q_interferes[head_other].add(head)
    q_movable: list[ValueId] = []
    seen: set[ValueId] = set()
    for vid in objective.movable:  # leaders of the movable values, first occurrence, preserving the deterministic order
        head = lead(vid)
        if head in q_pinned or head in seen:
            continue
        seen.add(head)
        q_movable.append(head)
    q_assign, nreg = color(
        ColoringProblem(
            movable=q_movable,
            pinned=q_pinned,
            interferes=q_interferes,
            consumer_ports=q_ports,
            producer_key={head: frozenset(producers) for head, producers in q_producers.items()},
            fresh_start=objective.fresh_start,
        )
    )
    assign = {vid: q_assign[lead(vid)] for vid in interferes}
    # Check the EXPANDED per-value coloring against the FULL (residual-install) interference -- stronger than a check
    # over the collapsed quotient, catching any unsound union or oracle drift. A conflict is returned (not raised) so
    # the slot-coalescing retry can back the offending slot register out and recolor.
    return assign, nreg, find_coloring_conflict(assign, interferes)


def _residual_installs(
    phi_nodes: Mapping[ValueId, MirPhi], coalesced: frozenset[tuple[int, ValueId]]
) -> dict[int, frozenset[ValueId]]:
    """Per predecessor block, the phi dests whose arm did NOT coalesce and so install by a pc-gated copy at its tail."""
    installs: dict[int, set[ValueId]] = {}
    for vid, phi in phi_nodes.items():
        for pred, _arm, _conditioner in phi.arms:
            if (pred, vid) not in coalesced:
                installs.setdefault(pred, set()).add(vid)
    return {pred: frozenset(dests) for pred, dests in installs.items()}


def coalesce_and_color(
    phi_nodes: Mapping[ValueId, MirPhi],
    phi_order: list[ValueId],
    candidate_arms: dict[ValueId, list[tuple[int, ValueId]]],
    pinned: dict[ValueId, int],
    reserved_regs: set[int],
    build_interferes: Callable[[dict[int, frozenset[ValueId]]], dict[ValueId, set[ValueId]]],
    objective: ColorObjective,
) -> tuple[dict[ValueId, int], int, _PhiCoalescing, int | None]:
    """
    Coalesce one bank's phi arms and color it, iterated to a soundness fixpoint. ``_coalesce_phis`` judges merges on the
    install-free oracle -- ``build_interferes({})``, the same interference graph with no residual installs -- which
    over-approximates coalescability (see its docstring); the final interference from the actual residual installs can
    therefore show a coalescing class interfering with itself -- a member still live where a residual sibling arm's
    install writes the merged register. When it does, every arm-merge of each offending class is FORBIDDEN and
    coalescing re-runs. Forbidding the whole class (not just the guilty merge) is an intentional sound-but-conservative
    choice: it cannot under-forbid, and the worst case (all arms forbidden) is the copy-everything baseline, which has
    no class-internal interference -- so the loop converges. The returned coalescing is the FINAL one; its ``coalesced``
    arms are exactly the copies the emitter elides. The fourth return is the register of an interfering co-assignment
    (from the pins) or None; the caller backs the offending slot out of in-place coalescing and recolors.
    """
    # The install-free baseline; deriving it here from the same builder keeps it in lockstep with the final graph.
    oracle = build_interferes({})
    forbidden: set[tuple[int, ValueId]] = set()
    arm_budget = sum(len(arms) for arms in candidate_arms.values())
    for _round in range(arm_budget + 1):  # forbidding grows by >= 1 each conflicting round; this bounds the fixpoint
        coalescing = _coalesce_phis(phi_nodes, phi_order, candidate_arms, oracle, pinned, reserved_regs, forbidden)
        interferes = build_interferes(_residual_installs(phi_nodes, coalescing.coalesced))
        bad_leaders: set[ValueId] = set()
        for vid, neighbours in interferes.items():
            head = coalescing.leader.get(vid, vid)
            if any(coalescing.leader.get(other, other) == head for other in neighbours):
                bad_leaders.add(head)  # this class interferes with itself under the final, install-aware graph
        if not bad_leaders:
            assign, nreg, conflict = _color_quotient(coalescing.leader, pinned, interferes, objective)
            return assign, nreg, coalescing, conflict
        forbidden |= {
            (pred, phi_vid)
            for pred, phi_vid in coalescing.coalesced
            if coalescing.leader.get(phi_vid, phi_vid) in bad_leaders
        }
    assert False, "phi-coalescing fixpoint did not converge"  # unreachable: forbidding is monotone and bounded

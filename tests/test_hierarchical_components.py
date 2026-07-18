"""
Public-API black-box tests for HIERARCHICAL components: a component that holds another component as a member and calls
it, so the child's persistent attributes become nested state. Every test drives ``holoso.synthesize(fn, ops)`` + the
elaborated simulator and checks output values against the same class run in Python across a stream, so the nested state
is proven to persist. Structural claims are limited to the state-slot COUNT (stable), not internal slot names.

The child-with-its-own-state case (iir1_hpf) is the concrete stage-7 example; the aliasing and self-reference cases pin
that object IDENTITY -- not attribute path -- is what defines a state slot, so two names for one child share one slot.
"""

from collections.abc import Callable

import pytest

import holoso
from holoso import (
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FloatFormat,
    FMulILog2OperatorFamily,
    FMulOperator,
    OpConfig,
    UnsupportedConstruct,
)

FMT = FloatFormat(6, 18)


def _ops() -> OpConfig:
    return OpConfig(
        FAddOperator(FMT), FMulOperator(FMT), FDivOperator(FMT), FMulILog2OperatorFamily(FMT), FCmpOperator(FMT)
    )


def _sim(fn: Callable[..., object], name: str) -> holoso.NumericalSimulator:
    return holoso.synthesize(fn, _ops(), name=name).numerical_model.elaborate()


class _Leaf:
    def __init__(self, alpha: float = 0.5) -> None:
        self.m = 0.0
        self.alpha = alpha

    def __call__(self, x: float) -> float:
        self.m = self.m + self.alpha * (x - self.m)
        return self.m


def test_component_with_a_stateful_child_persists_the_nested_state() -> None:
    class HighPass:
        def __init__(self) -> None:
            self.lpf = _Leaf(0.5)

        def __call__(self, x: float) -> float:
            bias = self.lpf(x)  # the child carries its own persistent m
            return x - bias

    sim = _sim(HighPass().__call__, "highpass")
    reference = HighPass()
    for x in (1.0, 1.0, 1.0, 0.0, -1.0, 2.0):
        got = float(sim.run(x)[0])
        want = float(reference(x))
        assert abs(got - want) <= 1e-3 * max(1.0, abs(want)), f"highpass x={x}: {got} vs {want}"


def test_two_member_names_for_one_child_share_a_single_state_slot() -> None:
    class Aliased:
        def __init__(self) -> None:
            shared = _Leaf(1.0)  # alpha 1 makes the child a running sum: m += (x - m) == x-carry, exact integers
            self.a = shared
            self.b = shared

        def __call__(self, x: float) -> float:
            return self.a(x) + self.b(x)  # both calls advance the SAME child

    sim = _sim(Aliased().__call__, "aliased_child")
    assert sum(1 for p in sim.outputs if p.name.startswith("state_")) == 1  # one child object -> one slot
    reference = Aliased()
    for x in (1.0, 2.0, 3.0, -4.0):
        assert float(sim.run(x)[0]) == float(reference(x)), f"aliased x={x}"


def test_self_referential_member_reads_through_the_alias() -> None:
    class SelfRef:
        def __init__(self) -> None:
            self.bias = 0.0
            self.me = self  # a cycle: me aliases the component itself

        def __call__(self, x: float) -> float:
            return x - self.me.bias  # reads self.bias through the alias

    sim = _sim(SelfRef().__call__, "self_ref")
    reference = SelfRef()
    for x in (1.0, 2.0, -3.0):
        assert float(sim.run(x)[0]) == float(reference(x)), f"self_ref x={x}"


def test_float_of_a_float_attribute_is_an_identity_no_op() -> None:
    # iir1_hpf spells ``x = float(x)`` defensively; on a value already typed float it is the identity, not a runtime
    # conversion, and must not block lowering.
    class Defensive:
        def __init__(self) -> None:
            self.lpf = _Leaf(0.5)

        def __call__(self, x: float) -> float:
            x = float(x)
            return x - self.lpf(x)

    sim = _sim(Defensive().__call__, "defensive_cast")
    reference = Defensive()
    for x in (1.0, 1.0, -2.0, 3.0):
        got = float(sim.run(x)[0])
        want = float(reference(x))
        assert abs(got - want) <= 1e-3 * max(1.0, abs(want)), f"defensive x={x}: {got} vs {want}"


class _Sum:
    def __init__(self) -> None:
        self.m = 0.0

    def __call__(self, x: float) -> float:
        self.m = self.m + x
        return self.m


def test_root_state_and_a_child_state_coexist_as_distinct_slots() -> None:
    # Root owns ``acc`` (bare slot ``acc``, preserving the single-owner ABI) while the child owns ``m`` (qualified slot
    # ``child__m``): two owners in one kernel, previously rejected, now compiled with injective names.
    class RootPlusChild:
        def __init__(self) -> None:
            self.acc = 0.0
            self.child = _Sum()

        def __call__(self, x: float) -> float:
            self.acc = self.acc + 1.0
            return self.child(x) + self.acc

    sim = _sim(RootPlusChild().__call__, "root_plus_child")
    slots = {p.name for p in sim.outputs if p.name.startswith("state_")}
    assert slots == {"state_acc", "state_child__m"}  # bare root slot + qualified child slot, no collision
    reference = RootPlusChild()
    for x in (1.0, 2.0, 3.0, -4.0):
        assert float(sim.run(x)[0]) == float(reference(x)), f"root_plus_child x={x}"


def test_two_children_fed_the_same_input_keep_independent_state() -> None:
    class TwoFilters:
        def __init__(self) -> None:
            self.a = _Sum()
            self.b = _Sum()

        def __call__(self, x: float) -> float:
            return self.a(x) - self.b(x)  # both advance on the SAME input, but their slots are independent

    sim = _sim(TwoFilters().__call__, "two_filters")
    slots = {p.name for p in sim.outputs if p.name.startswith("state_")}
    assert slots == {"state_a__m", "state_b__m"}
    reference = TwoFilters()
    for x in (1.0, 2.0, 3.0, -4.0):
        assert float(sim.run(x)[0]) == float(reference(x)), f"two_filters x={x}"


def test_rebinding_a_component_member_is_a_located_rejection() -> None:
    class Rebinds:
        def __init__(self) -> None:
            self.child = _Sum()
            self.other = _Sum()

        def __call__(self, x: float) -> float:
            self.child = self.other  # a per-transaction topology change
            return self.child(x)

    with pytest.raises(UnsupportedConstruct, match="cannot be rebound"):
        holoso.synthesize(Rebinds().__call__, _ops(), name="rebinds")

    class Constructs:
        def __init__(self) -> None:
            self.child = _Sum()

        def __call__(self, x: float) -> float:
            self.child = _Sum()  # in-kernel construction rejects at the call (the concrete-call whitelist)
            return self.child(x)

    with pytest.raises(UnsupportedConstruct, match="is not supported in a kernel"):
        holoso.synthesize(Constructs().__call__, _ops(), name="constructs")


def test_a_slot_name_collision_is_a_located_rejection() -> None:
    # A child ``x`` (slot ``x__m``) and a scalar attribute literally named ``x__m`` (slot ``x__m``) collide; the
    # double-underscore encoding is injective for every ordinary name, so this needs a deliberately dunder-ish clash.
    class Collides:
        def __init__(self) -> None:
            self.x = _Sum()
            self.x__m = 0.0

        def __call__(self, u: float) -> float:
            self.x__m = self.x__m + u
            return self.x(u) + self.x__m

    with pytest.raises(UnsupportedConstruct, match="collision"):
        holoso.synthesize(Collides().__call__, _ops(), name="collides")


def test_child_slot_name_is_canonical_regardless_of_discovery_order() -> None:
    # Regression (review): an aliased child reached first through a lexicographically-LARGER member name must still get
    # the canonical (shortest, then least) slot name, not the stale first-seen one. Provenance is a shortest-path
    # fixpoint over the member graph, so the emitted name cannot depend on which alias the body happens to touch first.
    class Mid:
        def __init__(self) -> None:
            self.leaf = _Sum()

        def __call__(self, x: float) -> float:
            return self.leaf(x)

    class DiscoverLargerFirst:
        def __init__(self) -> None:
            shared = Mid()
            self.zzz = shared
            self.aaa = shared  # aaa < zzz, so aaa__leaf__m is canonical even though zzz is used first

        def __call__(self, x: float) -> float:
            return self.zzz(x) + self.aaa.leaf.m

    class DiscoverSmallerFirst:
        def __init__(self) -> None:
            shared = Mid()
            self.zzz = shared
            self.aaa = shared

        def __call__(self, x: float) -> float:
            return self.aaa(x) + self.zzz.leaf.m

    for cls in (DiscoverLargerFirst, DiscoverSmallerFirst):
        sim = _sim(cls().__call__, cls.__name__.lower())
        slots = {p.name for p in sim.outputs if p.name.startswith("state_")}
        assert slots == {"state_aaa__leaf__m"}, f"{cls.__name__}: {slots}"


def test_state_port_order_follows_call_site_order_when_one_setter_inlines_twice() -> None:
    # Both stores originate from the SAME setter line, so the innermost origin frames tie and only the user call
    # sites can order the ports: first (the then-arm) must precede second (the else-arm) in the port ABI.
    class Child:
        def __init__(self) -> None:
            self.value = 0.0

        def set(self, v: float) -> None:
            self.value = v

    class TwoChildren:
        def __init__(self) -> None:
            self.first = Child()
            self.second = Child()

        def step(self, c: bool, x: float) -> float:
            if c:
                self.first.set(x)
            else:
                self.second.set(x)
            return self.first.value + self.second.value

    sim = _sim(TwoChildren().step, "two_children")
    assert [p.name for p in sim.outputs] == ["out_0", "state_first__value", "state_second__value"]
    reference = TwoChildren()
    for c, x in ((True, 1.0), (False, 2.0), (True, -3.0), (False, 0.5)):
        got = float(sim.run(c, x)[0])
        want = float(reference.step(c, x))
        assert got == want, f"step({c}, {x}): {got} vs {want}"

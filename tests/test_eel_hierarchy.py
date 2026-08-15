"""
Hierarchical state: kernel objects composing stateful sub-components, helper/inherited methods writing their
receiver, and the provenance guards that keep mid-statement state mutation CPython-consistent.
Every rejection here is a consensus-reviewed contract with a spelled rewrite.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pytest
from jaxtyping import Float64

import holoso
from holoso import UnsupportedConstruct
from holoso._eel import lower
from ._eeloracle import InputRow, assert_hir_matches_reference
from ._modelref import DEFAULT_UNROLL_MAX_TRIPS

_OPTIONS = holoso.Options(
    holoso.OperatorOptions(
        fadd=holoso.FAddOptions(),
        fmul=holoso.FMulOptions(),
        fdiv=holoso.FDivOptions(),
        fmul_ilog2=holoso.FMulILog2Options(),
        fcmp=holoso.FCmpOptions(),
    ),
    ffmt=holoso.FloatFormat(wexp=8, wman=36),
)


def _oracle(target: Callable[..., object], vectors: Sequence[InputRow]) -> None:
    assert_hir_matches_reference(
        lower(target, DEFAULT_UNROLL_MAX_TRIPS).hir, target, vectors, label=str(getattr(target, "__qualname__", target))
    )


def _rejects(target: object, match: str) -> None:
    assert callable(target)
    with pytest.raises(UnsupportedConstruct, match=match):
        holoso.synthesize(target, _OPTIONS)


class _Lpf:
    def __init__(self, *, alpha: float) -> None:
        self.alpha = alpha
        self.y = 0.0
        self._first = True

    def __call__(self, x: float) -> float:
        if self._first:
            self._first = False
            self.y = x
        else:
            self.y += self.alpha * (x - self.y)
        return self.y


class _Hpf:
    def __init__(self) -> None:
        self.lpf = _Lpf(alpha=2.0**-4)

    def step(self, x: float) -> float:
        return x - self.lpf(x)


def test_a_component_call_compiles_with_nested_slots_and_ports() -> None:
    _oracle(_Hpf().step, [{"x": v} for v in (1.0, 1.0, 5.0, 0.0, -2.0, 3.0)])
    result = holoso.synthesize(_Hpf().step, _OPTIONS)
    ports = [port.name for port in result.output_ports]
    assert "state_lpf_y" in ports, ports
    assert not any("_first" in name for name in ports), ports


class _TwoStage:
    def __init__(self) -> None:
        self.a = _Lpf(alpha=0.5)
        self.b = _Lpf(alpha=0.25)

    def step(self, x: float) -> float:
        return self.b(self.a(x))


def test_two_instances_of_one_component_class_hold_disjoint_slots() -> None:
    _oracle(_TwoStage().step, [{"x": v} for v in (1.0, 3.0, -2.0, 0.5)])


class _HelperWrites:
    def __init__(self) -> None:
        self.total = 0.0

    def _add(self, x: float) -> float:
        self.total = self.total + x
        return self.total

    def step(self, x: float) -> float:
        return self._add(x) * 2.0


class _Accumulator:
    def __init__(self) -> None:
        self.acc = 0.0

    def tick(self, x: float) -> float:
        self.acc = self.acc + x
        return self.acc


class _InheritsAccumulator(_Accumulator):
    def step(self, x: float) -> float:
        return self.tick(x) * 2.0


def test_helper_and_inherited_methods_write_receiver_state() -> None:
    _oracle(_HelperWrites().step, [{"x": v} for v in (1.0, 2.0, 3.5)])
    _oracle(_InheritsAccumulator().step, [{"x": v} for v in (1.0, 1.0, -0.5)])


class _DivergentLanes:
    def __init__(self) -> None:
        self.y = 0.0

    def _helper(self, c: bool, x: float) -> float:
        if c:
            self.y = x
            return 1.0
        self.y = -x
        return 2.0

    def step(self, c: bool, x: float) -> float:
        t = self._helper(c, x)
        return self.y + t


def test_divergent_state_across_helper_return_lanes_joins() -> None:
    _oracle(_DivergentLanes().step, [{"c": True, "x": 3.0}, {"c": False, "x": 3.0}, {"c": True, "x": -1.0}])


class _Child:
    def __init__(self, parent: object) -> None:
        self.parent = parent
        self.gain = 2.0

    def apply(self, x: float) -> float:
        self.parent.total = self.parent.total + x * self.gain  # type: ignore[attr-defined]
        return self.parent.total  # type: ignore[attr-defined,no-any-return]


class _ParentBackRef:
    def __init__(self) -> None:
        self.total = 0.0
        self.child = _Child(self)

    def step(self, x: float) -> float:
        return self.child.apply(x)


def test_a_parent_back_reference_write_names_the_canonical_slot() -> None:
    _oracle(_ParentBackRef().step, [{"x": 1.0}, {"x": 2.0}])
    result = holoso.synthesize(_ParentBackRef().step, _OPTIONS)
    assert "state_total" in [port.name for port in result.output_ports]


class _Cell:
    def __init__(self) -> None:
        self.y = 0.0

    def bump(self, x: float) -> float:
        self.y = self.y + x
        return self.y


class _TwoPathShared:
    def __init__(self) -> None:
        cell = _Cell()
        self.a = cell
        self.b = cell

    def step(self, x: float) -> float:
        return self.a.bump(x)


def test_a_multiply_referenced_stateful_component_is_rejected() -> None:
    _rejects(_TwoPathShared().step, "multiply-referenced component cannot hold state")


def _free_bump(cell: _Cell, x: float) -> float:
    cell.y = cell.y + x  # the path is seeded by _Cell.bump, so identity routing admits this spelling too
    return cell.y


class _FreeFunctionWrite:
    def __init__(self) -> None:
        self.cell = _Cell()

    def step(self, x: float) -> float:
        return _free_bump(self.cell, x)


class _LocalAliasWrite:
    def __init__(self) -> None:
        self.cell = _Cell()

    def step(self, x: float) -> float:
        handle = self.cell
        handle.y = handle.y + x
        return handle.y


def test_seeded_paths_admit_writes_through_any_spelling() -> None:
    _oracle(_FreeFunctionWrite().step, [{"x": 1.0}, {"x": 2.0}])
    _oracle(_LocalAliasWrite().step, [{"x": 1.0}, {"x": 2.0}])


def _free_poke(box: object, x: float) -> float:
    box.stash = x  # type: ignore[attr-defined]
    return x


class _Box:
    def __init__(self) -> None:
        self.stash = 0.0

    def read(self) -> float:
        return self.stash


class _UnseededWrite:
    def __init__(self) -> None:
        self.box = _Box()

    def step(self, x: float) -> float:
        return _free_poke(self.box, x) + self.box.read()


def test_a_write_no_method_spells_is_the_unseeded_refusal() -> None:
    _rejects(_UnseededWrite().step, "was not statically visible when state was seeded")


class _HostUtility:
    def __init__(self) -> None:
        self.y = 0.0
        self.callback = None

    def detach(self) -> None:
        self.callback = None
        self.history: list[float] = []

    def step(self, x: float) -> float:
        self.y = self.y + x
        return self.y


class _ReachedPoisonAbsent:
    def __init__(self) -> None:
        self.y = 0.0

    def step(self, x: float) -> float:
        self.started = True
        return x


def test_poisoned_paths_convict_only_when_reached() -> None:
    _oracle(_HostUtility().step, [{"x": 1.0}, {"x": 2.0}])
    _rejects(_ReachedPoisonAbsent().step, "has no value on the instance at synthesis time")


class _DeadArmPoisonedWrite:
    def __init__(self) -> None:
        self.y = 0.0
        self.mode = "fast"  # unrepresentable reset; the only write to it lies on a statically dead arm

    def step(self, x: float) -> float:
        if False:
            self.mode = 1.0
        self.y = self.y + x
        return self.y


def test_a_statically_pruned_write_to_a_poisoned_path_is_accepted() -> None:
    # The eager reset conviction moved to the reached write, so the pruned arm costs nothing. (A literal
    # `None` on the arm is different: the desugar whitelist rejects it, dead code enjoying no exemption.)
    _oracle(_DeadArmPoisonedWrite().step, [{"x": 1.0}, {"x": 2.0}])


class _TrimmedNested:
    def __init__(self) -> None:
        self.cell = _Cell()  # _Cell.bump seeds cell.y, but nothing here reaches it: the trim frees the slot

    def step(self, x: float) -> float:
        return self.cell.y + x


def test_an_unreached_nested_write_trims_and_the_read_folds_frozen() -> None:
    result = holoso.synthesize(_TrimmedNested().step, _OPTIONS)
    assert [port.name for port in result.output_ports] == ["out_0"]


class _AugBase:
    def __init__(self) -> None:
        self.x = 1.0

    def _bump(self) -> float:
        self.x = 5.0
        return 2.0


class _AugDirect(_AugBase):
    def step(self, v: float) -> float:
        self.x += self._bump()
        return self.x


class _AugWrapped(_AugBase):
    def step(self, v: float) -> float:
        self.x += self._bump() + 0.0
        return self.x


class _AugStaticFolded(_AugBase):
    def step(self, v: float) -> float:
        self.x += self._bump() * 0.0 + 2.0
        return self.x


class _AugHoisted(_AugBase):
    def step(self, v: float) -> float:
        t = self._bump()
        self.x += t
        return self.x


class _AugMarkCollision:
    def __init__(self) -> None:
        self.x = 1.0

    def helper(self, a: Float64[np.ndarray, "1"], v: float) -> float:
        self.x = v
        a[0] += 0.0  # the callee's own mark 0 must not clobber the caller's pending mark
        return v

    def update(self, v: float) -> float:
        self.x += self.helper(np.array([v]), v)
        return self.x


class _AugNonStateTarget:
    def __init__(self) -> None:
        self.gain = 1.0

    def _next(self) -> float:
        self.gain = self.gain * 2.0
        return self.gain

    def step(self, x: float) -> float:
        buf = np.array([x, x])
        buf[0] += self._next()  # a local tree is unreachable by state writes: no refusal
        return float(buf[0] + buf[1])


def test_augmented_stores_with_stateful_right_hand_sides_reject() -> None:
    for target in (_AugDirect().step, _AugWrapped().step, _AugStaticFolded().step, _AugMarkCollision().update):
        _rejects(target, "split into an explicit read and store")


def test_augmented_store_benign_spellings_compile() -> None:
    _oracle(_AugHoisted().step, [{"v": 0.0}, {"v": 1.0}])
    _oracle(_AugNonStateTarget().step, [{"x": 1.5}, {"x": -2.0}])


class _StaleHandle:
    def __init__(self) -> None:
        self.arr = np.array([1.0, 2.0])

    def _poke(self, x: float) -> int:
        self.arr[0] = x
        return 1

    def step(self, x: float) -> float:
        return float(self.arr[self._poke(x)])


class _RebindOnly:
    def __init__(self) -> None:
        self.arr = np.array([1.0, 2.0])

    def _swap(self, x: float) -> int:
        self.arr = np.array([x, x + 1.0])  # a rebind: Python's old handles keep the old object, and so do ours
        return 0

    def step(self, x: float) -> float:
        return float(self.arr[self._swap(x)])


class _FreshHandle:
    def __init__(self) -> None:
        self.arr = np.array([1.0, 2.0])

    def _poke(self, x: float) -> int:
        self.arr[0] = x
        return 0

    def step(self, x: float) -> float:
        i = self._poke(x)
        return float(self.arr[i])


def test_state_handles_across_mutating_calls() -> None:
    _rejects(_StaleHandle().step, "reload the handle after the call")
    _oracle(_RebindOnly().step, [{"x": 7.0}, {"x": -1.0}])
    _oracle(_FreshHandle().step, [{"x": 7.0}, {"x": -1.0}])


class _LocalAliasThenMutate:
    def __init__(self) -> None:
        self.arr = np.array([1.0, 2.0])

    def _poke(self, x: float) -> float:
        self.arr[0] = x
        return x

    def step(self, x: float) -> float:
        v = self.arr
        _ = self._poke(x)
        return float(v[0])


def test_a_shared_local_alias_blocks_the_mutation() -> None:
    _rejects(_LocalAliasThenMutate().step, "it is shared")


class _LoopCalls:
    def __init__(self) -> None:
        self.total = 0.0

    def _add(self, x: float) -> float:
        self.total = self.total + x
        return self.total

    def step(self, n: int, x: float) -> float:
        i = 0
        while i < n:
            _ = self._add(x)
            i = i + 1
        return self.total


class _LoopNestedSpelling:
    def __init__(self) -> None:
        self.cell = _Cell()

    def step(self, n: int, x: float) -> float:
        i = 0
        while i < n:
            self.cell.y = self.cell.y + x
            i = i + 1
        return self.cell.y


class _HeaderWrites:
    def __init__(self) -> None:
        self.count = 0

    def _tick(self, limit: int) -> bool:
        self.count = self.count + 1
        return self.count < limit

    def step(self, limit: int) -> int:
        while self._tick(limit):
            pass
        return self.count


def test_state_writes_reached_inside_residual_loops_are_carried() -> None:
    _oracle(_LoopCalls().step, [{"n": 3, "x": 2.0}, {"n": 0, "x": 5.0}, {"n": 2, "x": -1.0}])
    _oracle(_LoopNestedSpelling().step, [{"n": 3, "x": 2.0}, {"n": 1, "x": 0.5}])
    _oracle(_HeaderWrites().step, [{"limit": 4}, {"limit": 1}])


class _CrossingReturn:
    """
    A helper whose return sites straddle its own residual loop while writing state: the frame rows must
    carry the state live-outs, not only the result.
    """

    def __init__(self) -> None:
        self.y = 0.0

    def _search(self, n: int, x: float) -> float:
        i = 0
        while i < n:
            if x > 2.0:
                self.y = x
                return x * 10.0
            i = i + 1
        self.y = -x
        return 0.0

    def step(self, n: int, x: float) -> float:
        t = self._search(n, x)
        return self.y + t


def test_a_return_crossing_the_helpers_loop_carries_state_rows() -> None:
    _oracle(_CrossingReturn().step, [{"n": 3, "x": 5.0}, {"n": 3, "x": 1.0}, {"n": 0, "x": 9.0}])


def test_the_stateful_residual_loop_schedule_is_frozen() -> None:
    # No catalogued example combines persistent state with a residual back edge, so this directed row guards the
    # lean carry design (an untouched slot must not grow phis) permanently.
    result = holoso.synthesize(_LoopCalls().step, _OPTIONS, name="loop_calls")
    assert result.initiation_interval == (10, None), result.initiation_interval


class _NestedPromotion:
    def __init__(self) -> None:
        self.cell = _IntCell()

    def step(self, x: float) -> float:
        self.cell.n = self.cell.n + x
        return self.cell.n


class _IntCell:
    def __init__(self) -> None:
        self.n: float = 0  # the reset VALUE is an int, so the slot starts INT and promotes when it meets a float


def test_a_nested_int_leaf_promotes_to_float() -> None:
    _oracle(_NestedPromotion().step, [{"x": 0.5}, {"x": 1.25}])


class _ReturnsNestedLiveOut:
    def __init__(self) -> None:
        self.cell = _Cell()

    def step(self, x: float) -> float:
        return self.cell.bump(x)


def test_a_returned_nested_live_out_elides_into_its_state_port() -> None:
    result = holoso.synthesize(_ReturnsNestedLiveOut().step, _OPTIONS)
    assert [port.name for port in result.output_ports] == ["state_cell_y"]


class _DelegatingMiddle:
    """An honest delegation proxy: attribute access runs host code the structural store walk would skip."""

    def __init__(self) -> None:
        self.inner = _Cell()

    def __getattribute__(self, name: str) -> object:
        return object.__getattribute__(self, name)


class _StoreThroughProxy:
    def __init__(self) -> None:
        self.mid = _DelegatingMiddle()

    def step(self, x: float) -> float:
        self.mid.inner.y = x
        return x


def test_an_overridden_protocol_anywhere_on_a_store_chain_refuses() -> None:
    # The chain descends structurally, so an intermediate component running host code on attribute access
    # must poison the path -- lowering the write would diverge from CPython's routing (refuse, never diverge).
    _rejects(_StoreThroughProxy().step, "overrides __getattribute__")


class _SlottedChainStore:
    __slots__ = ("q",)

    def __init__(self) -> None:
        self.q = 0.0

    def step(self, x: float) -> float:
        self.a.b = x  # type: ignore[attr-defined]
        return x


def test_a_slots_receiver_chain_store_is_a_located_refusal() -> None:
    # The store gate's structural descent must survive a __dict__-less object: a clean refusal, never the
    # raw vars() TypeError.
    _rejects(_SlottedChainStore().step, "an attribute store through self.a is not supported")


class _HopProxy:
    """A delegation-style component: attribute reads run host code, so a store chain may not hop through it."""

    def __init__(self, parent: object) -> None:
        object.__setattr__(self, "parent", parent)

    def __getattribute__(self, name: str) -> object:
        return object.__getattribute__(self, name)


def _poke_through(mid: _HopProxy, x: float) -> float:
    mid.parent.total = x  # type: ignore[attr-defined]
    return x


class _OffCanonicalHop:
    def __init__(self) -> None:
        self.total = 0.0
        self.mid = _HopProxy(self)

    def _spell(self, x: float) -> None:
        self.total = x  # seeds ("total",) on the canonical (clean) chain

    def step(self, x: float) -> float:
        return _poke_through(self.mid, x) + self.total


def test_a_protocol_override_on_an_off_canonical_hop_refuses() -> None:
    # The seeded key's canonical chain is clean, but the SPELLED route hops through a component whose reads
    # run host code; the store walk must refuse the hop rather than resolve it structurally and diverge.
    _rejects(_OffCanonicalHop().step, "overrides __getattribute__, so reading 'parent'")


class _Shadowed:
    def __init__(self) -> None:
        self._actual = _Cell()
        self.__dict__["cell"] = _Cell()

    @property
    def cell(self) -> _Cell:
        return self._actual

    def step(self, x: float) -> float:
        self.cell.y = x
        return self.cell.y


def test_a_descriptor_shadowed_chain_edge_refuses() -> None:
    # CPython routes `self.cell` through the property while the structural walk would read the __dict__
    # shadow -- two different objects; the store must refuse the lying edge rather than diverge.
    _rejects(_Shadowed().step, "shadowed by a property/descriptor")


class _StaticCallable:
    @staticmethod
    def __call__(ignored: float, x: float = 10.0) -> float:
        return x


class _CallsStaticCallable:
    def __init__(self) -> None:
        self.child = _StaticCallable()

    def step(self, x: float) -> float:
        return self.child(x)


def test_a_staticmethod_dunder_call_receives_no_instance() -> None:
    # CPython's descriptor protocol calls a staticmethod __call__ without the instance, so `x` lands in
    # `ignored` and the default answers -- the inline must bind identically.
    _oracle(_CallsStaticCallable().step, [{"x": 3.0}, {"x": -2.0}])


class _VoidHelpers:
    def __init__(self) -> None:
        self.y = 0.0
        self.count = 0

    def reset(self) -> None:
        self.y = 0.0
        self.count = 0

    def push(self, x: float) -> None:
        self.y = self.y + x
        self.count = self.count + 1

    def step(self, x: float, clear: bool) -> float:
        if clear:
            self.reset()
        self.push(x)
        _ = self.push(x)  # type: ignore[func-returns-value]  # the bound None is judged only at use
        return self.y


def test_void_state_writing_helpers_are_callable_statements() -> None:
    _oracle(
        _VoidHelpers().step,
        [{"x": 1.0, "clear": False}, {"x": 2.0, "clear": False}, {"x": 5.0, "clear": True}, {"x": 1.5, "clear": False}],
    )


class _UsesVoidValue:
    def __init__(self) -> None:
        self.y = 0.0

    def _set(self, x: float) -> None:
        self.y = x

    def step(self, x: float) -> float:
        return self._set(x)  # type: ignore[func-returns-value,return-value]


def test_returning_a_void_helpers_none_is_a_located_refusal() -> None:
    _rejects(_UsesVoidValue().step, "returns no value .None. but its annotation declares one")


class _GetDescriptor:
    def __get__(self, instance: object, owner: type) -> object:
        return _redirected

    def __call__(self, x: float) -> float:
        return x + 1.0


def _redirected(x: float) -> float:
    return x + 10.0


class _DescriptorCallable:
    op = _GetDescriptor()

    def step(self, x: float) -> float:
        return self.op(x)  # type: ignore[operator,no-any-return]


def test_a_callable_non_data_descriptor_read_refuses() -> None:
    # CPython routes the read through __get__ (answering the redirect); inlining the descriptor's own
    # __call__ would answer 3.0 where Python answers 12.0 -- refuse the read instead.
    _rejects(_DescriptorCallable().step, "is a descriptor the compiler cannot read")


class _VoidKernel:
    def __init__(self) -> None:
        self.y = 0.0

    def _set(self, x: float) -> None:
        self.y = x

    def step(self, x: float) -> None:
        return self._set(x)


def test_a_none_kernel_may_tail_call_a_void_helper() -> None:
    # `return self._set(x)` in a `-> None` kernel is Python's idiomatic void tail call.
    result = holoso.synthesize(_VoidKernel().step, _OPTIONS)
    assert [port.name for port in result.output_ports] == ["state_y"]
    _oracle(_VoidKernel().step, [{"x": 2.0}, {"x": -1.0}])


class _LyingVoidAnnotation:
    def __init__(self) -> None:
        self.y = 0.0

    def helper(self, x: float) -> float:
        self.y = x
        return  # type: ignore[return-value]  # contradicts -> float on every path

    def step(self, x: float) -> float:
        self.helper(x)
        return self.y


def test_a_void_callee_declaring_a_value_is_refused_even_when_discarded() -> None:
    _rejects(_LyingVoidAnnotation().step, "returns no value on any path, but its return annotation declares one")


class _ItemStoreOnComponent:
    def __init__(self) -> None:
        self.sub = _Cell()

    def step(self, x: float) -> float:
        self.sub[0] = x  # type: ignore[index]
        return x


def test_an_item_store_on_a_component_names_the_right_mistake() -> None:
    # Whatever the spelling, the refusal must name the item-assignment mistake, never mislabel it as a
    # state-representation problem.
    _rejects(_ItemStoreOnComponent().step, "a component object does not support item assignment")

"""
Persistent state: the FULL install-disjointness matrix in BOTH directions (rejections and admissions -- an
implementation silently narrowing the design is a failure too), the assumed-state trim with its
conservative-rejection finality, the state aliasing events, early-return joins, and the slot loop carries.
"""

import dataclasses
import functools
import math
from collections.abc import Callable, Sequence

import numpy as np
import pytest
from jaxtyping import Float64 as Float64

import holoso
from holoso import FloatFormat, FloatType, UnsupportedConstruct
from holoso._eel import lower

from ._eeloracle import InputRow, assert_hir_matches_reference
from ._modelref import default_options, DEFAULT_UNROLL_MAX_TRIPS
from ._public import strip_locations

type _Row = InputRow

_OPTIONS = holoso.Options(holoso.OperatorOptions())
# The wide format keeps the derived integer word at the binary64 boundary, so the no-exact-float-image
# propositions below stay sharp (2**53+1 is the smallest integer without an image) instead of vacuous.
_WIDE_OPTIONS = holoso.Options(holoso.OperatorOptions(), ffmt=FloatFormat(11, 52))

_FMT = FloatFormat(8, 23)
_SYNTH_OPTIONS = default_options(_FMT)
_FFROMINT_OPTIONS = dataclasses.replace(
    _SYNTH_OPTIONS, operator=dataclasses.replace(_SYNTH_OPTIONS.operator, ffromint=holoso.FFromIntOptions())
)


def _oracle(target: Callable[..., object], vectors: Sequence[_Row]) -> None:
    label = getattr(target, "__qualname__", "kernel")
    compared = assert_hir_matches_reference(lower(target, DEFAULT_UNROLL_MAX_TRIPS).hir, target, vectors, label=label)
    assert compared == len(vectors)


def _rejects(target: object, match: str, options: holoso.Options = _OPTIONS) -> None:
    assert callable(target)
    with pytest.raises(UnsupportedConstruct, match=match):
        holoso.synthesize(target, options, name="k")


def _synthesized(target: Callable[..., object], options: holoso.Options = _SYNTH_OPTIONS) -> holoso.SynthesisResult:
    return holoso.synthesize(target, options, name="kernel")


def _residual(target: Callable[..., object], options: holoso.Options = _SYNTH_OPTIONS) -> str:
    return strip_locations(_synthesized(target, options).frontend_ir[-1])


# ---------------------------------------------------------------------- A5: install rejections

_SHARED_TABLE = [1.0, 2.0]
_ESCAPED_ARRAY = np.array([1.0, 2.0])


class _CrossTransactionAlias:
    def __init__(self) -> None:
        self.buf = np.zeros(2)
        self.snap = np.zeros(2)

    def step(self, x: float) -> float:
        self.snap = self.buf
        self.buf[0] = x
        return float(self.snap[0])


class _InternalRawAliasing:
    def __init__(self) -> None:
        self.buf = [[0.0]] * 2  # the reset snapshot aliases internally; the reached write convicts it

    def step(self, x: float) -> float:
        self.buf = [[x], [x]]
        return x


class _ImmutableLaundering:
    def __init__(self) -> None:
        inner = [0.0]
        self.s = ((inner,), (inner,))

    def step(self, x: float) -> float:
        self.s = (([x],), ([x],))
        return x


class _SameBoundaryDoubleInstall:
    def __init__(self) -> None:
        self.a = [0.0]
        self.b = [0.0]

    def step(self, x: float) -> float:
        t = [x]
        self.a = t
        self.b = t
        return x


class _SnapshotOverlapWithCapture:
    def __init__(self) -> None:
        self.buf = _SHARED_TABLE

    def step(self, x: float) -> float:
        self.buf = [x, x]
        return _SHARED_TABLE[0]


class _RawViewOverlapAcrossRoots:
    def __init__(self) -> None:
        base = np.zeros(3)
        self.a = base
        self.b = base[0:1]

    def step(self, x: float) -> float:
        self.a[0] = x
        self.b[0] = x
        return x


class _RawViewOverlapWithinOneTree:
    def __init__(self) -> None:
        base = np.zeros(3)
        self.a = [base, base[0:2]]

    def step(self, x: float) -> float:
        self.a = [np.array([x]), np.array([x])]
        return x


class _ZeroStrideSelfOverlap:
    def __init__(self) -> None:
        base = np.array([0.0])
        self.v = np.lib.stride_tricks.as_strided(base, shape=(2,), strides=(0,))

    def step(self, x: float) -> float:
        self.v[0] = x
        return float(self.v[1])


class _PartialByteSelfOverlap:
    def __init__(self) -> None:
        base = np.zeros(2, dtype=np.float64)
        self.v = np.lib.stride_tricks.as_strided(base, shape=(2,), strides=(4,))

    def step(self, x: float) -> float:
        self.v[0] = x
        return float(self.v[1])


class _InstallsEscapedGlobal:
    def __init__(self) -> None:
        self.buf = np.zeros(2)

    def step(self, x: float) -> float:
        self.buf = _ESCAPED_ARRAY
        return x


class _InstallsExtractedStateSubtree:
    def __init__(self) -> None:
        self.buf = np.zeros((2, 2))
        self.snap = np.zeros(2)

    def step(self, x: float) -> float:
        self.buf[0, 0] = x
        self.snap = self.buf[0]
        return x


class _InstallsFrozenCapturedSubtree:
    def __init__(self) -> None:
        self.buf = [[0.0], [1.0]]
        self.snap = [0.0]

    def step(self, x: float) -> float:
        self.snap = self.buf[0]
        return x


class _ConcatIntoState:
    def __init__(self) -> None:
        self.buf = [0.0, 0.0]

    def step(self, x: float) -> float:
        self.buf += [x]
        return x


class _InstallsRepeatedTensor:
    def __init__(self) -> None:
        self.grid = ((0.0,), (0.0,))

    def step(self, x: float) -> float:
        t = np.array([x])
        self.grid = (t, t)  # type: ignore[assignment]
        return x


class _InstallsJoinedPick:
    def __init__(self) -> None:
        self.a = [1.0]
        self.b = [2.0]
        self.dst = [0.0]

    def _pick(self, c: bool) -> tuple[float, ...]:
        if c:
            return self.a  # type: ignore[return-value]
        else:
            return self.b  # type: ignore[return-value]

    def step(self, c: bool, x: float) -> float:
        self.dst = self._pick(c)  # type: ignore[assignment]
        return x


class _InstallsRootLevelJoin:
    def __init__(self) -> None:
        self.dst = [0.0, 0.0]

    def step(self, c: bool, x: float) -> float:
        if c:
            t = [x, 0.0]
        else:
            t = [0.0, x]
        self.dst = t
        return x


class _InstallsWrongLength:
    def __init__(self) -> None:
        self.buf = [0.0, 0.0]

    def step(self, x: float) -> float:
        self.buf = [x]
        return x


class _InstallsTensorForSequence:
    def __init__(self) -> None:
        self.buf = [0.0, 0.0]

    def step(self, x: float) -> float:
        self.buf = np.array([x, x])  # type: ignore[assignment]
        return x


class _InstallsSequenceForTensor:
    def __init__(self) -> None:
        self.v = np.array([1.0, 2.0])

    def step(self, s: float) -> float:
        w = self.v * s
        self.v = [w[0], w[1]]  # type: ignore[assignment]
        return s


class _InstallsWrongShape:
    def __init__(self) -> None:
        self.P = np.zeros((2, 2))

    def step(self, a: float) -> float:
        self.P = self.P[0] * a
        return a


class _InstallsBooleanLeaves:
    def __init__(self) -> None:
        self.P = np.zeros((2, 2))

    def step(self, flag: bool) -> float:
        self.P = np.array([[flag, flag], [flag, flag]])
        return 1.0


class _ScalarReplacesAggregate:
    def __init__(self) -> None:
        self.buf = [0.0, 0.0]

    def step(self, x: float) -> float:
        self.buf = x  # type: ignore[assignment]
        return x


class _AggregateReplacesScalar:
    def __init__(self) -> None:
        self.y = 0.0

    def step(self, x: float) -> float:
        self.y = [x]  # type: ignore[assignment]
        return x


def test_the_a5_install_matrix_rejections() -> None:
    for target, match in [
        (_CrossTransactionAlias().step, "backs .or backed. the state attribute self.buf"),
        (_InternalRawAliasing().step, "reaches the same mutable object through more than one path within self.buf"),
        (_ImmutableLaundering().step, "reaches the same mutable object through more than one path within self.s"),
        (_SameBoundaryDoubleInstall().step, "backs .or backed. the state attribute self.a"),
        (_SnapshotOverlapWithCapture().step, "overlaps the environment name '_SHARED_TABLE'"),
        (_RawViewOverlapAcrossRoots().step, "shares storage with the poisoned attribute self.b"),
        (_RawViewOverlapWithinOneTree().step, "overlaps the storage of another array within self.a"),
        (_ZeroStrideSelfOverlap().step, "self-overlapping array view"),
        (_PartialByteSelfOverlap().step, "self-overlapping array view"),
        (_InstallsEscapedGlobal().step, "arrived from outside the kernel"),
        (_InstallsExtractedStateSubtree().step, "backs .or backed. the state attribute self.buf"),
        (_InstallsFrozenCapturedSubtree().step, "arrived from outside the kernel"),
        (_ConcatIntoState().step, "not supported on a sequence"),
        (_InstallsRepeatedTensor().step, "the same array is reachable through more than one path within it"),
        (_InstallsJoinedPick().step, "merged across runtime branches"),
        (_InstallsRootLevelJoin().step, "merged across runtime branches"),
        (_InstallsWrongLength().step, "its structure does not match the reset value's"),
        (_InstallsTensorForSequence().step, "its structure does not match the reset value's"),
        (_InstallsSequenceForTensor().step, "its structure does not match the reset value's"),
        (_InstallsWrongShape().step, "its structure does not match the reset value's"),
        (_InstallsBooleanLeaves().step, "an array must hold numbers, not booleans"),
        (_ScalarReplacesAggregate().step, "a scalar cannot replace it"),
        (_AggregateReplacesScalar().step, "its structure does not match the reset value's"),
    ]:
        _rejects(target, match)


# ---------------------------------------------------------------------- A5: admissions


class _PlainLocalSharing:
    def __init__(self) -> None:
        self.v = np.array([1.0, 2.0])

    def step(self, x: float) -> float:
        b = self.v
        return float(b[0] + b[1]) + x


class _InstallsExplicitCopies:
    def __init__(self) -> None:
        self.x = [0.0, 0.0]
        self.p = np.zeros(2)

    def step(self, x: float) -> float:
        t = [x, 2.0 * x]
        self.x = list(t)
        self.p = np.array(t)
        return self.x[0] + float(self.p[1])


class _UnaliasedElementStores:
    def __init__(self) -> None:
        self.buf = np.zeros(3)

    def step(self, i_gain: float, x: float) -> float:
        self.buf[0] = x
        self.buf[1] = self.buf[0] * i_gain
        self.buf[2] = self.buf[2] + x
        return float(self.buf[1] + self.buf[2])


class _ReadThenElementStore:
    def __init__(self) -> None:
        self.v = np.array([1.0, 2.0])

    def step(self, x: float) -> float:
        t = self.v[1]
        self.v[0] = x
        return float(t + self.v[0])


class _PureScalarRepeatedTuple:
    def __init__(self) -> None:
        t = (0.0,)
        self.s = (t, t)

    def step(self, x: float) -> float:
        self.s = ((x,), (x,))
        return self.s[0][0]


class _HelperBuiltInstall:
    def __init__(self) -> None:
        self.x = [0.0, 0.0]

    def _make(self, x: float) -> tuple[float, float]:
        return (x, x + 1.0)

    def step(self, x: float) -> float:
        self.x = self._make(x)  # type: ignore[assignment]
        return self.x[1]


class _FreshHelperReturnStaysMutable:
    def __init__(self) -> None:
        self.y = 0.0

    def _make(self) -> np.ndarray:
        m = np.zeros(2)
        m[0] = 1.0
        return m

    def step(self, x: float) -> float:
        m2 = self._make()
        m2[1] = x
        self.y = float(m2[0] + m2[1])
        return self.y


class _PropertyFreshTensorStaysMutable:
    def __init__(self) -> None:
        self.y = 0.0

    @property
    def fresh(self) -> np.ndarray:
        return np.zeros(2)

    def step(self, x: float) -> float:
        b = self.fresh
        b[0] = x
        self.y = float(b[0])
        return self.y


def test_the_a5_install_matrix_admissions() -> None:
    _oracle(_PlainLocalSharing().step, [{"x": 0.5}, {"x": -1.0}])
    _oracle(_InstallsExplicitCopies().step, [{"x": 1.5}, {"x": -2.0}, {"x": 0.0}])
    _oracle(_UnaliasedElementStores().step, [{"i_gain": 2.0, "x": 1.0}, {"i_gain": 0.5, "x": 3.0}])
    _oracle(_ReadThenElementStore().step, [{"x": 5.0}, {"x": -1.5}])
    _oracle(_PureScalarRepeatedTuple().step, [{"x": 2.0}, {"x": -3.0}])
    _oracle(_HelperBuiltInstall().step, [{"x": 1.0}, {"x": 4.0}])
    _oracle(_FreshHelperReturnStaysMutable().step, [{"x": 2.5}, {"x": -0.5}])
    _oracle(_PropertyFreshTensorStaysMutable().step, [{"x": 1.25}, {"x": -2.0}])


# ---------------------------------------------------------------------- the state aliasing events


class _AliasThenStore:
    def __init__(self) -> None:
        self.v = np.zeros(2)

    def step(self, x: float) -> float:
        b = self.v
        self.v[0] = x
        return float(b[0])


class _AliasThroughHelperArgument:
    def __init__(self) -> None:
        self.v = np.zeros(2)

    def _poke(self, buf: np.ndarray, x: float) -> float:
        buf[0] = x
        return float(buf[0])

    def step(self, x: float) -> float:
        t = self._poke(self.v, x)
        self.v[1] = t
        return float(self.v[1])


class _AliasHandedOutByHelper:
    def __init__(self) -> None:
        self.v = np.zeros(2)

    def _get(self) -> np.ndarray:
        return self.v

    def step(self, x: float) -> float:
        b = self._get()
        self.v[0] = x
        return float(b[0])


class _AsarrayLaunderedInstall:
    def __init__(self) -> None:
        self.buf = [np.array([0.0]), np.array([0.0])]

    def step(self, setup: bool, x: float) -> float:
        if setup:
            t = np.array([x])
            self.buf = [t, np.asarray(t)]
        else:
            self.buf[0][0] = x
        return float(self.buf[1][0])


def test_an_aggregate_state_read_landing_anywhere_persistent_shares_the_tree() -> None:
    _rejects(_AliasThenStore().step, "cannot store into self.v.0.: it is shared")
    _rejects(_AliasThroughHelperArgument().step, "cannot store into buf.0.: it is shared")
    _rejects(_AliasHandedOutByHelper().step, "cannot store into self.v.0.: it is shared")


def test_a_view_derivation_cannot_launder_an_install() -> None:
    _rejects(_AsarrayLaunderedInstall().step, "the same array is reachable through more than one path within it")


# ---------------------------------------------------------------------- receiver discipline


class _AliasRootedReceiverStore:
    def __init__(self) -> None:
        self.y = 0.0

    def step(self, x: float) -> float:
        s = self
        s.y = x
        return self.y


class _UnrepresentableStateObject:
    def __init__(self) -> None:
        self.mode: object = "fast"

    def step(self, x: float) -> float:
        self.mode = x
        return x


class _NestedAttributeStore:
    def __init__(self) -> None:
        self.a = [0.0]

    def step(self, x: float) -> float:
        self.a.b = x  # type: ignore[attr-defined, unused-ignore]
        return x


class _MissingReset:
    def step(self, x: float) -> float:
        self.n = x
        return x


class _ClassLevelDefaultReset:
    y = 0.0

    def step(self, x: float) -> float:
        self.y = self.y + x
        return self.y


class _NaNReset:
    def __init__(self) -> None:
        self.y = math.nan

    def step(self, x: float) -> float:
        self.y = x
        return self.y


class _BoolSlotTypeChange:
    def __init__(self) -> None:
        self.flag = False

    def step(self, x: float) -> float:
        self.flag = 1  # type: ignore[assignment]
        return x


class _FloatSlotGetsBool:
    def __init__(self) -> None:
        self.y = 0.0

    def step(self, flag: bool) -> float:
        self.y = flag
        return 1.0


class _SlotNameCollision:
    def __init__(self) -> None:
        self.x_0 = 0.0
        self.x = np.array([0.0])

    def step(self, a: float, b: float) -> float:
        self.x_0 = a
        self.x[0] = b
        return self.x_0 + float(self.x[0])


class _EmptyAggregateState:
    def __init__(self) -> None:
        self.buf: list[float] = []

    def step(self, x: float) -> float:
        self.buf = [x]
        return x


class _EmptyTensorState:
    def __init__(self) -> None:
        self.v = np.zeros(0)  # unrepresentable reset; the reached write below convicts it

    def step(self, x: float) -> float:
        self.v = np.array([x])
        return x


def test_receiver_discipline_rejections() -> None:
    for target, match in [
        (_AliasRootedReceiverStore().step, "was not statically visible when state was seeded"),
        (_UnrepresentableStateObject().step, "which the compiler cannot represent as state"),
        (_NestedAttributeStore().step, "an attribute store through self.a is not supported: it is not a component"),
        (_MissingReset().step, "has no value on the instance at synthesis time"),
        (_ClassLevelDefaultReset().step, "has no value on the instance at synthesis time"),
        (_NaNReset().step, "the reset value of self.y is NaN"),
        (_BoolSlotTypeChange().step, "bool state joins only with bool"),
        (_FloatSlotGetsBool().step, "would change type from float to bool"),
        (_SlotNameCollision().step, "decompose to the same slot name 'x_0'"),
        (_EmptyAggregateState().step, "is an empty aggregate"),
        (_EmptyTensorState().step, "must be a non-empty 1-D or 2-D array"),
    ]:
        _rejects(target, match)


class _HierarchicalComponent:
    class _Inner:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float) -> float:
            self.y = self.y + x
            return self.y

    def __init__(self) -> None:
        self.inner = _HierarchicalComponent._Inner()

    def step(self, x: float) -> float:
        return self.inner(x)


def test_a_component_instance_call_compiles_with_nested_state() -> None:
    # The sub-component's own register becomes the nested slot inner.y, updated by its inlined __call__.
    _oracle(_HierarchicalComponent().step, [{"x": 1.0}, {"x": 2.5}, {"x": -0.5}, {"x": 4.0}])


def _scale(factor: float, x: float) -> float:
    return factor * x


_PARTIAL = functools.partial(_scale, 0.5)


def _calls_an_unregistered_numpy_function(x: float) -> float:
    return float(np.sum(np.array([x, x])))


def _calls_a_partial(x: float) -> float:
    return _PARTIAL(x)


def test_an_unregistered_callable_is_not_mistaken_for_a_component_instance() -> None:
    # A component instance is one the compiler would have to inline: its class defines __call__ as a plain Python
    # function. numpy's dispatchers, ufuncs and functools.partial are callable objects with an instance __dict__ but a
    # C-level __call__, so they must draw the plain unregistered-callee refusal, not a message about hierarchical state.
    for kernel in (_calls_an_unregistered_numpy_function, _calls_a_partial):
        _rejects(kernel, "are not supported yet")
        with pytest.raises(UnsupportedConstruct) as excinfo:
            holoso.synthesize(kernel, _OPTIONS, name="k")
        assert "component instance" not in excinfo.value.message


# ---------------------------------------------------------------------- the trim and its finality


class _DeadArmWrite:
    def __init__(self) -> None:
        self.gain = 2.0

    def step(self, x: float) -> float:
        if x != x:  # statically false is not enough: the seed is syntactic, so use a residual-shaped dead arm
            pass
        if False:
            self.gain = 3.0
        return self.gain * x


class _IdentityWrite:
    def __init__(self) -> None:
        self.y = 1.5

    def step(self, x: float) -> float:
        self.y = self.y
        return self.y * x


class _ConservativeFinality:
    def __init__(self) -> None:
        self.n = 3

    def step(self, x: float) -> float:
        if False:
            self.n = 4
        v = (x, 2.0, 3.0)
        return v[self.n - 3]


def test_a_dead_arm_write_trims_back_to_a_frozen_constant() -> None:
    result = _synthesized(_DeadArmWrite().step)
    assert [(p.name, p.scalar_type) for p in result.output_ports] == [("out_0", FloatType(_FMT))]
    assert "2.0" in strip_locations(result.frontend_ir[-1])
    _oracle(_DeadArmWrite().step, [{"x": 3.0}])


def test_an_identity_write_keeps_its_slot() -> None:
    assert "state y: float reset 1.5" in _residual(_IdentityWrite().step)
    _oracle(_IdentityWrite().step, [{"x": 2.0}, {"x": -1.0}])


def test_a_conservative_rejection_is_final_and_names_the_pinning_write() -> None:
    with pytest.raises(UnsupportedConstruct) as info:
        holoso.synthesize(_ConservativeFinality().step, _OPTIONS, name="k")
    message = str(info.value)
    assert "must be a compile-time constant" in message
    assert "self.n is treated as persistent state because of the write at" in message
    assert "if such a write is unreachable, remove it" in message


# ---------------------------------------------------------------------- slot typing across the join


class _IntSlotStaysInt:
    def __init__(self) -> None:
        self.n = 0

    def step(self, up: bool) -> int:
        if up:
            self.n = self.n + 1
        return self.n


class _IntSlotPromotesToFloat:
    def __init__(self) -> None:
        self.acc = 0

    def step(self, x: float) -> float:
        self.acc = self.acc + x  # type: ignore[assignment]
        return self.acc


class _FloatSlotAcceptsIntWrites:
    def __init__(self) -> None:
        self.y = 0.0

    def step(self, reset: bool, x: float) -> float:
        if reset:
            self.y = 0
        else:
            self.y = self.y + x
        return self.y


def test_slot_types_join_by_the_one_rule() -> None:
    _oracle(_IntSlotStaysInt().step, [{"up": True}, {"up": True}, {"up": False}])
    assert "state acc: float reset 0.0" in _residual(_IntSlotPromotesToFloat().step)
    _oracle(_IntSlotPromotesToFloat().step, [{"x": 1.5}, {"x": 2.0}])
    _oracle(_FloatSlotAcceptsIntWrites().step, [{"reset": False, "x": 2.5}, {"reset": True, "x": 1.0}])


# ---------------------------------------------------------------------- early-return joins


class _AllArmsReturn:
    def __init__(self) -> None:
        self.count = 0

    def step(self, x: float) -> float:
        self.count = self.count + 1
        if x > 0.0:
            return x
        else:
            return -x


class _MixedReturn:
    def __init__(self) -> None:
        self.peak = 0.0

    def step(self, x: float) -> float:
        if x > self.peak:
            self.peak = x
            return x
        return self.peak


def _mixed_in_helper(x: float) -> float:
    if x > 0.0:
        return x
    return -x


def _calls_mixed_helper(x: float) -> float:
    return _mixed_in_helper(x) * 2.0


def test_early_return_joins() -> None:
    _oracle(_AllArmsReturn().step, [{"x": 2.0}, {"x": -3.0}, {"x": 0.0}])
    _oracle(_MixedReturn().step, [{"x": 1.0}, {"x": 0.5}, {"x": 2.0}, {"x": 1.5}])


def _mixed_chain_helper(x: float) -> float:
    if x > 2.0:
        return x
    y = x + 1.0
    if y > 2.0:
        return y * 2.0
    return -y


def _calls_mixed_chain(x: float) -> float:
    return _mixed_chain_helper(x) * 2.0


def _helper_with_loop_return(x: float) -> float:
    for i in range(4):
        if x > float(i):
            return x + float(i)
        x = x + 0.5
    return x


def _calls_helper_with_loop_return(x: float) -> float:
    return _helper_with_loop_return(x) - 1.0


class _MixedHelperInResidualLoop:
    def __init__(self) -> None:
        self.acc = 0.0

    def step(self, x: float) -> float:
        while x > 0.0:
            x = x - _mixed_in_helper(x - 2.0) - 1.0
        self.acc = self.acc + x
        return self.acc


def test_mixed_returns_in_inlined_callees_match_cpython() -> None:
    _oracle(_calls_mixed_helper, [{"x": 2.0}, {"x": -3.0}, {"x": 0.0}])
    _oracle(_calls_mixed_chain, [{"x": 3.0}, {"x": 1.5}, {"x": 0.0}, {"x": -4.0}])
    _oracle(_calls_helper_with_loop_return, [{"x": 2.5}, {"x": 0.5}, {"x": -2.0}, {"x": -0.25}])
    _oracle(_MixedHelperInResidualLoop().step, [{"x": 3.5}, {"x": 0.5}, {"x": 6.0}, {"x": -1.0}])
    _oracle(_ContinueThenBreakSlot().step, [{"x": -1.0}, {"x": 0.5}, {"x": 1.5}, {"x": 5.0}])


def _falls_through_helper(x: float) -> float:  # type: ignore[return]
    if x > 0.0:
        return x


def _calls_falling_helper(x: float) -> float:
    return _falls_through_helper(x) * 2.0


def _loop_return_helper(x: float) -> float:
    while x > 0.0:
        if x > 10.0:
            return x
        x = x - 1.0
    return x


def _calls_loop_return_helper(x: float) -> float:
    return _loop_return_helper(x) + 1.0


def _halve_into_band(x: float) -> float:
    while x > 1.0:
        if x < 2.0:
            return x
        x = x * 0.5
    return x


def _crossing_inside_enclosing_loop(a: float) -> float:
    acc = a
    while acc > 3.0:
        acc = acc - _halve_into_band(acc)
    return acc


def _header_band(x: float) -> float:
    while x > 4.0:
        if x < 8.0:
            return x
        x = x * 0.5
    return x


def _crossing_in_header(x: float) -> float:
    acc = x
    while _header_band(acc) > 0.5:
        acc = acc * 0.25
    return acc


def _static_pick(c: bool, x: float) -> int:
    while c:
        if x > 0.0:
            return 3
        c = False
    return 3


def _uses_static_pick(x: float, c: bool) -> float:
    # Both sites return the same constant, so the result must STAY static: a sequence subscript demands it.
    v = (x, x * 2.0, x * 3.0, x * 4.0)
    return v[_static_pick(c, x)]


def _mixed_sibling_and_crossing(x: float) -> float:
    if x < 0.0:
        return -1.0
    while x > 2.0:
        if x < 4.0:
            return x * 10.0
        x = x * 0.5
    return x


def _calls_mixed_sibling_and_crossing(x: float) -> float:
    return _mixed_sibling_and_crossing(x) + 0.125


def _int_float_sites(x: float) -> float:
    while x > 2.0:
        if x < 4.0:
            return 7
        x = x * 0.5
    return x


def _calls_int_float_sites(x: float) -> float:
    return _int_float_sites(x) * 2.0


def _pair_helper(x: float) -> tuple[float, float]:
    n = 0.0
    while x > 2.0:
        if x < 4.0:
            return x, n
        x = x * 0.5
        n = n + 1.0
    return x, n


def _calls_pair_helper(x: float) -> float:
    lo, count = _pair_helper(x)
    return lo + count * 100.0


def _exits_and_return_helper(x: float, cap: float) -> float:
    hits = 0.0
    while x > 0.0:
        x = x - 1.0
        if hits > cap:
            return hits * 1000.0
        if x > 6.0:
            hits = hits + 2.0
            continue
        if x < 1.0:
            break
        hits = hits + 1.0
    return hits + x


def _calls_exits_and_return(x: float, cap: float) -> float:
    return _exits_and_return_helper(x, cap) - 0.5


def _nested_loops_helper(x: float, n: float) -> float:
    while n > 0.0:
        n = n - 1.0
        y = x + n
        while y > 1.0:
            if y < 2.0:
                return y + n * 10.0
            y = y * 0.5
    return -n


def _calls_nested_loops(x: float, n: float) -> float:
    return _nested_loops_helper(x, n) + 0.25


def _outer_band(x: float) -> float:
    while x > 4.0:
        if x < 8.0:
            return _halve_into_band(x) + 1.0
        x = x * 0.5
    return _halve_into_band(x)


def _calls_nested_frames(x: float) -> float:
    return _outer_band(x) * 2.0


def _broadcast_host(x: float, y: float) -> float:
    for _ in range(1):
        if x > 0.0:
            return x
        if y > 1.0:
            break
    return _halve_into_band(y) + 0.5


def _calls_broadcast_host(x: float, y: float) -> float:
    return _broadcast_host(x, y)


def _all_return_loop_helper(c: bool) -> int:
    while c:
        return 3
    return 3


def _calls_all_return_loop_helper(c: bool) -> int:
    return _all_return_loop_helper(c)


def test_a_callee_that_can_fall_through_cannot_return_a_value() -> None:
    _rejects(_calls_falling_helper, "the call can complete without returning a value")


def test_returns_inside_callee_residual_loops_match_cpython() -> None:
    _oracle(_calls_loop_return_helper, [{"x": 15.0}, {"x": 5.0}, {"x": 0.0}, {"x": -3.0}, {"x": 10.5}])
    _oracle(_crossing_inside_enclosing_loop, [{"a": 20.0}, {"a": 3.0}, {"a": 3.5}, {"a": 100.0}])
    _oracle(_crossing_in_header, [{"x": 40.0}, {"x": 0.4}, {"x": 4.0}, {"x": 1000.0}])
    _oracle(_uses_static_pick, [{"x": 2.0, "c": True}, {"x": -1.0, "c": True}, {"x": 2.0, "c": False}])
    # The pre-loop sibling lane must keep its arm in the union: a fold that sealed it flat would leave the
    # in-loop lane's continuation unreachable.
    _oracle(_calls_mixed_sibling_and_crossing, [{"x": -5.0}, {"x": 1.0}, {"x": 3.0}, {"x": 50.0}])
    _oracle(_calls_int_float_sites, [{"x": 3.0}, {"x": 1.0}, {"x": 64.0}])
    _oracle(_calls_pair_helper, [{"x": 3.0}, {"x": 1.5}, {"x": 80.0}, {"x": -2.0}])
    _oracle(
        _calls_exits_and_return,
        [{"x": 10.0, "cap": 3.0}, {"x": 10.0, "cap": 100.0}, {"x": 4.0, "cap": 0.0}, {"x": 0.0, "cap": 1.0}],
    )
    _oracle(_calls_nested_loops, [{"x": 1.5, "n": 3.0}, {"x": 0.5, "n": 2.0}, {"x": 8.0, "n": 1.0}])
    # A frame nested inside another frame's own crossing lane, and a frame built while an unrelated
    # pending lane keeps two sinks open, so the wrap runs on a broadcast piece.
    _oracle(_calls_nested_frames, [{"x": 20.0}, {"x": 5.0}, {"x": 0.5}, {"x": 100.0}])
    _oracle(_calls_broadcast_host, [{"x": 1.0, "y": 8.0}, {"x": -1.0, "y": 8.0}, {"x": -1.0, "y": 0.5}])


def test_a_callee_loop_body_returning_on_every_path_cannot_iterate() -> None:
    _rejects(_calls_all_return_loop_helper, "returns on every path, so the loop cannot iterate")


def _return_shape_mismatch(c: bool, x: float, /) -> tuple[float, ...]:
    if c:
        return x, x
    return (x,)


def test_return_sites_must_agree_in_shape() -> None:
    _rejects(_return_shape_mismatch, "does not match the kernel's other return sites")


class _ReturnInResidualLoop:
    def __init__(self) -> None:
        self.y = 0.0

    def step(self, x: float) -> float:
        while x > 0.0:
            if x > 10.0:
                return x
            x = x - 1.0
        return x


class _SlotCommitAtLoopReturn:
    def __init__(self) -> None:
        self.hits = 0.0

    def step(self, x: float) -> float:
        while x > 0.0:
            if x > 10.0:
                self.hits = self.hits + 1.0
                return x
            x = x - 1.0
        return x


class _PromotedCarryWithLoopReturn:
    """
    The INT carry assumption meets a FLOAT back value (`k + 0.5`), so the body pass is discarded and
    re-run; the return site committed by the discarded pass must not poison the output table or the
    elision bookkeeping. Both sites return the public slot itself, so both must stay ELIDED.
    """

    def __init__(self) -> None:
        self.y = 0.0

    def step(self, x: float) -> float:
        k = 0
        while x > 0.0:
            if x > 10.0:
                self.y = x
                return self.y
            k = k + 0.5  # type: ignore[assignment]  # the int-to-float rebind IS the promotion trigger
            x = x - 1.0
        self.y = k
        return self.y


class _ContinueThenBreakSlot:
    def __init__(self) -> None:
        self.found = 0.0

    def step(self, x: float) -> float:
        for i in range(3):
            if x > float(i):
                continue
            self.found = float(i) * 10.0 + 1.0
            break
        return self.found


def test_returns_inside_residual_loops_match_cpython() -> None:
    _oracle(_ReturnInResidualLoop().step, [{"x": 12.0}, {"x": 3.5}, {"x": 0.0}, {"x": 20.0}, {"x": -1.0}])
    _oracle(_SlotCommitAtLoopReturn().step, [{"x": 12.0}, {"x": 3.5}, {"x": 15.0}, {"x": 0.0}])
    _oracle(_PromotedCarryWithLoopReturn().step, [{"x": 12.0}, {"x": 5.5}, {"x": 0.0}, {"x": 11.0}])
    # Both sites must stay elided (bare returns) through the promotion re-run's discarded commit. The
    # restore is invariant-protective: annotation conformance makes the discarded and kept passes commit
    # identical tables today, so no kernel can yet distinguish its absence -- this pins the machinery
    # running, not a divergence.
    result = _synthesized(_PromotedCarryWithLoopReturn().step)
    assert "return %" not in strip_locations(result.frontend_ir[-1])
    assert [p.name for p in result.output_ports] == ["state_y"]


# ---------------------------------------------------------------------- slots as residual-loop carries


class _TensorSlotLoopCarry:
    def __init__(self) -> None:
        self.acc = np.zeros(2)

    def step(self, x: float) -> float:
        while x > 0.0:
            self.acc[0] = self.acc[0] + x
            self.acc[1] = self.acc[1] + 1.0
            x = x - 1.0
        return float(self.acc[0] + self.acc[1])


class _InstallInsideResidualLoop:
    def __init__(self) -> None:
        self.buf = [0.0]

    def step(self, x: float) -> float:
        while x > 0.0:
            self.buf = [x]
            x = x - 1.0
        return float(self.buf[0])


def test_a_tensor_slot_carries_leafwise_through_a_residual_loop() -> None:
    _oracle(_TensorSlotLoopCarry().step, [{"x": 3.5}, {"x": 0.0}, {"x": 1.5}])


def test_an_install_inside_a_residual_loop_is_a_staged_gap() -> None:
    _rejects(_InstallInsideResidualLoop().step, "installing a new aggregate into the state attribute self.buf inside")


# ---------------------------------------------------------------------- elision across sites


class _ElisionKilledByDisagreeingSite:
    def __init__(self) -> None:
        self.y = 0.0

    def step(self, x: float) -> float:
        if x > 0.0:
            self.y = x
            return self.y
        return 100.0


def test_a_returned_leaf_matching_a_public_slot_at_every_site_elides_and_only_then() -> None:
    agreeing = _synthesized(_MixedReturn().step)  # every site returns the peak slot's live-out
    assert [(p.name, p.scalar_type) for p in agreeing.output_ports] == [("state_peak", FloatType(_FMT))]
    disagreeing = _synthesized(_ElisionKilledByDisagreeingSite().step)
    assert [(p.name, p.scalar_type) for p in disagreeing.output_ports] == [
        ("out_0", FloatType(_FMT)),
        ("state_y", FloatType(_FMT)),
    ]


# ---------------------------------------------------------------------- review-round pins (adopted defects)


class _PromotionIsLeafGranular:
    def __init__(self) -> None:
        self._pair = (0, 9007199254740993)

    def step(self, x: float) -> bool:
        self._pair = (x, self._pair[1])  # type: ignore[assignment]
        return self._pair[1] == 9007199254740992


class _TransientTypeChangesDoNotPromote:
    def __init__(self) -> None:
        self._n = 9007199254740993

    def step(self, x: float) -> bool:
        out = self._n == 9007199254740992
        self._n = x  # type: ignore[assignment]
        self._n = 9007199254740993
        return out


class _HelperBranchReadsCurrentState:
    def __init__(self) -> None:
        self._y = 1.0

    def _choose(self, positive: bool) -> float:
        if positive:
            z = self._y
        else:
            z = -self._y
        return z

    def step(self, positive: bool, x: float) -> float:
        self._y = x
        return self._choose(positive)


def test_promotion_is_leaf_granular_and_judged_at_the_commit() -> None:
    _oracle(_PromotionIsLeafGranular().step, [{"x": 1.0}, {"x": 2.5}])
    _oracle(_TransientTypeChangesDoNotPromote().step, [{"x": 1.5}, {"x": 0.5}])


def test_a_helper_branch_reads_the_current_slot_values() -> None:
    _oracle(
        _HelperBranchReadsCurrentState().step,
        [{"positive": True, "x": 3.0}, {"positive": False, "x": -2.0}, {"positive": True, "x": 0.5}],
    )


_TRANSPOSE_SOURCE = np.array([[1.0]])


class _TransposeLaunderedInstall:
    def __init__(self) -> None:
        self._a = np.zeros((1, 1))

    def step(self, setup: bool, x: float) -> float:
        if setup:
            self._a = _TRANSPOSE_SOURCE.T
        else:
            self._a[0, 0] = x
        return float(_TRANSPOSE_SOURCE[0, 0])


def test_a_transpose_carries_its_source_storage_into_the_install_gate() -> None:
    _rejects(_TransposeLaunderedInstall().step, "arrived from outside the kernel")


class _ReturnsStateAggregate:
    def __init__(self) -> None:
        self._buf = np.zeros(1)

    def step(self, x: float) -> "Float64[np.ndarray, '1']":
        self._buf[0] = x
        return self._buf


class _ReturnsStateCopy:
    def __init__(self) -> None:
        self._buf = np.zeros(2)

    def step(self, x: float) -> "Float64[np.ndarray, '2']":
        self._buf[0] = x
        self._buf[1] = self._buf[1] + x
        return np.array(self._buf)


def test_returning_a_state_aggregate_rejects_but_a_copy_is_the_blessed_spelling() -> None:
    _rejects(_ReturnsStateAggregate().step, "would hand out a live alias")
    _oracle(_ReturnsStateCopy().step, [{"x": 1.5}, {"x": -0.5}, {"x": 2.0}])


class _InstallThenElementStore:
    def __init__(self) -> None:
        self.p = np.zeros(2)

    def step(self, x: float) -> float:
        self.p = np.array([x, x])
        self.p[0] = x + 1.0
        return float(self.p[0])


class _ChainedStateReadModifyWrite:
    def __init__(self) -> None:
        self.m = np.zeros((2, 2))

    def step(self, x: float) -> float:
        self.m[0][0] = self.m[0][0] + x
        return float(self.m[0, 0])


def test_state_store_rejections_carry_followable_advice() -> None:
    _rejects(_InstallThenElementStore().step, "installed into the state attribute self.p this transaction")
    _rejects(_ChainedStateReadModifyWrite().step, "the augmented .\\+=. and multi-index .m.i, j.. spellings")


class _BranchInstallOfRetiredStateTree:
    def __init__(self) -> None:
        self.a = np.array([1.0])
        self.b = np.array([2.0])

    def step(self, c: bool, x: float) -> float:
        if c:
            self.a[0] = x
            self.a = np.array([0.0])
        else:
            self.b = self.a
        return float(self.b[0]) + 1.0


class _SiblingTensorKeepsIntPrecision:
    def __init__(self) -> None:
        self._s = (0, np.array([9007199254740993], dtype=np.int64))

    def step(self, x: float) -> bool:
        out = self._s[1][0] == 9007199254740992
        self._s = (x, np.array([9007199254740993]))  # type: ignore[assignment]
        return bool(out)


class _JoinedStateReturn:
    def __init__(self) -> None:
        self.buf = np.zeros(1)

    def step(self, c: bool, x: float) -> "Float64[np.ndarray, '1']":
        if c:
            self.buf = np.array([x])
        return self.buf


def _one_arm_return_with_else_binding(c: bool, x: float, /) -> float:
    if c:
        return x
    else:
        y = x + 1.0
    return y


class _ScalarSlotPromotesInResidualLoop:
    def __init__(self) -> None:
        self.acc = 0

    def step(self, x: float) -> float:
        while x > 0.0:
            self.acc = self.acc + x  # type: ignore[assignment]
            x = x - 1.0
        return float(self.acc)


def test_a_branch_install_cannot_unprotect_the_sibling_arms_state_tree() -> None:
    _rejects(_BranchInstallOfRetiredStateTree().step, "backs .or backed. the state attribute self.a")


def test_promoting_one_leaf_leaves_sibling_tensor_resets_exact() -> None:
    _oracle(_SiblingTensorKeepsIntPrecision().step, [{"x": 1.5}, {"x": 2.0}])


def test_a_branch_joined_state_tree_still_cannot_be_returned() -> None:
    _rejects(_JoinedStateReturn().step, "would hand out a live alias")


def test_a_one_arm_return_leaves_the_surviving_arms_bindings_live() -> None:
    _oracle(_one_arm_return_with_else_binding, [{"c": True, "x": 5.0}, {"c": False, "x": 5.0}])


def test_a_scalar_slot_promotes_across_a_residual_loop_back_edge() -> None:
    _oracle(_ScalarSlotPromotesInResidualLoop().step, [{"x": 0.0}, {"x": 2.5}, {"x": 1.0}])


class _ReceiverSubscriptStore:
    def __init__(self) -> None:
        self.y = 0.0

    def step(self, x: float) -> float:
        self.y = x
        self[0] = x  # type: ignore[index]
        return x


def _guarded_raise_after_partial_return(c: bool, x: float, /) -> float:
    if c:
        return x
    raise ValueError("negative input")


class _SelfInstallNoOp:
    def __init__(self) -> None:
        self.v = [0.0]

    def step(self, x: float) -> float:
        self.v = self.v
        return self.v[0] + x


def test_receiver_subscript_store_is_a_located_rejection() -> None:
    _rejects(_ReceiverSubscriptStore().step, "a component object does not support item assignment")


def test_a_raise_after_a_partial_return_is_judged_data_dependent() -> None:
    _rejects(_guarded_raise_after_partial_return, "a raise on a data-dependent path")


def test_rebinding_an_attribute_to_its_own_tree_is_a_no_op() -> None:
    _oracle(_SelfInstallNoOp().step, [{"x": 2.0}, {"x": -1.0}])


class _PromotedInexactReset:
    def __init__(self) -> None:
        self._n = 9007199254740993

    def step(self, x: float) -> bool:
        out = self._n == 9007199254740992
        self._n = x  # type: ignore[assignment]
        return out


class _PromotedOverflowingReset:
    def __init__(self) -> None:
        self._n = 10**400

    def step(self, x: float) -> float:
        self._n = x  # type: ignore[assignment]
        return x


class _PromotedExactReset:
    def __init__(self) -> None:
        self._n = 3

    def step(self, x: float) -> float:
        self._n = self._n + x  # type: ignore[assignment]
        return self._n


class _TrimClearsConservativePromotion:
    def __init__(self) -> None:
        self._n = 5
        self._gate = False

    def step(self, up: bool) -> float:
        if self._gate:
            self._n = 0.5  # type: ignore[assignment]
        if False:
            self._gate = True
        if up:
            self._n = self._n + 1
        return float(self._n)


class _DeadCodeFrozenAlias:
    def __init__(self) -> None:
        shared = np.zeros(1)
        self.buf = shared
        self.alias = shared

    def step(self, x: float) -> float:
        if False:
            return float(self.alias[0])
        self.buf[0] = x
        return x + 0.25


def test_a_promotion_with_an_inexact_reset_image_is_a_located_rejection() -> None:
    _rejects(_PromotedInexactReset().step, "has no exact float image", _WIDE_OPTIONS)
    _rejects(_PromotedOverflowingReset().step, "has no exact float image", _WIDE_OPTIONS)
    _oracle(_PromotedExactReset().step, [{"x": 0.5}, {"x": 1.0}])


def test_a_trim_retires_promotions_made_under_the_conservative_assumption() -> None:
    assert "state _n: int reset 5" in _residual(_TrimClearsConservativePromotion().step, _FFROMINT_OPTIONS)
    _oracle(_TrimClearsConservativePromotion().step, [{"up": True}, {"up": False}, {"up": True}])


def test_a_frozen_alias_of_state_rejects_even_when_only_dead_code_reads_it() -> None:
    _rejects(_DeadCodeFrozenAlias().step, "shares storage with the frozen attribute self.alias")


class _TransientLoopFloatRestoredToInt:
    def __init__(self) -> None:
        self._n = 0

    def step(self, x: float) -> bool:
        out = self._n == 9007199254740992
        while x > 0.0:
            self._n = x  # type: ignore[assignment]
            x = x - 1.0
        self._n = 9007199254740993
        return out


def _nested_partial_return_then_raise(c: bool, d: bool, x: float, /) -> float:
    if c:
        if d:
            return x
        y = 1.0
    else:
        y = 2.0
    raise ValueError("nope")


def test_a_transient_loop_float_does_not_promote_a_slot_restored_to_int() -> None:
    _oracle(_TransientLoopFloatRestoredToInt().step, [{"x": 1.0}, {"x": 0.0}])


def test_a_raise_after_a_nested_partial_return_is_judged_data_dependent() -> None:
    _rejects(_nested_partial_return_then_raise, "a raise on a data-dependent path")


class _PromotedSlotElisionCandidate:
    def __init__(self) -> None:
        self.n = 0

    def step(self, c: bool, x: float) -> float:
        if c:
            self.n = x  # type: ignore[assignment]
        return float(self.n)


def test_a_promoted_slot_never_claims_the_elision() -> None:
    _oracle(_PromotedSlotElisionCandidate().step, [{"c": False, "x": 1.5}, {"c": True, "x": 2.5}])


class _FloatArrayInstallOverIntReset:
    def __init__(self) -> None:
        self.v = np.array([0], dtype=np.int64)

    def step(self, install: bool, x: float) -> float:
        if install:
            self.v = np.array([x])
            return float(self.v[0])
        self.v[0] = x
        return float(self.v[0])


class _IntArrayInstallOverFloatReset:
    def __init__(self) -> None:
        self.buf = np.array([1.0, 2.0])

    def step(self, sel: bool, x: float) -> float:
        if sel:
            self.buf = np.array([3, 4])
        else:
            self.buf[0] = x
        return float(self.buf[0]) + float(self.buf[1])


def test_an_array_install_cannot_change_the_element_family_in_either_direction() -> None:
    _rejects(_FloatArrayInstallOverIntReset().step, "element family .float. differs")
    _rejects(_IntArrayInstallOverFloatReset().step, "element family .int. differs")


_MODULE_SHARED = np.array([0.0])


class _StateAliasesUnreadGlobal:
    def __init__(self) -> None:
        self.v = _MODULE_SHARED

    def step(self, x: float) -> float:
        self.v[0] = x
        return x


def _int_array_into_float_helper(v: "Float64[np.ndarray, '2']") -> float:
    v[0] = 3.5
    return float(v[0] + v[1])


def _calls_float_helper_with_int_array(x: float) -> float:
    return _int_array_into_float_helper(np.array([1, 2])) + x


class _AnnotationLaunderedInstall:
    def __init__(self) -> None:
        self._v = np.array([0.0])

    def _make(self) -> "Float64[np.ndarray, '1']":
        return np.array([1])

    def step(self, setup: bool, x: float) -> float:
        if setup:
            self._v = self._make()
        else:
            self._v[0] = x
        return float(self._v[0])


def test_a_state_tree_aliasing_an_unread_environment_aggregate_rejects_at_conversion() -> None:
    _rejects(_StateAliasesUnreadGlobal().step, "overlaps the environment name '_MODULE_SHARED'")


def test_an_annotation_never_converts_an_array_family() -> None:
    _rejects(_calls_float_helper_with_int_array, "an annotation does not convert an array")
    _rejects(_AnnotationLaunderedInstall().step, "an annotation does not convert an array")


class _StateAliasesParameterDefault:
    shared = np.array([0.0])

    def __init__(self) -> None:
        self.buf = self.shared

    def step(self, x: float, unused: "Float64[np.ndarray, '1']" = shared) -> float:
        self.buf[0] = x
        return x


def test_a_state_tree_aliasing_a_parameter_default_rejects_at_conversion() -> None:
    _rejects(_StateAliasesParameterDefault().step, "overlaps the environment name 'a parameter default'")

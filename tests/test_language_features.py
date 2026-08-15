"""
Public-API, black-box behavioral tests for front-end language features: the boolean `^` operator, instance/
inherited/static method calls, `@property` reads, descriptor and attribute-access-protocol boundaries, and
module-level constant resolution. Every test drives the compiler ONLY through `holoso.synthesize(fn, ops)` and
exercises the elaborated numerical model, asserting on observable output values, so the tests survive a refactor of
the front end. The rejection checks guard soundness boundaries where a silent miscompile would otherwise diverge
from Python -- behavioral, not mere input validation.
"""

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import holoso
from holoso import (
    FAddOptions,
    FCmpOptions,
    FDivOptions,
    FMulILog2Options,
    FMulOptions,
    FloatFormat,
    OperatorOptions,
    Options,
    UnsupportedConstruct,
)

_FMT = FloatFormat(4, 8)


def _ops() -> Options:
    return Options(
        OperatorOptions(
            fadd=FAddOptions(),
            fmul=FMulOptions(),
            fdiv=FDivOptions(),
            fmul_ilog2=FMulILog2Options(),
            fcmp=FCmpOptions(),
        ),
        ffmt=_FMT,
    )


def _model(target: Callable[..., object]) -> holoso.NumericalSimulator:
    return holoso.synthesize(target, _ops()).numerical_model.elaborate()


def _xor_chain(a: bool, b: bool, c: bool, d: bool) -> bool:
    return a ^ b ^ c ^ d


def _float_xor(x: float, y: float) -> float:
    return x ^ y  # type: ignore[operator, no-any-return]  # deliberately ill-typed: exercises rejection of float ^


def test_bool_xor_chain_is_parity() -> None:
    sim = _model(_xor_chain)
    for bits in itertools.product((False, True), repeat=4):
        assert bool(sim.run(*bits)[0]) == (sum(bits) % 2 == 1), f"parity {bits}"


def test_xor_on_floats_is_rejected() -> None:
    # `^` requires boolean operands; a float operand must fail loudly, not silently lower to some float op.
    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(_float_xor, _ops())


class _ParityBase:
    def __init__(self, odd: bool) -> None:
        self._odd = odd

    def _polarized_parity(self, a: bool, b: bool, c: bool) -> bool:
        even = a ^ b ^ c
        return (not even) if self._odd else even  # self._odd is a folded read-only attribute


class _ParityUser(_ParityBase):
    def __call__(self, a: bool, b: bool, c: bool) -> bool:
        return self._polarized_parity(a, b, c)  # an INHERITED method call, resolved through the MRO


def test_inherited_method_call_even_and_odd() -> None:
    even = _model(_ParityUser(False).__call__)
    odd = _model(_ParityUser(True).__call__)
    for bits in itertools.product((False, True), repeat=3):
        want_even = sum(bits) % 2 == 1
        assert bool(even.run(*bits)[0]) == want_even, f"even {bits}"
        assert bool(odd.run(*bits)[0]) == (not want_even), f"odd {bits}"


class _Thresholded:
    def __init__(self, base: float) -> None:
        self._base = base

    @property
    def _threshold(self) -> float:
        return self._base * 2  # a computed read-only value, derived from frozen configuration

    def __call__(self, x: float) -> bool:
        return x >= self._threshold


def test_property_read_folds_from_configuration() -> None:
    sim = _model(_Thresholded(3.0).__call__)
    for x in (0.0, 5.0, 6.0, 7.0, 10.0):
        assert bool(sim.run(x)[0]) == (x >= 6.0), f"x={x}"


_GAIN = 4.0
_LIMIT = 10  # an int module constant resolves as a float in a value position


def _uses_module_constants(x: float) -> tuple[float, bool]:
    scaled = x * _GAIN
    return scaled, scaled >= _LIMIT


def _local_shadows_module_constant(x: float) -> float:
    _GAIN = 1.0  # noqa: F841  -- a local binding shadows the module constant of the same name
    return x * _GAIN


def test_module_constants_resolve_in_value_position() -> None:
    sim = _model(_uses_module_constants)
    for x in (0.0, 1.0, 2.0, 3.0, 5.0):
        scaled, over = sim.run(x)
        assert float(scaled) == x * 4.0, f"gain x={x}"
        assert bool(over) == (x * 4.0 >= 10.0), f"limit x={x}"


def test_local_name_shadows_module_constant() -> None:
    sim = _model(_local_shadows_module_constant)
    for x in (1.0, 2.0, 5.0):
        assert float(sim.run(x)[0]) == x * 1.0, f"shadow x={x}"


_TRIPS = 3  # a module-level int constant used in BOTH a static-int position and a value position


def _module_constant_as_int_and_value(x: float) -> float:
    acc = 0.0
    for _ in range(_TRIPS):  # static-int position: the range bound (resolved by the static-int evaluator)
        acc = acc + x
    return acc + x**_TRIPS  # value position: same constant as a ** exponent and a folded literal


def test_module_constant_in_static_int_and_value_positions() -> None:
    # The same module constant must resolve in a static-int position (range bound, ** exponent) AND a value position
    # without collision -- the static-int path and the value-position literal path read it consistently.
    sim = _model(_module_constant_as_int_and_value)
    for x in (0.5, 1.0, 2.0, 3.0):
        assert float(sim.run(x)[0]) == x * _TRIPS + x**_TRIPS, f"x={x}"


class _Scaler:
    def __init__(self, bias: float) -> None:
        self._bias = bias

    @staticmethod
    def _double(x: float) -> float:
        return x * 2

    def __call__(self, x: float) -> float:
        return self._double(x) + self._bias


def test_staticmethod_call_binds_all_arguments() -> None:
    sim = _model(_Scaler(3.0).__call__)
    for x in (0.0, 1.0, 2.5, -4.0):
        assert float(sim.run(x)[0]) == x * 2 + 3.0, f"x={x}"


def _instance_shadow(x: float) -> float:
    return x + 5  # the callable an instance attribute is bound to, shadowing the class method below


class _ShadowedMethod:
    def __init__(self) -> None:
        self._mix = _instance_shadow  # type: ignore[method-assign]  # shadows the same-named method below

    def _mix(self, x: float) -> float:  # shadowed at runtime by the __init__ binding
        return x * 2

    def __call__(self, x: float) -> float:
        return self._mix(x)  # Python calls the instance attribute (x + 5), NOT the method (x * 2)


def test_method_shadowed_by_instance_attribute_follows_python() -> None:
    # Python resolves the instance attribute first (a method is a non-data descriptor), so the call must reach the
    # stored callable (x + 5) rather than the same-named class method (x * 2) it shadows.
    sim = holoso.synthesize(_ShadowedMethod().__call__, _ops()).numerical_model.elaborate()
    for x in (3.0, -1.5, 0.0):
        assert float(sim.run(x)[0]) == _ShadowedMethod()(x)


def _bool_eq_ne(a: bool, b: bool) -> tuple[bool, bool]:
    return a == b, a != b


def _bool_ordering(a: bool, b: bool) -> bool:
    return a < b  # ordering on booleans is meaningless here and must be rejected


def _bool_eq_branch(a: bool, b: bool, x: float) -> float:
    if a == b:  # a runtime boolean equality as a branch condition (lowers via the xnor)
        y = x * 2
    else:
        y = x
    return y


def test_bool_eq_is_xnor_and_ne_is_xor() -> None:
    sim = _model(_bool_eq_ne)
    for a, b in itertools.product((False, True), repeat=2):
        eq, ne = sim.run(a, b)
        assert bool(eq) == (a == b), f"eq {a} {b}"
        assert bool(ne) == (a != b), f"ne {a} {b}"


def test_bool_ordering_is_rejected() -> None:
    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(_bool_ordering, _ops())


def test_bool_eq_as_runtime_branch_condition() -> None:
    sim = _model(_bool_eq_branch)
    for a, b in itertools.product((False, True), repeat=2):
        for x in (1.0, 3.0):
            assert float(sim.run(a, b, x)[0]) == (x * 2 if a == b else x), f"{a} {b} {x}"


_FEATURE_ON = True
_THRESHOLD = 2.0


def _module_bool_branch(x: float) -> float:
    if _FEATURE_ON:  # folds to True at compile time, so the early return on the live path is allowed
        return x * 2
    return x  # statically dead -- never lowered, so no spurious state/branch


def _module_float_branch(x: float) -> float:
    if _THRESHOLD < 0.0:  # folds to False (the module float resolves in the static comparison)
        return x
    return x * _THRESHOLD


def test_module_bool_constant_folds_branch_reachability() -> None:
    sim = _model(_module_bool_branch)
    for x in (1.0, 3.0):
        assert float(sim.run(x)[0]) == x * 2, f"x={x}"


def test_module_float_constant_folds_in_static_comparison() -> None:
    sim = _model(_module_float_branch)
    for x in (1.0, 2.0):
        assert float(sim.run(x)[0]) == x * 2.0, f"x={x}"


_CHAIN_A = True
_CHAIN_B = True
_NP_TWO = np.float64(2.0)  # a numpy float module constant, which must resolve like a plain float


def _chained_static_bool(x: float) -> float:
    if _CHAIN_A == _CHAIN_B == _FEATURE_ON:  # a chain of compile-time bools folds, so the early return is allowed
        return x * 3
    return x


def _np_float_constant(x: float) -> float:
    if _NP_TWO > 1.0:  # an np.float64 module constant resolves in the static comparison (folds True)
        return x * _NP_TWO
    return x


def test_chained_static_bool_comparison_folds() -> None:
    # A chain `a == b == c` of module-level bools must fold (else the return inside the branch over-rejects).
    sim = _model(_chained_static_bool)
    for x in (1.0, 2.0):
        assert float(sim.run(x)[0]) == x * 3, f"x={x}"


def test_np_float64_module_constant_resolves_in_static_comparison() -> None:
    sim = _model(_np_float_constant)
    for x in (1.0, 2.0):
        assert float(sim.run(x)[0]) == x * 2.0, f"x={x}"


class _ReadOnlyAttrEqPoison:
    def __init__(self) -> None:
        self._flag = True  # read-only -> the else below is statically dead
        self._other = True  # written ONLY in that dead else, so it must stay read-only

    def __call__(self, x: float) -> float:
        if self._flag == True:  # noqa: E712 -- a read-only-attr equality whose else is dead
            pass
        else:
            self._other = False
        if self._other:  # _other stays read-only True -> folds, so the return is the only reachable path
            return x + 1.0
        return x - 1.0


class _ReadOnlyAttrCmpPoison:
    def __init__(self) -> None:
        self._thresh = 1.0  # read-only float -> the else is dead
        self._other = True

    def __call__(self, x: float) -> float:
        if self._thresh > 0.5:
            pass
        else:
            self._other = False
        if self._other:
            return x + 2.0
        return x - 2.0


def test_read_only_attr_equality_does_not_poison_later_folds() -> None:
    # A read-only-attr `==` whose dead arm writes another attribute must not mark that attribute written; otherwise a
    # later fold on it is blocked and the return over-rejects. The read-only fixpoint resolves it for bool and float.
    for kernel, delta in ((_ReadOnlyAttrEqPoison().__call__, 1.0), (_ReadOnlyAttrCmpPoison().__call__, 2.0)):
        sim = _model(kernel)
        for x in (1.0, 5.0):
            assert float(sim.run(x)[0]) == x + delta, f"{kernel} x={x}"


class _HelperGuardedStateWrite:
    """
    A called helper whose self-write hides behind a guard the reset snapshot would fold dead -- stale once the entry
    method writes that guard at runtime. The syntactic seed (dead arms included) pins both attributes as state, so
    the helper's guarded write compiles into the same latch CPython computes; a reachability-folded seed would have
    pruned the write and silently diverged.
    """

    def __init__(self) -> None:
        self._flag = False  # snapshot False -> a reachability-folded check folds the guard dead and prunes the write...
        self._x = False  # ...so its assignment below would go undetected; pure-syntactic detection still sees it

    def _arm(self) -> bool:
        if self._flag:
            self._x = True  # a guarded self-write in a called helper -> only the entry method may write state
        return self._x

    def __call__(self, p: bool) -> bool:
        self._flag = p  # the entry owns state; setting the guard at runtime makes the helper's snapshot fold stale
        return self._arm()


def test_guarded_helper_state_write_compiles_and_latches() -> None:
    # The seed must stay purely syntactic even for called helpers: the guarded write pins self._x as state although
    # the reset snapshot folds the guard dead, so the latch behaves exactly as CPython's across transactions.
    sim = _model(_HelperGuardedStateWrite().__call__)
    reference = _HelperGuardedStateWrite()
    for p in (False, False, True, False, True, False):
        assert bool(sim.run(p)[0]) == reference(p), f"p={p}"


class _PropertyShadowsDict:
    """
    A class `property` whose name also has an instance `__dict__` entry: the data descriptor shadows the dict
    slot, so a read of `self._mode` must resolve through the getter (True), not the snapshot value (False).
    """

    @property
    def _mode(self) -> bool:
        return True

    def __init__(self) -> None:
        self.__dict__["_mode"] = (
            False  # a same-named snapshot entry the static folds must NOT read in the getter's place
        )

    def __call__(self, x: float) -> float:
        y = x
        if self._mode:  # via the property (True) -> multiply runs; a snapshot-reading fold would wrongly skip it
            y = x * 2.0
        return y


def test_property_shadowing_dict_entry_resolves_via_getter() -> None:
    # A class property takes precedence over a same-named __dict__ entry (Python data-descriptor rule). The static folds
    # read the snapshot directly, so they must defer such a name to the getter, else the branch folds the wrong way.
    sim = _model(_PropertyShadowsDict().__call__)
    for x in (1.0, 2.0, 3.0):
        assert float(sim.run(x)[0]) == x * 2.0, f"x={x}"


class _PropertySetterWrite:
    """
    A property with a setter, shadowed by a same-named `__dict__` entry. `self.flag = x` invokes the setter in
    Python (descriptor precedence) but would be a plain state-slot store to the dead snapshot entry in the compiler --
    a silent divergence, since the reader inlines the getter. The write must be rejected, not miscompiled.
    """

    @property
    def flag(self) -> bool:
        return self._flag

    @flag.setter
    def flag(self, value: bool) -> None:
        self._flag = value

    def __init__(self) -> None:
        self._flag = False
        self.__dict__["flag"] = False  # a dead shadow entry that descriptor precedence keeps Python from ever reading

    def __call__(self, x: bool, /) -> bool:
        self.flag = x  # Python: setter -> self._flag = x; a plain state-slot store would silently diverge
        return self.flag


def test_property_setter_assignment_is_rejected() -> None:
    # Writing through a property setter is not supported; the assignment must be rejected rather than silently lowered
    # to a state-slot store (which, against a same-named __dict__ shadow, would diverge from the getter-inlined read).
    with pytest.raises(UnsupportedConstruct, match="descriptor"):
        holoso.synthesize(_PropertySetterWrite().__call__, _ops())


class _DataDescriptor:
    """A minimal data descriptor (defines `__set__`, so it takes precedence over a same-named instance `__dict__`)."""

    def __get__(self, instance: object, owner: object) -> bool:
        return True

    def __set__(self, instance: object, value: bool) -> None:
        instance._written = value  # type: ignore[attr-defined]


class _DataDescriptorWrite:
    """
    A custom (non-property) data descriptor shadowed by a __dict__ entry. Like the property setter, the descriptor
    wins for read and write in Python; treating the shadow as a state slot would diverge, so the write is rejected.
    """

    _flag = _DataDescriptor()

    def __init__(self) -> None:
        self._written = False
        self.__dict__["_flag"] = False

    def __call__(self, x: bool, /) -> bool:
        self._flag = x  # Python: descriptor __set__; a state-slot store would diverge
        return self._written


class _DataDescriptorRead:
    """
    A custom data descriptor that is only read. The write check never fires, so the read path must reject it -- its
    getter is arbitrary code, not a stored value, and reading the dead __dict__ shadow as a state slot would diverge.
    """

    _flag = _DataDescriptor()

    def __init__(self) -> None:
        self._written = False
        self.__dict__["_flag"] = False

    def __call__(self, x: float, /) -> float:
        return x * 2.0 if self._flag else x


def test_data_descriptor_write_is_rejected() -> None:
    # A class data descriptor (any object with __set__/__delete__, not only @property) takes precedence over a
    # same-named __dict__ entry; writing it as a plain state slot would diverge from Python's dispatch. Reject it.
    with pytest.raises(UnsupportedConstruct, match="descriptor"):
        holoso.synthesize(_DataDescriptorWrite().__call__, _ops())


def test_data_descriptor_read_is_rejected() -> None:
    # A read-only custom data descriptor is not caught by the write check, so the read path must reject it (only
    # @property getters are synthesizable); else its dead __dict__ shadow would be folded/read as a stored value.
    with pytest.raises(UnsupportedConstruct, match="descriptor"):
        holoso.synthesize(_DataDescriptorRead().__call__, _ops())


class _Meta(type):
    @property
    def flag(cls) -> bool:
        return False  # a metaclass property: governs `Class.flag`, NOT instance access


class _MetaclassPropertyShadow(metaclass=_Meta):
    flag: bool  # declared only for typing; set via __dict__ below, never through this class's own namespace

    def __init__(self) -> None:
        self.__dict__["flag"] = True  # a LIVE instance attribute: the metaclass property does not govern instances

    def __call__(self, x: float, /) -> float:
        return x * 2.0 if self.flag else x  # reads the instance True, not the metaclass property False


class _InterceptingRead:
    """A receiver whose `__getattribute__` rescales one attribute -- an honest idiom, but not one Holoso can read."""

    def __init__(self) -> None:
        self.gain = 2.0

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        return value * 3.0 if name == "gain" else value

    def __call__(self, x: float, /) -> float:
        return x * self.gain  # Python reads 6.0 through the override; the structural read would answer 2.0


class _InterceptingReadWithState(_InterceptingRead):
    def __init__(self) -> None:
        super().__init__()
        self.acc = 0.0

    def __call__(self, x: float, /) -> float:
        self.acc = self.acc + x * self.gain
        return self.acc


def test_intercepting_read_protocol_is_rejected_with_or_without_state() -> None:
    # Attributes are resolved STRUCTURALLY (class metadata then the instance __dict__), never by calling into the
    # object, so an overridden read protocol makes the compiler answer past the value Python would produce. The
    # refusal must not depend on the receiver also writing state: a stateless receiver reads attributes just the same.
    for target in (_InterceptingRead(), _InterceptingReadWithState()):
        with pytest.raises(UnsupportedConstruct, match="__getattribute__"):
            holoso.synthesize(target.__call__, _ops())


def test_metaclass_property_does_not_shadow_instance_attribute() -> None:
    # A descriptor lookup via getattr_static(type(instance), attr) would find the METACLASS property, but a metaclass
    # descriptor governs class access, not instance access -- so the instance __dict__ entry is live ordinary state.
    sim = _model(_MetaclassPropertyShadow().__call__)
    for x in (1.0, 2.0, 3.0):
        assert float(sim.run(x)[0]) == x * 2.0, f"x={x}"


class _CustomSetattr:
    def __init__(self) -> None:
        self._v = False

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)  # routes every write through code the direct-state model cannot mirror

    def __call__(self, x: bool, /) -> bool:
        self._v = x
        return self._v


def test_custom_attribute_access_protocol_is_rejected() -> None:
    # A class overriding the attribute-access protocol (__setattr__ here) routes self.<attr> through arbitrary code the
    # state model cannot mirror; it must be rejected up front rather than silently lowered as direct state access.
    with pytest.raises(UnsupportedConstruct, match="overrides"):
        holoso.synthesize(_CustomSetattr().__call__, _ops())


class _Slotted:
    __slots__ = ("_v",)  # no instance __dict__: the reset-state snapshot cannot read the attributes via vars()

    def __init__(self) -> None:
        self._v = False

    def __call__(self, x: bool, /) -> bool:
        self._v = x
        return self._v


def test_slots_instance_without_dict_is_rejected() -> None:
    # A __slots__ instance has no __dict__, so the reset snapshot (vars(instance)) cannot read its attributes; this must
    # be a clean UnsupportedConstruct, not the raw TypeError that vars() would otherwise raise.
    with pytest.raises(UnsupportedConstruct, match="__slots__|__dict__"):
        holoso.synthesize(_Slotted().__call__, _ops())


def _bool_eq_chain(a: bool, b: bool, c: bool) -> bool:
    return a == b == c  # a runtime 3-link chain: (a == b) and (b == c), each link an xnor


def _bool_mixed_chain(a: bool, b: bool, c: bool) -> bool:
    return a != b == c  # mixed links: (a != b) and (b == c) -- xor then xnor


def test_runtime_bool_comparison_chains() -> None:
    # The chain conjunction (a == b == c -> (a==b) and (b==c)) is a distinct path from the 2-operand compare; a
    # regression mishandling link associativity or applying a float relation to a bool link would miscompile silently.
    eq = _model(_bool_eq_chain)
    mixed = _model(_bool_mixed_chain)
    for a, b, c in itertools.product((False, True), repeat=3):
        assert bool(eq.run(a, b, c)[0]) == (a == b == c), f"eq {a}{b}{c}"
        assert bool(mixed.run(a, b, c)[0]) == (a != b == c), f"mixed {a}{b}{c}"


class _RunningHalf:
    def __init__(self) -> None:
        self._acc = 0.0

    @property
    def _half(self) -> float:
        return self._acc / 2.0  # a property over MUTABLE state -- its value changes within a call and across calls

    def __call__(self, x: float) -> tuple[float, float]:
        before = self._half
        self._acc = self._acc + x
        after = self._half  # the getter must be re-inlined, not reused, so this sees the updated state
        return before, after


def test_property_over_written_state_recomputes_each_read() -> None:
    # A property whose getter reads written state must be inlined fresh at each use (not CSE'd across the intervening
    # write), so the two reads in one call see the old and new state; persistent state also carries across calls.
    sim = _model(_RunningHalf().__call__)
    ref = _RunningHalf()
    for x in (2.0, 4.0, 1.0, 3.0):
        got = tuple(float(v) for v in sim.run(x))
        assert got == ref(x), f"x={x}: {got} != {ref(x)}"

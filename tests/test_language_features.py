"""
Public-API, black-box behavioral tests for the front-end language features landed alongside the UART example: the
boolean ``^`` operator, instance/inherited method calls, ``@property`` reads, and module-level numeric/boolean
constant resolution. Every test drives the compiler ONLY through ``holoso.synthesize(fn, ops)`` and exercises the
elaborated numerical model, asserting on observable output values, so the tests survive a refactor of the front end.

The two rejection checks (``^`` on floats, a state-writing helper) guard real soundness boundaries: without them a
float ``^`` would silently miscompile and a state-mutating helper would be inlined past the entry method's state-slot
analysis -- both behavioral, not mere input validation.
"""

import itertools

import numpy as np
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
)
from holoso._errors import UnsupportedConstruct

_FMT = FloatFormat(4, 8)


def _ops() -> OpConfig:
    return OpConfig(
        FAddOperator(_FMT), FMulOperator(_FMT), FDivOperator(_FMT), FMulILog2OperatorFamily(_FMT), FCmpOperator(_FMT)
    )


def _model(target: object):  # type: ignore[no-untyped-def]
    return holoso.synthesize(target, _ops()).numerical_model.elaborate()


# --------------------------------------------------------------------------------------------------------------------
# Boolean exclusive-or (``^``)
# --------------------------------------------------------------------------------------------------------------------


def _xor2(a: bool, b: bool) -> bool:
    return a ^ b


def _xor_chain(a: bool, b: bool, c: bool, d: bool) -> bool:
    return a ^ b ^ c ^ d  # left-associative chain of the binary operator


def _float_xor(x: float, y: float) -> float:
    return x ^ y  # nonsensical: ``^`` is boolean only


def test_bool_xor_truth_table() -> None:
    sim = _model(_xor2)
    for a, b in itertools.product((False, True), repeat=2):
        assert bool(sim.run(a, b)[0]) == (a != b), f"xor {a} {b}"


def test_bool_xor_chain_is_parity() -> None:
    sim = _model(_xor_chain)
    for bits in itertools.product((False, True), repeat=4):
        assert bool(sim.run(*bits)[0]) == (sum(bits) % 2 == 1), f"parity {bits}"


def test_xor_on_floats_is_rejected() -> None:
    # ``^`` requires boolean operands; a float operand must fail loudly, not silently lower to some float op.
    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(_float_xor, _ops())


# --------------------------------------------------------------------------------------------------------------------
# Instance / inherited method calls
# --------------------------------------------------------------------------------------------------------------------


class _ParityBase:
    """A shared base providing a parity-with-polarity helper, inherited by the synthesized subclass."""

    def __init__(self, odd: bool) -> None:
        self._odd = odd

    def _polarized_parity(self, a: bool, b: bool, c: bool) -> bool:
        even = a ^ b ^ c
        return (not even) if self._odd else even  # self._odd is a folded read-only attribute


class _ParityUser(_ParityBase):
    def __call__(self, a: bool, b: bool, c: bool) -> bool:
        return self._polarized_parity(a, b, c)  # an INHERITED method call, resolved through the MRO


class _StateWriter:
    def __init__(self) -> None:
        self._latch = False

    def _absorb(self, x: bool) -> bool:
        self._latch = x  # a method assigning self state -- must be rejected
        return x

    def __call__(self, x: bool) -> bool:
        return self._absorb(x)


def test_inherited_method_call_even_and_odd() -> None:
    even = _model(_ParityUser(False).__call__)
    odd = _model(_ParityUser(True).__call__)
    for bits in itertools.product((False, True), repeat=3):
        want_even = sum(bits) % 2 == 1
        assert bool(even.run(*bits)[0]) == want_even, f"even {bits}"
        assert bool(odd.run(*bits)[0]) == (not want_even), f"odd {bits}"


def test_method_writing_self_state_is_rejected() -> None:
    # A called method may read self but not write it (the entry method owns the state-slot analysis).
    with pytest.raises(UnsupportedConstruct, match="self attribute"):
        holoso.synthesize(_StateWriter().__call__, _ops())


# --------------------------------------------------------------------------------------------------------------------
# ``@property`` reads
# --------------------------------------------------------------------------------------------------------------------


class _Thresholded:
    def __init__(self, base: float) -> None:
        self._base = base

    @property
    def _threshold(self) -> float:
        return self._base * 2  # a computed read-only value, derived from frozen configuration

    def __call__(self, x: float) -> bool:
        return x >= self._threshold  # reads the property like an attribute


def test_property_read_folds_from_configuration() -> None:
    sim = _model(_Thresholded(3.0).__call__)  # threshold == 6.0
    for x in (0.0, 5.0, 6.0, 7.0, 10.0):
        assert bool(sim.run(x)[0]) == (x >= 6.0), f"x={x}"


# --------------------------------------------------------------------------------------------------------------------
# Module-level constant resolution (and local shadowing)
# --------------------------------------------------------------------------------------------------------------------

_GAIN = 4.0  # a module-level float constant
_LIMIT = 10  # a module-level int constant (resolves as a float in a value position)


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
        # the LOCAL _GAIN (1.0) wins, not the module _GAIN (4.0): x * 1.0, not x * 4.0
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


# --------------------------------------------------------------------------------------------------------------------
# ``@staticmethod`` calls
# --------------------------------------------------------------------------------------------------------------------


class _Scaler:
    def __init__(self, bias: float) -> None:
        self._bias = bias

    @staticmethod
    def _double(x: float) -> float:
        return x * 2  # a static method: no receiver, all arguments bound

    def __call__(self, x: float) -> float:
        return self._double(x) + self._bias  # a ``self.<staticmethod>(...)`` call


def test_staticmethod_call_binds_all_arguments() -> None:
    sim = _model(_Scaler(3.0).__call__)
    for x in (0.0, 1.0, 2.5, -4.0):
        assert float(sim.run(x)[0]) == x * 2 + 3.0, f"x={x}"


def _instance_shadow(x: float) -> float:
    return x + 5  # the callable an instance attribute is bound to, shadowing the class method below


class _ShadowedMethod:
    def __init__(self) -> None:
        self._mix = _instance_shadow  # an instance attribute that shadows the same-named method

    def _mix(self, x: float) -> float:  # type: ignore[no-redef]  # shadowed at runtime by the __init__ binding
        return x * 2

    def __call__(self, x: float) -> float:
        return self._mix(x)  # Python calls the instance attribute (x + 5), NOT the method (x * 2)


def test_method_shadowed_by_instance_attribute_is_rejected() -> None:
    # Python resolves the instance attribute first (a method is a non-data descriptor), so inlining the class method
    # would diverge from Python; the stored attribute is not a synthesizable callable, so the call must be rejected
    # rather than silently miscompiled to the shadowed method.
    with pytest.raises(UnsupportedConstruct, match="stored instance attribute"):
        holoso.synthesize(_ShadowedMethod().__call__, _ops())


# --------------------------------------------------------------------------------------------------------------------
# Boolean ``==`` / ``!=`` (mapping to xnor / xor)
# --------------------------------------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------------------------------------
# Module-level bool/float constants in compile-time-static positions (branch reachability)
# --------------------------------------------------------------------------------------------------------------------

_FEATURE_ON = True  # a module-level bool constant used as a branch condition
_THRESHOLD = 2.0  # a module-level float constant used in a static comparison


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


_CHAIN_A = True  # module bools for a chained static comparison
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
    """A called helper whose self-write hides behind a guard the reset snapshot would fold dead -- stale once the entry
    method writes that guard at runtime. A reachability-folded self-write check prunes the dead write and wrongly
    accepts the helper (then drops the write, diverging from Python); pure-syntactic detection must reject it."""

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


def test_guarded_helper_state_write_is_rejected() -> None:
    # The self-write detection on a called helper must be purely syntactic: a `self.x =` anywhere in the helper rejects
    # it, even under a guard the snapshot would fold dead. A reachability-folded check would prune and silently accept.
    with pytest.raises(UnsupportedConstruct, match="self attribute"):
        holoso.synthesize(_HelperGuardedStateWrite().__call__, _ops())


class _PropertyShadowsDict:
    """A class ``property`` whose name also has an instance ``__dict__`` entry: the data descriptor shadows the dict
    slot, so a read of ``self._mode`` must resolve through the getter (True), not the snapshot value (False)."""

    @property
    def _mode(self) -> bool:
        return True

    def __init__(self) -> None:
        self.__dict__["_mode"] = (
            False  # a same-named snapshot entry the static folds must NOT read in the getter's place
        )

    def __call__(self, x: float) -> float:
        y = x
        if self._mode:  # resolves through the property (True); before the fix the static fold read the snapshot False
            y = x * 2.0
        return y


def test_property_shadowing_dict_entry_resolves_via_getter() -> None:
    # A class property takes precedence over a same-named __dict__ entry (Python data-descriptor rule). The static folds
    # read the snapshot directly, so they must defer such a name to the getter, else the branch folds the wrong way.
    sim = _model(_PropertyShadowsDict().__call__)
    for x in (1.0, 2.0, 3.0):
        assert float(sim.run(x)[0]) == x * 2.0, f"x={x}"


class _PropertySetterWrite:
    """A property with a setter, shadowed by a same-named ``__dict__`` entry. ``self.flag = x`` invokes the setter in
    Python (descriptor precedence) but would be a plain state-slot store to the dead snapshot entry in the compiler --
    a silent divergence, since the reader inlines the getter. The write must be rejected, not miscompiled."""

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
    """A minimal data descriptor (defines ``__set__``, so it takes precedence over a same-named instance __dict__)."""

    def __get__(self, instance: object, owner: object) -> bool:
        return True

    def __set__(self, instance: object, value: bool) -> None:
        instance._written = value  # type: ignore[attr-defined]


class _DataDescriptorWrite:
    """A custom (non-property) data descriptor shadowed by a __dict__ entry. Like the property setter, the descriptor
    wins for read and write in Python; treating the shadow as a state slot would diverge, so the write is rejected."""

    _flag = _DataDescriptor()

    def __init__(self) -> None:
        self._written = False
        self.__dict__["_flag"] = False

    def __call__(self, x: bool, /) -> bool:
        self._flag = x  # Python: descriptor __set__; a state-slot store would diverge
        return self._written


class _DataDescriptorRead:
    """A custom data descriptor that is only read. The write check never fires, so the read path must reject it -- its
    getter is arbitrary code, not a stored value, and reading the dead __dict__ shadow as a state slot would diverge."""

    _flag = _DataDescriptor()

    def __init__(self) -> None:
        self._written = False
        self.__dict__["_flag"] = False

    def __call__(self, x: float, /) -> float:
        return x * 2.0 if self._flag else x  # reads the data descriptor


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
        return False  # a metaclass property: governs ``Class.flag``, NOT instance access


class _MetaclassPropertyShadow(metaclass=_Meta):
    def __init__(self) -> None:
        self.__dict__["flag"] = True  # a LIVE instance attribute: the metaclass property does not govern instances

    def __call__(self, x: float, /) -> float:
        return x * 2.0 if self.flag else x  # reads the instance True, not the metaclass property False


def test_metaclass_property_does_not_shadow_instance_attribute() -> None:
    # A descriptor lookup via getattr_static(type(instance), attr) would find the METACLASS property, but a metaclass
    # descriptor governs class access, not instance access -- so the instance __dict__ entry is live ordinary state.
    sim = _model(_MetaclassPropertyShadow().__call__)
    for x in (1.0, 2.0, 3.0):
        assert (
            float(sim.run(x)[0]) == x * 2.0
        ), f"x={x}"  # flag is the instance True -> doubled, not the metaclass False


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


# --------------------------------------------------------------------------------------------------------------------
# Runtime boolean comparison chains and a property over written state (isolated coverage of two new front-end paths)
# --------------------------------------------------------------------------------------------------------------------


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
        before = self._half  # reads acc/2 before the write
        self._acc = self._acc + x
        after = self._half  # reads the updated acc/2 -- the getter must be re-inlined, not reused
        return before, after


def test_property_over_written_state_recomputes_each_read() -> None:
    # A property whose getter reads written state must be inlined fresh at each use (not CSE'd across the intervening
    # write), so the two reads in one call see the old and new state; persistent state also carries across calls.
    sim = _model(_RunningHalf().__call__)
    ref = _RunningHalf()
    for x in (2.0, 4.0, 1.0, 3.0):
        got = tuple(float(v) for v in sim.run(x))
        assert got == ref(x), f"x={x}: {got} != {ref(x)}"

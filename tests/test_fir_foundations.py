"""Golden unit tests for the FIR semantic foundations: name resolution, the static value domain, and OpSemantics."""

import dataclasses
import math
import symtable
import textwrap
import warnings

import numpy as np
import pytest

from holoso._frontend._fir._opsem import BinOp, UnOp, static_binop, static_compare, static_truth, static_unop
from holoso._frontend._fir._resolve import (
    Builtin,
    Free,
    Global,
    Local,
    Missing,
    NameResolver,
    UnboundCell,
    comprehension_only_targets,
)
from holoso._frontend._fir._fact import Reference
from holoso._frontend._fir._value import (
    MetaInt,
    NpBool,
    NpFloat,
    NpInt,
    StaticArray,
    StaticBool,
    StaticFloat,
    StaticRecord,
    StaticSeq,
    admit,
    as_python,
    same,
)
from holoso._util import RelationalOp

_MODULE_CONSTANT = 42.0
_SHADOWED = "module"


def test_resolver_classifies_params_and_locals() -> None:
    def fn(x: float) -> float:
        y = x + 1.0
        return y

    resolver = NameResolver(fn)
    assert resolver.resolve("x") == Local("x")
    assert resolver.resolve("y") == Local("y")


def test_resolver_reads_module_globals_live() -> None:
    def fn() -> float:
        return _MODULE_CONSTANT

    assert NameResolver(fn).resolve("_MODULE_CONSTANT") == Global("_MODULE_CONSTANT", 42.0)


def test_resolver_prefers_closure_over_same_named_global() -> None:
    # Regression (TODO): the old frontend resolved a captured name to a same-named module global; Python's lookup
    # order puts the closure cell first.
    def outer() -> object:
        _SHADOWED = "closure"  # noqa: F841 -- captured below

        def inner() -> str:
            return _SHADOWED

        return inner

    resolution = NameResolver(outer()).resolve("_SHADOWED")
    assert resolution == Free("_SHADOWED", "closure")


def test_resolver_prefers_captured_builtin_rebinding_over_the_builtin() -> None:
    # Regression (TODO): a captured rebinding of range/len must not be treated as the builtin.
    def outer() -> object:
        def range(n: int) -> list[int]:  # noqa: A001
            return [n]

        def inner() -> object:
            return range(2)

        return inner

    resolution = NameResolver(outer()).resolve("range")
    rebound = resolution.value if isinstance(resolution, Free) else None
    assert callable(rebound) and rebound(2) == [2]


def test_resolver_classifies_builtins_and_missing() -> None:
    def fn(x: float) -> float:
        return abs(x)

    resolver = NameResolver(fn)
    assert resolver.resolve("abs") == Builtin("abs", abs)
    assert resolver.resolve("nonexistent_name") == Missing("nonexistent_name")


def test_resolver_del_makes_a_name_local() -> None:
    # Regression (TODO): `del NAME` anywhere makes the name local throughout in CPython (a later read is
    # UnboundLocalError, never the module global); the code object already encodes that.
    def fn() -> float:
        del _MODULE_CONSTANT  # noqa: F821
        return _MODULE_CONSTANT  # noqa: F821

    assert NameResolver(fn).resolve("_MODULE_CONSTANT") == Local("_MODULE_CONSTANT")


def test_resolver_unbound_cell_raises_its_own_error() -> None:
    def outer() -> object:
        late: float

        def inner() -> float:
            return late

        if False:
            late = 1.0  # noqa: F841
        return inner

    with pytest.raises(UnboundCell):
        NameResolver(outer()).resolve("late")


def test_resolver_pep709_comprehension_target_is_a_function_local() -> None:
    def fn(x: float) -> list[float]:
        return [x for x in (1.0, 2.0)]

    assert NameResolver(fn).resolve("x") == Local("x")


def test_resolver_locals_agree_with_symtable_on_own_source() -> None:
    # symtable on the function's own (module-context) source must agree with the code object about locals; the code
    # object stays authoritative in production because an isolated getsourcelines fragment loses enclosing scopes.
    source = textwrap.dedent("""
        def fn(a):
            b = a + 1
            del b
            c = [i for i in range(3)]
            return c
        """)
    namespace: dict[str, object] = {}
    exec(compile(source, "<fir-test>", "exec"), namespace)  # noqa: S102
    resolver = NameResolver(namespace["fn"])
    table = symtable.symtable(source, "<fir-test>", "exec").get_children()[0]
    symtable_locals = {
        s.get_name() for s in table.get_symbols() if s.is_local() and not s.get_name().startswith(".")
    }  # symtable reports hidden dotted temporaries (e.g. the inlined comprehension's) the code object never names
    assert symtable_locals <= set(resolver.local_names)


def test_admit_keeps_scalar_provenance() -> None:
    assert admit(True) == StaticBool(True)
    assert admit(np.bool_(False)) == NpBool(False)
    assert admit(7) == MetaInt(7)
    assert admit(np.int64(7)) == NpInt(7)
    assert admit(1.5) == StaticFloat(1.5)
    assert admit(np.float64(1.5)) == NpFloat(1.5)
    assert isinstance(as_python(NpInt(7)), np.int64)


def test_admit_containers_and_records() -> None:
    record_type = dataclasses.make_dataclass("Point", [("x", float), ("y", float)])
    admitted = admit([record_type(1.0, 2.0), (3, 4.0)])
    assert isinstance(admitted, StaticSeq) and admitted.is_list
    record, pair = admitted.items
    assert isinstance(record, StaticRecord) and record.klass is record_type
    assert isinstance(pair, StaticSeq) and not pair.is_list and pair.items == (MetaInt(3), StaticFloat(4.0))


def test_admit_refuses_the_open_world() -> None:
    class Arbitrary:
        pass

    assert admit(Arbitrary()) is None
    assert admit(np.array(["a"])) is None  # non-numeric dtype
    assert admit({1: 2}) is None
    assert admit(math.sin) is None  # a callable is never a value; it references through the fact sort


def test_admitted_array_is_frozen_but_the_source_stays_writable() -> None:
    source = np.array([1.0, 2.0])
    admitted = admit(source)
    assert isinstance(admitted, StaticArray)
    with pytest.raises(ValueError):
        admitted.array[0] = 9.0
    source[0] = 9.0  # the caller's array is untouched


def test_same_is_tagged_and_bitwise() -> None:
    assert not same(StaticBool(True), MetaInt(1))  # True is not 1
    assert not same(MetaInt(1), NpInt(1))  # provenance is part of the identity
    assert not same(StaticFloat(0.0), StaticFloat(-0.0))  # signed zero is a change
    assert same(StaticFloat(float("nan")), StaticFloat(float("nan")))  # a NaN cannot oscillate a fixpoint
    assert same(admit(np.array([1, 2])), admit(np.array([1, 2])))  # type: ignore[arg-type]
    assert not same(admit(np.array([1, 2])), admit(np.array([1.0, 2.0])))  # type: ignore[arg-type]  # dtype differs
    assert not same(admit([1.0]), admit((1.0,)))  # type: ignore[arg-type]  # list vs tuple flavor


def test_static_binop_follows_each_provenance() -> None:
    huge = 2**53
    assert static_binop(BinOp.ADD, MetaInt(huge), MetaInt(1)) == MetaInt(huge + 1)  # exact bigint
    assert static_binop(BinOp.DIV, MetaInt(1), MetaInt(3)) == StaticFloat(1 / 3)  # Python true division
    assert static_binop(BinOp.FLOORDIV, MetaInt(7), MetaInt(2)) == MetaInt(3)
    assert static_binop(BinOp.MUL, StaticFloat(0.5), MetaInt(4)) == StaticFloat(2.0)
    mixed = static_binop(BinOp.ADD, NpInt(1), StaticFloat(0.5))
    assert mixed == NpFloat(1.5)  # the result carries numpy provenance, as numpy's own result would


def test_static_binop_defers_errors_and_nan_to_runtime() -> None:
    assert static_binop(BinOp.DIV, MetaInt(1), MetaInt(0)) is None
    assert static_binop(BinOp.DIV, StaticFloat(0.0), StaticFloat(0.0)) is None  # NaN never folds (ZKF has none)
    assert static_binop(BinOp.ADD, StaticSeq((), is_list=True), MetaInt(1)) is None  # sequences are not arithmetic


def test_static_compare_is_exact_for_python_ints_and_numpy_for_np_ints() -> None:
    big, big_float = 2**53 + 1, float(2**53)
    assert static_compare(RelationalOp.EQ, MetaInt(big), StaticFloat(big_float)) == StaticBool(False)  # Python exact
    assert static_compare(RelationalOp.EQ, NpInt(big), StaticFloat(big_float)) == NpBool(True)  # numpy converts
    assert static_compare(RelationalOp.LT, StaticBool(False), StaticBool(True)) is None  # bool ordering not static
    assert static_compare(RelationalOp.NE, StaticBool(False), StaticBool(True)) == StaticBool(True)
    assert static_compare(RelationalOp.EQ, StaticBool(True), MetaInt(1)) is None  # never conflated


def test_static_unop_and_truth() -> None:
    assert static_unop(UnOp.NEG, MetaInt(2**53 + 1)) == MetaInt(-(2**53) - 1)
    assert static_truth(StaticFloat(0.0)) is False
    assert static_truth(MetaInt(-1)) is True
    assert static_truth(StaticSeq((), is_list=False)) is False


def test_admit_restricts_numpy_scalars_to_default_widths() -> None:
    # A narrower numpy dtype wraps at its own width (np.int8(100) + np.int8(100) is -56), which the domain does not
    # model, and a uint64 beyond int64 cannot reconstitute; both are simply not static.
    assert admit(np.int8(100)) is None
    assert admit(np.uint64(2**63 + 5)) is None
    assert admit(np.float32(1.5)) is None
    assert admit(np.int64(7)) == NpInt(7)
    assert admit(np.float64(1.5)) == NpFloat(1.5)


def test_np_float_comparisons_follow_numpy_conversion_rules() -> None:
    big = 2**53
    assert static_compare(RelationalOp.EQ, NpFloat(float(big)), MetaInt(big + 1)) == NpBool(True)  # numpy converts
    assert static_compare(RelationalOp.EQ, StaticFloat(float(big)), MetaInt(big + 1)) == StaticBool(False)  # exact


def test_np_int_zero_division_defers_instead_of_folding_garbage() -> None:
    assert static_binop(BinOp.DIV, NpInt(1), NpInt(0)) is None
    assert static_binop(BinOp.FLOORDIV, NpInt(1), NpInt(0)) is None
    assert static_binop(BinOp.MOD, NpInt(1), NpInt(0)) is None


def test_np_int_wraparound_folds_faithfully() -> None:
    wrapped = static_binop(BinOp.ADD, NpInt(2**62), NpInt(2**62))
    assert wrapped == NpInt(-(2**63))  # numpy's own 64-bit wraparound is the value numpy code would compute


def test_huge_integer_power_defers_instead_of_exhausting_the_compiler() -> None:
    assert static_binop(BinOp.POW, MetaInt(2), MetaInt(10**9)) is None
    assert static_binop(BinOp.POW, MetaInt(2), MetaInt(64)) == MetaInt(2**64)
    assert static_binop(BinOp.POW, MetaInt(2), MetaInt(-2)) == StaticFloat(0.25)


def test_admitted_array_is_a_snapshot() -> None:
    source = np.array([1.0, 2.0])
    admitted = admit(source)
    assert isinstance(admitted, StaticArray)
    source[0] = 9.0
    assert admitted.array[0] == 1.0  # later mutation of the caller's array must not move a folded value


def test_records_capture_init_false_fields_and_refuse_cycles() -> None:
    @dataclasses.dataclass
    class WithInitFalse:
        x: float
        y: float = dataclasses.field(default=0.0, init=False)

    admitted = admit(WithInitFalse(1.0))
    assert admitted == StaticRecord(WithInitFalse, (("x", StaticFloat(1.0)), ("y", StaticFloat(0.0))))

    @dataclasses.dataclass
    class Lazy:
        x: float = dataclasses.field(init=False)

    assert admit(Lazy()) is None  # the unset field raises AttributeError on read: refused, not crashed

    cyclic: list[object] = [1.0]
    cyclic.append(cyclic)
    assert admit(cyclic) is None


def test_admit_normalizes_scalar_subclasses() -> None:
    import enum

    class Flag(enum.IntEnum):
        A = 3

    admitted = admit(Flag.A)
    assert isinstance(admitted, MetaInt) and admitted.value == 3 and type(admitted.value) is int


def test_resolver_pep709_comprehension_only_target_defers_to_enclosing_scope() -> None:
    # The outermost iterable of a comprehension is evaluated in the enclosing scope, so with BOUND bound only as a
    # comprehension target, range(BOUND) reads the module global -- the code object alone would misclassify it Local.
    def fn(x: float) -> list[float]:
        return [x for BOUND in range(BOUND)]  # noqa: F821

    import ast as ast_module
    import inspect
    import textwrap as tw

    fndef = ast_module.parse(tw.dedent(inspect.getsource(fn))).body[0]
    assert isinstance(fndef, ast_module.FunctionDef)
    only = comprehension_only_targets(fndef)
    assert only == frozenset({"BOUND"})
    resolver = NameResolver(fn, comprehension_only=only)
    assert isinstance(resolver.resolve("BOUND"), Global)
    assert resolver.resolve("x") == Local("x")  # a parameter reused as a target elsewhere stays local


BOUND = 5


def test_snapshot_cannot_be_unfrozen_through_as_python() -> None:
    admitted = admit(np.array([1.0, 2.0]))
    assert isinstance(admitted, StaticArray)
    exposed = as_python(admitted)
    assert isinstance(exposed, np.ndarray)
    with pytest.raises(ValueError):
        exposed.setflags(write=True)  # a view of the read-only snapshot cannot be made writable


def test_np_float64_subclass_is_not_static() -> None:
    class SubFloat(np.float64):
        pass

    assert admit(SubFloat(1.5)) is None  # neither exact np.float64 nor exact float: numpy semantics must not be lost


def test_admit_refuses_operator_overriding_subclasses() -> None:
    class WeirdInt(int):
        def __add__(self, other: object) -> int:
            return 0

    class FalsyList(list[float]):
        def __bool__(self) -> bool:
            return False

    class LoudStr(str):
        pass

    assert admit(WeirdInt(7)) is None
    assert admit(FalsyList([1.0])) is None  # its truth would fold from len(), contradicting the object's own __bool__
    assert admit(LoudStr("x")) is None
    assert admit((WeirdInt(7),)) is None  # containers must not smuggle a refused element through


def test_records_capture_observed_state_without_running_constructors() -> None:
    runs: list[int] = []

    @dataclasses.dataclass
    class Doubling:
        x: float

        def __init__(self, x: float) -> None:
            runs.append(0)
            self.x = x * 2.0

    original = Doubling(1.0)
    admitted = admit(original)
    assert admitted == StaticRecord(Doubling, (("x", StaticFloat(2.0)),))  # the OBSERVED state, exactly
    rebuilt = as_python(admitted)
    assert isinstance(rebuilt, Doubling) and rebuilt.x == 2.0 and rebuilt is not original
    assert len(runs) == 1  # neither admission nor reconstruction executes user constructors

    @dataclasses.dataclass
    class Normalizing:
        x: float

        def __post_init__(self) -> None:
            self.x = self.x * 2.0

    assert admit(Normalizing(1.0)) == StaticRecord(Normalizing, (("x", StaticFloat(2.0)),))

    @dataclasses.dataclass(frozen=True)
    class PositionalOnly:
        x: float

        def __init__(self, x: float, /) -> None:
            object.__setattr__(self, "x", x)

    assert admit(PositionalOnly(3.0)) == StaticRecord(PositionalOnly, (("x", StaticFloat(3.0)),))


def test_records_with_state_beyond_fields_are_refused() -> None:
    @dataclasses.dataclass
    class Sneaky:
        x: float
        secret: dataclasses.InitVar[int] = 0

        def __post_init__(self, secret: int) -> None:
            self.stash = secret

    assert admit(Sneaky(1.0, 41)) is None  # no field captures the stash: a rebuild would silently drop it

    @dataclasses.dataclass(slots=True)
    class CleanSlots:
        x: float

    assert admit(CleanSlots(1.0)) == StaticRecord(CleanSlots, (("x", StaticFloat(1.0)),))


def test_private_names_resolve_with_cpython_mangling() -> None:
    class Kernel:
        def fn(self) -> object:
            __local = 1.0  # noqa: F841
            return [__t for __t in range(2)], __VALUE  # noqa: F821

    resolver = NameResolver(Kernel.fn, comprehension_only=frozenset({"__t"}))
    assert resolver.resolve("__VALUE") == Global("_Kernel__VALUE", 23)  # runtime reads the MANGLED global
    assert resolver.resolve("__local") == Local("_Kernel__local")
    assert not isinstance(resolver.resolve("__t"), Local)  # the carve-out subtracts the mangled spelling
    assert resolver.resolve("__dunder__") == Missing("__dunder__")  # dunder spellings never mangle


_Kernel__VALUE = 23
__VALUE = 17  # the unmangled decoy: resolving to this one would repeat the defect


def test_shared_nodes_admit_linearly() -> None:
    node: object = (1.0,)
    for _ in range(60):
        node = (node, node)  # 2**60 paths; only linear-cost traversal can survive this
    wide = admit(node)
    assert isinstance(wide, StaticSeq)
    assert same(wide, wide)
    assert same(wide, admit(node))  # type: ignore[arg-type]  # distinct-but-equal DAGs compare via the pair memo


def test_shared_list_field_aliasing_survives_reconstruction() -> None:
    @dataclasses.dataclass
    class Shared:
        a: list[float]
        b: list[float]

    payload = [1.0, 2.0]
    record = admit(Shared(payload, payload))
    assert isinstance(record, StaticRecord)
    rebuilt = as_python(record)
    assert isinstance(rebuilt, Shared)
    assert rebuilt.a is rebuilt.b  # sharing within one value survives one reconstruction call


def test_records_with_property_shadowed_fields_are_refused() -> None:
    writes: list[float] = []

    @dataclasses.dataclass
    class Shadowed:
        x: float

    def _get(self: object) -> float:
        stored = self.__dict__["x"]
        assert isinstance(stored, float)
        return stored

    def _set(self: object, value: float) -> None:
        writes.append(value)
        self.__dict__["x"] = value * 100.0 if len(writes) > 2 else value

    Shadowed.x = property(_get, _set)  # type: ignore[assignment]
    instance = Shadowed(2.0)
    assert admit(instance) is None  # a faithful-until-the-third-write accessor must never reach as_python()


def test_records_with_convenience_properties_admit() -> None:
    @dataclasses.dataclass
    class Params:
        kp: float
        ki: float

        @property
        def ratio(self) -> float:
            return self.kp / self.ki

    admitted = admit(Params(2.0, 4.0))
    assert admitted == StaticRecord(Params, (("kp", StaticFloat(2.0)), ("ki", StaticFloat(4.0))))
    rebuilt = as_python(admitted)
    assert isinstance(rebuilt, Params) and rebuilt.ratio == 0.5  # field-derived properties survive the rebuild


def test_admit_survives_hostile_metaclasses() -> None:
    class Hostile(type):
        def __getattr__(cls, name: str) -> object:
            raise RuntimeError(name)

    class Weird(metaclass=Hostile):
        pass

    assert admit(Weird()) is None  # refused, not crashed: even is_dataclass() can raise through a metaclass


def test_mangling_ignores_forged_qualname_metadata() -> None:
    def wrapper() -> object:
        return __VALUE  # noqa: F821

    wrapper.__qualname__ = "Kernel.fn"  # what functools.wraps copies from a wrapped method
    assert NameResolver(wrapper).resolve("__VALUE") == Global("__VALUE", 17)  # co_qualname is the lexical truth


def test_admit_bounds_sequence_length() -> None:
    assert admit([0.0] * ((1 << 20) + 1)) is None
    assert isinstance(admit([0.0] * 8), StaticSeq)


def test_resolution_equality_and_hash_key_on_payload_identity() -> None:
    table = np.array([1.0, 2.0])
    assert Global("a", table) == Global("a", table)
    assert isinstance(Global("a", table) == Global("a", table), bool)  # never an elementwise ndarray
    assert Global("a", table) != Global("a", np.array([1.0, 2.0]))  # identity-keyed, like Reference
    assert Global("a", table) != Builtin("a", table)  # the resolution kind is part of the identity
    assert len({Global("a", table), Global("a", table), Free("a", table)}) == 2  # hash is total


def test_float_pow_edges_defer_or_fold_per_provenance() -> None:
    assert static_binop(BinOp.POW, StaticFloat(-8.0), StaticFloat(0.5)) is None  # a complex result never folds
    assert static_binop(BinOp.POW, StaticFloat(1e308), MetaInt(2)) is None  # Python float ** raises OverflowError
    assert static_binop(BinOp.POW, NpFloat(1e308), MetaInt(2)) == NpFloat(math.inf)  # numpy returns inf and folds


def test_reference_equality_and_hash_key_on_referent_identity() -> None:
    payload = [1.0, 2.0]
    assert Reference(payload) == Reference(payload)  # same referent: equal
    assert Reference([1.0, 2.0]) != Reference([1.0, 2.0])  # equal-valued but distinct referents: not equal
    array_ref = Reference(np.array([1.0, 2.0]))
    assert isinstance(array_ref == array_ref, bool)  # never an elementwise ndarray
    assert len({Reference(payload), Reference(payload), Reference(payload)}) == 1  # hash is total and identity-keyed


def test_record_admission_never_runs_constructors_when_nested() -> None:
    runs: list[int] = []

    @dataclasses.dataclass
    class Node:
        child: object

        def __init__(self, child: object) -> None:
            runs.append(0)
            self.child = child

    node: object = 1.0
    for _ in range(12):
        node = Node(node)
    runs.clear()
    assert admit(node) is not None
    assert not runs  # constructor-driven validation was exponential in nesting depth on top of the replay itself


def test_admit_restricts_array_dtypes_to_default_widths() -> None:
    assert admit(np.array([100], dtype=np.int8)) is None
    assert admit(np.array([2**63 + 5], dtype=np.uint64)) is None
    assert isinstance(admit(np.array([1], dtype=np.int64)), StaticArray)
    assert isinstance(admit(np.array([1.0])), StaticArray)


def test_compare_defers_when_numpy_conversion_overflows() -> None:
    assert static_compare(RelationalOp.LT, NpFloat(1.0), MetaInt(2**2000)) is None
    assert static_compare(RelationalOp.GT, MetaInt(2**2000), NpFloat(1.0)) is None
    assert static_compare(RelationalOp.LT, StaticFloat(1.0), MetaInt(2**2000)) == StaticBool(True)  # exact in Python


def test_static_ops_are_immune_to_ambient_numpy_error_state() -> None:
    saved = np.seterr(all="raise")
    try:
        wrapped = static_unop(UnOp.NEG, NpInt(-(2**63)))
        assert wrapped == NpInt(-(2**63))  # numpy wraps int64 negation overflow; ambient seterr must not leak in
        overflowed = static_binop(BinOp.MUL, NpFloat(1e308), NpFloat(1e308))
        assert overflowed == NpFloat(math.inf)
    finally:
        np.seterr(**saved)


def test_subscript_store_target_reads_do_not_count_as_bindings() -> None:
    import ast

    src = textwrap.dedent("""
        def fn(a):
            a[x] = 0
            return [1.0 for x in range(3)]
        """)
    fndef = ast.parse(src).body[0]
    assert isinstance(fndef, ast.FunctionDef)
    assert comprehension_only_targets(fndef) == frozenset({"x"})  # a[x] READS x; only the comprehension binds it


def test_metaint_folds_are_width_bounded_beyond_pow() -> None:
    big = MetaInt(2**60_000)
    assert static_binop(BinOp.MUL, big, big) is None  # a compact squaring chain must not exhaust the compiler
    assert static_binop(BinOp.ADD, big, big) == MetaInt(2**60_001)  # near the bound still folds
    colossal = MetaInt(1 << 8_000_000)
    assert static_binop(BinOp.MUL, colossal, colossal) is None  # the operand pre-check defers before burning CPU
    assert static_unop(UnOp.NEG, colossal) is None


def test_snapshot_cannot_be_unfrozen_through_view_bases() -> None:
    admitted = admit(np.array([1.0, 2.0]))
    assert isinstance(admitted, StaticArray)
    chain: object = as_python(admitted)
    while isinstance(chain, np.ndarray):
        with pytest.raises(ValueError):
            chain.setflags(write=True)
        chain = chain.base


def test_nan_never_participates_in_folds() -> None:
    nan = admit(float("nan"))
    assert isinstance(nan, StaticFloat)
    assert static_truth(nan) is None  # Python says bool(nan) is True, but a static NaN never reaches a decision
    assert static_binop(BinOp.POW, nan, MetaInt(0)) is None  # Python folds nan**0 to 1.0; we defer uniformly
    assert static_unop(UnOp.NEG, nan) is None
    assert static_compare(RelationalOp.EQ, nan, nan) is None


def test_enclosing_scope_stores_do_not_defeat_the_carve_out() -> None:
    import ast

    for source in (
        "def fn(unused=(BOUND := 4)):\n    return [BOUND for BOUND in range(BOUND)]\n",
        "def fn():\n    global BOUND\n    BOUND = 4\n    return [BOUND for BOUND in range(BOUND)]\n",
    ):
        fndef = ast.parse(source).body[0]
        assert isinstance(fndef, ast.FunctionDef)
        only = comprehension_only_targets(fndef)
        assert only == frozenset({"BOUND"})  # neither a default walrus nor a global store is a LOCAL binding
        namespace: dict[str, object] = {}
        exec(compile(source, "<fir-test>", "exec"), namespace)  # noqa: S102
        fn = namespace["fn"]
        assert fn() == [0, 1, 2, 3]  # type: ignore[operator]  # CPython itself reads the module-level BOUND (= 4)
        assert NameResolver(fn, comprehension_only=only).resolve("BOUND") == Global("BOUND", 4)


def test_python_equality_and_hash_are_boolean_for_arrays() -> None:
    a = admit(np.array([1.0, 2.0]))
    b = admit(np.array([1.0, 2.0]))
    assert isinstance(a, StaticArray) and isinstance(b, StaticArray)
    assert a == b and hash(a) == hash(b) and len({a, b}) == 1
    assert admit((np.array([1.0]),)) == admit((np.array([1.0]),))  # no ambiguous-truth crash through containers
    assert a != admit(np.array([1, 2]))  # dtype differs
    assert same(a, b) and not same(a, admit(np.array([1, 2])))  # type: ignore[arg-type]


def test_snapshot_metadata_is_isolated_from_consumers() -> None:
    admitted = admit(np.array([1.0, 2.0, 3.0, 4.0]))
    assert isinstance(admitted, StaticArray)
    exposed = as_python(admitted)
    assert isinstance(exposed, np.ndarray) and exposed is not admitted.array
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # the attack is deprecated in numpy, yet still works
        exposed.dtype = np.int64  # type: ignore[misc]  # runtime allows this reinterpretation despite the stubs
    assert admitted.array.dtype == np.dtype(np.float64)  # the snapshot's metadata cannot be moved by a consumer
    assert same(admitted, admit(np.array([1.0, 2.0, 3.0, 4.0])))  # type: ignore[arg-type]


def test_admit_bounds_logical_array_size() -> None:
    huge = np.broadcast_to(np.float64(1.0), (100_000_000,))
    assert admit(huge) is None  # 8 bytes of storage, 800 MB logical: refused before materialization
    assert isinstance(admit(np.zeros(8)), StaticArray)


def test_resolver_accepts_mappingproxy_builtins() -> None:
    import builtins
    import types

    source = "def fn(x):\n    return abs(x)\n"
    namespace: dict[str, object] = {"__builtins__": types.MappingProxyType(vars(builtins))}
    exec(compile(source, "<fir-test>", "exec"), namespace)  # noqa: S102
    assert NameResolver(namespace["fn"]).resolve("abs") == Builtin("abs", abs)


def test_nested_scope_bindings_do_not_shadow_comprehension_only_targets() -> None:
    import ast

    src = textwrap.dedent("""
        def fn():
            def helper():
                y = 1
                return y
            return [helper() for y in range(2)]
        """)
    fndef = ast.parse(src).body[0]
    assert isinstance(fndef, ast.FunctionDef)
    assert comprehension_only_targets(fndef) == frozenset({"y"})  # helper's own y is not a binding of fn

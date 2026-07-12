"""
The closed static-value domain of the FIR analyzer. Admission is a whitelist: only objects the compiler can evaluate
with defined semantics become static values, everything else is not static (never an error here -- the caller decides).
Provenance is part of the domain because it decides semantics: a Python int compares exactly, while numpy's default
64-bit scalars follow numpy's own conversion rules in mixed comparisons (``np.int64(2**53 + 1) == float(2**53)`` and
``np.float64(2**53) == 2**53 + 1`` are both True there), so they are distinct variants; narrower numpy dtypes carry
width-dependent wraparound the domain does not model, so they are simply not static. Fixed-point equality is tagged
and structural, never Python ``==``: ``True == 1`` and array-valued ndarray comparisons would lie to a convergence
check, and floats compare by bit pattern so a signed zero or a NaN cannot oscillate.
"""

import enum
import math
import struct
import types
from dataclasses import dataclass, fields, is_dataclass

import numpy as np

_MAX_DEPTH = 64
_MAX_ELEMENTS = 1 << 20


@dataclass(frozen=True, slots=True)
class StaticBool:
    value: bool


@dataclass(frozen=True, slots=True)
class MetaInt:
    """An exact arbitrary-precision Python integer; never rounded implicitly."""

    value: int


@dataclass(frozen=True, slots=True)
class NpInt:
    """A numpy int64 scalar: exact in arithmetic (wrapping at 64 bits as numpy does), numpy rules when mixed."""

    value: int


@dataclass(frozen=True, slots=True)
class StaticFloat:
    value: float


@dataclass(frozen=True, slots=True)
class NpFloat:
    """A numpy float64 scalar: bit-identical values to a Python float, numpy's conversion rules when mixed."""

    value: float


@dataclass(frozen=True, slots=True)
class StaticStr:
    value: str


@dataclass(frozen=True, slots=True)
class StaticRange:
    start: int
    stop: int
    step: int


@dataclass(frozen=True, slots=True, eq=False)
class StaticArray:
    """
    A frozen ndarray SNAPSHOT (a private read-only copy): plain ``np.ndarray`` of a numeric dtype only.
    Equality and hash are hand-written because the dataclass-generated ones delegate to the ndarray, whose ``==``
    yields an ndarray (poisoning ``in``/``==`` on any enclosing value) and whose hash raises.
    """

    array: np.ndarray

    def __post_init__(self) -> None:
        assert type(self.array) is np.ndarray and not self.array.flags.writeable

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StaticArray):
            return NotImplemented
        return (
            self.array.dtype == other.array.dtype
            and self.array.shape == other.array.shape
            and self.array.tobytes() == other.array.tobytes()
        )

    def __hash__(self) -> int:
        return hash((str(self.array.dtype), self.array.shape, self.array.tobytes()))


@dataclass(frozen=True, slots=True)
class StaticSeq:
    """A Python list or tuple of static values; ``is_list`` keeps the two flavors apart (list ``+`` concatenates)."""

    items: tuple["StaticValue", ...]
    is_list: bool


@dataclass(frozen=True, slots=True)
class StaticRecord:
    """A dataclass instance: the class by identity plus its admitted field values in declaration order."""

    klass: type
    field_values: tuple[tuple[str, "StaticValue"], ...]


@dataclass(frozen=True, slots=True, eq=False)
class ObjectRef:
    """
    An identity-keyed reference: a callable, module, class, or stateful component object. Equality and hash are
    hand-written to key on the REFERENT's identity: the dataclass-generated ones would call the referent's own
    ``==`` (an ndarray poisons enclosing comparisons) and are partial under hashing (an unhashable referent raises).
    """

    obj: object

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObjectRef):
            return NotImplemented
        return self.obj is other.obj

    def __hash__(self) -> int:
        return hash(id(self.obj))


type StaticValue = (
    StaticBool
    | MetaInt
    | NpInt
    | StaticFloat
    | NpFloat
    | StaticStr
    | StaticRange
    | StaticArray
    | StaticSeq
    | StaticRecord
    | ObjectRef
)


def _mro_attribute(klass: type, name: str) -> object | None:
    # MRO-dict lookup, not getattr: getattr on a class falls through to the metaclass, which never governs
    # instance semantics.
    return next((c.__dict__[name] for c in klass.__mro__ if name in c.__dict__), None)


def admit(obj: object) -> StaticValue | None:
    """
    The static value of a Python object, or None when the object is outside the closed domain. Scalars normalize to
    their base type (an IntEnum member admits as its plain int) and keep their provenance; numpy scalars admit only
    at the default 64-bit widths (a narrower dtype wraps at its own width, which the domain does not model); numpy
    unsigned values beyond int64 admit not at all. Containers admit only if every element does, cycles and
    beyond-depth nesting are refused rather than overflowed, and a dataclass admits only when reconstructible from
    its fields alone. Callables, modules, and classes become identity references only via :func:`admit_ref` -- a
    plain :func:`admit` refuses them so an arbitrary object cannot slip into arithmetic. Aliasing is flattened to
    values -- the kernel subset treats aggregates as immutable, so sharing is unobservable -- but shared nodes are
    admitted once (a DAG costs linear time, not exponential).
    """
    return _admit(obj, {}, frozenset(), _MAX_DEPTH)


def _admit(
    obj: object, memo: dict[int, tuple[object, "StaticValue"]], visiting: frozenset[int], depth: int
) -> StaticValue | None:
    hit = memo.get(id(obj))
    if hit is not None and hit[0] is obj:
        return hit[1]
    value = _admit_uncached(obj, memo, visiting, depth)
    if value is not None:
        memo[id(obj)] = (obj, value)  # the obj reference pins the id against reuse for the memo's lifetime
    return value


def _admit_uncached(
    obj: object, memo: dict[int, tuple[object, "StaticValue"]], visiting: frozenset[int], depth: int
) -> StaticValue | None:
    if depth == 0:
        return None
    # Exact-type checks throughout: a subclass may override operators, and evaluating those inside the compiler
    # would leak foreign semantics into folds. Enum members are the sanctioned subclass exception (inputs are
    # trusted: an enum that redefines arithmetic is not an honest mistake worth modeling); they normalize to the
    # base type on admission.
    if type(obj) is bool or type(obj) is np.bool_:
        return StaticBool(bool(obj))
    if type(obj) is np.int64:
        return NpInt(int(obj))
    if type(obj) is np.float64:
        return NpFloat(float(obj))
    if type(obj) is int or isinstance(obj, enum.IntEnum):
        return MetaInt(int(obj))
    if type(obj) is float:
        return StaticFloat(float(obj))
    if type(obj) is str or isinstance(obj, enum.StrEnum):
        return StaticStr(str(obj))
    if type(obj) is range:
        return StaticRange(obj.start, obj.stop, obj.step)
    if type(obj) is np.ndarray:
        if obj.dtype not in (np.dtype(np.int64), np.dtype(np.float64)):
            return None
        if obj.size > _MAX_ELEMENTS:
            return None  # the LOGICAL size: a zero-stride view is small in memory yet snapshots at full size
        # A snapshot over an immutable bytes buffer: numpy refuses setflags(write=True) anywhere in the view chain,
        # so no consumer can unfreeze it -- not even through .base -- and later caller mutation cannot move a fold.
        snapshot = np.frombuffer(obj.tobytes(), dtype=obj.dtype).reshape(obj.shape)
        return StaticArray(snapshot)
    if isinstance(obj, (list, tuple)) and type(obj) in (list, tuple):  # the isinstance only narrows for mypy
        if id(obj) in visiting or len(obj) > _MAX_ELEMENTS:
            return None
        inner = visiting | {id(obj)}
        items: list[StaticValue] = []
        for element in obj:
            admitted = _admit(element, memo, inner, depth - 1)
            if admitted is None:
                return None
            items.append(admitted)
        return StaticSeq(tuple(items), is_list=type(obj) is list)
    try:
        is_record = is_dataclass(obj) and not isinstance(obj, type)  # a framework metaclass can raise even here
    except Exception:
        return None
    if is_record:
        assert is_dataclass(obj) and not isinstance(obj, type)  # re-narrows for the type checker; proven above
        if id(obj) in visiting:
            return None
        inner = visiting | {id(obj)}
        try:
            field_names = {field.name for field in fields(obj)}
            # Only the declared fields are captured, so instance state beyond them (an honest __post_init__ cache,
            # say) would be silently dropped by the field-only rebuild in _rebuild_record and must refuse instead.
            if set(getattr(obj, "__dict__", ())) - field_names:
                return None
            # A field SHADOWED by a data descriptor (the field+property pattern) reads through user code whose
            # result a field-only rebuild cannot reproduce. Plain non-field properties derive from the fields and
            # survive the rebuild, so they stay admissible; slots member descriptors ARE the fields themselves.
            for name in field_names:
                attr = _mro_attribute(type(obj), name)
                if (
                    attr is not None
                    and not isinstance(attr, types.MemberDescriptorType)
                    and (hasattr(type(attr), "__set__") or hasattr(type(attr), "__delete__"))
                ):
                    return None
            field_values: list[tuple[str, StaticValue]] = []
            for field in fields(obj):
                admitted = _admit(getattr(obj, field.name), memo, inner, depth - 1)
                if admitted is None:
                    return None
                field_values.append((field.name, admitted))
        except Exception:
            return None
        return StaticRecord(type(obj), tuple(field_values))
    return None


def _rebuild_record(
    klass: type, field_values: tuple[tuple[str, "StaticValue"], ...], memo: dict[int, object]
) -> object:
    # By memory, never by protocol: object.__new__ + object.__setattr__ reproduce exactly the admitted fields with
    # no user constructor run, so reconstruction is deterministic and side-effect-free even for frozen/slots classes.
    instance: object = object.__new__(klass)
    for name, item in field_values:
        object.__setattr__(instance, name, _as_python(item, memo))
    return instance


def admit_ref(obj: object) -> ObjectRef:
    """
    A total identity reference: any object outside the value domain (a callable, module, class, stateful component,
    or anything else) is tracked by identity. It never enters arithmetic -- only :func:`admit` produces foldable
    values -- so totality is safe and spares every caller a partiality case.
    """
    return ObjectRef(obj)


def _float_bits(value: float) -> bytes:
    return struct.pack("<d", value)


def same(a: StaticValue, b: StaticValue) -> bool:
    """
    Tagged structural equality for fixed-point convergence. Distinct tags are never equal (True is not 1); floats
    compare by bit pattern (a signed zero flip or a NaN must read as a change exactly once, not oscillate); arrays
    compare by dtype, shape, and contents bits; references compare by identity. Shared nodes compare once (linear
    on a DAG): identical objects are bitwise-equal by construction, and proven-equal pairs are not re-descended.
    """
    return _same(a, b, set())


def _same(a: StaticValue, b: StaticValue, proven: set[tuple[int, int]]) -> bool:
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    key = (id(a), id(b))
    if key in proven:
        return True
    result: bool
    match a, b:
        case (StaticFloat(value=x), StaticFloat(value=y)) | (NpFloat(value=x), NpFloat(value=y)):
            result = _float_bits(x) == _float_bits(y)
        case (StaticSeq(items=x, is_list=lx), StaticSeq(items=y, is_list=ly)):
            result = lx == ly and len(x) == len(y) and all(_same(p, q, proven) for p, q in zip(x, y))
        case (StaticRecord(klass=kx, field_values=x), StaticRecord(klass=ky, field_values=y)):
            result = (
                kx is ky
                and len(x) == len(y)
                and all(nx == ny and _same(p, q, proven) for (nx, p), (ny, q) in zip(x, y))
            )
        case _:
            result = a == b  # scalars, arrays, and references: their own __eq__ is already the doctrine
    if result:
        proven.add(key)
    return result


def as_python(value: StaticValue) -> object:
    """
    The plain Python object a static value denotes, for concrete evaluation through real Python/numpy. The scalar
    provenance survives the round trip (an NpInt reconstitutes as np.int64), so evaluating with the host interpreter
    applies exactly the semantics the variant encodes. The one exception is np.bool_, which admits as a plain
    StaticBool and reconstitutes as Python bool; sound while no fold evaluates bool arithmetic (static_binop
    refuses bools), so a consumer that changes that must first split the variant. A node shared within one value
    reconstructs once per call (linear cost on a DAG, aliasing preserved within the call).
    """
    return _as_python(value, {})


def _as_python(value: StaticValue, memo: dict[int, object]) -> object:
    hit = memo.get(id(value))
    if hit is not None:
        return hit
    result: object
    match value:
        case StaticBool(value=v):
            result = v
        case MetaInt(value=v):
            result = v
        case NpInt(value=v):
            result = np.int64(v)
        case StaticFloat(value=v):
            result = v
        case NpFloat(value=v):
            result = np.float64(v)
        case StaticStr(value=v):
            result = v
        case StaticRange(start=start, stop=stop, step=step):
            result = range(start, stop, step)
        case StaticArray(array=array):
            result = array.view()  # metadata isolation: reassigning the view's shape/dtype cannot touch the snapshot
        case StaticSeq(items=items, is_list=is_list):
            elements = [_as_python(item, memo) for item in items]
            result = elements if is_list else tuple(elements)
        case StaticRecord(klass=klass, field_values=field_values):
            result = _rebuild_record(klass, field_values, memo)
        case ObjectRef(obj=obj):
            result = obj
    memo[id(value)] = result
    return result


def is_nan_free(value: StaticValue) -> bool:
    """ZKF has no NaN, so a NaN constant can never enter the datapath; containers check their leaves once each."""
    return _is_nan_free(value, set())


def _is_nan_free(value: StaticValue, clean: set[int]) -> bool:
    if id(value) in clean:
        return True
    result: bool
    match value:
        case StaticFloat(value=v) | NpFloat(value=v):
            result = not math.isnan(v)
        case StaticArray(array=array):
            result = not bool(np.isnan(array).any()) if array.dtype.kind == "f" else True
        case StaticSeq(items=items):
            result = all(_is_nan_free(item, clean) for item in items)
        case StaticRecord(field_values=field_values):
            result = all(_is_nan_free(item, clean) for _, item in field_values)
        case _:
            result = True
    if result:
        clean.add(id(value))
    return result

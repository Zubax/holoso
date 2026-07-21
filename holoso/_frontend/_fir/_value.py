"""
The closed static-value domain of the FIR analyzer. Admission is a whitelist: only objects the compiler can evaluate
with defined semantics become static values, everything else is not static (never an error here -- the caller decides).
Provenance is part of the domain because it decides semantics: a Python int compares exactly, while numpy scalars
follow numpy's own conversion rules in mixed comparisons (``np.int64(2**53 + 1) == float(2**53)`` and
``np.float64(2**53) == 2**53 + 1`` are both True there), so they are distinct variants. Numeric WIDTH, by contrast,
is immaterial: a narrower numpy scalar or array admits by EXACT value embedding into its category's 64-bit carrier
(bool_/int64/float64), so width-dependent arithmetic artifacts -- int8 wraparound, float32 intermediate rounding --
are not emulated, consistent with the datapath computing in the configured format rather than the source width;
a uint64 beyond the signed-64 range and a longdouble have no exact embedding and stay non-static. Fixed-point
equality is tagged and structural, never Python ``==``: ``True == 1`` and array-valued ndarray comparisons would
lie to a convergence check, and floats compare by bit pattern so a signed zero or a NaN cannot oscillate.
"""

import enum
import math
import struct
import types
from dataclasses import dataclass, fields, is_dataclass

import numpy as np

_MAX_DEPTH = 64
_MAX_ELEMENTS = 1 << 20

# The exact-type membership keeps the subclass doctrine: a user subclass of a numpy scalar never admits.
# The platform alias types (longlong/ulonglong) are distinct exact types on some builds, same-dtype as int64.
_NP_INTEGER_TYPES = (
    np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.longlong, np.ulonglong,
)  # fmt: skip
_NP_FLOAT_TYPES = (np.float16, np.float32, np.float64)  # longdouble has no exact float64 embedding


class SemType(enum.Enum):
    """The runtime semantic kind of a scalar value: float, bool, or signed integer."""

    FLOAT = "float"
    BOOL = "bool"
    INT = "int"


@dataclass(frozen=True, slots=True)
class StaticBool:
    value: bool


@dataclass(frozen=True, slots=True)
class NpBool:
    """
    A numpy boolean, kept distinct from the Python bool exactly as NpInt/NpFloat keep their provenance: numpy 2
    stripped np.bool_ of __index__, so a subscript or repeat count spelled np.True_ is a Python TypeError while the
    plain True is legal, and a reconstruction must reproduce numpy's own arithmetic (np.True_ + np.True_ stays a
    boolean, never 2).
    """

    value: bool


@dataclass(frozen=True, slots=True)
class MetaInt:
    """
    An exact arbitrary-precision Python integer: exact while static or integer-typed, rounding only at a typed
    int-to-float promotion point (accepted C-style under the fastmath charter).
    """

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


@dataclass(frozen=True, slots=True)
class StaticSlice:
    """A slice with integer (or absent) bounds: a pure value, usable as a subscript key like any other Known."""

    start: int | None
    stop: int | None
    step: int | None


@dataclass(frozen=True, slots=True, eq=False)
class StaticArray:
    """
    A frozen ndarray SNAPSHOT (a private read-only copy): plain ``np.ndarray`` of a numeric dtype only.
    Snapshots normalize to C-contiguous layout: memory-order-sensitive operations (``ravel(order="K")`` on a
    transposed table) may observe a different order than the original object -- a documented value-semantics
    deviation.
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


type StaticValue = (
    StaticBool
    | NpBool
    | MetaInt
    | NpInt
    | StaticFloat
    | NpFloat
    | StaticStr
    | StaticRange
    | StaticSlice
    | StaticArray
    | StaticSeq
    | StaticRecord
)


def datapath_value(value: StaticValue) -> bool:
    """
    Whether a static value can become a datapath constant. A non-datapath value (a string, a range, a slice, an
    aggregate) stays fact-only: its every use folds during analysis, so defining a cell for it would only force a
    spurious materialization rejection.
    """
    return isinstance(value, (StaticBool, NpBool, MetaInt, NpInt, StaticFloat, NpFloat))


def value_kind(value: StaticValue) -> SemType:
    """The scalar kind a datapath constant materializes in."""
    assert datapath_value(value)
    if isinstance(value, (StaticBool, NpBool)):
        return SemType.BOOL
    return SemType.INT if isinstance(value, (MetaInt, NpInt)) else SemType.FLOAT


def conform_value(value: StaticValue, kind: SemType) -> StaticValue:
    """
    A datapath constant as the TARGET side of a transfer sees it. Only the integer-into-float store edge changes
    anything: the binary64 image is the value the slot or variable actually holds afterwards, so recording the
    source integer beside a float kind would leave the target-side value un-derivable from the row alone. The
    rounding is exact here because an inexact one is a storage-schema violation refused during analysis.
    """
    if kind is SemType.FLOAT and isinstance(value, (MetaInt, NpInt)):
        image = float(value.value)
        assert int(image) == value.value, "an inexact integer store edge is refused by the storage schema"
        return NpFloat(image) if isinstance(value, NpInt) else StaticFloat(image)
    assert value_kind(value) is kind
    return value


def _mro_attribute(klass: type, name: str) -> object | None:
    # MRO-dict lookup, not getattr: getattr on a class falls through to the metaclass, which never governs
    # instance semantics.
    return next((c.__dict__[name] for c in klass.__mro__ if name in c.__dict__), None)


def admit(obj: object) -> StaticValue | None:
    """
    The static value of a Python object, or None when the object is outside the closed domain. Scalars normalize to
    their base type (an IntEnum member admits as its plain int, full stop); numpy scalars keep their kind but admit
    only at the default 64-bit widths (a narrower dtype wraps at its own width, which the domain does not model); numpy
    unsigned values beyond int64 admit not at all. Containers admit only if every element does, cycles and
    beyond-depth nesting are refused rather than overflowed, and a dataclass admits only when reconstructible from
    its fields alone. Callables, modules, and classes are refused -- references are a separate fact sort, never a
    value -- so an arbitrary object cannot slip into arithmetic. Aliasing is flattened to
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
    # trusted: an enum that redefines arithmetic, shadows its base type's methods, or shadows dataclass fields
    # with descriptors is not an honest mistake worth modeling); they normalize to their plain base value on
    # admission, full stop, so arithmetic folds with base-type semantics and no member identity survives.
    if type(obj) is bool:
        return StaticBool(obj)
    if type(obj) is np.bool_:
        return NpBool(bool(obj))
    if type(obj) in _NP_INTEGER_TYPES:
        assert isinstance(obj, np.integer)
        embedded = int(obj)
        if not -(2**63) <= embedded < 2**63:
            return None  # a uint64 beyond the signed carrier has no exact embedding
        return NpInt(embedded)
    if type(obj) in _NP_FLOAT_TYPES:
        assert isinstance(obj, np.floating)
        return NpFloat(float(obj))
    if type(obj) is int:
        return MetaInt(int(obj))
    if isinstance(obj, enum.IntEnum):
        return MetaInt(int(obj))
    if type(obj) is float:
        return StaticFloat(float(obj))
    if type(obj) is str:
        return StaticStr(str(obj))
    if isinstance(obj, enum.StrEnum):
        return StaticStr(str(obj))
    if type(obj) is range:
        return StaticRange(obj.start, obj.stop, obj.step)
    if type(obj) is slice:
        bounds = (obj.start, obj.stop, obj.step)
        if all(bound is None or type(bound) is int for bound in bounds):
            return StaticSlice(*bounds)
        return None  # a non-integer slice never resolves a supported subscript
    if type(obj) is np.ndarray:
        if obj.ndim == 0:
            return None  # trimmed (scope ruling T3): a 0-d array is an accident, not an idiom; the scalar is
        if obj.size > _MAX_ELEMENTS:
            return None  # the LOGICAL size: a zero-stride view is small in memory yet snapshots at full size
        carrier: type[np.generic]
        # Gates are dtype-KIND based: issubdtype(timedelta64, np.integer) is True yet a duration is not an
        # integer (scaling one rounds to integral units), and dtype equality misses byte-swapped spellings
        # (dtype(">u8") != np.uint64), which must still hit the unsigned boundary below.
        if obj.dtype.kind == "b":
            carrier = np.bool_
        elif obj.dtype.kind in "iu":
            if obj.dtype.kind == "u" and obj.dtype.itemsize == 8 and obj.size and int(obj.max()) >= 2**63:
                return None  # beyond the signed carrier: no exact embedding
            carrier = np.int64
        elif obj.dtype.kind == "f" and obj.dtype.itemsize <= 8:
            carrier = np.float64
        else:
            return None
        # A snapshot over an immutable bytes buffer: numpy refuses setflags(write=True) anywhere in the view chain,
        # so no consumer can unfreeze it -- not even through .base -- and later caller mutation cannot move a fold.
        # The width collapse happens here, once: the snapshot IS the exact carrier embedding of the source values.
        snapshot = np.frombuffer(obj.astype(carrier).tobytes(), dtype=carrier).reshape(obj.shape)
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
    if is_dataclass(obj) and not isinstance(obj, type):
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


def _float_bits(value: float) -> bytes:
    return struct.pack("<d", value)


def same(a: StaticValue, b: StaticValue) -> bool:
    """
    Tagged structural equality for fixed-point convergence. Distinct tags are never equal (True is not 1); floats
    compare by bit pattern (a signed zero flip or a NaN must read as a change exactly once, not oscillate); arrays
    compare by dtype, shape, and contents bits. Shared nodes compare once (linear on a DAG): identical objects
    are bitwise-equal by construction, and proven-equal pairs are not re-descended.
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
            result = a == b  # scalars and arrays: their own __eq__ is already the doctrine
    if result:
        proven.add(key)
    return result


def as_python(value: StaticValue) -> object:
    """
    The plain Python object a static value denotes, for concrete evaluation through real Python/numpy. The scalar
    provenance survives the round trip (an NpInt reconstitutes as np.int64, an NpBool as np.bool_), so evaluating
    with the host interpreter applies exactly the semantics the variant encodes. A node shared within one value
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
        case NpBool(value=v):
            result = np.bool_(v)
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
        case StaticSlice(start=start_bound, stop=stop_bound, step=step_bound):
            result = slice(start_bound, stop_bound, step_bound)
        case StaticArray(array=array):
            result = array.view()  # metadata isolation: reassigning the view's shape/dtype cannot touch the snapshot
        case StaticSeq(items=items, is_list=is_list):
            elements = [_as_python(item, memo) for item in items]
            result = elements if is_list else tuple(elements)
        case StaticRecord(klass=klass, field_values=field_values):
            result = _rebuild_record(klass, field_values, memo)
    memo[id(value)] = result
    return result

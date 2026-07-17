"""Frontend tests: kernel signature contracts -- parameters, ports, and return annotations."""

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from holoso import UnsupportedConstruct
from holoso._frontend import lower
from holoso._frontend._ast_support import port_name
from holoso._hir import BoolType, FloatAdd, FloatDiv, FloatMul, FloatNeg, FloatType, InPort

from ._modelref import arith_count as _arith_count, flatten_value, output_names


def test_scalar_is_output_zero() -> None:
    assert output_names(3.14) == ["out_0"]


def test_flat_sequence_is_positional() -> None:
    assert output_names((1.0, 2.0, 3.0)) == ["out_0", "out_1", "out_2"]


def test_nested_list_row_major_like_ekf1_stateless() -> None:
    # ekf1_stateless's update_x_P returns a 9x1 nested list -> out_0_0 .. out_8_0
    matrix = [[float(i)] for i in range(9)]
    assert output_names(matrix) == [f"out_{i}_0" for i in range(9)]


def test_matrix_n_by_m() -> None:
    matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert output_names(matrix) == ["out_0_0", "out_0_1", "out_0_2", "out_1_0", "out_1_1", "out_1_2"]


def test_dataclass_fields_and_nesting() -> None:
    @dataclasses.dataclass
    class Foo:
        bar: float

    @dataclasses.dataclass
    class Baz:
        foo: Foo

    assert output_names((Baz(Foo(1.0)), 2.0)) == ["out_0_foo_bar", "out_1"]


def test_bare_dataclass_uses_field_names() -> None:
    @dataclasses.dataclass
    class Out:
        x: float
        y: float

    assert output_names(Out(1.0, 2.0)) == ["out_x", "out_y"]


def test_port_name_paths() -> None:
    assert port_name([0]) == "out_0"
    assert port_name([0, "foo", "bar"]) == "out_0_foo_bar"
    assert port_name([3, 1]) == "out_3_1"


def test_flatten_value_returns_leaves() -> None:
    leaves = flatten_value([[1.5], [2.5]])
    assert [value for _, value in leaves] == [1.5, 2.5]


def test_small_kernel_inputs_outputs_and_ops() -> None:
    def kernel(a: float, b: float) -> float:
        return (a - b) * 0.25 + a * b

    hir = lower(kernel)
    assert hir.input_names() == ["a", "b"]
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert _arith_count(hir, FloatMul) == 2
    assert _arith_count(hir, FloatAdd) == 2  # subtraction (add+neg) and the final add
    assert _arith_count(hir, FloatNeg) == 1  # the negation introduced by subtraction


def test_bool_parameter_annotation_becomes_bool_input() -> None:
    def passthrough(flag: bool) -> bool:
        return flag

    hir = lower(passthrough)
    assert hir.input_names() == ["flag"]
    node = hir.nodes[hir.input_ids[0]]
    assert isinstance(node, InPort)
    assert isinstance(node.type, BoolType)
    assert [o.name for o in hir.outputs] == ["out_0"]


def test_float_parameter_annotation_remains_float_input() -> None:
    def passthrough(value: float) -> float:
        return value

    hir = lower(passthrough)
    node = hir.nodes[hir.input_ids[0]]
    assert isinstance(node, InPort)
    assert isinstance(node.type, FloatType)


def test_unsupported_scalar_parameter_annotation_is_rejected() -> None:
    def passthrough(value: str) -> float:  # int is now a supported scalar; str/bytes/complex remain rejected
        return float(len(value))

    with pytest.raises(UnsupportedConstruct, match="parameter annotation"):
        lower(passthrough)


def test_missing_parameter_annotation_is_rejected() -> None:
    # An unannotated parameter is rejected: there is no implicit float default.
    def passthrough(value):  # type: ignore[no-untyped-def]
        return value

    with pytest.raises(UnsupportedConstruct, match="requires an explicit type annotation"):
        lower(passthrough)


def test_ekf1_stateless_structure() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    hir = lower(ekf1_stateless.update_x_P)
    assert len(hir.input_ids) == 17
    assert [o.name for o in hir.outputs] == [f"out_{i}_0" for i in range(9)]
    assert _arith_count(hir, FloatDiv) == 1  # only x22 = 1 / x21


def test_keyword_only_params_become_inputs() -> None:
    def f(a: float, *, b: float, c: float) -> float:
        return a + b + c

    assert lower(f).input_names() == ["a", "b", "c"]


def test_missing_return_annotation_is_rejected() -> None:
    def f(a: float):  # type: ignore[no-untyped-def]
        return a + 1.0

    with pytest.raises(UnsupportedConstruct, match="return type must be explicitly annotated"):
        lower(f)


def test_return_annotation_scalar_type_mismatch_is_rejected() -> None:
    def f(a: float) -> bool:
        return a + 1.0  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="return type mismatch"):
        lower(f)


def test_return_annotation_bool_declared_float_inferred_is_rejected() -> None:
    def f(a: float) -> float:
        return a > 0.0

    with pytest.raises(UnsupportedConstruct, match="return type mismatch"):
        lower(f)


def test_unsupported_return_annotation_is_rejected() -> None:
    def f(a: float) -> int:
        return a  # type: ignore[return-value]  # int is now valid, but a float value cannot match a declared int return

    with pytest.raises(UnsupportedConstruct, match="return type mismatch"):
        lower(f)


def test_scalar_declared_but_tuple_returned_is_rejected() -> None:
    def f(a: float) -> float:
        return a, a  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="declared float, returns an aggregate"):
        lower(f)


def test_return_tuple_arity_mismatch_is_rejected() -> None:
    def f(a: float) -> tuple[float, float, float]:
        return a, a  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="arity mismatch"):
        lower(f)


def test_return_none_declared_but_value_returned_is_rejected() -> None:
    # The return annotation is validated, per the design contract: a ``-> None`` kernel that returns a value is a
    # located mismatch, never silently lowered against its own signature.
    def f(a: float) -> None:
        return a  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="-> None"):
        lower(f)


def test_return_value_declared_but_method_returns_nothing_is_rejected() -> None:
    class Acc:
        def __init__(self) -> None:
            self._acc = 0.0

        def update(self, x: float) -> float:  # type: ignore[return]
            self._acc = self._acc + x

    with pytest.raises(UnsupportedConstruct, match="returns nothing"):
        lower(Acc().update)


def test_tuple_return_annotation_accepted() -> None:
    def f(a: float, b: float) -> tuple[float, bool]:
        return a + b, a > b

    assert [port.name for port in lower(f).outputs] == ["out_0", "out_1"]


def test_variadic_tuple_return_annotation_accepted() -> None:
    def f(a: bool, b: bool) -> tuple[bool, ...]:
        return a, b, a and b

    assert [port.name for port in lower(f).outputs] == ["out_0", "out_1", "out_2"]


def test_list_return_annotation_accepted() -> None:
    def f(a: float, b: float) -> list[float]:
        return [a, b]

    assert [port.name for port in lower(f).outputs] == ["out_0", "out_1"]


def test_none_return_annotation_accepted_for_stateful_method() -> None:
    class Acc:
        def __init__(self) -> None:
            self._acc = 0.0

        def update(self, x: float) -> None:
            self._acc = self._acc + x

    lower(Acc().update)


def test_scalar_returned_but_tuple_declared_is_rejected() -> None:
    # The return annotation is validated, per the design contract: an aggregate annotation demands an aggregate
    # value, so a scalar return under ``tuple[...]`` is a located mismatch.
    def f(a: float) -> tuple[float, float]:
        return a  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="aggregate return"):
        lower(f)


_STATIC_PAIR = np.array([1.0, 2.0])


def test_shaped_array_ports_decompose_per_leaf() -> None:
    # A fixed-shape jaxtyping parameter decomposes into one float input port per leaf, and an array return
    # flattens onto out_* ports, both row-major under the shared indexed-name convention.
    from jaxtyping import Float64

    def array_parameter(v: Float64[np.ndarray, "3"]) -> float:
        return v[0]  # type: ignore[no-any-return]

    def array_return(x: float) -> Float64[np.ndarray, "2"]:
        return _STATIC_PAIR

    assert lower(array_parameter).input_names() == ["v_0", "v_1", "v_2"]
    assert [o.name for o in lower(array_return).outputs] == ["out_0", "out_1"]


def test_malformed_list_return_annotation_is_rejected() -> None:
    def f(a: float) -> list[float, float]:  # type: ignore[type-arg]
        return [a, a]

    with pytest.raises(UnsupportedConstruct, match="exactly one element type"):
        lower(f)


def test_nested_tuple_return_annotation_accepted() -> None:
    def f(a: float, b: float) -> tuple[tuple[float, float], bool]:
        return (a, b), a > b

    assert [port.name for port in lower(f).outputs] == ["out_0_0", "out_0_1", "out_1"]


def test_nested_tuple_return_shape_mismatch_is_rejected() -> None:
    def f(a: float, b: float) -> tuple[tuple[float, float], float]:
        return a, a + b  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="declared a tuple, returns a scalar"):
        lower(f)


def test_explicit_return_none_is_accepted_for_stateful_method() -> None:
    class Acc:
        def __init__(self) -> None:
            self._acc = 0.0

        def update(self, x: float) -> None:
            self._acc = self._acc + x
            return None

    lower(Acc().update)


def test_a_non_datapath_leaf_under_a_scalar_contract_names_the_leaf() -> None:
    # Review round 6: the str leaf hit the generic "cannot materialize" message; the leaf-kind check now runs
    # first, so the rejection names the diverging leaf and the declared kind.
    def kernel(x: float) -> tuple[float, float]:
        return ("tag", x)  # type: ignore[return-value]

    with pytest.raises(
        UnsupportedConstruct, match=r"return type mismatch at leaf \[0\]: declared float, returns a str"
    ):
        lower(kernel)


def test_reference_returns_reject_with_named_contracts_not_assertions() -> None:
    # A reference leaf inside a returned aggregate crashed emission with a raw AssertionError ("read of an
    # undefined place"); a root object return under a value contract claimed "returns nothing". Both are named
    # contract mismatches now.
    def ref_leaf(x: float) -> tuple[float, float]:
        return (x, math)  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="returns an object"):
        lower(ref_leaf)

    def ref_root(x: float) -> float:
        return math  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="returns an object"):
        lower(ref_root)


def test_void_annotated_object_return_is_named() -> None:
    # A non-None object returned under '-> None' was silently discarded as if the kernel returned nothing.
    def kernel(x: float) -> None:
        return math  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="returns an object"):
        lower(kernel)


class _Indented:
    """The kernel is a METHOD so its source is indented, which is what the reparse path must survive."""

    def multiline_literal(self, x: float) -> float:
        table = """
        123456"""
        return x + float(len(table))

    def zero_column_docstring(self, x: float) -> float:
        """A docstring whose continuation sits at
        column zero, defeating a textual dedent."""
        return x + 1.0


def test_indented_kernel_multiline_literal_survives_reparse() -> None:
    # Regression (A1): the source was textually dedented before parsing, stripping the common indent from the
    # INTERIOR of multiline string literals, so len(table) folded to a different constant than Python computes.
    from holoso._hir import FloatConst

    expected = float(len("\n        123456"))  # the kernel's literal: newline + its 8-space interior indent + digits
    hir = lower(_Indented().multiline_literal)
    assert expected in {node.value for node in hir.nodes.values() if isinstance(node, FloatConst)}


def test_indented_kernel_with_zero_column_docstring_line_parses() -> None:
    # Regression (A1): a column-zero line inside the docstring made the textual dedent a no-op, so the still-
    # indented def reached ast.parse and escaped as a bare IndentationError.
    assert [o.name for o in lower(_Indented().zero_column_docstring).outputs] == ["out_0"]


def test_unresolvable_annotation_is_a_located_rejection() -> None:
    # Regression (D2): under PEP 649 the fallback read of __annotations__ evaluated the deferred annotation and
    # let the raw NameError escape as a traceback instead of a located rejection.
    def typoed(x: flaot) -> float:  # type: ignore[name-defined]  # noqa: F821
        return x  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="annotation"):
        lower(typoed)


def _helper_with_typoed_annotation(a: float, b: flaot) -> float:  # type: ignore[name-defined]  # noqa: F821
    return a + b  # type: ignore[no-any-return]


def test_callee_annotations_are_documentation_never_evaluated() -> None:
    # A callee's annotations are documentation, never a lowering directive (and Python never evaluates them
    # either): a typo'd annotation on an inlined helper must not reject the kernel.
    def kernel(x: float) -> float:
        return _helper_with_typoed_annotation(x, 2.0)

    assert [o.name for o in lower(kernel).outputs] == ["out_0"]


class _AnnotatedReceiver:
    def kernel(self: MissingReceiverType, x: float) -> float:  # type: ignore[name-defined]  # noqa: F821
        return x + 2.0


def test_receiver_annotation_is_never_evaluated() -> None:
    # The receiver is not a port, so its annotation is documentation Python itself never evaluates: a kernel
    # whose self-annotation names a missing type must lower exactly as it runs.
    assert _AnnotatedReceiver().kernel(1.0) == 3.0
    assert [o.name for o in lower(_AnnotatedReceiver().kernel).outputs] == ["out_0"]


@dataclasses.dataclass
class _BaseParams:
    gain: float


@dataclasses.dataclass
class _DerivedParams(_BaseParams):
    offset: float


def test_inherited_record_fields_form_ports() -> None:
    # Regression (C3): annotation resolution was own-class-only while dataclasses.fields includes inherited
    # fields, so a derived record parameter rejected at the boundary despite working inside the kernel.
    def kernel(p: _DerivedParams) -> float:
        return p.gain + p.offset

    hir = lower(kernel)
    assert hir.input_names() == ["p_gain", "p_offset"]


@dataclasses.dataclass
class _WithDimsField:
    v: float
    dims: float = 2.0


def test_a_dataclass_with_a_dims_field_is_a_record_not_an_array() -> None:
    # Regression (C4): array detection keyed on hasattr(annotation, "dims"), so a defaulted field named dims
    # (which becomes a class attribute) flipped a working record parameter into a rejected array annotation.
    def kernel(p: _WithDimsField) -> float:
        return p.v + p.dims

    assert lower(kernel).input_names() == ["p_v", "p_dims"]


def test_nested_none_return_annotation_is_a_located_rejection() -> None:
    # Regression (D1): a None nested inside a return annotation hit an internal assertion instead of a located
    # rejection; None is only meaningful as the whole return annotation.
    def kernel(a: float) -> tuple[float, None]:
        return a, None

    with pytest.raises(UnsupportedConstruct, match="None is only meaningful"):
        lower(kernel)


@dataclasses.dataclass
class _RecordWithNoneField:
    x: float
    v: None


@dataclasses.dataclass
class _RecordWithOptionalField:
    x: float
    v: float | None


@dataclasses.dataclass
class _CycleThroughTuple:
    t: "tuple[_CycleThroughTuple, float]"


_CycleThroughTuple.__annotations__["t"] = tuple[_CycleThroughTuple, float]


def test_record_fields_are_component_positions() -> None:
    # Review round: record fields re-entered the top-level parser, so a None field crashed the internal
    # assertion, an X|None field silently unwrapped where a tuple component rejects, and a cycle through a
    # container detour escaped as a raw RecursionError.
    def none_field(a: float) -> _RecordWithNoneField:
        return _RecordWithNoneField(a, None)

    with pytest.raises(UnsupportedConstruct, match="None is only meaningful"):
        lower(none_field)

    def optional_field(a: float) -> _RecordWithOptionalField:
        return _RecordWithOptionalField(a, a)

    with pytest.raises(UnsupportedConstruct, match="expected float, bool"):
        lower(optional_field)

    def cyclic(a: float) -> _CycleThroughTuple:
        return _CycleThroughTuple((_CycleThroughTuple((None, 0.0)), a))  # type: ignore[arg-type]

    with pytest.raises(UnsupportedConstruct, match="recursively contains itself"):
        lower(cyclic)


@dataclasses.dataclass
class _DerivedReturn(_BaseParams):
    offset: float


def test_inherited_record_fields_flatten_on_return_ports() -> None:
    def kernel(a: float) -> _DerivedReturn:
        return _DerivedReturn(a, a + 1.0)

    assert [o.name for o in lower(kernel).outputs] == ["out_gain", "out_offset"]

"""
Desugarer behavior pins: name classification, evaluation-order materialization, guarded conditional shapes, and the
dead-arm narrowing. Assertions are on the canonical printed Eel text — the desugarer's public contract — observed
through the public `frontend_ir[0]` artifact wherever the kernel synthesizes; kernels the pipeline refuses for
reasons unrelated to the printed shape under test (sequence parameter annotations, data-dependent raises,
out-of-word literals) pin the same text on the direct desugar dump.
"""

import inspect
import types
from collections.abc import Callable

import numpy as np
import pytest
from jaxtyping import Float64

import holoso
from holoso import FAddOptions, FCmpOptions, FMulILog2Options, FMulOptions, UnsupportedConstruct
from holoso._eel._desugar import desugar
from holoso._eel._print import print_eel

from ._public import strip_locations

GAIN = 100.0
RANGE_TABLE = (7.0, 8.0)

_OPTIONS = holoso.Options(
    holoso.OperatorOptions(fadd=FAddOptions(), fmul=FMulOptions(), fmul_ilog2=FMulILog2Options(), fcmp=FCmpOptions())
)


def _text(target: object) -> str:
    assert callable(target)
    return strip_locations(holoso.synthesize(target, _OPTIONS, name="k").frontend_ir[0])


def _lines(target: object) -> list[str]:
    return _text(target).splitlines()


def _fn(target: object) -> types.FunctionType:
    fn = target.__func__ if inspect.ismethod(target) else target
    assert isinstance(fn, types.FunctionType)
    return fn


def _desugared_text(target: object) -> str:
    """For the kernels the pipeline refuses for reasons unrelated to the printed shape under test."""
    return print_eel(desugar(_fn(target)))


def _desugared_lines(target: object) -> list[str]:
    return _desugared_text(target).splitlines()


def test_closure_shadowed_by_module_global_is_free() -> None:
    def outer() -> Callable[..., float]:
        GAIN = 2.0  # noqa: F841 -- captured by the kernel, shadowing the module global

        def kernel(x: float) -> float:
            return x * GAIN

        return kernel

    text = _text(outer())
    assert "free GAIN" in text
    assert "env GAIN" not in text


def test_captured_range_rebinding_is_free_not_builtin() -> None:
    def outer() -> Callable[..., float]:
        range = (1.0, 2.0)  # noqa: F841 A001 -- captured, shadowing the builtin

        def kernel(x: float) -> float:
            acc = x
            for step in range:
                acc = acc + step
            return acc

        return kernel

    text = _text(outer())
    assert "free range" in text
    assert "env range" not in text


def test_pep709_comp_iterable_reads_enclosing_scope() -> None:
    def kernel() -> tuple[float, ...]:
        return [RANGE_TABLE * 2.0 for RANGE_TABLE in RANGE_TABLE]  # type: ignore[return-value]  # noqa: B020

    text = _text(kernel)
    assert "env RANGE_TABLE" in text  # the iterable: the module global, per PEP 709
    assert "for RANGE_TABLE$0 in" in text
    assert "RANGE_TABLE$0 * 2.0" in text


def test_assert_captured_cellvar_is_still_local() -> None:
    def kernel(x: float) -> float:
        tol = x * 0.5
        assert (lambda: tol > 0.0)()
        return tol

    code = kernel.__code__
    assert "tol" in code.co_cellvars and "tol" not in code.co_varnames
    text = _text(kernel)
    assert "return tol" in text
    assert "env tol" not in text


def test_assert_only_binding_is_local_not_global() -> None:
    def kernel() -> float:
        assert (w := 1.0) > 0.0
        return w

    text = _desugared_text(kernel)  # synthesize refuses the unrelated unbound read of the assert-only binding
    assert "return w" in text
    assert "env w" not in text


def test_comp_target_shadowing_local_is_renamed() -> None:
    def kernel(vs: Float64[np.ndarray, "2"], s: float) -> float:
        v = 3.0
        r = [v * s for v in vs]
        return v + r[0]  # type: ignore[no-any-return]

    text = _text(kernel)
    assert "for v$0 in vs" in text
    assert "v$0 * s" in text
    assert "v + " in text  # the outer local, untouched by the comprehension


def test_callee_captured_before_arguments() -> None:
    def f1(v: float) -> float:
        return v

    def f2(v: float) -> float:
        return v

    def outer() -> Callable[..., float]:
        def kernel(x: float) -> float:
            return (f := f1)((f := f2)(x))  # noqa: F841

        return kernel

    lines = _lines(outer())
    first = next(i for i, line in enumerate(lines) if line.endswith("= free f1"))
    second = next(i for i, line in enumerate(lines) if line.endswith("= free f2"))
    assert first < second
    callee = lines[first].split(" = ")[0].strip()
    assert lines[-2].strip().startswith("%") and f"call {callee}(" in lines[-2]


def test_starred_argument_evaluates_before_keyword_value() -> None:
    def f4(a: float, b: float, *, k: float) -> float:
        return a + b + k

    def outer() -> Callable[..., float]:
        def kernel(y: float, z: float) -> float:
            return f4(k=z * 2.0, *(y, y))

        return kernel

    lines = _lines(outer())
    star = next(i for i, line in enumerate(lines) if "(y, y)" in line)
    kw = next(i for i, line in enumerate(lines) if "z * 2.0" in line)
    assert star < kw  # CPython evaluates positional/starred before keyword values
    assert any("call" in line and "*%" in line and "k=%" in line for line in lines)


def test_multi_target_assignment_uses_one_temp() -> None:
    def kernel(x: float) -> float:
        a = b = x + 1.0
        return a * b

    lines = _lines(kernel)
    assert any(line.strip() == "%0 = x + 1.0" for line in lines)
    assert any(line.strip() == "a = %0" for line in lines)
    assert any(line.strip() == "b = %0" for line in lines)


def test_mixed_multi_target_index_reads_new_binding() -> None:
    def kernel(t: list[float], i: int) -> None:
        i = t[i] = i + 1

    lines = [line.strip() for line in _desugared_lines(kernel)]
    assert lines.index("i = %0") < lines.index("t[i] = %0")  # the store's index read sees the rebound i


def test_plain_store_evaluates_value_before_index() -> None:
    def kernel(t: list[float], i: int, v: float) -> None:
        t[i + 1] = v + 1.0

    lines = [line.strip() for line in _desugared_lines(kernel)]
    assert lines.index("%0 = v + 1.0") < lines.index("%1 = i + 1")
    assert "t[%1] = %0" in lines


def test_aug_store_evaluates_index_before_value() -> None:
    def kernel(t: list[float], i: int, v: float) -> None:
        t[i + 1] += v + 1.0

    lines = [line.strip() for line in _desugared_lines(kernel)]
    assert lines.index("%0 = i + 1") < lines.index("mark !0") < lines.index("%1 = v + 1.0")
    assert "t[%0] += %1 !0" in lines


def test_attribute_aug_store() -> None:
    class C:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self, x: float) -> float:
            self.y += x * 2.0
            return self.y

    lines = [line.strip() for line in _lines(C().step)]
    assert "self.y += %0 !0" in lines
    assert lines.index("mark !0") < lines.index("%0 = x * 2.0")


def test_assert_is_ignored_wholesale() -> None:
    def kernel(x: float) -> float:
        assert x > 0.0 and {x: x}[x] == x, "unsupported constructs in here are exempt"
        return x

    text = _text(kernel)
    assert "assert" not in text
    assert ">" not in text


def test_dead_arm_still_rejects() -> None:
    def kernel(x: float) -> float:
        if False:
            import os  # noqa: F401
        return x

    with pytest.raises(UnsupportedConstruct, match="unsupported statement: Import") as info:
        holoso.synthesize(kernel, _OPTIONS, name="k")
    assert info.value.location is not None


def test_rejection_reports_outer_construct_before_inner_walrus() -> None:
    def kernel(ys: tuple[float, ...]) -> float:
        return sum(w := y for y in ys)  # noqa: F841

    with pytest.raises(UnsupportedConstruct, match="generator expressions"):
        holoso.synthesize(kernel, _OPTIONS, name="k")


def test_conditional_expression_guards_arms() -> None:
    def kernel(c: bool, a: float, b: float) -> float:
        return a * 2.0 if c else b * 3.0

    lines = [line.strip() for line in _lines(kernel)]
    assert "if c:" in lines
    assert "else:" in lines
    assert sum(1 for line in lines if line.endswith("* 2.0")) == 1
    assert sum(1 for line in lines if line.endswith("* 3.0")) == 1


def test_comparison_chain_guards_suffix_and_shares_operand() -> None:
    def kernel(a: float, b: float, c: float) -> bool:
        return a < b * 2.0 < c

    lines = [line.strip() for line in _lines(kernel)]
    assert lines.count("%0 = b * 2.0") == 1  # the shared operand evaluates once
    assert "%1 = a < %0" in lines
    assert "if %1:" in lines
    assert "%2 = %0 < c" in lines  # inside the guard: evaluated only when the running result holds
    assert "%2 = False" in lines


def test_while_header_reevaluates_condition_temps() -> None:
    def kernel(x: float) -> float:
        while x * 2.0 < 10.0:
            x = x + 1.0
        return x

    lines = [line.strip() for line in _lines(kernel)]
    header_at = lines.index("while:")
    assert lines[header_at + 1] == "%0 = x * 2.0"
    assert lines[header_at + 2] == "%1 = %0 < 10.0"
    assert lines[header_at + 3] == "do %1:"


def test_atomic_while_condition_prints_plainly() -> None:
    def kernel(flag: bool) -> bool:
        while flag:
            flag = False
        return flag

    assert "while flag:" in _text(kernel)


def test_walrus_in_while_test_is_accepted() -> None:
    def kernel(x: float) -> float:
        while (d := x) > 0.0:
            x = x - d
        return x

    lines = [line.strip() for line in _lines(kernel)]
    header_at = lines.index("while:")
    assert lines[header_at + 1] == "d = x"  # rebinds every iteration, and the name persists after the loop


def test_annotated_assignment_forms() -> None:
    def kernel(x: float) -> float:
        v: float = x * 2.0
        w: float  # noqa: F842 -- a bare declaration: no runtime effect
        return v

    text = _text(kernel)
    assert "v = x * 2.0" in text
    assert "w" not in text


def test_negative_literals_fold() -> None:
    def kernel(x: float) -> float:
        return x * -2.5

    assert "x * -2.5" in _text(kernel)


def test_raise_fstring_parts_are_hoisted() -> None:
    def kernel(x: float) -> float:
        if x < 0.0:
            raise ValueError(f"negative input: {x}")
        return x

    text = _desugared_text(kernel)  # synthesize refuses the unrelated data-dependent raise
    assert "raise ValueError 'negative input: ' x" in text


def test_multi_target_atomic_rhs_still_evaluates_once() -> None:
    def kernel(t: tuple[float, float]) -> tuple[float, float]:
        t, second = out = t  # type: ignore[assignment]  # noqa: F841
        return out

    lines = [line.strip() for line in _desugared_lines(kernel)]
    assert "%0 = t" in lines  # the atomic RHS is captured before the first target rebinds t
    assert "t, second = %0" in lines
    assert "out = %0" in lines


class _TripleQuotedRaise:
    def step(self, x: float) -> float:
        if x < 0.0:
            raise ValueError("""one
    two""")
        return x


def test_string_literal_content_survives_indented_source() -> None:
    text = _desugared_text(_TripleQuotedRaise().step)  # synthesize refuses the unrelated data-dependent raise
    assert repr("one\n    two") in text  # dedenting the retrieved source must not reach inside the literal


# fmt: off
class _ColumnZeroContinuation:
    def step(self, x: float) -> float:
        y = (x +
1.0)
        return y
# fmt: on


def test_column_zero_continuation_line_desugars() -> None:
    lines = [line.strip() for line in _lines(_ColumnZeroContinuation().step)]
    assert "y = x + 1.0" in lines


class _ColumnProbe:
    def step(self, x: float) -> float:
        s = {x}  # noqa: F841
        return x


def test_diagnostic_column_matches_original_source() -> None:
    with pytest.raises(UnsupportedConstruct, match="set displays") as info:
        holoso.synthesize(_ColumnProbe().step, _OPTIONS, name="k")
    location = info.value.location
    assert location is not None
    assert location.line is not None
    assert location.line[location.col] == "{"  # columns index the ORIGINAL line, not a dedented copy


def test_diagnostic_column_is_character_exact_after_non_ascii() -> None:
    def kernel(x: float) -> float:
        café = x
        y = café + {x}  # type: ignore[operator]
        return y

    with pytest.raises(UnsupportedConstruct, match="set displays") as info:
        holoso.synthesize(kernel, _OPTIONS, name="k")
    location = info.value.location
    assert location is not None and location.line is not None
    assert location.line[location.col] == "{"  # AST columns are UTF-8 byte offsets; ours must be characters


_M_GLOBAL__x = 42.0


class _MangledCompTarget:
    def step(self) -> float:
        assert [0.0 for __x in (1,)]
        return _M_GLOBAL__x


def test_mangled_assert_comp_target_does_not_shadow_global() -> None:
    code = _MangledCompTarget.step.__code__
    assert "_MangledCompTarget__x" in code.co_varnames  # the comp target, mangled by CPython
    text = _text(_MangledCompTarget().step)
    assert "env _M_GLOBAL__x" in text  # the global read stays a global read, as in CPython


def test_single_target_unpack_prints_distinctly() -> None:
    def kernel(t: tuple[float]) -> float:
        (x,) = t
        return x

    lines = [line.strip() for line in _desugared_lines(kernel)]
    assert "x, = t" in lines  # must not print identically to the plain assignment `x = t`


def test_huge_integer_literals_print_without_a_digit_limit() -> None:
    def kernel() -> int:
        return 0x1_0000_0000_0000_0000_0000  # past 64 bits: prints hex, immune to the CPython repr digit cap

    assert "0x100000000000000000000" in _desugared_text(kernel)  # synthesize refuses the out-of-word literal


def test_nonfinite_float_literal_prints_unambiguously() -> None:
    def kernel(inf: float, c: bool) -> float:
        return 1e309 if c else inf

    text = _text(kernel)
    assert "float('inf')" in text  # the literal folds to infinity, which must not print like the local `inf`


def test_negated_negative_literal_prints_readably() -> None:
    def kernel(x: float) -> float:
        return -(-2.5) * x

    assert "-(-2.5)" in _text(kernel)


def test_multi_dimensional_subscripts_are_retained() -> None:
    def kernel(m: Float64[np.ndarray, "3 3"]) -> tuple[float, Float64[np.ndarray, "3"]]:
        return m[0, 1], m[:, 2]

    lines = [line.strip() for line in _lines(kernel)]
    assert "%0 = m[0, 1]" in lines
    assert "%1 = m[:, 2]" in lines


def test_display_splices_are_retained() -> None:
    def kernel(x: float) -> tuple[float, ...]:
        a = (x, 1.0)
        b = (2.0,)
        return [0.0, *a, *b]  # type: ignore[return-value]

    assert "[0.0, *a, *b]" in _text(kernel)


def test_outer_genexp_rejection_precedes_walrus_collision() -> None:
    def kernel(ys: tuple[float, ...]) -> float:
        w = 0.0
        return sum((w := w + y) for y in ys)

    with pytest.raises(UnsupportedConstruct, match="generator expressions"):
        holoso.synthesize(kernel, _OPTIONS, name="k")

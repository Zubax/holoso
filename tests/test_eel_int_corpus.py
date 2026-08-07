"""
The corpus differential: every kernel is oracle-verified against CPython -- exact ints, multi-transaction,
state included -- and then lowered. The float kernels are the two residual-loop exit shapes plus the FIR and
biquad examples, whose exactness against CPython this suite adds on top of the tolerance-based example matrix.

The integer kernels go one layer further: each is re-run through :class:`MirInterpreter` and compared against the
HIR evaluator transaction by transaction, so selection is judged against the same graph CPython already vouched
for. The two agree only while nothing reaches a rail -- HIR is unbounded, the machine saturates -- which is what
the generous word width below buys.
"""

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

import holoso
from holoso._eel import lower
from holoso._hir import HirEvaluator, optimize
from holoso._mir import MirInterpreter, lower as lower_to_mir
from holoso._value import FloatValue, IntValue, ScalarValue

from ._modelref import build_lir, build_ops
from ._eel_corpus import Crc8, Debouncer, IntUartRx, IntUartTx, Lfsr16, NcoPhase, PriorityEncoder, Pwm
from ._eel_corpus import band_scan, convergence_steps
from ._eeloracle import InputRow, assert_hir_matches_reference

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from biquad import Biquad  # noqa: E402
from fir import Fir4  # noqa: E402


def _uart_tx_vectors() -> list[InputRow]:
    idle = 16 * 11 + 4
    rows: list[InputRow] = [{"start": True, "char": 0xA5}]
    rows += [{"start": False, "char": 0}] * idle
    rows += [{"start": True, "char": 0x0F}, {"start": True, "char": 0xFF}]
    rows += [{"start": False, "char": 0}] * idle
    return rows


def _uart_rx_vectors() -> list[InputRow]:
    def frame(char: int, parity: bool, stop: bool) -> list[bool]:
        bits = [False] + [(char >> i) & 1 == 1 for i in range(8)] + [parity, stop]
        return [level for bit in bits for level in [bit] * OVERSAMPLE]

    OVERSAMPLE = 16
    line = [True] * 8 + frame(0x5A, True, True) + [True] * 20 + frame(0xC3, False, False) + [True] * 20
    line += [False] * 4 + [True] * 20  # a false start: the line recovers before the first mid-bit sample
    return [{"rx": level} for level in line]


def _rows(name: str, values: Sequence[float | bool | int]) -> list[InputRow]:
    return [{name: value} for value in values]


_INT_CASES: list[tuple[str, Callable[[], Callable[..., object]], list[InputRow]]] = [
    ("int_uart_tx_8e1", lambda: IntUartTx(parity=False).__call__, _uart_tx_vectors()),
    ("int_uart_tx_8n1", lambda: IntUartTx(parity=None).__call__, _uart_tx_vectors()),
    ("int_uart_rx_8e1", lambda: IntUartRx(parity=False).__call__, _uart_rx_vectors()),
    ("int_uart_rx_8o1", lambda: IntUartRx(parity=True).__call__, _uart_rx_vectors()),
    ("int_uart_rx_8n1", lambda: IntUartRx(parity=None).__call__, _uart_rx_vectors()),
    ("crc8", lambda: Crc8().step, _rows("byte", [0x31, 0x32, 0x33, 0xFF, 0x00, 0x80, 0x01])),
    ("lfsr16", lambda: Lfsr16().step, _rows("advance", [True] * 20 + [False] * 2 + [True] * 3)),
    ("nco_phase", lambda: NcoPhase().step, _rows("increment", [0x40000000] * 5 + [0x3FFFFFFF] * 3 + [1, 0])),
    ("pwm", lambda: Pwm(top=5).step, _rows("duty", [3] * 12 + [0] * 3 + [5] * 6)),
    ("priority_encoder", lambda: PriorityEncoder().step, _rows("bits", [0b1000, 0b0101, 0b0000, 0xFF, 0x80, 1])),
    (
        "debouncer",
        lambda: Debouncer(n=3).step,
        _rows("raw", [False, True, False, True, True, True, True, False, False, False, True, False, False, False]),
    ),
]

_FLOAT_CASES: list[tuple[str, Callable[[], Callable[..., object]], list[InputRow]]] = [
    ("fir4", lambda: Fir4().__call__, _rows("x", [1.0, 2.0, -1.0, 0.5, 3.0, -2.5, 0.0])),
    ("biquad", lambda: Biquad().__call__, _rows("x", [1.0, 0.0, 0.0, 2.0, -1.0, 0.5, 0.0])),
    (
        "convergence_steps",
        lambda: convergence_steps,
        [{"x": 100.0, "tol": 1.0}, {"x": 3.0, "tol": 8.0}, {"x": 500.0, "tol": 0.0}, {"x": -1.0, "tol": 0.5}],
    ),
    (
        "band_scan",
        lambda: band_scan,
        [{"x": 20.0, "floor": 0.5}, {"x": 1.5, "floor": 0.0}, {"x": 0.25, "floor": 1.0}, {"x": 2.0, "floor": -1.0}],
    ),
]

_CASES = _INT_CASES + _FLOAT_CASES


@pytest.mark.parametrize("name,make,vectors", _CASES, ids=[name for name, _, _ in _CASES])
def test_corpus_oracle(name: str, make: Callable[[], Callable[..., object]], vectors: list[InputRow]) -> None:
    target = make()
    compared = assert_hir_matches_reference(lower(target).hir, target, vectors, label=name)
    assert compared == len(vectors)


def _ops() -> holoso.Options:
    fmt = holoso.FloatFormat(wexp=8, wman=23)
    return holoso.Options(
        holoso.OperatorOptions(
            fadd=holoso.FAddOptions(),
            fmul=holoso.FMulOptions(),
            fdiv=holoso.FDivOptions(),
            fmul_ilog2=holoso.FMulILog2Options(),
            fcmp=holoso.FCmpOptions(),
        ),
        ffmt=fmt,
    )


@pytest.mark.parametrize("name,make,vectors", _FLOAT_CASES, ids=[name for name, _, _ in _FLOAT_CASES])
def test_float_corpus_lowers_through_mir(
    name: str, make: Callable[[], Callable[..., object]], vectors: list[InputRow]
) -> None:
    lower_to_mir(lower(make()).hir, build_ops(_ops()), _ops().ffmt, _ops().ifmt)


def _int_ops() -> holoso.Options:
    """
    ``NcoPhase`` masks with ``0xFFFFFFFF`` and adds a ``2**30`` increment to a value already that wide, so the word
    must hold ``2**33`` for the sum to stay exact -- anything narrower saturates and the comparison below is no
    longer against CPython's arithmetic but against the rails.
    """
    return holoso.Options(
        holoso.OperatorOptions(
            fadd=holoso.FAddOptions(),
            fmul=holoso.FMulOptions(),
            fdiv=holoso.FDivOptions(),
            fmul_ilog2=holoso.FMulILog2Options(),
            fcmp=holoso.FCmpOptions(),
            ffromint=holoso.FFromIntOptions(),
            ftoint=holoso.FToIntOptions(),
        ),
        ffmt=holoso.FloatFormat(wexp=8, wman=23),
        wint_min=34,
    )


def _plain(value: ScalarValue) -> float | bool | int:
    match value:
        case bool():
            return value
        case IntValue():
            return int(value)
        case FloatValue():
            return float(value)


@pytest.mark.parametrize("name,make,vectors", _INT_CASES, ids=[name for name, _, _ in _INT_CASES])
def test_int_corpus_selects_and_agrees_with_the_hir_oracle(
    name: str, make: Callable[[], Callable[..., object]], vectors: list[InputRow]
) -> None:
    options = _int_ops()
    hir = lower(make()).hir
    mir = lower_to_mir(optimize(hir, options.ifconv_max_ops), build_ops(options), options.ffmt, options.ifmt)
    build_lir(mir, name)  # the carriage half: everything selection emits must reach LIR intact
    evaluator, interpreter = HirEvaluator(hir), MirInterpreter(mir)
    names = hir.input_names()
    assert [out.name for out in mir.outputs] == [out.name for out in hir.outputs]
    for index, row in enumerate(vectors):
        arguments = [row[port] for port in names]
        expected = evaluator.run(*arguments)
        assert [_plain(value) for value in interpreter.run(*arguments)] == expected, f"{name}[{index}]"

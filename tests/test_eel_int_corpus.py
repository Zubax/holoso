"""
The corpus differential: every integer kernel is oracle-verified against CPython -- exact ints,
multi-transaction, state included -- proving integers flow through Eel into HIR, and each one's MIR
refusal is recorded (``_reject_integers`` runs until the integer hardware sprint). The float FIR and
biquad kernels ride the same oracle; they lower all the way and graduate to ``examples/`` in M11.
"""

from collections.abc import Callable, Sequence

import pytest

import holoso
from holoso._eel import lower
from holoso._errors import UnsupportedConstruct
from holoso._mir import lower as lower_to_mir

from ._eel_corpus import Biquad, Crc8, Debouncer, Fir4, IntUartRx, IntUartTx, Lfsr16, NcoPhase, PriorityEncoder, Pwm
from ._eeloracle import InputRow, assert_hir_matches_reference


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
    ("fir4", lambda: Fir4((0.25, 0.5, 0.25, 0.125)).step, _rows("x", [1.0, 2.0, -1.0, 0.5, 3.0, -2.5, 0.0])),
    ("biquad", lambda: Biquad((0.2, 0.4, 0.2), (-0.5, 0.25)).step, _rows("x", [1.0, 0.0, 0.0, 2.0, -1.0, 0.5, 0.0])),
]

_CASES = _INT_CASES + _FLOAT_CASES


@pytest.mark.parametrize("name,make,vectors", _CASES, ids=[name for name, _, _ in _CASES])
def test_corpus_oracle(name: str, make: Callable[[], Callable[..., object]], vectors: list[InputRow]) -> None:
    target = make()
    compared = assert_hir_matches_reference(lower(target), target, vectors, label=name)
    assert compared == len(vectors)


def _ops() -> holoso.OpConfig:
    fmt = holoso.FloatFormat(wexp=8, wman=23)
    return holoso.OpConfig(
        holoso.FAddOperator(fmt),
        holoso.FMulOperator(fmt),
        holoso.FDivOperator(fmt),
        holoso.FMulILog2OperatorFamily(fmt),
        holoso.FCmpOperator(fmt),
    )


@pytest.mark.parametrize("name,make,vectors", _INT_CASES, ids=[name for name, _, _ in _INT_CASES])
def test_int_corpus_rejects_at_mir(
    name: str, make: Callable[[], Callable[..., object]], vectors: list[InputRow]
) -> None:
    with pytest.raises(UnsupportedConstruct, match="not yet lowerable to hardware"):
        lower_to_mir(lower(make()), _ops())


@pytest.mark.parametrize("name,make,vectors", _FLOAT_CASES, ids=[name for name, _, _ in _FLOAT_CASES])
def test_float_corpus_lowers_through_mir(
    name: str, make: Callable[[], Callable[..., object]], vectors: list[InputRow]
) -> None:
    lower_to_mir(lower(make()), _ops())

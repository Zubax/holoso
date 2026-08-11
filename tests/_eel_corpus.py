"""
The integer corpus: the kernels the float examples stand in for (``examples/uart.py``'s TODO(integers) notes),
plus the residual-loop exit kernels. This module is the single owner of the integer acceptance matrix
(``INT_CASES`` with its vectors and ``int_corpus_options``); the public model-vs-CPython acceptance, the retained
UART HIR-oracle rows, the MIR-interpreter qualification, and the cosim subset all feed off it.
"""

from collections.abc import Callable, Sequence

import holoso

from ._eeloracle import InputRow

OVERSAMPLE = 16
LAST_PHASE = OVERSAMPLE - 1
HALF_BIT = OVERSAMPLE // 2 - 1


class _IntUartFrame:
    def __init__(self, parity: bool | None) -> None:
        self._parity_present = parity is not None
        self._parity_odd = bool(parity)

    @property
    def _last_index(self) -> int:
        return 10 if self._parity_present else 9

    def _parity_bit(self, char: int) -> bool:
        rest = char
        parity = self._parity_odd
        for _ in range(8):
            parity = parity ^ ((rest & 1) == 1)
            rest = rest >> 1
        return parity


class IntUartTx(_IntUartFrame):
    """
    The integer rewrite of ``examples/uart.py``'s transmitter: the byte shifts out of an int register LSB
    first, so the float version's bit peeling and byte reversal disappear, and the counters are ints.
    """

    def __init__(self, parity: bool | None) -> None:
        super().__init__(parity)
        self._busy = False
        self._phase = 0
        self._index = 0
        self._shift = 0
        self._parity = False

    def __call__(self, start: bool, char: int, /) -> tuple[bool, bool]:
        if not self._busy:
            tx = True
            if start:
                self._busy = True
                self._phase = LAST_PHASE
                self._index = 0
                self._shift = char
                self._parity = self._parity_bit(char)
        else:
            if self._index <= 0:
                tx = False
            elif self._index <= 8:
                tx = (self._shift & 1) == 1
            elif self._index <= 9:
                tx = self._parity if self._parity_present else True
            else:
                tx = True
            if self._phase <= 0:
                if (self._index >= 1) and (self._index <= 8):
                    self._shift = self._shift >> 1
                if self._index >= self._last_index:
                    self._busy = False
                else:
                    self._index = self._index + 1
                    self._phase = LAST_PHASE
            else:
                self._phase = self._phase - 1
        return tx, self._busy


class IntUartRx(_IntUartFrame):
    """The integer receiver: each sampled bit lands in bit 7 and shifts down, rebuilding the byte LSB first."""

    def __init__(self, parity: bool | None) -> None:
        super().__init__(parity)
        self._busy = False
        self._count = 0
        self._index = 0
        self._char = 0
        self._parity_rx = False

    def __call__(self, rx: bool, /) -> tuple[bool, int, bool, bool]:
        valid = False
        parity_error = False
        frame_error = False
        if not self._busy:
            if not rx:
                self._busy = True
                self._count = HALF_BIT
                self._index = 0
                self._char = 0
        elif self._count <= 0:
            if self._index <= 0:
                if rx:
                    self._busy = False
                else:
                    self._count = LAST_PHASE
                    self._index = 1
            elif self._index <= 8:
                self._char = (self._char >> 1) | (int(rx) << 7)
                self._count = LAST_PHASE
                self._index = self._index + 1
            elif self._index < self._last_index:
                self._parity_rx = rx
                self._count = LAST_PHASE
                self._index = self._index + 1
            else:
                valid = True
                self._busy = False
                frame_error = not rx
                parity_error = (self._parity_rx ^ self._parity_bit(self._char)) if self._parity_present else False
        else:
            self._count = self._count - 1
        return valid, self._char, parity_error, frame_error


class Crc8:
    """CRC-8/ATM (polynomial 0x07), one byte per transaction, MSB first."""

    def __init__(self) -> None:
        self.crc = 0

    def step(self, byte: int) -> int:
        acc = self.crc ^ byte
        for _ in range(8):
            if (acc & 0x80) == 0x80:
                acc = ((acc << 1) ^ 0x07) & 0xFF
            else:
                acc = (acc << 1) & 0xFF
        self.crc = acc
        return acc


class Lfsr16:
    """Galois LFSR with taps 0xB400 (maximal 16-bit); ``advance`` gates the shift."""

    def __init__(self) -> None:
        self.state = 0xACE1

    def step(self, advance: bool) -> int:
        if advance:
            if (self.state & 1) == 1:
                self.state = (self.state >> 1) ^ 0xB400
            else:
                self.state = self.state >> 1
        return self.state & 1


class NcoPhase:
    """A 32-bit phase accumulator; the MSB is the square-wave output."""

    def __init__(self) -> None:
        self.phase = 0

    def step(self, increment: int) -> bool:
        self.phase = (self.phase + increment) & 0xFFFFFFFF
        return ((self.phase >> 31) & 1) == 1


class Pwm:
    """A modulo-TOP counter compared against the duty input."""

    def __init__(self, top: int) -> None:
        self._top = top
        self.counter = 0

    def step(self, duty: int) -> bool:
        out = self.counter < duty
        self.counter = self.counter + 1
        if self.counter >= self._top:
            self.counter = 0
        return out


class Debouncer:
    """A counting debouncer: N consecutive disagreeing samples flip the reported level."""

    def __init__(self, n: int) -> None:
        self._n = n
        self.level = False
        self.count = 0

    def step(self, raw: bool) -> bool:
        if raw == self.level:
            self.count = 0
        else:
            self.count = self.count + 1
            if self.count >= self._n:
                self.level = raw
                self.count = 0
        return self.level


class PriorityEncoder:
    """Finds the lowest set bit with a guarded break -- the corpus's loop-exit shape."""

    def __init__(self) -> None:
        self.last = 0

    def step(self, bits: int) -> int:
        position = -1
        for i in range(8):
            if (bits >> i) & 1 == 1:
                position = i
                break
        self.last = position
        return position


def _into_band(x: float) -> float:
    while x > 1.0:
        if x < 2.0:
            return x
        x = x * 0.5
    return x


def band_scan(x: float, floor: float) -> float:
    """A helper returning from inside its own data-dependent scan; its sites converge at the frame exit."""
    return _into_band(x) + floor


def convergence_steps(x: float, tol: float) -> float:
    """Early-exit iteration: halve toward the tolerance, skim the coarse steps, stop after six tallies."""
    err = x
    steps = 0.0
    while err > tol:
        err = err * 0.5
        if err > 16.0:
            continue
        steps = steps + 1.0
        if steps > 6.0:
            break
    return err + steps * 0.001


def rows(name: str, values: Sequence[float | bool | int]) -> list[InputRow]:
    return [{name: value} for value in values]


def uart_tx_vectors() -> list[InputRow]:
    idle = 16 * 11 + 4
    result: list[InputRow] = [{"start": True, "char": 0xA5}]
    result += [{"start": False, "char": 0}] * idle
    result += [{"start": True, "char": 0x0F}, {"start": True, "char": 0xFF}]
    result += [{"start": False, "char": 0}] * idle
    return result


def uart_rx_vectors() -> list[InputRow]:
    def frame(char: int, parity: bool, stop: bool) -> list[bool]:
        bits = [False] + [(char >> i) & 1 == 1 for i in range(8)] + [parity, stop]
        return [level for bit in bits for level in [bit] * OVERSAMPLE]

    line = [True] * 8 + frame(0x5A, True, True) + [True] * 20 + frame(0xC3, False, False) + [True] * 20
    line += [False] * 4 + [True] * 20  # a false start: the line recovers before the first mid-bit sample
    return [{"rx": level} for level in line]


INT_CASES: list[tuple[str, Callable[[], Callable[..., object]], list[InputRow]]] = [
    ("int_uart_tx_8e1", lambda: IntUartTx(parity=False).__call__, uart_tx_vectors()),
    ("int_uart_tx_8n1", lambda: IntUartTx(parity=None).__call__, uart_tx_vectors()),
    ("int_uart_rx_8e1", lambda: IntUartRx(parity=False).__call__, uart_rx_vectors()),
    ("int_uart_rx_8o1", lambda: IntUartRx(parity=True).__call__, uart_rx_vectors()),
    ("int_uart_rx_8n1", lambda: IntUartRx(parity=None).__call__, uart_rx_vectors()),
    ("crc8", lambda: Crc8().step, rows("byte", [0x31, 0x32, 0x33, 0xFF, 0x00, 0x80, 0x01])),
    ("lfsr16", lambda: Lfsr16().step, rows("advance", [True] * 20 + [False] * 2 + [True] * 3)),
    ("nco_phase", lambda: NcoPhase().step, rows("increment", [0x40000000] * 5 + [0x3FFFFFFF] * 3 + [1, 0])),
    ("pwm", lambda: Pwm(top=5).step, rows("duty", [3] * 12 + [0] * 3 + [5] * 6)),
    ("priority_encoder", lambda: PriorityEncoder().step, rows("bits", [0b1000, 0b0101, 0b0000, 0xFF, 0x80, 1])),
    (
        "debouncer",
        lambda: Debouncer(n=3).step,
        rows("raw", [False, True, False, True, True, True, True, False, False, False, True, False, False, False]),
    ),
]


def int_corpus_options() -> holoso.Options:
    """
    ``NcoPhase`` masks with ``0xFFFFFFFF`` and adds a ``2**30`` increment to a value already that wide, so the word
    must hold ``2**33`` for the sum to stay exact -- anything narrower saturates and a comparison against CPython's
    arithmetic becomes a comparison against the rails.
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

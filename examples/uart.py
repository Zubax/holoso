#!/usr/bin/env python3
"""
An ordinary UART receiver/transmitter. This is not a typical use case of Holoso but it illustrates its capabilities.
"""

from pathlib import Path

import holoso

OVERSAMPLE = 16  # ticks per bit period
LAST_PHASE = OVERSAMPLE - 1  # the phase/data countdown runs LAST_PHASE..0 inclusive
HALF_BIT = OVERSAMPLE // 2 - 1  # countdown after start detection, placing the first sample mid-start-bit


class _UartFrame:
    def __init__(self, parity: bool | None) -> None:
        # Frozen at construction and never reassigned by the traced method, so the framing folds away at compile time.
        self._parity_present = parity is not None
        self._parity_odd = bool(parity)

    @property
    def _last_index(self) -> int:
        """The frame-bit index of the stop bit: a parity frame carries one bit more than a bare 8N1 one."""
        return 10 if self._parity_present else 9

    def _parity_bit(self, char: int) -> bool:
        rest = char
        parity = self._parity_odd  # seeding with the polarity leaves the odd chain inverted for free
        for _ in range(8):  # unrolls into a plain eight-input exclusive-or tree
            parity = parity ^ ((rest & 1) == 1)
            rest = rest >> 1
        return parity


class UartTx(_UartFrame):
    """
    A 16x-oversampling UART receiver and transmitter supporting the three common single-byte framings: 8N1 (no parity),
    8E1 (even parity), and 8O1 (odd parity). The host invokes the kernel once per oversample tick and the class drives
    the bit-level state machine internally, the transmitter serialising a latched byte into a start/data/parity/stop
    frame and the receiver detecting the start edge, sampling each bit at its midpoint, and flagging framing/parity
    errors.

    While `busy` is low, assert `start` for one tick with the byte on `char` to begin a frame. `tx` then carries the
    serial line -- idle high, start low, eight data bits LSB first, optional parity, stop high -- holding each bit for
    OVERSAMPLE ticks, and `busy` stays high until the stop bit completes.
    """

    def __init__(self, parity: bool | None) -> None:
        super().__init__(parity)
        self._busy = False
        self._phase = 0  # sub-bit countdown within the current frame bit
        self._index = 0  # which frame bit is on the wire: 0 start, 1..8 data, then parity/stop
        self._shift = 0  # the byte being shifted out, current bit in the LSB, which is UART wire order
        self._parity = False  # computed once at latch

    def tick(self, start: bool, char: int, /) -> tuple[bool, bool]:
        if not self._busy:
            tx = True  # idle line is high
            if start:
                self._busy = True  # the frame begins on the next tick
                self._phase = LAST_PHASE
                self._index = 0
                self._shift = char
                self._parity = self._parity_bit(char)
        else:
            if self._index <= 0:
                tx = False  # start bit
            elif self._index <= 8:
                tx = (self._shift & 1) == 1  # data bit
            elif self._index <= 9:
                tx = self._parity if self._parity_present else True  # parity bit (E/O) or stop bit (N)
            else:
                tx = True  # stop bit (E/O)
            if self._phase <= 0:
                if (self._index >= 1) and (self._index <= 8):
                    self._shift = self._shift >> 1
                if self._index >= self._last_index:
                    self._busy = False  # frame complete; the next tick is idle
                else:
                    self._index += 1
                    self._phase = LAST_PHASE
            else:
                self._phase -= 1
        return tx, self._busy


class UartRx(_UartFrame):
    """
    Feed the serial line on `rx` once per oversample tick. On the stop bit `valid` rises for one tick with the
    recovered byte on `char`, `parity_error` set iff the received parity bit disagrees with the recomputed one (always
    low for 8N1), and `frame_error` set iff the stop bit was not high.
    """

    def __init__(self, parity: bool | None) -> None:
        super().__init__(parity)
        self._busy = False
        self._count = 0  # ticks remaining until the next mid-bit sample
        self._index = 0  # which bit is being sampled: 0 start, 1..8 data, then parity/stop
        self._char = 0  # accumulated bit by bit; only meaningful on the tick `valid` is high
        self._parity_rx = False  # the parity bit as sampled off the wire (E/O only)

    def tick(self, rx: bool, /) -> tuple[bool, int, bool, bool]:
        valid = False
        parity_error = False
        frame_error = False
        if not self._busy:
            if not rx:  # falling edge into the start bit
                self._busy = True
                self._count = HALF_BIT
                self._index = 0
                self._char = 0
        elif self._count <= 0:  # mid-bit sample of frame bit `index`
            if self._index <= 0:
                if rx:
                    self._busy = False  # high mid-start-bit: a false start, abort
                else:
                    self._count = LAST_PHASE
                    self._index = 1
            elif self._index <= 8:
                self._char = (self._char >> 1) | (int(rx) << 7)  # the wire delivers LSB first
                self._count = LAST_PHASE
                self._index += 1
            elif self._index < self._last_index:  # the only bit between data and stop is parity (E/O only)
                self._parity_rx = rx
                self._count = LAST_PHASE
                self._index += 1
            else:
                valid = True  # stop bit: the byte is complete, report status and return to idle
                self._busy = False
                frame_error = not rx
                parity_error = (self._parity_rx ^ self._parity_bit(self._char)) if self._parity_present else False
        else:
            self._count -= 1
        return valid, self._char, parity_error, frame_error


def main() -> None:
    options = holoso.Options(holoso.OperatorOptions())  # bits, counters and flags only: no float operator is built
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    for label, target in (
        ("uart_tx", UartTx(parity=False).tick),  # 8E1: even parity
        ("uart_rx", UartRx(parity=False).tick),
    ):
        result = holoso.synthesize(target, options, name=label)
        for filename, path in result.write(out_dir / label).items():
            print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

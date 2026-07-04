#!/usr/bin/env python3
"""
A 16x-oversampling UART receiver and transmitter supporting the three common single-byte framings -- 8N1 (no parity),
8E1 (even parity), and 8O1 (odd parity). The host invokes ``update()`` once per oversample tick (sixteen ticks per bit
period) and the class drives the bit-level state machine internally: the transmitter serialises a latched byte into a
start / data / parity / stop frame, and the receiver detects the start edge, samples each bit at its midpoint, recovers
the byte, and flags framing/parity errors.

TODO(integers): There is no integer type yet, so the byte and the counters are carried as floats.
FIXME: Every bit-level operation here is a stand-in for an integer bitwise/shift op and collapses to a few lines once
an integer type is available.
"""

from pathlib import Path

import holoso

OVERSAMPLE = 16  # ticks per bit period
LAST_PHASE = OVERSAMPLE - 1  # the phase/data countdown runs LAST_PHASE..0 inclusive (OVERSAMPLE sub-bit ticks)
HALF_BIT = OVERSAMPLE // 2 - 1  # countdown after start detection: places the first sample at the start bit's middle
MSB = 128  # weight of bit 7, the bit a shift register exposes first; only needed until we have integer support


class _UartFrame:
    """
    Construction-frozen frame configuration shared by both directions, selected by ``parity``: ``None`` for 8N1 (no
    parity), ``False`` for 8E1 (even), ``True`` for 8O1 (odd). It is decomposed at construction into two read-only
    booleans -- presence and polarity -- which the traced method never reassigns, so the front-end folds them to
    constants and the framing choice disappears at compile time. ``_parity_bit`` and ``_last_index`` are inherited by
    both ends.
    """

    def __init__(self, parity: bool | None) -> None:
        self._parity_present = parity is not None
        self._parity_odd = bool(parity)

    @property
    def _last_index(self) -> int:
        """The frame-bit index of the stop bit: an even/odd-parity frame carries one extra bit over a bare 8N1 one."""
        return 10 if self._parity_present else 9

    def _parity_bit(self, char: float) -> bool:
        """
        The parity bit for one byte: the exclusive-or of its eight bits (even parity), inverted for odd parity. The
        reduction seeds the accumulator with the polarity (odd seeds True, so the chain ends inverted) and peels one
        bit per turn by testing the MSB and shifting left; XOR is order-independent, so MSB-first peeling is fine.
        TODO(integers): this shift-and-test loop is a population-count parity that an integer type collapses to one op.
        """
        rest = char
        parity = self._parity_odd
        for _ in range(8):
            bit = rest >= MSB
            rest = (rest - MSB if bit else rest) * 2
            parity = parity ^ bit
        return parity


class UartTx(_UartFrame):
    """
    UART transmitter. While ``busy`` is low, assert ``start`` for one tick with the byte on ``char`` to begin a frame;
    ``tx`` then carries the serial line (idle high, start low, eight data bits LSB first, optional parity, stop high),
    holding each bit for OVERSAMPLE ticks, and ``busy`` stays high until the stop bit completes.
    """

    def __init__(self, parity: bool | None) -> None:
        super().__init__(parity)
        self._busy = False
        self._phase = 0  # sub-bit countdown LAST_PHASE..0 within the current frame bit
        self._index = 0  # which frame bit is on the wire: 0 start, 1..8 data, then parity/stop
        self._shift = 0.0  # the byte being shifted out, current bit in the MSB
        self._parity = False  # the polarized parity bit, computed once at latch

    @staticmethod
    def _reverse_byte(char: float) -> float:
        """
        Reverse the eight bits of a byte: peel ``char`` most-significant bit first and shift each peeled bit into the
        top of the result, so the original LSB ends up in the MSB. Shifting the reversed byte out MSB first then emits
        the original byte LSB first (standard UART order). TODO(integers): a one-instruction bit-reversal with integers.
        """
        rest = char
        rev = 0.0
        for _ in range(8):
            bit = rest >= MSB
            rest = (rest - MSB if bit else rest) * 2
            rev = rev / 2 + (MSB if bit else 0)
        return rev

    def __call__(self, start: bool, char: float, /) -> tuple[bool, bool]:
        if not self._busy:
            tx = True  # idle line is high
            if start:
                self._busy = True  # the frame begins (start bit) on the next tick
                self._phase = LAST_PHASE
                self._index = 0
                self._shift = self._reverse_byte(char)  # reversed so the MSB-first shift-out emits the byte LSB first
                self._parity = self._parity_bit(char)
        else:
            if self._index <= 0:
                tx = False  # start bit
            elif self._index <= 8:
                # the reversed byte's MSB is the original LSB, so the wire carries the byte LSB first
                tx = self._shift >= MSB  # data bit
            elif self._index <= 9:
                tx = self._parity if self._parity_present else True  # parity bit (E/O) or stop bit (N)
            else:
                tx = True  # stop bit (E/O)
            if self._phase <= 0:
                if (self._index >= 1) and (self._index <= 8):
                    # drop the bit just sent and expose the next in the MSB
                    self._shift = (self._shift - MSB if self._shift >= MSB else self._shift) * 2
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
    UART receiver. Feed the serial line on ``rx`` once per oversample tick. On the falling start edge the machine arms,
    samples each subsequent bit at its midpoint, and on the stop bit raises ``valid`` for one tick with the recovered
    byte on ``char``, ``parity_error`` set iff the received parity bit disagrees with the recomputed one (always low for
    8N1), and ``frame_error`` set iff the stop bit was not high.
    """

    def __init__(self, parity: bool | None) -> None:
        super().__init__(parity)
        self._busy = False
        self._count = 0  # ticks remaining until the next mid-bit sample
        self._index = 0  # which bit is being sampled: 0 start, 1..8 data, then parity/stop
        self._char = 0.0  # the byte, accumulated bit by bit; only meaningful on the tick ``valid`` is high
        self._parity_rx = False  # the parity bit as sampled off the wire (E/O only)

    def __call__(self, rx: bool, /) -> tuple[bool, float, bool, bool]:
        valid = False
        parity_error = False
        frame_error = False
        if not self._busy:
            if not rx:  # falling edge into the start bit
                self._busy = True
                self._count = HALF_BIT  # the first sample lands at the middle of the start bit
                self._index = 0
                self._char = 0  # begin a fresh byte
        elif self._count <= 0:
            # Mid-bit sample of frame bit ``index``.
            if self._index <= 0:
                if rx:
                    self._busy = False  # the line is high at the middle of the start bit: a false start, abort
                else:
                    self._count = LAST_PHASE
                    self._index = 1
            elif self._index <= 8:
                self._char = self._char / 2 + float(rx) * MSB  # shift the data bit into the top: rebuilds LSB first
                self._count = LAST_PHASE
                self._index += 1
            elif self._index < self._last_index:
                # the only bit between the data and the stop bit is the parity bit (E/O frames only)
                self._parity_rx = rx
                self._count = LAST_PHASE
                self._index += 1
            else:
                # Stop bit: the byte is complete in ``char``; report status and return to idle.
                valid = True
                self._busy = False
                frame_error = not rx  # the stop bit must be high
                # a parity error is a mismatch between the received and recomputed bit: exactly an exclusive-or
                parity_error = (self._parity_rx ^ self._parity_bit(self._char)) if self._parity_present else False
        else:
            self._count -= 1
        return valid, self._char, parity_error, frame_error


def main() -> None:
    # The narrowest ZKF format that holds a 0..255 byte exactly: wman=8 gives an 8-bit significand
    # (integers 0..256 exact), and wexp=4 is the smallest exponent field reaching 2^7..2^8
    # (largest finite magnitude 255).
    # TODO this is temporary until we have integer support.
    float_fmt = holoso.FloatFormat(wexp=4, wman=8)
    ops = holoso.OpConfig(
        holoso.FAddOperator(float_fmt),
        holoso.FMulOperator(float_fmt),
        holoso.FDivOperator(float_fmt),
        holoso.FMulILog2OperatorFamily(float_fmt),
        holoso.FCmpOperator(float_fmt),
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    for label, target in (
        ("uart_tx", UartTx(parity=False).__call__),  # 8E1: even parity
        ("uart_rx", UartRx(parity=False).__call__),
    ):
        result = holoso.synthesize(target, ops, name=label)
        for filename, path in result.write(out_dir / label).items():
            print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

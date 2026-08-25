#!/usr/bin/env python3
"""
Automatic gain control for a 1920x1200 image sensor: pixels are corrected as they stream in, sixteen per
transaction, while the statistics of the frame accumulate behind them. The mean brightness a frame measures sets
the gain applied to the frames after it, as the sensor's own gain would be, so no frame is ever buffered.

A beat that ends no frame takes ~51 cycles, or 255 ns at 200 MHz, and a frame is 1920*1200/16 = 144000 beats, so
36.7 ms and 27 FPS. A wider beat buys frame rate at the cost of kernel area, which is often a better trade than a
conventional II=1 pipeline: time-sharing one set of operators utilizes the fabric better.
"""

from pathlib import Path

import numpy as np
from jaxtyping import UInt8

import holoso

PIXEL_MAX = 255
GAIN_MAX = 16.0
FRAME_PIXELS = 1920 * 1200

type PixelBeat = UInt8[np.ndarray, "16"]


class ImageAgc:
    def __init__(self, *, frame_pixels: int = FRAME_PIXELS) -> None:
        self._frame_pixels = frame_pixels
        self.gain: float = 1.0
        self._sum: int = 0
        self._count: int = 0

    @property
    def _mean_brightness(self) -> float:
        return self._sum / self._frame_pixels  # a constant divisor: the mean costs a scale, not a division

    def __call__(self, pixels: PixelBeat, target: int) -> PixelBeat:
        out = np.array([min(round(pixel * self.gain), PIXEL_MAX) for pixel in pixels])
        self._sum += pixels.sum()
        self._count += len(pixels)
        if self._count >= self._frame_pixels:
            self.gain = min(target / max(self._mean_brightness, 1.0), GAIN_MAX)  # the floor: a black frame reads zero
            self._sum = self._count = 0
        return out


def main() -> None:
    options = holoso.Options(
        holoso.OperatorOptions(
            fmul=holoso.FMulOptions(),
            fdiv=holoso.FDivOptions(),
            fsort=holoso.FSortOptions(),
            ffromint=holoso.FFromIntOptions(),
            ftoint=holoso.FToIntOptions(),
        ),
        ffmt=holoso.FloatFormat(wexp=8, wman=36),
        wint_min=31,  # the frame sum reaches 1920*1200*255, plus the sign bit
    )
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem
    result = holoso.synthesize(ImageAgc().__call__, options)
    for filename, path in result.write(out_dir).items():
        print(f"{filename}: {path}")


if __name__ == "__main__":
    main()

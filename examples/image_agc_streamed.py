#!/usr/bin/env python3
"""
Automatic exposure and gain control for a 1920x1200 image sensor: pixels are corrected as they stream in, sixteen
per transaction, while the statistics of the frame accumulate behind them, so no frame is ever buffered.

The controller is a feedback loop in stops (log2 units) with each frame moving the demand a fraction of the way toward
the target, which keeps the loop stable through the frame or two of latency a sensor takes to apply a new exposure,
and a deadband stops it dithering once it is close. The demand is served by three actuators in order of noise cost:
exposure time first, then analog gain, digital gain last; the two sensor-side commands are read from the state ports
and forwarded to the sensor by the host. Metering is centre-weighted, and a frame that saturates more than a small
fraction of its pixels refuses any further increase, so highlights are not traded for the mean.

A beat that ends no frame takes 39 cycles, or 195 ns at 200 MHz, and a frame is 1920*1200/16 = 144000 beats, so
28.1 ms and 35 FPS max. A wider beat buys frame rate at the cost of kernel area, which is often a better trade than a
conventional II=1 pipeline because time-sharing one set of operators utilizes the fabric better.
"""

import math
from pathlib import Path

import numpy as np
from jaxtyping import UInt8

import holoso

PIXEL_MAX = 255
WIDTH = 1920
HEIGHT = 1200
BEAT = 16

EXPOSURE_MIN_s = 1 / 8000
EXPOSURE_STOPS = 8.0  # up to 1/31 s
ANALOG_STOPS = 4.0  # up to 16x
DIGITAL_STOPS = 2.0  # up to 4x
TOTAL_STOPS = EXPOSURE_STOPS + ANALOG_STOPS + DIGITAL_STOPS

DAMPING = 0.25  # the fraction of the error corrected per frame
DEADBAND_STOPS = 1 / 16
CENTRE_WEIGHT = 0.75  # the weight of the central quarter of the frame in the metered brightness
CLIP_FRACTION_MAX = 0.01

type PixelBeat = UInt8[np.ndarray, "16"]


class ImageAgc:
    """
    Each call takes one beat of raw pixels in scan order and the target brighness the loop steers toward on the
    same 0..255 scale as the pixels; returns the same beat with the digital gain applied.
    The sensor-side commands are the public state: `exposure_s` and `analog_gain`, updated once per frame.
    """

    def __init__(self, *, width: int = WIDTH, height: int = HEIGHT) -> None:
        self._beats_per_row = width // BEAT
        self._rows = height
        self._centre_x0, self._centre_x1 = self._beats_per_row // 4, 3 * self._beats_per_row // 4
        self._centre_y0, self._centre_y1 = height // 4, 3 * height // 4
        self._frame_pixels = width * height
        self._centre_pixels = (self._centre_x1 - self._centre_x0) * BEAT * (self._centre_y1 - self._centre_y0)
        self._clip_limit = int(CLIP_FRACTION_MAX * self._frame_pixels)
        self.exposure_s: float = EXPOSURE_MIN_s
        self.analog_gain: float = 1.0
        self.digital_gain: float = 1.0
        self._demand_stops: float = 0.0
        self._clip_threshold: int = PIXEL_MAX  # the raw level from which the output saturates
        self._x: int = 0
        self._y: int = 0
        self._sum: int = 0
        self._centre_sum: int = 0
        self._clipped: int = 0

    @property
    def _in_centre(self) -> bool:
        return self._centre_x0 <= self._x < self._centre_x1 and self._centre_y0 <= self._y < self._centre_y1

    @property
    def _metered_brightness(self) -> float:
        mean = self._sum / self._frame_pixels
        centre_mean = self._centre_sum / self._centre_pixels
        return (1.0 - CENTRE_WEIGHT) * mean + CENTRE_WEIGHT * centre_mean

    def __call__(self, pixels: PixelBeat, target: int) -> PixelBeat:
        out = np.array([round(min(pixel * self.digital_gain, PIXEL_MAX)) for pixel in pixels])
        beat = pixels.sum()
        self._sum += beat
        if self._in_centre:
            self._centre_sum += beat
        self._clipped += np.array([int(pixel >= self._clip_threshold) for pixel in pixels]).sum()
        self._x += 1
        if self._x == self._beats_per_row:
            self._x = 0
            self._y += 1
            if self._y == self._rows:
                self._y = 0
                self._end_of_frame(target)
        return out

    def _end_of_frame(self, target: int) -> None:
        """A costly update that only runs once per frame, costs a few dozen extra cycles."""
        brightness = self._metered_brightness * self.digital_gain
        error = math.log2(max(target, 1) / max(brightness, 1.0))  # both floored at one LSB, so black stays finite
        if self._clipped > self._clip_limit:
            error = min(error, 0.0)
        if abs(error) > DEADBAND_STOPS:
            self._demand_stops = min(max(self._demand_stops + DAMPING * error, 0.0), TOTAL_STOPS)
        exposure = min(self._demand_stops, EXPOSURE_STOPS)
        analog = min(self._demand_stops - exposure, ANALOG_STOPS)
        self.exposure_s = EXPOSURE_MIN_s * 2.0**exposure
        self.analog_gain = 2.0**analog
        self.digital_gain = 2.0 ** (self._demand_stops - exposure - analog)
        self._clip_threshold = math.ceil(PIXEL_MAX / self.digital_gain)
        self._sum = self._centre_sum = self._clipped = 0


def main() -> None:
    options = holoso.Options(
        holoso.OperatorOptions(
            fadd=holoso.FAddOptions(),
            fmul=holoso.FMulOptions(),
            fdiv=holoso.FDivOptions(),
            fmul_ilog2=holoso.FMulILog2Options(),
            fcmp=holoso.FCmpOptions(),
            fsort=holoso.FSortOptions(),
            fexp2=holoso.FExp2Options(),
            flog2=holoso.FLog2Options(),
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

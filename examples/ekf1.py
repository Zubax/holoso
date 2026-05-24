#!/usr/bin/env python3
"""CLI example: synthesize the bundled EKF update kernel and write the artifacts to disk.

The kernel itself lives in ``holoso.demos.ekf_update`` (the single source shared with the test suite and the web UI);
this script just drives ``synthesize`` + ``write_artifacts`` to show the end-to-end flow.
"""

from pathlib import Path

import holoso
from holoso.demos.ekf_update import update_x_P


def main() -> None:
    float_format = holoso.FloatFormat(wexp=6, wman=18)  # Use 24-bit float with 6-bit exponent and 18-bit significand.
    out_dir = Path(__file__).resolve().parent / "build" / Path(__file__).stem

    # Generated Verilog module name defaults to the function name, unless overridden explicitly.
    result = holoso.synthesize(update_x_P, float_format=float_format)
    paths = holoso.write_artifacts(result, out_dir)
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()

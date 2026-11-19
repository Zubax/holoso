"""
The float operator configurations the tests build on, spelled out once: every optional stage off, every count and
level zero, except an output register where an operator would otherwise be combinational, so that the suite's schedules
exercise operator latency (`combinational_options` in `_modelref.py` covers the combinational case), and the CORDIC
product's one register a rotation needs. Each takes knob overrides by name, so a test states only what it needs to
differ.
"""

from typing import Any

from holoso import (
    FAddOptions,
    FCmpOptions,
    FCordicOptions,
    FDivsqrtOptions,
    FExp2Options,
    FFmaOptions,
    FFromIntOptions,
    FILog2Options,
    FLog2Options,
    FMulILog2Options,
    FMulOptions,
    FRintOptions,
)

WMULTIPLIER = 18
"""The multiplier width of every test machine that does not say otherwise: the DSP operand width of ECP5 and Artix-7."""


def fadd(**knobs: Any) -> FAddOptions:
    lean: dict[str, Any] = {"stage_input": 0, "stage_core": FAddOptions.CoreStages.NONE, "stage_output": 1}
    return FAddOptions(**(lean | knobs))


def fmul(**knobs: Any) -> FMulOptions:
    lean: dict[str, Any] = {"stage_input": 0, "stage_core_mul": 0, "stage_output": 1}
    return FMulOptions(**(lean | knobs))


def ffma(**knobs: Any) -> FFmaOptions:
    lean: dict[str, Any] = {
        "stage_input": 0,
        "stage_core_mul": 0,
        "stage_core_add": FAddOptions.CoreStages.NONE,
        "stage_output": 1,
    }
    return FFmaOptions(**(lean | knobs))


def fdivsqrt(**knobs: Any) -> FDivsqrtOptions:
    lean: dict[str, Any] = {
        "stage_input": 0,
        "stage_core": FDivsqrtOptions.CoreStages.NONE,
        "stage_output": 0,
        "fast": True,
    }
    return FDivsqrtOptions(**(lean | knobs))


def fmul_ilog2(**knobs: Any) -> FMulILog2Options:
    lean: dict[str, Any] = {"stage_input": 0, "stage_core": FMulILog2Options.CoreStages.NONE, "stage_output": 1}
    return FMulILog2Options(**(lean | knobs))


def filog2(**knobs: Any) -> FILog2Options:
    lean: dict[str, Any] = {"stage_input": 0, "stage_output": 1}
    return FILog2Options(**(lean | knobs))


def fcmp(**knobs: Any) -> FCmpOptions:
    lean: dict[str, Any] = {"stage_input": 0, "stage_output": 1}
    return FCmpOptions(**(lean | knobs))


def frint(**knobs: Any) -> FRintOptions:
    lean: dict[str, Any] = {"stage_input": 0, "stage_core": FRintOptions.CoreStages.NONE, "stage_output": 1}
    return FRintOptions(**(lean | knobs))


def ffromint(**knobs: Any) -> FFromIntOptions:
    lean: dict[str, Any] = {"stage_input": 0, "stage_core": FFromIntOptions.CoreStages.NONE, "stage_output": 1}
    return FFromIntOptions(**(lean | knobs))


def fexp2(**knobs: Any) -> FExp2Options:
    lean: dict[str, Any] = {
        "stage_input": 0,
        "stage_core_kernel_level": 0,
        "stage_core": FExp2Options.CoreStages.NONE,
        "stage_output": 0,
        "fast": True,
    }
    return FExp2Options(**(lean | knobs))


def flog2(**knobs: Any) -> FLog2Options:
    lean: dict[str, Any] = {
        "stage_input": 0,
        "stage_core_kernel_level": 0,
        "stage_core_mul": 0,
        "stage_core": FLog2Options.CoreStages.NONE,
        "stage_output": 0,
        "fast": True,
    }
    return FLog2Options(**(lean | knobs))


def fcordic(**knobs: Any) -> FCordicOptions:
    lean: dict[str, Any] = {
        "stage_input": 0,
        "stage_core_mul": 1,
        "stage_core": FCordicOptions.CoreStages.NONE,
        "stage_output": 0,
        "fast": True,
    }
    return FCordicOptions(**(lean | knobs))

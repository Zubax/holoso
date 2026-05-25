"""Unit tests for the pure-Python verification core."""

import sys
from pathlib import Path

import numpy as np

from holoso import FAddOp, FDivOp, FloatFormat, FMulILog2GenericOp, FMulOp, OpConfig
from holoso._frontend import lower
from holoso._passes import run
from holoso._verify import default_tolerance, evaluate_reference, unit_roundoff

from _modelref import (
    bounded,
    encode_inputs,
    evaluate_opgraph,
    log_uniform_positive,
    random_legal_bits,
    spd_matrix,
    within,
)

F32 = FloatFormat(8, 24)
FMT = FloatFormat(6, 18)
OPS = OpConfig(FAddOp(), FMulOp(), FDivOp(), FMulILog2GenericOp())


def test_codec_known_binary32_values() -> None:
    assert F32.encode(1.0) == 0x3F800000
    assert F32.encode(2.0) == 0x40000000
    assert F32.encode(0.5) == 0x3F000000
    assert F32.encode(-1.0) == 0xBF800000
    assert F32.encode(0.0) == 0
    assert F32.decode(0x3F800000) == 1.0
    assert F32.decode(0) == 0.0


def test_codec_round_trip_within_unit_roundoff() -> None:
    rng = np.random.default_rng(1)
    for fmt in (F32, FMT):
        u = unit_roundoff(fmt)
        for _ in range(500):
            x = float(rng.uniform(-100.0, 100.0))
            y = fmt.decode(fmt.encode(x))
            assert abs(y - x) <= u * abs(x) + 1e-30


def test_codec_exact_powers_and_simple_fractions() -> None:
    for value in (3.0, 0.25, -7.5, 16.0, 0.125):
        assert FMT.decode(FMT.encode(value)) == value


def test_is_legal_rejects_subnormal_and_negative_zero() -> None:
    # exp == 0 with nonzero fraction is subnormal; sign bit with zero magnitude is negative zero.
    assert not FMT.is_legal(0b1)  # subnormal
    neg_zero = 1 << (FMT.width - 1)
    assert not FMT.is_legal(neg_zero)
    assert FMT.is_legal(FMT.encode(1.0))


def test_reference_evaluates_and_flattens() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return [a + b, a * b]

    assert evaluate_reference(f, {"a": 2.0, "b": 3.0}) == [5.0, 6.0]


def test_opgraph_matches_original_small_kernels() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    hir = run(lower(f, FMT), OPS)
    inputs = {"a": 1.25, "b": -3.5}
    ref = evaluate_reference(f, inputs)
    got = evaluate_opgraph(hir, inputs)
    assert all(within(g, r, 1e-12, 1e-15) for g, r in zip(got, ref))


def test_opgraph_matches_original_ekf1() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    rng = np.random.default_rng(12345)
    cov = spd_matrix(rng, 3, 0.5, 2.0)
    inputs = {
        "P00": float(cov[0, 0]),
        "P01": float(cov[0, 1]),
        "P02": float(cov[0, 2]),
        "P11": float(cov[1, 1]),
        "P12": float(cov[1, 2]),
        "P22": float(cov[2, 2]),
        "Q_R": log_uniform_positive(rng, 1e-3, 1e-1),
        "Q_g": log_uniform_positive(rng, 1e-3, 1e-1),
        "Q_i": log_uniform_positive(rng, 1e-3, 1e-1),
        "R_ct": log_uniform_positive(rng, 1e-1, 1.0),
        "R_shunt": log_uniform_positive(rng, 1e-1, 1.0),
        "dt": bounded(rng, 1e-3, 1e-2),
        "x_R": bounded(rng, -1.0, 1.0),
        "x_g": bounded(rng, -1.0, 1.0),
        "x_i": bounded(rng, -1.0, 1.0),
        "z_ct": bounded(rng, -1.0, 1.0),
        "z_shunt": bounded(rng, -1.0, 1.0),
    }
    hir = run(lower(ekf1.update_x_P, FMT), OPS)
    ref = evaluate_reference(ekf1.update_x_P, inputs)
    got = evaluate_opgraph(hir, inputs)
    assert len(ref) == 9 and all(np.isfinite(ref))
    assert all(within(g, r, 1e-9, 1e-12) for g, r in zip(got, ref))


def test_tolerance_predicate() -> None:
    assert within(1.0, 1.0, 0.0, 0.0)
    assert within(1.001, 1.0, 0.01, 0.0)
    assert not within(1.1, 1.0, 0.01, 0.0)
    assert within(float("inf"), float("inf"), 1.0, 1.0)
    assert not within(float("inf"), 1.0, 1.0, 1.0)


def test_default_tolerance_scales_with_format_and_size() -> None:
    coarse = default_tolerance(FMT, 100)[0]
    fine = default_tolerance(F32, 100)[0]
    assert coarse > fine  # 6/18 has a larger unit roundoff than 8/24
    assert default_tolerance(FMT, 200)[0] > default_tolerance(FMT, 10)[0]


def test_sampling_legal_and_spd() -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        bits = random_legal_bits(FMT, rng)
        assert FMT.is_legal(bits) and FMT.is_finite(bits)
    cov = spd_matrix(rng, 3)
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)
    encoded = encode_inputs(FMT, {"a": 1.0, "b": 2.0})
    assert set(encoded) == {"a", "b"} and encoded["a"] == FMT.encode(1.0)

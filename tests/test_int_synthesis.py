"""
Black-box integer synthesis through the public API: each kernel here was refused while the LIR integer gate stood,
so each pins the lifted gate -- all four backends must build, and the model answers with saturating arithmetic.
"""

import pickle

import pytest

import holoso

_OPTIONS = holoso.Options(holoso.OperatorOptions())


def _as_int(value: object) -> int:
    assert isinstance(value, holoso.IntValue)
    return int(value)


class Accumulator:
    def __init__(self) -> None:
        self.total = 0

    def step(self, x: int) -> int:
        self.total = self.total + x
        return abs(self.total)


@pytest.fixture(scope="module")
def accumulator_result() -> holoso.SynthesisResult:
    return holoso.synthesize(Accumulator().step, _OPTIONS, name="IntAccumulator")


def test_an_integer_state_kernel_synthesizes_and_the_model_matches_python(
    accumulator_result: holoso.SynthesisResult,
) -> None:
    fmt = _OPTIONS.ifmt
    model = pickle.loads(pickle.dumps(accumulator_result.numerical_model))  # the blob a generated bench embeds
    assert [(port.name, port.scalar_type) for port in model.inputs] == [("x", holoso.IntType(fmt))]
    assert [(port.name, port.scalar_type) for port in model.outputs] == [
        ("out_0", holoso.IntType(fmt)),
        ("state_total", holoso.IntType(fmt)),
    ]
    sim = model.elaborate()
    total = 0
    for x in [5, -12, 7, fmt.max, fmt.max, -3, fmt.min, fmt.min, 100]:
        total = fmt.saturate(total + x)
        assert [_as_int(value) for value in sim.run(x)] == [fmt.saturate(abs(total)), total]


def mixed_constants(x: float, n: int) -> tuple[float, int, int]:
    """``1`` beside ``1.0`` pools a FloatValue and an IntValue side by side; ``7`` reads at a non-value index."""
    return x + 1.0, n + 1, n + 7


def test_a_mixed_family_kernel_synthesizes_end_to_end() -> None:
    options = holoso.Options(holoso.OperatorOptions(fadd=holoso.FAddOptions()))
    result = holoso.synthesize(mixed_constants, options, name="MixedConstants")
    sim = result.numerical_model.elaborate()
    for x, n in [(2.5, 40), (-1.0, -3), (0.0, options.ifmt.max)]:
        out_x, out_n1, out_n7 = sim.run(x, n)
        assert isinstance(out_x, holoso.FloatValue)
        assert (float(out_x), _as_int(out_n1), _as_int(out_n7)) == (
            x + 1.0,
            options.ifmt.saturate(n + 1),
            options.ifmt.saturate(n + 7),
        )


def divmod_pair(a: int, b: int) -> tuple[int, int]:
    return a // b, a % b


def test_a_divider_kernel_model_matches_python() -> None:
    """The RTL cosim shares the model's LIR, so a swapped quotient/remainder lane is visible only upstream."""
    result = holoso.synthesize(divmod_pair, _OPTIONS, name="DivmodPair")
    sim = result.numerical_model.elaborate()
    fmt = _OPTIONS.ifmt
    for a, b in [(7, 3), (-7, 3), (7, -3), (-7, -3), (fmt.max, 2), (fmt.min, 7), (0, 5)]:
        assert [_as_int(value) for value in sim.run(a, b)] == [a // b, a % b], (a, b)
    assert [_as_int(value) for value in sim.run(fmt.min, -1)] == [fmt.max, 0]


def test_an_integer_input_rejects_a_bool(accumulator_result: holoso.SynthesisResult) -> None:
    """``bool`` is an ``int`` subclass, so a silently float-free miscoercion -- not a crash -- is the failure mode."""
    sim = accumulator_result.numerical_model.elaborate()
    with pytest.raises(TypeError, match=r"input 0 must be IntValue or int"):
        sim.run(True)

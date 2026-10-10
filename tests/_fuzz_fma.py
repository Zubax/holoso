"""
The fused multiply-add batch of the differential fuzzer.

The passes that read inside an fma -- strength reduction, the linear pass, the rescaling of a constant the format
cannot hold, and the expansion on a machine without the operator -- all sit in the front and mid end the interpreter
shares with the model, where the campaign's primary check cannot see them. So each kernel here spells one operand shape
and carries a twin (see `_fuzz`) that restates each fma as what the documented reading leaves of it. Each shape also
states how many fused operations the machine with the operator must be left with, so a fold that silently stops
applying -- or starts applying where the fma was to stay fused -- fails even where the bits agree.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from holoso._type import FloatFormat

from ._fuzz import (
    _Emitter,
    _finish_function_kernel,
    _max_normal_exp,
    _min_normal_exp,
    _power2_literal,
    _run_campaign_kernel,
    _seed_rng,
    CampaignStats,
    Divergence,
    GeneratedKernel,
    Mode,
    Shape,
)

# The edge vectors reach the format's largest value, and a kernel held to the bit may not leave the finite range, so
# every operand of the fma batch is clamped to this power of two in magnitude. The smallest values are let through:
# what a fold makes of the lower rail is the reference's to say.
_FMA_OPERAND_BOUND_EXP = 2


@dataclass(frozen=True, slots=True)
class _FmaShape:
    lanes: list[str]  # the names returned, each an output port of its own
    fused: int  # the fused multiply-adds the machine with the operator must be left with


def _signed(sign: int, operand: str) -> str:
    assert sign in {1, -1}
    return operand if sign > 0 else f"-{operand}"


def _exponent_of(value: float) -> int | None:
    """`k` where the magnitude is `2**k`, else `None`."""
    mantissa, exponent = math.frexp(abs(value))
    return exponent - 1 if mantissa == 0.5 else None


def _scaled(x: str, constant: float) -> str:
    """
    What a twin states for `x` times a constant, in the one shape the compiler gives a constant scaling: the exponent
    scaler's step where the constant is a power of two, the sign riding outside it, and the multiplication otherwise.
    """
    exponent = _exponent_of(constant)
    if exponent is None:
        return f"machine.mul({x}, {constant!r})"
    return _signed(1 if constant > 0.0 else -1, f"machine.scale({x}, {exponent})")


class _FmaEmitter(_Emitter):
    """The statements of an fma shape, each written into the kernel and stated to its twin."""

    def sign(self) -> int:
        return 1 if self.chance(0.5) else -1

    def shuffled(self, a: str, b: str) -> tuple[str, str]:
        return (a, b) if self.chance(0.5) else (b, a)

    def short_constant(self) -> float:
        """
        A positive constant that is no power of two, of so few significand bits that the sum and the product of two
        are exact in float64 and in the format alike: a twin can then name the coefficient a fold arrives at.
        """
        return math.ldexp(2 * self.randint(1, 3) + 1, self.randint(-2, 1))

    def ordinary_constant(self) -> float:
        """
        A positive constant that is no power of two: a short one, or one the format rounds. The latter stands only
        where the constant a fold arrives at is one float64 operation on what was written, which a twin can repeat.
        """
        return self.short_constant() if self.chance(0.7) else [0.1, 0.001, 1.3, 2.7][self.randint(0, 3)]

    def fma_operands(self, n: int) -> list[str]:
        """`n` distinct inputs, each clamped by exact selects to where no shape of the fma batch can overflow."""
        assert len(self._floats) >= n, "an operand standing in twice would let a sum cancel that no shape intends"
        bound = _power2_literal(_FMA_OPERAND_BOUND_EXP)
        operands: list[str] = []
        for source in self.pick_floats(n):
            below = self.fresh("lim")
            operand = self.fresh("x")
            self.emit(f"{below} = ({bound} if {source} > {bound} else {source})")
            self.emit(f"{operand} = (-{bound} if {below} < -{bound} else {below})")
            operands.append(operand)
        return operands

    def rounded(self, prefix: str, written: str, restated: str) -> str:
        """One rounding statement bound to a fresh name: as the kernel writes it, and as its twin states it."""
        name = self.fresh(prefix)
        self.emit(f"{name} = {written}", restated=f"{name} = {restated}")
        return name

    def written_product(self, a: str, b: str) -> str:
        return self.rounded("p", f"{a} * {b}", f"machine.mul({a}, {b})")

    def constant_scaling(self, x: str, constant: float) -> str:
        return self.rounded("s", f"{x} * {constant!r}", _scaled(x, constant))

    def spelled_fma(self, a: str, b: str, c: str, restated: str | None = None) -> str:
        """
        Unless restated, the twin takes the fma as written, which is the whole reference wherever what a fold drops is
        exact; only a fold that moves a rounding, or meets a rail of the format, has to be restated.
        """
        return self.rounded("f", f"math.fma({a}, {b}, {c})", restated or f"machine.fma({a}, {b}, {c})")

    def fma_by_constant(self, x: str, constant: float, z: str) -> str:
        """
        What a twin states for an fma by a constant the format may not hold. One that encodes to a finite nonzero value
        is held as a literal is: in the half binade under the smallest normal that means rounded up to it, however far
        from the number written. One that encodes to zero or to an infinity is carried instead as a significand and an
        exponent, the exponent an exact scaling of the other factor, and the product by the significand stays fused.
        """
        bits = self._fmt.encode(constant)
        if bits != 0 and self._fmt.is_finite(bits):
            return f"machine.fma({x}, {constant!r}, {z})"
        mantissa, exponent = math.frexp(constant)
        return f"machine.fma(machine.scale({x}, {exponent - 1}), {2.0 * mantissa!r}, {z})"

    def leaning_operand(self, u: str, v: str, lean: int) -> str:
        """
        An unknown `lean` binades off one that is no constant scaling of a single value: a select between two inputs
        scaled a binade apart. A constant factor meeting it finds no layer under it to compose with, so the constant
        reaches the format as written.
        """
        one = self.constant_scaling(u, math.ldexp(1.0, lean))
        other = self.constant_scaling(v, math.ldexp(1.0, lean - 1))
        operand = self.fresh("q")
        self.emit(f"{operand} = ({one} if {u} < {v} else {other})")
        return operand

    def rail_reach(self) -> tuple[int, int]:
        """
        A constant's exponent at or past a rail of the format, from its last binade to a few beyond, as `(exponent,
        lean)`: `lean` is the exponent an operand must carry for its product by such a constant to land mid-range,
        clear of both rails with room for a sum.
        """
        max_exp, min_exp = _max_normal_exp(self._fmt), _min_normal_exp(self._fmt)
        reach = max_exp // 4
        lean = (2 * max_exp) // 3
        # Past the format's own exponent span the constant is refused rather than rescaled, and the product has to land
        # where a few binades of operand, significand and carry still fit.
        assert max_exp + reach <= (1 << self._fmt.wexp) - 3 and reach + _FMA_OPERAND_BOUND_EXP + 4 < lean < max_exp
        # The rail's own binade and the two past it are drawn more often than the farther ones: the first one under the
        # smallest normal is where the format still holds a constant, by rounding it up.
        past = self.randint(0, 2) if self.chance(0.6) else self.randint(3, reach)
        return (max_exp + past, -lean) if self.chance(0.5) else (min_exp - past, lean)

    @property
    def fmt(self) -> FloatFormat:
        return self._fmt


def _emit_fma_zero_factor(em: _FmaEmitter) -> _FmaShape:
    """A zero factor leaves the addend, a zero the graph arrives at by its own identities like one written."""
    x, z, w = em.fma_operands(3)
    zero = em.choice(["0.0", f"({w} * 0.0)", f"({w} - {w})"])
    return _FmaShape([em.spelled_fma(*em.shuffled(x, zero), z)], fused=0)


def _emit_fma_zero_addend(em: _FmaEmitter) -> _FmaShape:
    """A zero addend leaves the product, which a constant factor makes the scaling a written product by it is."""
    x, y = em.fma_operands(2)
    constant = em.ordinary_constant() if em.chance(0.6) else math.ldexp(1.0, em.exact_scale_shift())
    other = y if em.chance(0.4) else repr(em.sign() * constant)
    zero = em.choice(["0.0", f"({y} - {y})"])
    return _FmaShape([em.spelled_fma(*em.shuffled(x, other), zero)], fused=0)


def _emit_fma_unit_factor(em: _FmaEmitter) -> _FmaShape:
    """A factor of plus or minus one leaves the plain sum, which is exact wherever the fused one is."""
    x, z = em.fma_operands(2)
    return _FmaShape([em.spelled_fma(*em.shuffled(x, repr(float(em.sign()))), z)], fused=0)


def _emit_fma_power_of_two_factor(em: _FmaEmitter) -> _FmaShape:
    """
    A factor of plus or minus a power of two leaves the plain sum, over whatever power-of-two scaling lies under it.
    The product is exact only short of the format's rails: at the lower one the scaler rounds on its own what the fused
    sum would have rounded once, so the twin states the sum the documented reading leaves rather than the fma as
    written. The two scalings compose as exponents, so either may lie past the format where the other brings it back.
    """
    x, z = em.fma_operands(2)
    step = em.exact_scale_shift()
    sign = em.sign()
    base, under = x, 0
    if em.chance(0.5):
        under = em.exact_scale_shift() if em.chance(0.5) else em.rail_reach()[0]
        if abs(under) > _max_normal_exp(em.fmt) // 2:
            step -= under
        base = em.constant_scaling(x, math.ldexp(1.0, under))
    addend = z if em.chance(0.8) else repr(em.sign() * em.ordinary_constant())
    restated = f"machine.add({addend}, {_signed(sign, f'machine.scale({x}, {under + step})')})"
    factor = repr(sign * math.ldexp(1.0, step))
    return _FmaShape([em.spelled_fma(*em.shuffled(base, factor), addend, restated)], fused=0)


def _emit_fma_known_factors(em: _FmaEmitter) -> _FmaShape:
    """
    Two known factors fold into the constant the addend is summed with. The fold is the host's arithmetic and not the
    format's, so the twin states the sum with the product as Python forms it.
    """
    (z,) = em.fma_operands(1)
    a, b = (em.sign() * em.ordinary_constant() for _ in range(2))
    return _FmaShape([em.spelled_fma(repr(a), repr(b), z, f"machine.add({z}, {a * b!r})")], fused=0)


def _emit_fma_constant_factor(em: _FmaEmitter) -> _FmaShape:
    """
    An ordinary constant factor stays fused, and one the format does not hold exactly rounds into it once, as a
    literal does.
    """
    x, z = em.fma_operands(2)
    addend = z if em.chance(0.8) else repr(em.sign() * em.ordinary_constant())
    return _FmaShape([em.spelled_fma(*em.shuffled(x, repr(em.sign() * em.ordinary_constant())), addend)], fused=1)


def _emit_fma_unknown_factors(em: _FmaEmitter) -> _FmaShape:
    """Between two unknowns nothing is read into the fma, an operand standing in two places included."""
    x, y, z = em.fma_operands(3)
    a, b, c = [(x, y, z), (x, x, z), (x, y, x), (x, y, repr(em.sign() * em.short_constant()))][em.randint(0, 3)]
    return _FmaShape([em.spelled_fma(a, b, c)], fused=1)


def _emit_fma_signed_operands(em: _FmaEmitter) -> _FmaShape:
    """
    Each operand's own chain of negations and absolute values folds into its sign control, and so does the chain over
    the result into whatever reads it: the fma stays one operation and no sign costs another.
    """
    operands = em.fma_operands(3)
    chains = ["-{}", "abs({})", "-abs({})", "{}"]
    conditioned = [em.choice(chains).format(operand) for operand in operands]
    if conditioned == operands:
        conditioned[em.randint(0, 2)] = f"-{operands[0]}"
    return _FmaShape([em.choice(chains).format(em.spelled_fma(*conditioned))], fused=1)


def _emit_fma_scaled_factor(em: _FmaEmitter, *, kept: bool) -> _FmaShape:
    """
    A constant factor composes with the constant scaling under it, a layer another consumer keeps included: the fused
    product is by the one constant the two amount to, so the layer's own rounding never reaches it. Two constants
    whose product the host rounds to a power of two amount to that power, and leave the plain sum as it does.
    """
    x, z = em.fma_operands(2)
    outer = em.sign() * em.ordinary_constant()
    inner = em.sign() * (em.ordinary_constant() if em.chance(0.6) else math.ldexp(1.0, em.exact_scale_shift()))
    layer = em.constant_scaling(x, inner)
    factor, composed = layer, inner * outer
    if em.chance(0.3):
        factor, composed = f"-{layer}", -composed
    stays_fused = _exponent_of(composed) is None
    restated = f"machine.fma({x}, {composed!r}, {z})" if stays_fused else f"machine.add({z}, {_scaled(x, composed)})"
    fused = em.spelled_fma(*em.shuffled(factor, repr(outer)), z, restated)
    return _FmaShape([fused, layer] if kept else [fused], fused=int(stays_fused))


def _emit_fma_factor_at_the_rails(em: _FmaEmitter) -> _FmaShape:
    """
    A constant factor at or past the format's exponent range, which the twin states as the format takes it (see
    `fma_by_constant`): held, or carried with its exponent moved onto the other factor. A power of two is an exponent
    at any distance, so there the plain sum is left.
    """
    u, v, w = em.fma_operands(3)
    exponent, lean = em.rail_reach()
    q = _signed(em.sign(), em.leaning_operand(u, v, lean))
    z = em.constant_scaling(w, math.ldexp(1.0, exponent + lean))
    sign = em.sign()
    factor = sign * math.ldexp(1.0 + 0.25 * em.randint(0, 3), exponent)
    if _exponent_of(factor) is not None:
        restated = f"machine.add({z}, {_signed(sign, f'machine.scale({q}, {exponent})')})"
        return _FmaShape([em.spelled_fma(*em.shuffled(q, repr(factor)), z, restated)], fused=0)
    return _FmaShape([em.spelled_fma(*em.shuffled(q, repr(factor)), z, em.fma_by_constant(q, factor, z))], fused=1)


def _emit_fma_composed_across_the_rails(em: _FmaEmitter) -> _FmaShape:
    """
    Scalings compose as exponents, so the format's range is asked only of what the composition leaves. A factor at
    or past the range is an ordinary constant once the scaling under it brings it back, and two constants the format
    holds compose into one that may lie past it, which the format then takes as it takes one written there.
    """
    u, v, w = em.fma_operands(3)
    exponent, lean = em.rail_reach()
    z = em.constant_scaling(w, math.ldexp(1.0, exponent + lean))
    sign = em.sign()
    if em.chance(0.5):
        outer = sign * math.ldexp(1.0 + 0.25 * em.randint(1, 3), exponent)
        layer = em.constant_scaling(u, math.ldexp(1.0, lean))
        restated = f"machine.fma({u}, {math.ldexp(outer, lean)!r}, {z})"
        return _FmaShape([em.spelled_fma(*em.shuffled(layer, repr(outer)), z, restated)], fused=1)
    q = em.leaning_operand(u, v, lean)
    inner = math.ldexp(1.0 + 0.25 * em.randint(1, 2), exponent // 2)
    outer = sign * math.ldexp(1.0 + 0.25 * em.randint(1, 2), exponent - exponent // 2)
    layer = em.constant_scaling(q, inner)
    restated = em.fma_by_constant(q, inner * outer, z)
    return _FmaShape([em.spelled_fma(*em.shuffled(layer, repr(outer)), z, restated)], fused=1)


def _emit_fma_addend_scaled_alike(em: _FmaEmitter) -> _FmaShape:
    """
    A fused product by a constant and an addend scaled by the same constant are one scaling of their sum, the signs
    riding the addition: the sum rounds first and nothing is left fused. The factor's own scale is read through a
    power-of-two layer under it, and the addend's through a stack of layers or a negation spelled over it.
    """
    x, y = em.fma_operands(2)
    scale = em.ordinary_constant() if em.chance(0.7) else math.ldexp(1.0, em.exact_scale_shift())
    factor_sign, addend_sign = em.sign(), em.sign()
    base, under = x, 1.0
    if em.chance(0.4):
        under = math.ldexp(1.0, em.exact_scale_shift())
        base = em.constant_scaling(x, under)
    match em.randint(0, 2):
        case 0:
            addend = em.constant_scaling(y, addend_sign * scale * under)
        case 1:
            addend = em.constant_scaling(em.constant_scaling(y, addend_sign * scale), under)
        case _:
            addend = f"-{em.constant_scaling(y, -addend_sign * scale * under)}"
    total = f"machine.add({_signed(addend_sign, y)}, {_signed(factor_sign, x)})"
    fused = em.spelled_fma(*em.shuffled(base, repr(factor_sign * scale)), addend, _scaled(total, scale * under))
    return _FmaShape([fused], fused=0)


def _emit_fma_addend_scaled_alike_and_kept(em: _FmaEmitter) -> _FmaShape:
    """An addend's scaling that another consumer keeps is not the fma's to replace, so the fma stays as written."""
    x, y = em.fma_operands(2)
    scale = em.ordinary_constant()
    addend = em.constant_scaling(y, em.sign() * scale)
    return _FmaShape([em.spelled_fma(*em.shuffled(x, repr(em.sign() * scale)), addend), addend], fused=1)


def _emit_fma_addend_scaled_apart(em: _FmaEmitter) -> _FmaShape:
    """Only at the same exponent do two scalings become one: a binade or more apart, the fma stays as written."""
    x, y = em.fma_operands(2)
    scale = em.ordinary_constant()
    addend = em.constant_scaling(y, em.sign() * math.ldexp(scale, em.exact_scale_shift()))
    return _FmaShape([em.spelled_fma(*em.shuffled(x, repr(em.sign() * scale)), addend)], fused=1)


def _emit_fma_gathering_addend(em: _FmaEmitter) -> _FmaShape:
    """
    An fma by a constant is the linear terms it adds, so an addend that is a multiple of the factor's own value gathers
    with it into one scaling of that value, or into nothing where the two cancel. The coefficients add exactly and
    the one left rounds once, which is what the host's own addition of the two does.
    """
    (x,) = em.fma_operands(1)
    factor = em.sign() * em.ordinary_constant()
    other = [-factor, 1.0, -1.0, em.sign() * em.ordinary_constant()][em.randint(0, 3)]
    if other == -factor and em.chance(0.5):
        addend = f"-{em.constant_scaling(x, factor)}"
    elif abs(other) == 1.0:
        addend = _signed(int(other), x)
    else:
        addend = em.constant_scaling(x, other)
    total = factor + other
    restated = "0.0" if total == 0.0 else _scaled(x, total)
    return _FmaShape([em.spelled_fma(*em.shuffled(x, repr(factor)), addend, restated)], fused=0)


def _emit_fma_cancelled_from_outside(em: _FmaEmitter) -> _FmaShape:
    """
    A sum cancels across an fma by a constant: taking the addend back leaves the product, and taking the product back
    leaves the addend, each answered whatever else reads the fma, which stays fused for another reader alone.
    """
    x, z = em.fma_operands(2)
    factor = em.sign() * em.ordinary_constant()
    fused = em.spelled_fma(*em.shuffled(x, repr(factor)), z)
    lanes: list[str] = []
    if em.chance(0.6):
        lanes.append(em.rounded("g", f"{fused} - {z}", _scaled(x, factor)))
    if not lanes or em.chance(0.4):
        lanes.append(em.rounded("g", f"{fused} - {em.constant_scaling(x, factor)}", z))
    kept = em.chance(0.5)
    return _FmaShape([*lanes, fused] if kept else lanes, fused=int(kept))


def _emit_fma_cancelled_through_a_sum(em: _FmaEmitter) -> _FmaShape:
    """
    The linear reading runs through the sums an fma by a constant touches: a term of the addend, or the addend
    against a term of the factor, cancels and leaves a single scaling of the term that is left.
    """
    x, y = em.fma_operands(2)
    factor = em.sign() * em.ordinary_constant()
    opposite = em.constant_scaling(x, -factor)
    if em.chance(0.5):
        addend = em.rounded("t", f"{y} + {opposite}", f"machine.add({y}, {opposite})")
        return _FmaShape([em.spelled_fma(*em.shuffled(x, repr(factor)), addend, y)], fused=0)
    total = em.rounded("t", f"{x} + {y}", f"machine.add({x}, {y})")
    return _FmaShape([em.spelled_fma(*em.shuffled(total, repr(factor)), opposite, _scaled(y, factor))], fused=0)


def _emit_fma_cancelled_through_a_chain(em: _FmaEmitter) -> _FmaShape:
    """
    The linear reading runs through an fma by a constant into the one under it: an outer addend that cancels the inner
    product leaves a single scaling of the inner addend. The constants are short, since the term cancels only where
    the product of the two factors is exact.
    """
    x, z = em.fma_operands(2)
    inner_factor, outer_factor = em.sign() * em.short_constant(), em.sign() * em.short_constant()
    inner = em.spelled_fma(*em.shuffled(x, repr(inner_factor)), z)
    addend = em.constant_scaling(x, -(inner_factor * outer_factor))
    outer = em.spelled_fma(*em.shuffled(inner, repr(outer_factor)), addend, _scaled(z, outer_factor))
    kept = em.chance(0.5)
    return _FmaShape([outer, inner] if kept else [outer], fused=1 if kept else 0)


def _emit_fma_product_error(em: _FmaEmitter) -> _FmaShape:
    """
    Between two unknowns the product is no value of the graph, so nothing cancels against it: fused with its own
    rounded and negated product the fma is that product's rounding error, while the machine without the operator, for
    which the same spelling is the product less itself, answers zero.
    """
    x, y = em.fma_operands(2)
    product = em.written_product(x, y)
    error = em.spelled_fma(x, y, f"-{product}")
    return _FmaShape([error, product] if em.chance(0.5) else [error], fused=1)


def _emit_fma_difference_of_products(em: _FmaEmitter) -> _FmaShape:
    """Kahan's difference of two products as `difference_of_products_float` spells it; both fmas stay fused."""
    x, y, z, w = em.fma_operands(4)
    if em.chance(0.2):
        z, w = x, y
    product = em.written_product(x, y)
    error = em.spelled_fma(x, y, f"-{product}")
    rest = em.spelled_fma(f"-{z}", w, product)
    return _FmaShape([em.rounded("g", f"{rest} + {error}", f"machine.add({rest}, {error})")], fused=2)


def _emit_fma_shared_product(em: _FmaEmitter) -> _FmaShape:
    """
    A written product another consumer reads, standing as a factor of the fma or beside an fma of the same two
    operands: the reader sees it rounded, and the fused sum either takes that rounded value or never rounds its own.
    """
    x, y, z, w = em.fma_operands(4)
    product = em.written_product(x, y)
    fused = em.spelled_fma(*em.shuffled(product, z), w) if em.chance(0.5) else em.spelled_fma(x, y, z)
    return _FmaShape([fused, product], fused=1)


def _emit_fma_negated_factor(em: _FmaEmitter) -> _FmaShape:
    """
    A negated factor negates the product however the negation is spelled, so a written product by it is the plain
    one negated, and the fma by it the one whose factor carries the sign.
    """
    x, y, z = em.fma_operands(3)
    negated = em.choice(["-{}", "(0.0 - {})", "({} * -1.0)"]).format(y)
    lanes = [em.rounded("n", f"{negated} * {x}", f"machine.mul(-{y}, {x})"), em.written_product(y, x)]
    with_fma = em.chance(0.6)
    if with_fma:
        lanes.append(em.spelled_fma(*em.shuffled(x, negated), z, f"machine.fma({x}, -{y}, {z})"))
    return _FmaShape(lanes, fused=int(with_fma))


def _emit_fma_chain(em: _FmaEmitter) -> _FmaShape:
    """
    An fma feeding an fma between unknowns, as a factor or as the addend, constant coefficients included as a Horner
    evaluation spells them: each is one fused operation and reads the one before it as a value like any other.
    """
    x, y, z, w = em.fma_operands(4)
    coefficients = [repr(em.sign() * em.ordinary_constant()) for _ in range(3)]
    match em.randint(0, 2):
        case 0:
            first = em.spelled_fma(coefficients[0], x, coefficients[1])
            second = em.spelled_fma(first, x, coefficients[2])
        case 1:
            first = em.spelled_fma(x, y, z)
            second = em.spelled_fma(*em.shuffled(first, w), x)
        case _:
            first = em.spelled_fma(x, y, z)
            second = em.spelled_fma(y, w, first)
    return _FmaShape([second, first] if em.chance(0.5) else [second], fused=2)


def _emit_fma_horner_loop(em: _FmaEmitter) -> _FmaShape:
    """
    A polynomial evaluated by an unrolled loop of fmas, as `np.polyval` spells it, from zero or from a coefficient: the
    step from zero is its addend, and every other one is fused.
    """
    x, y = em.fma_operands(2)
    steps = em.randint(2, 5)
    start = ["0.0", repr(em.sign() * em.ordinary_constant()), y][em.randint(0, 2)]
    coefficient = repr(em.sign() * em.ordinary_constant())
    acc = em.fresh("acc")
    em.emit(f"{acc} = {start}")
    em.emit(f"for _i in range({steps}):")
    em.emit(f"{acc} = math.fma({acc}, {x}, {coefficient})", 2, f"{acc} = machine.fma({acc}, {x}, {coefficient})")
    return _FmaShape([acc], fused=steps - 1 if start == "0.0" else steps)


# One operand shape per kernel: the linear pass and the scaling reader see through sums, scalings and fmas by a constant
# alike, so two shapes sharing operands in one kernel could cancel or share a layer in ways neither twin states.
_FMA_TEMPLATES: list[Callable[[_FmaEmitter], _FmaShape]] = [
    _emit_fma_zero_factor,
    _emit_fma_zero_addend,
    _emit_fma_unit_factor,
    _emit_fma_power_of_two_factor,
    _emit_fma_known_factors,
    _emit_fma_constant_factor,
    _emit_fma_unknown_factors,
    _emit_fma_signed_operands,
    lambda em: _emit_fma_scaled_factor(em, kept=False),
    lambda em: _emit_fma_scaled_factor(em, kept=True),
    _emit_fma_factor_at_the_rails,
    _emit_fma_composed_across_the_rails,
    _emit_fma_addend_scaled_alike,
    _emit_fma_addend_scaled_alike_and_kept,
    _emit_fma_addend_scaled_apart,
    _emit_fma_gathering_addend,
    _emit_fma_cancelled_from_outside,
    _emit_fma_cancelled_through_a_sum,
    _emit_fma_cancelled_through_a_chain,
    _emit_fma_product_error,
    _emit_fma_difference_of_products,
    _emit_fma_shared_product,
    _emit_fma_negated_factor,
    _emit_fma_chain,
    _emit_fma_horner_loop,
]


def _generate_fma_kernel(
    name: str,
    master_seed: int,
    index: int,
    template: Callable[[_FmaEmitter], _FmaShape],
    fmt: FloatFormat,
) -> GeneratedKernel:
    """
    A kernel consisting SOLELY of one fma operand shape, every lane returned verbatim, so its twin names each output's
    bits and the kernel is EXACT on every machine.
    """
    params = ["a", "b", "c", "d"]
    em = _FmaEmitter(_seed_rng(master_seed, index), fmt)
    for param in params:
        em.add_float(param)
    shape = template(em)
    if len(shape.lanes) == 1:
        em.return_line = f"return {shape.lanes[0]}"
    else:
        em.return_line = f"return ({', '.join(shape.lanes)},)"
        em.return_annotation = f"tuple[{', '.join('float' for _ in shape.lanes)}]"
    assert em.has_twin
    return _finish_function_kernel(
        name=name,
        master_seed=master_seed,
        index=index,
        params=params,
        bool_set=set(),
        em=em,
        shapes=frozenset({Shape.FMA}),
        mode=Mode.EXACT,
        fused_operations=shape.fused,
    )


def run_fma_batch(
    stats: CampaignStats,
    n_kernels: int,
    n_vectors: int,
    master_seed: int,
    effort: str,
    fmt: FloatFormat,
    on_divergence: Callable[[Divergence], None],
) -> None:
    """
    The templates are taken in turn and the batch is never smaller than their number, so each runs in every campaign.
    """
    for j in range(max(len(_FMA_TEMPLATES), n_kernels // 2)):
        template = _FMA_TEMPLATES[j % len(_FMA_TEMPLATES)]
        kernel = _generate_fma_kernel(f"fuzz_fma_{master_seed:x}_{j}", master_seed, 0xF3A00000 + j, template, fmt)
        stats.fma_batch_kernels += 1
        _run_campaign_kernel(kernel, stats, fmt, effort, n_vectors, on_divergence)

"""
Every scalar meaning the library serves, each declared whole: the stubs that lower it, then its spellings, each named
once beside the only thing that varies between them -- whether it also maps over an array. DESIGN.md states the model.
"""

import math
import operator

import numpy as np

from .._ir import BinaryOp, CompareOp, UnaryOp
from . import _intrinsics, _numpy, _pow
from ._registry import meaning

# ARITHMETIC, BITWISE AND SHIFTS

meaning(_intrinsics.add_int, _intrinsics.add_float, elementwise=[BinaryOp.ADD, np.add, operator.add])
meaning(_intrinsics.subtract_int, _numpy.subtract_float, elementwise=[BinaryOp.SUB, np.subtract, operator.sub])
meaning(_intrinsics.multiply_int, _intrinsics.multiply_float, elementwise=[BinaryOp.MUL, np.multiply, operator.mul])
meaning(_intrinsics.divide, elementwise=[BinaryOp.DIV, np.divide, np.true_divide, operator.truediv])
meaning(_intrinsics.floor_divide, elementwise=[BinaryOp.FLOORDIV, np.floor_divide, operator.floordiv])
meaning(_intrinsics.remainder, elementwise=[BinaryOp.MOD, np.remainder, np.mod, operator.mod])
meaning(_intrinsics.left_shift, elementwise=[BinaryOp.LSHIFT, np.left_shift, np.bitwise_left_shift, operator.lshift])
meaning(_intrinsics.right_shift, elementwise=[BinaryOp.RSHIFT, np.right_shift, np.bitwise_right_shift, operator.rshift])
meaning(_intrinsics.and_bool, _intrinsics.and_int, elementwise=[BinaryOp.BITAND, np.bitwise_and, operator.and_])
meaning(_intrinsics.or_bool, _intrinsics.or_int, elementwise=[BinaryOp.BITOR, np.bitwise_or, operator.or_])
meaning(_intrinsics.xor_bool, _intrinsics.xor_int, elementwise=[BinaryOp.BITXOR, np.bitwise_xor, operator.xor])
meaning(_intrinsics.negative_int, _intrinsics.negative_float, elementwise=[UnaryOp.NEG, np.negative, operator.neg])
meaning(_numpy.identity_int, _numpy.identity_float, elementwise=[UnaryOp.POS, np.positive, operator.pos])
meaning(
    _intrinsics.invert,
    elementwise=[UnaryOp.INVERT, np.invert, np.bitwise_not, np.bitwise_invert, operator.invert, operator.inv],
)
meaning(_numpy.square_int, _numpy.square_float, elementwise=[np.square])
meaning(_numpy.sign_int, _numpy.sign_float, elementwise=[np.sign])
meaning(_intrinsics.abs_float, _intrinsics.abs_int, elementwise=[abs, operator.abs, np.abs, np.absolute])
meaning(_intrinsics.abs_float, scalar=[math.fabs], elementwise=[np.fabs])
meaning(_intrinsics.popcount, scalar=[int.bit_count], elementwise=[np.bitwise_count])

meaning(
    _pow.pow_chain_int,
    _pow.pow_chain_float,
    _pow.pow_reciprocal,
    _pow.pow_root,
    _pow.pow_,
    elementwise=[BinaryOp.POW, pow, np.power, np.pow, operator.pow],
)
meaning(
    _pow.pow_chain_float, _pow.pow_reciprocal, _pow.pow_root, _pow.pow_, scalar=[math.pow], elementwise=[np.float_power]
)

# BOOLEAN GATES AND COMPARISONS

meaning(_intrinsics.and_bool, scalar=[BinaryOp.AND, np.logical_and])
meaning(_intrinsics.or_bool, scalar=[BinaryOp.OR, np.logical_or])
meaning(_intrinsics.xor_bool, scalar=[np.logical_xor])
meaning(_intrinsics.not_bool, scalar=[UnaryOp.NOT, np.logical_not, operator.not_])
meaning(_intrinsics.less_int, _intrinsics.less_float, scalar=[CompareOp.LT, np.less, operator.lt])
meaning(_intrinsics.less_equal_int, _intrinsics.less_equal_float, scalar=[CompareOp.LE, np.less_equal, operator.le])
meaning(_intrinsics.greater_int, _intrinsics.greater_float, scalar=[CompareOp.GT, np.greater, operator.gt])
meaning(
    _intrinsics.greater_equal_int, _intrinsics.greater_equal_float, scalar=[CompareOp.GE, np.greater_equal, operator.ge]
)
meaning(_intrinsics.equal_int, _intrinsics.equal_float, _numpy.equal_bool, scalar=[CompareOp.EQ, np.equal, operator.eq])
meaning(
    _intrinsics.xor_bool,
    _intrinsics.not_equal_int,
    _intrinsics.not_equal_float,
    scalar=[CompareOp.NE, np.not_equal, operator.ne],
)
# The NaN-suppressing twins fmin/fmax are the same operation under the no-NaN policy.
meaning(_intrinsics.min_float, _numpy.min_int, scalar=[min], elementwise=[np.minimum, np.fmin])
meaning(_intrinsics.max_float, _numpy.max_int, scalar=[max], elementwise=[np.maximum, np.fmax])
meaning(_intrinsics.isfinite, scalar=[math.isfinite, np.isfinite])
meaning(_intrinsics.isinf, scalar=[math.isinf, np.isinf])
meaning(_intrinsics.isposinf, scalar=[np.isposinf])
meaning(_intrinsics.isneginf, scalar=[np.isneginf])

# ROUNDING

# np.rint answers a float on an integer, as np.fabs does above, so each is a meaning of its own, without the integer
# identity the rest of its family carries; and on a float the math spellings answer an int where numpy's answer a float.

meaning(_intrinsics.floor, _numpy.identity_int, elementwise=[np.floor])
meaning(_intrinsics.ceil, _numpy.identity_int, elementwise=[np.ceil])
meaning(_intrinsics.trunc, _numpy.identity_int, elementwise=[np.trunc, np.fix])
meaning(_intrinsics.round_, _numpy.identity_int, elementwise=[np.round, np.around])
meaning(_intrinsics.round_, elementwise=[np.rint])
meaning(_numpy.identity_int, _numpy.floor_to_int, scalar=[math.floor])
meaning(_numpy.identity_int, _numpy.ceil_to_int, scalar=[math.ceil])
meaning(_numpy.identity_int, _intrinsics.int_from_float, scalar=[math.trunc])
meaning(_numpy.identity_int, _numpy.round_to_int, scalar=[round])

# TRANSCENDENTALS

meaning(_intrinsics.sqrt, scalar=[math.sqrt], elementwise=[np.sqrt])
meaning(_numpy.cbrt, scalar=[math.cbrt], elementwise=[np.cbrt])
meaning(_numpy.exp, scalar=[math.exp], elementwise=[np.exp])
meaning(_intrinsics.exp2, scalar=[math.exp2], elementwise=[np.exp2])
meaning(_numpy.expm1, scalar=[math.expm1], elementwise=[np.expm1])
meaning(_numpy.log, scalar=[math.log], elementwise=[np.log])
meaning(_intrinsics.log2, scalar=[math.log2], elementwise=[np.log2])
meaning(_numpy.log10, scalar=[math.log10], elementwise=[np.log10])
meaning(_numpy.log1p, scalar=[math.log1p], elementwise=[np.log1p])
meaning(_intrinsics.sin, scalar=[math.sin], elementwise=[np.sin])
meaning(_intrinsics.cos, scalar=[math.cos], elementwise=[np.cos])
meaning(_numpy.tan, scalar=[math.tan], elementwise=[np.tan])
meaning(_numpy.asin, scalar=[math.asin], elementwise=[np.arcsin, np.asin])
meaning(_numpy.acos, scalar=[math.acos], elementwise=[np.arccos, np.acos])
meaning(_numpy.atan, scalar=[math.atan], elementwise=[np.arctan, np.atan])
meaning(_intrinsics.atan2, scalar=[math.atan2], elementwise=[np.arctan2, np.atan2])
meaning(_intrinsics.hypot_pair, elementwise=[np.hypot])
meaning(_numpy.sinh, scalar=[math.sinh], elementwise=[np.sinh])
meaning(_numpy.cosh, scalar=[math.cosh], elementwise=[np.cosh])
meaning(_numpy.tanh, scalar=[math.tanh], elementwise=[np.tanh])
meaning(_numpy.asinh, scalar=[math.asinh], elementwise=[np.arcsinh, np.asinh])
meaning(_numpy.acosh, scalar=[math.acosh], elementwise=[np.arccosh, np.acosh])
meaning(_numpy.atanh, scalar=[math.atanh], elementwise=[np.arctanh, np.atanh])
meaning(_numpy.degrees, scalar=[math.degrees], elementwise=[np.degrees, np.rad2deg])
meaning(_numpy.radians, scalar=[math.radians], elementwise=[np.radians, np.deg2rad])
meaning(_intrinsics.fma, scalar=[math.fma])

# CASTS

meaning(_numpy.identity_int, _intrinsics.int_from_float, _intrinsics.int_from_bool, scalar=[int], protocol="__int__")
meaning(_numpy.identity_float, _intrinsics.float_from_int, _intrinsics.float_from_bool, scalar=[float])
meaning(_numpy.identity_bool, _intrinsics.bool_from_int, _intrinsics.bool_from_float, scalar=[bool])

"""The registry only decides WHICH callees mean a structural conversion or reshape; the semantics are the evaluator's."""

import numpy as np

from ._registry import conversion, reshape

conversion(np.array, copies=True)
conversion(np.asarray, np.asanyarray, copies=False)
reshape(np.reshape, np.ndarray.reshape)

"""The registry only decides WHICH callees mean a structural conversion; the semantics are the evaluator's."""

import numpy as np

from ._registry import conversion

conversion(np.array, copies=True)
conversion(np.asarray, np.asanyarray, copies=False)

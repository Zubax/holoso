"""
The Eel frontend differential oracle over example kernels: CPython executing the original kernel against the
evaluator running the unoptimized HIR lowered by ``holoso._eel``. ``ORACLE_COVERED`` is the set of example
names verified here; the parity table asserts membership before a row may claim ORACLE_OK, so the claim cannot
be made without the verification actually existing.
"""

import pytest

from holoso._eel import lower

from ._eeloracle import assert_hir_matches_reference
from ._examples import SPECS, ExampleSpec

ORACLE_COVERED = frozenset({"madd", "poly3", "signal_window"})

_SPECS = [spec for spec in SPECS if spec.name in ORACLE_COVERED]


def test_every_covered_name_is_a_spec() -> None:
    assert ORACLE_COVERED <= {spec.name for spec in SPECS}


@pytest.mark.parametrize("spec", _SPECS, ids=[spec.name for spec in _SPECS])
def test_eel_oracle_on_examples(spec: ExampleSpec) -> None:
    hir = lower(spec.make_kernel())
    vectors = spec.reference_vectors()
    compared = assert_hir_matches_reference(hir, spec.make_kernel(), vectors, label=spec.name)
    assert compared == len(vectors)

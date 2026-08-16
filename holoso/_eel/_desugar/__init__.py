"""
Sublayer 1: CPython `ast` -> Eel. Owns lexical binding, source evaluation order, and lossless syntax
normalization; zero types, zero shapes, zero evaluation. A whitelist over the ORIGINAL AST shapes,
reject-by-default with located diagnostics — including inside statically-dead arms; `assert` contents alone
are exempt (accepted and ignored whole, though their binding sites still shape name classification).

The accepted grammar is the whitelist in `_transform.py` (which also carries the ANF and evaluation-order
rationale), pinned positively by `tests/test_eel_desugar.py` and negatively — one located diagnostic per
banned family — by `tests/test_eel_reject.py`. Name classification lives in `_classify.py`, source
retrieval and location mapping in `_source.py`.
"""

from ._transform import desugar as desugar

"""
The Eel front end: Python -> Eel -> residual Eel -> HIR, in three stages with one representation.
Runs in shadow next to the primary frontend until cutover; ``run_shadow`` is the only integration point.
"""

from ._shadow import run_shadow as run_shadow

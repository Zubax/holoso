"""The Eel front end: Python -> Eel -> residual Eel -> HIR, in three stages with one representation."""

from ._lower import lower as lower
from ._names import spelled as spelled

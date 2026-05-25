"""Front end: lower a Python function object into HIR, plus the shared output-port naming convention."""

from ._lower import lower as lower
from ._shape import flatten_value as flatten_value, port_name as port_name

"""
``HolosoError`` is the root of every exception the package DEFINES, so a synthesis failure is catchable as one kind.
An exception that forgets to derive from it escapes such a handler as if it came from elsewhere. Ordinary misuse of
an API still raises the builtin the situation calls for (``FloatFormat(1, 4)`` is a ``ValueError``), which is why
this is a rule about the hierarchy rather than about everything a call can raise.
"""

import importlib
import inspect
import pkgutil

import holoso
from holoso import HolosoError


def test_every_exception_derives_from_the_public_root() -> None:
    modules = [holoso] + [
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(holoso.__path__, prefix=f"{holoso.__name__}.")
    ]
    offenders = sorted(  # a class re-exported by several modules is reached several times
        {
            f"{obj.__module__}.{obj.__qualname__}"
            for module in modules
            for obj in vars(module).values()
            if inspect.isclass(obj)
            and issubclass(obj, BaseException)
            and obj.__module__.split(".")[0] == holoso.__name__
            and not issubclass(obj, HolosoError)
        }
    )
    assert not offenders, "exceptions outside the public hierarchy:\n" + "\n".join(offenders)

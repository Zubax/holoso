"""
This is the central verification entry point for the project.
Tests may take a long time to run; if there is no output, assume they are still running, not stuck.

Important: When running locally instead of CI, export HOLOSO_REGALLOC_EFFORT=10 to speed up test execution.
This speeds up iteration significantly, at the cost of poorer register allocation.
"""

from pathlib import Path
import shutil
import nox

nox.options.reuse_existing_virtualenvs = True


@nox.session(python=False, default=False)
def clean(session):
    pats = [
        "dist",
        "build",
        "*/build",
        "html*",
        ".coverage*",
        ".*cache",
        "src/*.egg-info",
        "*.log",
        "*.tmp",
        ".nox",
        "*.history",
    ]
    for w in pats:
        for f in Path.cwd().glob(w):
            session.log(f"Removing: {f}")
            if f.is_dir():
                shutil.rmtree(f, ignore_errors=True)
            else:
                f.unlink(missing_ok=True)
    for f in Path.cwd().rglob("__pycache__"):
        session.log(f"Removing: {f}")
        shutil.rmtree(f, ignore_errors=True)


@nox.session
def tests(session: nox.Session) -> None:
    """Fast unit tests; the slow cosimulation, fuzzer, and example synthesis matrix live in their own sessions."""
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-m", "not cosim and not fuzz and not synth", "tests")


@nox.session
def cosim_examples(session: nox.Session) -> None:
    """Long-running end-to-end cocotb cosimulation of the bundled examples across stage configurations."""
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-m", "cosim", "tests")


@nox.session
def fuzz(session: nox.Session) -> None:
    """End-to-end blackbox differential fuzzing of the compiler; slow, no simulator. Scaled by HOLOSO_FUZZ_* knobs."""
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-s", "-m", "fuzz", "tests")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.install("-e", ".", "mypy~=2.1")
    session.run("mypy", *session.posargs)


@nox.session
def black(session: nox.Session) -> None:
    session.install("black~=26.5")
    default = ("--check", "holoso", "tests", "synth", "examples", "tools", "noxfile.py")
    session.run("python", "-m", "black", *(session.posargs or default))


@nox.session
def synth_examples(session: nox.Session) -> None:
    """
    Out-of-context FPGA synthesis (f_max/fabric) of the bundled example matrix across the available tools.
    This one takes a long time.

    The worker count is pinned to 2 rather than auto-scaled: the wide e8m36 datapaths are RAM-bound under
    place-and-route, so matching the host core count would exhaust memory before saturating it (pytest-enabler,
    which would otherwise force ``-n auto``, is disabled here so the cap holds). Override on a high-memory host,
    e.g. ``nox -s synth_examples -- -n 4 tests``.
    """
    session.install("-e", ".[test]")
    session.run(
        "python", "-m", "pytest", "-p", "no:enabler", "-s", "-m", "synth", *(session.posargs or ("-n", "2", "tests"))
    )


@nox.session
def synth(session: nox.Session) -> None:
    """Run external FPGA synthesis/place-and-route checks."""
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-s", *(session.posargs or ("synth",)))

import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["tests", "typecheck", "black"]
PYTHON_PATHS = ("holoso", "tests", "synth", "examples", "noxfile.py")


@nox.session
def tests(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-q", *(session.posargs or ("-m", "not slow", "tests")))


@nox.session
def synth_examples(session: nox.Session) -> None:
    """Long-running end-to-end synthesis cosimulation of the bundled examples (e.g. ekf1)."""
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-q", "-m", "slow", "tests")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.install("-e", ".[typecheck]")
    session.run("mypy", *session.posargs)


@nox.session
def black(session: nox.Session) -> None:
    session.install("-e", ".[format]")
    session.run("python", "-m", "black", *(session.posargs or ("--check", *PYTHON_PATHS)))


@nox.session
def synth(session: nox.Session) -> None:
    """Run external FPGA synthesis/place-and-route checks."""
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-q", "-s", *(session.posargs or ("synth",)))

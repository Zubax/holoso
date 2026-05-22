import nox

nox.options.reuse_existing_virtualenvs = True
PYTHON_PATHS = ("holoso", "tests", "examples", "noxfile.py")


@nox.session
def tests(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-q", "tests")


@nox.session
def black(session: nox.Session) -> None:
    session.install("-e", ".[format]")
    session.run("python", "-m", "black", *(session.posargs or ("--check", *PYTHON_PATHS)))


@nox.session(venv_backend="none")
def synth(session: nox.Session) -> None:
    """
    TODO: set up validation example synthesis and pnr using Yosys for different platforms:
        - ECP5, speed grade 6
        - Spartan 7
    To ensure synthesizability and verify timings and resource usage.
    """

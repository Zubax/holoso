import nox


nox.options.reuse_existing_virtualenvs = True


@nox.session
def tests(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-q", "tests")


@nox.session(venv_backend="none")
def synth(session: nox.Session) -> None:
    """
    TODO: set up validation example synthesis and pnr using Yosys for different platforms:
        - ECP5, speed grade 6
        - Spartan 7
    To ensure synthesizability and verify timings and resource usage.
    """

import nox

nox.options.reuse_existing_virtualenvs = True


@nox.session
def tests(session: nox.Session) -> None:
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-q", *(session.posargs or ("-m", "not slow", "tests")))


@nox.session
def cosim_examples(session: nox.Session) -> None:
    """Long-running end-to-end cocotb cosimulation of the bundled examples (e.g. ekf1)."""
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-q", "-m", "slow", "tests")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.install("-e", ".[typecheck]")
    session.run("mypy", *session.posargs)


@nox.session
def black(session: nox.Session) -> None:
    session.install("-e", ".[format]")
    default = ("--check", "holoso", "tests", "synth", "examples", "noxfile.py")
    session.run("python", "-m", "black", *(session.posargs or default))


@nox.session
def synth(session: nox.Session) -> None:
    """Run external FPGA synthesis/place-and-route checks."""
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-q", "-s", *(session.posargs or ("synth",)))


@nox.session(default=False)
def synth_examples(session: nox.Session) -> None:
    """Out-of-context FPGA synthesis (f_max/fabric) of the bundled examples across the available tools."""
    session.install("-e", ".[test]")

    def syn(*args: str) -> None:
        session.run("python", "-m", "synth", *args, "--rtl", "lib/kulibin/float/hdl")

    # TODO: the frequency is currently set to a very low setting; we will focus on timing closure a bit later.
    syn("examples/ekf1.py", "update_x_P", "--name", "ekf1", "--freq", "30.0")

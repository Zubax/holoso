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
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-q", "tests")


@nox.session
def cosim_examples(session: nox.Session) -> None:
    """Long-running end-to-end cocotb cosimulation of the bundled examples (e.g. ekf1)."""
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-q", "tests/test_cosim.py")


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


@nox.session
def synth_examples(session: nox.Session) -> None:
    """
    Out-of-context FPGA synthesis (f_max/fabric) of the bundled examples across the available tools.
    This one takes a long time.
    """
    session.install("-e", ".[test]")

    def syn(source: str, target: str, flows: list[str]) -> None:
        flows = [flow for f in flows for flow in ("--flow", f)]
        session.run("python", "-m", "synth", source, target, *flows, "--rtl", "lib/kulibin/float/hdl")

    syn("examples/madd.py", "madd", ["yosys-ecp5:freq=100", "diamond-ecp5:freq=100", "vivado:freq=150"])
    syn("examples/poly3.py", "poly3", ["yosys-ecp5:freq=100", "diamond-ecp5:freq=100", "vivado:freq=150"])
    syn(
        "examples/ekf1.py",
        "update_x_P",
        [
            "yosys-ecp5:freq=100",
            "diamond-ecp5:freq=100",
            "vivado:freq=150",
        ],
    )

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
    """Fast unit tests; the slow cocotb cosimulation and the differential fuzzer live in their own sessions."""
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-m", "not cosim and not fuzz", "tests")


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
    session.install("-e", ".[typecheck]")
    session.run("mypy", *session.posargs)


@nox.session
def black(session: nox.Session) -> None:
    session.install("-e", ".[format]")
    default = ("--check", "holoso", "tests", "synth", "examples", "tools", "noxfile.py")
    session.run("python", "-m", "black", *(session.posargs or default))


@nox.session
def synth(session: nox.Session) -> None:
    """Run external FPGA synthesis/place-and-route checks."""
    session.install("-e", ".[test]")
    session.run("python", "-m", "pytest", "-s", *(session.posargs or ("synth",)))


@nox.session
def synth_examples(session: nox.Session) -> None:
    """
    Out-of-context FPGA synthesis (f_max/fabric) of the bundled examples across the available tools.
    This one takes a long time.
    """
    session.install("-e", ".[test]")

    def syn(
        source: str,
        target: str,
        flows: list[str],
        *,
        wexp: int = 6,
        wman: int = 18,
        name: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        prefix = "python", "-m", "synth"
        flow_args = [arg for f in flows for arg in ("--flow", f)]
        name_args = ["--name", name] if name is not None else []
        session.run(*prefix, source, target, f"--wexp={wexp}", f"--wman={wman}", *name_args, *flow_args, env=env)

    # Wide scalars require extra stages to close timings. If closure fails, feel free to throw in more stages here.
    op_integrator_wide_ecp5 = (
        "fadd.stage_decode=1,fadd.stage_align=1,fadd.stage_normalize=1,fadd.stage_pack=1,"
        "fmul.stage_input=1,fmul.stage_pack=1"
    )
    syn(
        "examples/trapezoidal_leaky_streaming_integrator.py",
        "TrapezoidalLeakyStreamingIntegrator().__call__",
        [
            f"yosys-ecp5:freq=100,{op_integrator_wide_ecp5}",
            f"diamond-ecp5:freq=100,{op_integrator_wide_ecp5}",
            "vivado:freq=150",
        ],
        name="TrapezoidalLeakyStreamingIntegrator",
    )
    syn(
        "examples/madd.py",
        "madd",
        ["yosys-ecp5:freq=100,fmul.stage_pack=1", "diamond-ecp5:freq=100", "vivado:freq=150"],
    )
    syn("examples/poly3.py", "poly3", ["yosys-ecp5:freq=100", "diamond-ecp5:freq=100", "vivado:freq=150"])

    syn(
        "examples/ekf1_stateless.py",
        "update_x_P",
        [
            "yosys-ecp5:freq=100,fadd.stage_align=1",
            "diamond-ecp5:freq=100,fadd.stage_align=1",
            "vivado:freq=150",
        ],
        name="ekf1_stateless_e6m18",
    )

    op_ekf1_wide_stateless = (
        "fadd.stage_decode=1,fadd.stage_align=1,fadd.stage_normalize=1,fadd.stage_pack=1,"
        "fmul.stage_input=1,fmul.stage_product=1,fmul.stage_pack=1,"
        "fdiv.stage_input=1,fdiv.stage_pack=1,fdiv.stage_output=1"
    )
    op_ekf1_wide_stateless_yosys = (
        "fadd.stage_decode=1,fadd.stage_align=1,fadd.stage_normalize=1,fadd.stage_pack=1,"
        "fmul.stage_input=1,fmul.stage_product=2,fmul.stage_pack=1,"
        "fdiv.stage_input=1,fdiv.stage_pack=1,fdiv.stage_output=1,"
        "fmul_ilog2.stage_decode=1"
    )
    op_ekf1_wide_stateful = (
        "fadd.stage_decode=1,fadd.stage_align=1,fadd.stage_normalize=2,fadd.stage_pack=1,"
        "fmul.stage_input=1,fmul.stage_product=1,fmul.stage_pack=1,"
        "fdiv.stage_input=1,fdiv.stage_pack=1,fdiv.stage_output=1"
    )
    op_ekf1_wide_stateful_yosys = (
        "fadd.stage_decode=1,fadd.stage_align=1,fadd.stage_normalize=2,fadd.stage_pack=1,"
        "fmul.stage_input=1,fmul.stage_product=2,fmul.stage_pack=1,"
        "fdiv.stage_input=1,fdiv.stage_pack=1,fdiv.stage_output=1,"
        "fmul_ilog2.stage_decode=1"
    )
    syn(
        "examples/ekf1_stateless.py",
        "update_x_P",
        [
            f"yosys-ecp5:freq=100,{op_ekf1_wide_stateless_yosys}",
            f"diamond-ecp5:freq=100,{op_ekf1_wide_stateless},fmul.stage_output=1",
            f"vivado:freq=150,{op_ekf1_wide_stateless}",
        ],
        wexp=8,
        wman=36,
        name="ekf1_stateless_e8m36",
        env={"HOLOSO_DIAMOND_HARD": "1"},
    )
    syn(
        "examples/ekf1_stateful.py",
        "Ekf1().update",
        [
            "yosys-ecp5:freq=100,fadd.stage_decode=1,fadd.stage_align=1",
            "diamond-ecp5:freq=100,fadd.stage_decode=1,fadd.stage_align=1",
            "vivado:freq=150",
        ],
        name="ekf1_stateful_e6m18",
    )
    syn(
        "examples/ekf1_stateful.py",
        "Ekf1().update",
        [
            f"yosys-ecp5:freq=100,{op_ekf1_wide_stateful_yosys}",
            f"diamond-ecp5:freq=100,{op_ekf1_wide_stateful},fadd.stage_input=1",
            f"vivado:freq=150,{op_ekf1_wide_stateful}",
        ],
        wexp=8,
        wman=36,
        name="ekf1_stateful_e8m36",
        env={"HOLOSO_DIAMOND_HARD": "1"},
    )

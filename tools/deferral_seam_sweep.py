"""
Sweep the analyzer's deferral/grafting seam and report accept/refuse/crash counts per kernel family.

The seam is a known-defective area (TODO.md "Known defects"): a call deferred behind a transiently pending
store violation leaves stale reachability behind, which costs accepts and, on one surviving route, miscompiles.
Every change near it has to be judged on whether it moves those counts, and prose claims about "the loop family"
or "the dead-arm family" are unfalsifiable without the corpus that produced them. This generates both families
into real files -- exec-compiled kernels cannot lower, they raise SourceUnavailable -- and tallies outcomes.

Bind a specific worktree with --tree: it is inserted at sys.path[0] before holoso is imported and the binding
is checked. PYTHONPATH alone is not enough, though not for the reason one might assume -- it does precede the
editable-install finder, but the interpreter puts the script's directory (or the cwd, under -c) ahead of it, so
running from inside another checkout silently binds that one instead.

    python tools/deferral_seam_sweep.py --tree .            # this checkout
    python tools/deferral_seam_sweep.py --tree /path/to/wt  # compare against another commit
"""

import argparse
import importlib.util
import itertools
import pathlib
import sys
import tempfile

_PROLOGUE = """import numpy as np


class K:
    def __init__(self) -> None:
        self.t = 0.0
        self.s = 0.0
        self.n = 0

    def step(self, x: float, flag: bool) -> float:
        if flag:
            u = 1.0
            q = 1.0
        else:
            u = {u}
            q = {q}
        self.t = u
"""

# The deferral trigger is an inexact int reaching a float state slot; the narrow feed is the control.
_FEEDS = {"wide": ("2**53 + 1", "2**64"), "narrow": ("2.0", "3.0")}

# What defers. np.array is a CONVERSION and never grafts, yet still opens the seam.
_CALLS = {
    "array": "        a = np.array([q, x])",
    "dot": "        a = np.array([q, x])\n        y = np.dot(a, a)  # noqa: F841",
    "twice": "        a = np.array([q, x])\n        b = np.array([x, x])  # noqa: F841",
}

# Dead-arm family: a guard that folds false once the feed promotes, with varying content behind it.
_DEAD_ARM = {
    "inert": "        if a.shape[0] > 5:\n            pass\n        return x + 1.0",
    "store": "        if a.shape[0] > 5:\n            self.s = 7.0\n        return x + self.s",
    "store_both": (
        "        if a.shape[0] > 5:\n"
        "            if flag:\n                self.s = 7.0\n            else:\n                self.s = 8.0\n"
        "        return x + self.s"
    ),
    "raise": "        if a.shape[0] > 5:\n            raise ValueError('unreachable')\n        return x + 1.0",
    "int_state": "        if a.shape[0] > 5:\n            self.n = 1\n        return x + self.n",
    "loop": "        if a.shape[0] > 5:\n            for k in range(3):\n                self.s = float(k)\n        return x + self.s",
    "live_else": "        if a.shape[0] > 1:\n            r = 1.0\n        else:\n            r = 2.0\n        return x + r",
    # A store AFTER the guard, on the path both arms reconverge onto. The shipped rule refuses these, which is a
    # real cost -- the store does run on the taken path regardless. Sparing them was tried and readmitted a
    # miscompile, so they stay here to keep that cost visible rather than to assert it should be zero.
    "inert_then_store": "        if a.shape[0] > 5:\n            pass\n        self.s = x * 2.0\n        return self.s",
    "settles_true_then_store": "        if a.shape[0] > 1:\n            pass\n        self.s = x * 2.0\n        return self.s",
}

# Loop family: the same trigger with the branch INSIDE a loop body, where withholding an edge starves the
# fixed point -- the shape that disqualified two candidate fixes.
_LOOP = {
    "while_if": (
        "        acc = 0.0\n        run = flag\n        while run:\n"
        "            if a.shape[0] > 1:\n                acc = acc + 1.0\n            run = False\n"
        "        return x + acc"
    ),
    "while_plain": "        acc = 0.0\n        run = flag\n        while run:\n            acc = acc + 1.0\n            run = False\n        return x + acc",
    "for_if": (
        "        acc = 0.0\n        for _ in range(2):\n"
        "            if a.shape[0] > 1:\n                acc = acc + 1.0\n"
        "        return x + acc"
    ),
    "for_store": "        acc = 0.0\n        for _ in range(2):\n            self.s = acc\n            acc = acc + 1.0\n        return x + acc",
    "nested": (
        "        acc = 0.0\n        for _ in range(2):\n            for _j in range(2):\n"
        "                if a.shape[0] > 1:\n                    acc = acc + 1.0\n"
        "        return x + acc"
    ),
}


# Kernels whose accept must be checked against Python, not merely counted. The outcome alphabet alone cannot
# see a miscompile -- a wrong answer tallies as a good accept, which is exactly how two of them shipped. These
# carry a reset that is inexact in the target carrier, so if a speculated arm promotes it to runtime state the
# reset re-materializes narrower and a guard reading it flips. Each entry is (source, args, expected).
# Routes documented as OPEN in TODO.md, one entry per open route -- an oracle carrying only some of them cannot
# support a claim about the seam as a whole. The value is the WRONG answer the hardware currently returns, not
# merely the fact that it is wrong: recording only "diverges" would let one wrong answer silently become a
# different wrong answer. They miscompile today, so the tool reports them without failing, and fails on ANY
# change of outcome -- a different wrong value, the right value, or a refusal alike -- because each means the
# record has stopped describing the code, which is how this seam's earlier claims decayed.
_KNOWN_OPEN = {
    "live_in_poisoned_across_rounds": 30.0,
    "phantom_environment_keeps_a_stale_gate": 22.0,
    "self_assignment_defeats_the_runtime_state_check": 20.0,
}

_VALUE_ORACLE = {
    "dead_store": (
        """import numpy as np


class K:
    def __init__(self) -> None:
        self.t = 0.0
        self.s = 1 + 2**-30

    def step(self, x: float, flag: bool) -> float:
        if flag:
            u = 1.0
            q = 1.0
        else:
            u = 2**53 + 1
            q = 2**64
        self.t = u
        a = np.array([q, x])
        if a.shape[0] > 5:
            self.s = 7.0
        if self.s > 1:
            return 10.0
        return 20.0
""",
        (2.0, False),
    ),
    "dead_store_in_loop": (
        """import numpy as np


class K:
    def __init__(self) -> None:
        self.t = 0.0
        self.s = 1 + 2**-30

    def step(self, x: float, flag: bool, run: bool) -> float:
        if flag:
            u = 1.0
            q = 1.0
        else:
            u = 2**53 + 1
            q = 2**64
        while run:
            self.t = u
            a = np.array([q, x])
            if a.shape[0] > 5:
                self.s = 7.0
            run = False
        if self.s > 1:
            return 10.0
        return 20.0
""",
        (2.0, False, True),
    ),
    # No store anywhere on either arm: the merge phi ALONE carries the inexact constant, so the speculated arm
    # rounds it without promoting anything. The store-scoped narrowing accepted this and returned 20.0.
    "merge_phi_rounds_a_constant": (
        """import numpy as np


class K:
    def __init__(self) -> None:
        self.t = 0.0

    def step(self, x: float, flag: bool) -> float:
        if flag:
            u = 1.0
            q = 1.0
        else:
            u = 2**53 + 1
            q = 2**64
        self.t = u
        a = np.array([q, x])
        if a.shape[0] > 5:
            c = 1.0
        else:
            c = 1 + 2**-30
        if c > 1.0:
            return 10.0
        return 20.0
""",
        (2.0, False),
    ),
    # Discovered as runtime state on a round whose store a later round proves unreachable. Invisible to every
    # check on the final graph, because the final graph is correct -- only the accumulated W set is stale.
    "runtime_state_from_a_dead_round": (
        """import numpy as np


class K:
    def __init__(self) -> None:
        self.mode = True
        self.t = 0.0
        self.s = 1 + 2**-30

    def step(self, x: float, new_mode: bool) -> float:
        if self.mode:
            u, q = 2**53 + 1, 2**64
        else:
            u, q = x, x
        self.t = u
        a = np.array([q, x])
        if a.shape[0] > 5:
            self.s = 7.0
        self.mode = new_mode
        return 10.0 if self.s > 1.0 else 20.0
""",
        (2.0, False),
    ),
    # The other half of the W/D accumulator: a round-1 speculated arm poisons the live-in map, round 2 prunes
    # that arm, and a trailing store keeps the leaf in first_store so the runtime-state check passes too. OPEN.
    "live_in_poisoned_across_rounds": (
        """import numpy as np


class K:
    def __init__(self) -> None:
        self.mode = True
        self.t = 0.0
        self.s = 1 + 2**-30

    def step(self, x: float, flag: bool) -> float:
        if self.mode:
            u: float = 2**53 + 1
            q: float = 2**64
        else:
            u = x
            q = x
        self.t = u
        a = np.array([q, x])
        if a.shape[0] > 5:
            self.mode = False
        if self.mode:
            r = 10.0
        else:
            self.s = 7.0
            r = 20.0
        if flag:
            self.mode = True
        return r if self.s > 1.0 else 30.0
""",
        (2.0, False),
    ),
    # The phantom environment keeps a stale `gate` alive, so the condition settles as a runtime bool and the
    # Python-dead arm is genuinely live to the analyzer -- no contradiction for the gate to detect.
    "phantom_environment_keeps_a_stale_gate": (
        """import numpy as np


class K:
    def __init__(self) -> None:
        self.t = 0.0
        self.gate = False
        self.s = 1 + 2**-30

    def helper(self, a: float, b: float) -> float:
        self.gate = True
        return a * b

    def step(self, x: float, flag: bool, pick: bool) -> float:
        if flag:
            u = 1.0
            q = 1.0
        else:
            u = 2**53 + 1
            q = 2**64
        self.t = u
        args = np.array([q, x])
        self.helper(*args)
        if pick:
            pad = 1.0
        else:
            pad = 2.0
        if self.gate:
            marker = 0.0
        else:
            self.s = 7.0
            marker = 100.0
        if self.s > 1:
            return 10.0 + pad + marker
        return 20.0 + pad + marker
""",
        (3.0, False, False),
    ),
    # One line of ordinary Python defeats the runtime-state rule: the self-assignment keeps a store for the leaf
    # in the stable graph, so the check finds nothing and the dead arm's promotion goes through unchanged.
    "self_assignment_defeats_the_runtime_state_check": (
        """import numpy as np


class K:
    def __init__(self) -> None:
        self.mode = True
        self.t = 0.0
        self.s = 1 + 2**-30

    def step(self, x: float, new_mode: bool) -> float:
        if self.mode:
            u: float = 2**53 + 1
            q: float = 2**64
        else:
            u = x
            q = x
        self.t = u
        a = np.array([q, x])
        if a.shape[0] > 5:
            self.s = 7.0
        self.s = self.s
        self.mode = new_mode
        return 10.0 if self.s > 1.0 else 20.0
""",
        (2.0, False),
    ),
    "inert_arm_poisons_later_guard": (
        """import numpy as np


class K:
    def __init__(self) -> None:
        self.t = 0.0
        self.s = 1 + 2**-30

    def step(self, x: float, flag: bool) -> float:
        if flag:
            u = 1.0
            q = 1.0
        else:
            u = 2**53 + 1
            q = 2**64
        self.t = u
        a = np.array([q, x])
        if a.shape[0] > 5:
            g = True
        else:
            g = False
        if g:
            self.s = 7.0
        if self.s > 1:
            return 10.0
        return 20.0
""",
        (2.0, False),
    ),
}

# A name in _KNOWN_OPEN that no oracle kernel answers to would measure nothing while reading as coverage --
# the same silent decay the recorded state exists to prevent.
assert _KNOWN_OPEN.keys() <= _VALUE_ORACLE.keys()


# The loop family above puts the trigger and the call BEFORE the loop, which cannot exhibit the starvation that
# disqualified both rejected fixes: those fail only when the deferring call is INSIDE the body, so the branch
# they hold back precedes the body's own back-edge. These kernels are whole templates for that reason -- a
# family whose numbers cannot move under the regression it exists to guard is not evidence.
_LOOP_INNER = {
    "while_call_inside": """import numpy as np


class K:
    def __init__(self) -> None:
        self.t = 0.0
        self.s = 0.0

    def step(self, x: float, run: bool) -> float:
        first = True
        while run:
            self.t = ({wide}) if first else x
            a = np.array([({wider}) if first else x, x])
            np.dot(a, a)
            first = False
            run = False
        return x + self.t
""",
    "while_call_inside_branch": """import numpy as np


class K:
    def __init__(self) -> None:
        self.t = 0.0
        self.s = 0.0

    def step(self, x: float, run: bool) -> float:
        first = True
        acc = 0.0
        while run:
            self.t = ({wide}) if first else x
            a = np.array([({wider}) if first else x, x])
            y = np.dot(a, a)
            if y > 0.0:
                acc = acc + 1.0
            first = False
            run = False
        return x + self.t + acc
""",
    "for_call_inside": """import numpy as np


class K:
    def __init__(self) -> None:
        self.t = 0.0
        self.s = 0.0

    def step(self, x: float, flag: bool) -> float:
        acc = 0.0
        for i in range(2):
            self.t = ({wide}) if i == 0 else x
            a = np.array([({wider}) if i == 0 else x, x])
            y = np.dot(a, a)
            if y > 0.0:
                acc = acc + 1.0
        return x + self.t + acc
""",
}


def _families() -> dict[str, dict[str, str]]:
    return {"dead_arm": _DEAD_ARM, "loop": _LOOP}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", default=".", help="worktree to bind (inserted at sys.path[0])")
    parser.add_argument("--verbose", action="store_true", help="one line per kernel")
    args = parser.parse_args()

    tree = pathlib.Path(args.tree).resolve()
    sys.path.insert(0, str(tree))
    import holoso
    from holoso import FloatFormat

    bound = pathlib.Path(holoso.__file__).resolve()
    assert bound.is_relative_to(tree), f"bound {bound}, expected under {tree}"
    from tests._modelref import default_ops  # noqa: PLC0415 -- must follow the sys.path binding

    print(f"bound: {bound}")
    ops = default_ops(FloatFormat(8, 23))
    totals: dict[str, dict[str, int]] = {}
    miscompiles = 0
    stale_record = 0
    refused = 0
    with tempfile.TemporaryDirectory() as scratch:
        for label, (source, arguments) in _VALUE_ORACLE.items():
            path = pathlib.Path(scratch) / f"k_oracle_{label}.py"
            path.write_text(source)
            spec = importlib.util.spec_from_file_location(path.stem, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            try:
                expected = module.K().step(*arguments)
                built = holoso.synthesize(module.K().step, ops, name=path.stem)
                # NOT run(...)[0]: when the return is value-identical to a persisted slot the exit dedups it and
                # no out_0 exists, so index 0 is an unrelated state port and the comparison is meaningless.
                outputs = dict(
                    zip((port.name for port in built.output_ports), built.numerical_model.elaborate().run(*arguments))
                )
                actual = float(outputs["out_0"])
            except holoso.HolosoError:
                refused += 1
                if label in _KNOWN_OPEN:
                    # Safer than the recorded miscompile, but the record now describes behavior the code no
                    # longer has. Failing is the same rule as the FIXED arm below: any outcome change on a
                    # documented route gets reconciled with TODO.md rather than silently absorbed.
                    stale_record += 1
                    print(f"  oracle {label:32s} REFUSED but recorded as an open miscompile -- update TODO.md")
                else:
                    print(f"  oracle {label:32s} refused (safe, but proves nothing about values)")
                continue
            except Exception as error:  # noqa: BLE001
                miscompiles += 1
                print(f"  oracle {label:32s} CRASH:{type(error).__name__}: {str(error)[:60]}")
                continue
            diverged = actual != expected
            if diverged and label in _KNOWN_OPEN and actual == _KNOWN_OPEN[label]:
                # A documented open route (TODO.md), returning exactly the wrong value on record. Reported, but
                # it does not fail the run: the tool gates on CHANGE, so a known miscompile does not mask a new
                # one while still leaving room for the routes that are known to be open.
                print(f"  oracle {label:32s} accepted python={expected} hardware={actual}  KNOWN-OPEN miscompile")
            elif diverged and label in _KNOWN_OPEN:
                stale_record += 1
                print(
                    f"  oracle {label:32s} accepted python={expected} hardware={actual}  "
                    f"CHANGED from the recorded {_KNOWN_OPEN[label]} -- update TODO.md"
                )
            elif diverged:
                miscompiles += 1
                print(f"  oracle {label:32s} accepted python={expected} hardware={actual}  *** MISCOMPILE ***")
            elif label in _KNOWN_OPEN:
                stale_record += 1  # a documented route stopped miscompiling: the record is stale, not the code
                print(f"  oracle {label:32s} accepted python={expected} hardware={actual}  FIXED -- update TODO.md")
            else:
                print(f"  oracle {label:32s} accepted python={expected} hardware={actual}  OK")
        tally: dict[str, int] = {}
        for label, template in _LOOP_INNER.items():
            for feed_name, (wide, wider) in {"wide": ("2**53 + 1", "2**64"), "narrow": ("2.0", "3.0")}.items():
                path = pathlib.Path(scratch) / f"k_loop_inner_{feed_name}_{label}.py"
                path.write_text(template.format(wide=wide, wider=wider))
                spec = importlib.util.spec_from_file_location(path.stem, path)
                assert spec is not None and spec.loader is not None
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                try:
                    holoso.synthesize(module.K().step, ops, name=path.stem)
                    outcome = "accept"
                except holoso.HolosoError:
                    outcome = "refuse"
                except Exception as error:  # noqa: BLE001
                    outcome = f"CRASH:{type(error).__name__}"
                tally[outcome] = tally.get(outcome, 0) + 1
                if args.verbose:
                    print(f"  loop_inner {feed_name}_{label:26s} {outcome}")
        totals["loop_inner"] = tally
        for family, bodies in _families().items():
            tally = {}
            product = itertools.product(_FEEDS.items(), _CALLS.items(), bodies.items())
            for (feed_name, feed), (call_name, call), (body_name, body) in product:
                label = f"{feed_name}_{call_name}_{body_name}"
                path = pathlib.Path(scratch) / f"k_{family}_{label}.py"
                path.write_text(_PROLOGUE.format(u=feed[0], q=feed[1]) + call + "\n" + body + "\n")
                spec = importlib.util.spec_from_file_location(path.stem, path)
                assert spec is not None and spec.loader is not None
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                try:
                    holoso.synthesize(module.K().step, ops, name=path.stem)
                    outcome = "accept"
                except holoso.HolosoError:
                    outcome = "refuse"
                except Exception as error:  # noqa: BLE001 -- a raw escape is exactly what the sweep looks for
                    outcome = f"CRASH:{type(error).__name__}"
                tally[outcome] = tally.get(outcome, 0) + 1
                if args.verbose:
                    print(f"  {family:9s} {label:24s} {outcome}")
            totals[family] = tally
    for family, tally in totals.items():
        size = sum(tally.values())
        summary = "  ".join(f"{name}={count}" for name, count in sorted(tally.items()))
        print(f"{family:9s} ({size:3d} kernels)  {summary}")
    if miscompiles:
        print(f"{miscompiles} value oracle kernel(s) diverged from Python -- a wrong answer, not a refusal")
    if stale_record:
        print(f"{stale_record} recorded open route(s) changed outcome -- the record no longer describes the code")
    if refused:
        # Said out loud because a refusal is not evidence of value correctness: the oracle contributes signal
        # only where the kernel is accepted, and a gate that refuses everything would score a clean run.
        print(f"{refused} of {len(_VALUE_ORACLE)} value oracle kernel(s) were refused, so their values are unproven")
    crashed = any(name.startswith("CRASH") for tally in totals.values() for name in tally)
    return 1 if (miscompiles or crashed or stale_record) else 0


if __name__ == "__main__":
    sys.exit(main())

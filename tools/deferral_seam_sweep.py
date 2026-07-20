"""
Sweep the analyzer's deferral/grafting seam and report accept/refuse/crash counts per kernel family.

The seam is a known-defective area (TODO.md "Known defects"): a call deferred behind a transiently pending
store violation leaves stale reachability behind, which costs accepts and, on one surviving route, miscompiles.
Every change near it has to be judged on whether it moves those counts, and prose claims about "the loop family"
or "the dead-arm family" are unfalsifiable without the corpus that produced them. This generates both families
into real files -- exec-compiled kernels cannot lower, they raise SourceUnavailable -- and tallies outcomes.

Bind a specific worktree with --tree; note PYTHONPATH does NOT override an editable install, so the tree is
inserted at sys.path[0] before holoso is imported and the binding is asserted.

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
    with tempfile.TemporaryDirectory() as scratch:
        for family, bodies in _families().items():
            tally: dict[str, int] = {}
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
    return 1 if any(name.startswith("CRASH") for tally in totals.values() for name in tally) else 0


if __name__ == "__main__":
    sys.exit(main())

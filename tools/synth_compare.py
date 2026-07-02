#!/usr/bin/env python3
"""
Before/after synthesis comparison for the bundled example matrix.

``capture`` synthesizes every TARGET on the current checkout and writes a JSON of per-target cycle latency (min II and
last PC), f_max, slack, fabric area, and pass/fail. ``render`` reads a BEFORE and an AFTER JSON and emits a side-by-side
HTML report with deltas, pass/fail badges, headline totals, and a flag on every target whose operator stage knobs were
retuned (its f_max delta then reflects change+retune, not an isolated knob-fixed comparison; retuning is detected by a
changed ops repr). Report-only tooling -- not part of the compiler, no tests, no design-doc coupling.

Usage:
    python tools/synth_compare.py capture --out before.json
    python tools/synth_compare.py render --before before.json --after after.json --out report.html
"""

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import holoso  # noqa: E402
from holoso._frontend import lower  # noqa: E402
from holoso._hir import optimize  # noqa: E402
from holoso._lir import build  # noqa: E402
from holoso._mir import lower as lower_to_mir  # noqa: E402
from synth.flows import make_flow  # noqa: E402
from tests._synth_targets import TARGETS, TARGET_ENV_KEYS  # noqa: E402

# Per-flow resource-primitive names for the LUT/FF/DSP/BRAM report columns; each tool names them differently.
_RES_KEYS = {
    "yosys-ecp5": (("TRELLIS_COMB", "LUT"), ("TRELLIS_FF", "FF"), ("MULT18X18D", "DSP"), ("DP16KD", "BRAM")),
    "diamond-ecp5": (("LUT4", "LUT"), ("Registers", "FF"), ("MULT18X18D", "DSP"), ("DP16KD", "BRAM")),
    "vivado-artix7": (("Slice LUTs", "LUT"), ("Slice Registers", "FF"), ("DSPs", "DSP"), ("RAMB18", "BRAM")),
}


def _apply_env(env: dict) -> None:
    for key in TARGET_ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in env.items():
        os.environ[key] = value


def capture(out_path: str) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout)
    rows = []
    for target in TARGETS:
        _apply_env(target.env)
        row = {
            "label": target.label,
            "name": target.name,
            "example": target.example,
            "flow": target.flow.value,
            "target_MHz": target.target_frequency_MHz,
            "ops": repr(target.ops),
            "env": dict(target.env),
        }
        try:
            lir = build(lower_to_mir(optimize(lower(target.kernel())), target.ops), target.name, fetch_stages=3)
            row["min_ii"] = lir.min_initiation_interval
            row["last_pc"] = lir.last_pc
            flow = make_flow(target.flow, target.target_frequency_MHz)
            if not flow.available():
                row["available"] = False
            else:
                row["available"] = True
                result = holoso.synthesize(target.kernel(), target.ops, name=target.name)
                directory = REPO / "build" / "synth_compare" / target.label
                shutil.rmtree(directory, ignore_errors=True)
                report = flow.prepare(result).synthesize(directory)
                row["fmax_MHz"] = report.fmax_MHz
                row["slack_ns"] = report.slack_ns
                row["resources"] = {n: {"used": r.used, "available": r.available} for n, r in report.resources.items()}
                row["passed"] = report.fmax_MHz >= target.target_frequency_MHz
                row["critical_path"] = _critical_path(directory, target.flow.value)
        except Exception as exc:  # noqa: BLE001 -- a single target's failure must not abort the whole sweep
            row["error"] = f"{type(exc).__name__}: {exc}"
        marker = f"{row.get('fmax_MHz'):.2f} MHz" if "fmax_MHz" in row else row.get("error", "skipped (tool absent)")
        print(f"  {target.label:42} II={row.get('min_ii','?'):>4}  {marker}", flush=True)
        rows.append(row)
        payload = {"commit": commit, "dirty": dirty, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}
        Path(out_path).write_text(json.dumps(payload, indent=2))  # incremental: crash-safe over a long sweep
    print(f"\nWrote {out_path}: {len(rows)} targets @ {commit}{'+dirty' if dirty else ''}")


def _delta_cell(before, after, *, lower_is_better: bool, unit: str = "", fmt: str = "{:.2f}") -> str:
    if before is None or after is None:
        b = fmt.format(before) if before is not None else "—"
        a = fmt.format(after) if after is not None else "—"
        return f"<td class='num'>{b}{unit} → {a}{unit}</td>"
    d = after - before
    good = (d < 0) if lower_is_better else (d > 0)
    cls = "good" if (d and good) else ("bad" if d else "same")
    sign = "+" if d > 0 else ""
    return (
        f"<td class='num'>{fmt.format(before)}{unit} → {fmt.format(after)}{unit}"
        f"<span class='delta {cls}'>{sign}{fmt.format(d)}</span></td>"
    )


def _resource_total(row, key):
    res = row.get("resources") or {}
    item = res.get(key)
    return item["used"] if item else None


def _critical_path(directory: Path, flow: str) -> str | None:
    """Extract the place-and-route worst (critical) timing path report for the flow's tool, as plain text."""
    try:
        if "diamond" in flow:
            twr = next(iter((directory / "impl1").glob("*_impl1.twr")), None)
            if twr is None:
                return None
            lines = twr.read_text(errors="ignore").splitlines()
            for i, ln in enumerate(lines):
                if "Logical Details:" in ln:
                    out: list[str] = []
                    for sub in lines[i:]:
                        out.append(sub.rstrip())
                        if "(to clk_c)" in sub:
                            return "\n".join(out)
        elif "yosys" in flow:
            log = directory / "nextpnr.log"
            if not log.exists():
                return None
            lines = log.read_text(errors="ignore").splitlines()
            for i, ln in enumerate(lines):
                if "Critical path report" in ln:
                    out: list[str] = []
                    for sub in lines[i:]:
                        out.append(sub.replace("Info: ", "").rstrip())
                        if "ns logic" in sub and "ns rout" in sub:  # the path's terminating summary line
                            return "\n".join(out)
                    return "\n".join(out[:80])
        elif "vivado" in flow:
            rpt = directory / "worst_path.rpt"
            if not rpt.exists():
                return None
            lines = rpt.read_text(errors="ignore").splitlines()
            for i, ln in enumerate(lines):
                if "Slack (" in ln:
                    return "\n".join(s.rstrip() for s in lines[i : i + 12])
    except OSError:
        return None
    return None


def render(before_path: str, after_path: str, out_path: str) -> None:
    before_doc = json.loads(Path(before_path).read_text())
    after_doc = json.loads(Path(after_path).read_text())
    before = {r["label"]: r for r in before_doc["rows"]}
    after = {r["label"]: r for r in after_doc["rows"]}
    labels = [r["label"] for r in after_doc["rows"]]

    cycles_saved = 0
    fmax_deltas = []
    flips_to_pass = 0
    flips_to_fail = 0
    body = []
    regressed = []
    for label in labels:
        a = after[label]
        b = before.get(label, {})
        retuned = (b.get("ops") is not None and b.get("ops") != a.get("ops")) or (
            b.get("env") is not None and b.get("env") != a.get("env")
        )
        ii_b, ii_a = b.get("min_ii"), a.get("min_ii")
        if ii_b is not None and ii_a is not None:
            cycles_saved += ii_b - ii_a
        fmb, fma = b.get("fmax_MHz"), a.get("fmax_MHz")
        if fmb is not None and fma is not None:
            fmax_deltas.append(fma - fmb)
        pb = b.get("passed")
        a_failed = a.get("passed") is False or (a.get("passed") is None and a.get("error") is not None)
        if pb is False and a.get("passed") is True:
            flips_to_pass += 1
        if pb is True and a_failed:
            flips_to_fail += 1
        badge = (
            "<span class='fail'>❌ FAIL</span>"
            if a_failed
            else ("<span class='pass'>✅ pass</span>" if a.get("passed") else "<span class='skip'>— n/a</span>")
        )
        retune_tag = " <span class='retune'>retuned</span>" if retuned else ""
        cells = [
            f"<td class='name'>{a.get('example') or a['name']}{retune_tag}<br><span class='flow'>{a['flow']}"
            f" @ {a['target_MHz']:.0f} MHz</span></td>",
            _delta_cell(ii_b, ii_a, lower_is_better=True, fmt="{:.0f}"),
            _delta_cell(b.get("last_pc"), a.get("last_pc"), lower_is_better=True, fmt="{:.0f}"),
            _delta_cell(fmb, fma, lower_is_better=False, unit="", fmt="{:.1f}"),
            f"<td>{badge}</td>",
        ]
        for key, lbl in _RES_KEYS.get(a["flow"], _RES_KEYS["yosys-ecp5"]):
            cells.append(
                _delta_cell(_resource_total(b, key), _resource_total(a, key), lower_is_better=True, fmt="{:.0f}")
            )
        cells.append(f"<td class='err'>{html.escape(a.get('error', ''))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
        ii_up = ii_b is not None and ii_a is not None and ii_a > ii_b
        fmax_down = fmb is not None and fma is not None and fma < fmb - 1.0
        if retuned or ii_up or fmax_down or a_failed:
            regressed.append(label)

    cp_blocks = []
    for label in regressed:
        a, b = after[label], before.get(label, {})
        fmb, fma = b.get("fmax_MHz"), a.get("fmax_MHz")
        fm_b = f"{fmb:.1f}" if fmb is not None else "—"
        fm_a = f"{fma:.1f}" if fma is not None else "—"
        cp_b = html.escape(b.get("critical_path") or "(not captured)")
        cp_a = html.escape(a.get("critical_path") or "(not captured)")
        cp_blocks.append(
            f"<details><summary><b>{label}</b> &nbsp;·&nbsp; f_max {fm_b}→{fm_a} MHz &nbsp;·&nbsp; "
            f"II {b.get('min_ii')}→{a.get('min_ii')}</summary>"
            f"<div class='cp'><div><div class='cphdr'>before <code>{before_doc.get('commit','?')}</code></div>"
            f"<pre>{cp_b}</pre></div><div><div class='cphdr'>after <code>{after_doc.get('commit','?')}</code></div>"
            f"<pre>{cp_a}</pre></div></div></details>"
        )
    critical_paths = "\n".join(cp_blocks) if cp_blocks else "<p class='sub'>No timing-regressed modules.</p>"

    fmax_summary = (
        f"min {min(fmax_deltas):+.1f}, median {sorted(fmax_deltas)[len(fmax_deltas)//2]:+.1f}, "
        f"max {max(fmax_deltas):+.1f} MHz"
        if fmax_deltas
        else "n/a"
    )
    headline = (
        f"<b>{cycles_saved}</b> total II cycles reclaimed across {len(labels)} targets &nbsp;·&nbsp; "
        f"f_max Δ: {fmax_summary} &nbsp;·&nbsp; pass↔fail flips: "
        f"<span class='good'>{flips_to_pass}→pass</span>, <span class='bad'>{flips_to_fail}→fail</span>"
    )
    page = _PAGE.format(
        before_commit=before_doc.get("commit", "?"),
        after_commit=after_doc.get("commit", "?"),
        before_ts=before_doc.get("timestamp", ""),
        after_ts=after_doc.get("timestamp", ""),
        headline=headline,
        rows="\n".join(body),
        critical_paths=critical_paths,
    )
    Path(out_path).write_text(page)
    print(f"Wrote {out_path}")


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Holoso synthesis: before vs after</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;color:#1a1a2e;background:#fafaff}}
 h1{{font-size:1.4rem;margin:0 0 .3rem}}
 .sub{{color:#667;margin-bottom:1rem}}
 .headline{{background:#eef;border:1px solid #cce;border-radius:8px;padding:.7rem 1rem;margin-bottom:1.2rem}}
 table{{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 4px #0001;border-radius:8px;overflow:hidden}}
 th,td{{padding:.45rem .7rem;text-align:left;border-bottom:1px solid #eee}}
 th{{background:#1a1a2e;color:#fff;font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em}}
 td.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
 td.name{{font-weight:600}} .flow{{font-weight:400;color:#778;font-size:.8rem}}
 .delta{{margin-left:.5rem;font-size:.82rem;padding:.05rem .35rem;border-radius:4px}}
 .delta.good{{background:#dcfce7;color:#166534}} .delta.bad{{background:#fee2e2;color:#991b1b}}
 .delta.same{{color:#aab}}
 .pass{{color:#166534;font-weight:600}} .fail{{color:#991b1b;font-weight:700}} .skip{{color:#99a}}
 .retune{{background:#fef3c7;color:#92400e;font-size:.72rem;padding:.05rem .35rem;border-radius:4px;font-weight:600}}
 .err{{color:#991b1b;font-size:.8rem}}
 tr:hover td{{background:#f6f6ff}}
 details{{background:#fff;border:1px solid #e3e3ee;border-radius:8px;margin:.5rem 0;padding:.4rem .9rem;box-shadow:0 1px 4px #0001}}
 summary{{cursor:pointer;font-variant-numeric:tabular-nums;font-weight:600}}
 .cp{{display:flex;gap:1rem;margin-top:.6rem}} .cp>div{{flex:1;min-width:0}}
 .cphdr{{font-size:.76rem;color:#667;font-weight:600;text-transform:uppercase;letter-spacing:.04em;margin-bottom:.25rem}}
 .cp pre{{background:#0d1117;color:#c9d1d9;padding:.6rem .8rem;border-radius:6px;font-size:.73rem;overflow-x:auto;white-space:pre;line-height:1.35}}
</style></head><body>
<h1>Holoso synthesis — before vs after the <code>ucode[0]</code> NOP reclaim</h1>
<div class="sub">before <code>{before_commit}</code> ({before_ts}) &nbsp;→&nbsp; after <code>{after_commit}</code> ({after_ts})</div>
<div class="headline">{headline}</div>
<table><thead><tr>
 <th>kernel / flow</th><th>min II</th><th>last PC</th><th>f_max (MHz)</th><th>closure</th>
 <th>LUT</th><th>FF</th><th>DSP</th><th>BRAM</th><th>note</th>
</tr></thead><tbody>
{rows}
</tbody></table>
<p class="sub" style="margin-top:1rem">Lower is better for II / last PC / area (green Δ); higher is better for f_max.
A <span class="retune">retuned</span> tag marks a target whose operator stage knobs changed, so its f_max/area delta
reflects the change <em>and</em> the retune, not an isolated comparison.</p>
<h2 style="font-size:1.1rem;margin:1.8rem 0 .3rem">Critical-path reports — timing-regressed / retuned modules</h2>
<p class="sub">Every module the change pushed off its closure target, or whose latency (II) or f_max regressed, with the
place-and-route worst path <em>before</em> and <em>after</em>. The bottleneck shift shows what the added register
stages addressed (or why a routing-dominated path needed placement effort instead of pipelining).</p>
{critical_paths}
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--out", required=True)
    ren = sub.add_parser("render")
    ren.add_argument("--before", required=True)
    ren.add_argument("--after", required=True)
    ren.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cmd == "capture":
        capture(args.out)
    else:
        render(args.before, args.after, args.out)


if __name__ == "__main__":
    main()

"""
Out-of-context (OOC) synthesis-evaluation harness for Holoso-generated modules.

The public surface is what this module re-exports below. Concrete flows are imported per tool so pulling in one
does not require the others; the caller supplies the DUT's RTL dependencies (e.g. the Kulibin float primitives)::

    from synth.flows.yosys import YosysEcp5Flow, Ecp5Device
    result: SynthesisResult = ...  # See holoso API
    artifact = YosysEcp5Flow(device=Ecp5Device(), target_frequency_MHz=100.0).prepare(result, extra_rtl)
    report = artifact.synthesize()
    print(report.fmax_MHz, report.slack_ns, report.resources["DSP"].used)
"""

from ._ooc import build_ooc_wrapper as build_ooc_wrapper
from ._synth import SynthReport as SynthReport

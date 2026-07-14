"""
Out-of-context (OOC) synthesis-evaluation harness for Holoso-generated modules.

Concrete flows are imported per tool so pulling in one does not require the others. A generated module's only
dependency is the bundled support library, so the flow needs nothing beyond the synthesis result::

    from holoso import SynthesisResult
    from synth import build_compiler_ooc_design
    from synth.flows.yosys import YosysEcp5Flow, Ecp5Device
    result: SynthesisResult = ...  # See holoso API
    design = build_compiler_ooc_design(result)
    artifact = YosysEcp5Flow(device=Ecp5Device(), target_frequency_MHz=100.0).prepare(design)
    report = artifact.synthesize()
    print(report.fmax_MHz, report.slack_ns, report.resources["DSP"].used)

Tool availability is discovered at runtime by checking the `$PATH` and searching a few predefined locations such as
`/usr` and `/opt`. To ensure predictable results it is recommended to ensure the tool of interest is on `$PATH`.
"""

from ._ooc import build_ooc_wrapper as build_ooc_wrapper
from ._synth import OocDesign as OocDesign
from ._synth import SourceFile as SourceFile
from ._synth import build_compiler_ooc_design as build_compiler_ooc_design
from ._synth import SynthReport as SynthReport

"""
The Verilog backend renders a finished LIR into the ZISC kernel module and its private support modules,
all in a single .v file named after the original kernel.
"""

from ._emit import VerilogOutput as VerilogOutput, generate as generate, private_module as private_module

"""The Verilog backend: render a finished Lir into a synthesizable ZISC module plus its shared support HDL."""

from .verilog import VerilogOutput as VerilogOutput, generate as generate

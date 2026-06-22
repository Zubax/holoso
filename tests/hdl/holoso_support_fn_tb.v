// Test-only harness: the combinational helpers in holoso_support.vh are Verilog functions (a generated module
// `include`s the header and invokes them by name), so they cannot be a cocotb toplevel directly. These thin wrapper
// modules expose them as instantiable modules with x/y ports for the HDL tests; they ship only with the tests.

`default_nettype none

module holoso_fisfinite_tb #(parameter WEXP = 6, parameter WMAN = 18) (
    input  wire [WEXP+WMAN-1:0] x,
    output wire                 y
);
    localparam W = WEXP + WMAN;
    `include "holoso_support.vh"
    assign y = holoso_fisfinite(x);
endmodule

module holoso_fsaturate_tb #(parameter WEXP = 6, parameter WMAN = 18) (
    input  wire [WEXP+WMAN-1:0] x,
    output wire [WEXP+WMAN-1:0] y
);
    localparam W = WEXP + WMAN;
    `include "holoso_support.vh"
    assign y = holoso_fsaturate(x);
endmodule

`default_nettype wire

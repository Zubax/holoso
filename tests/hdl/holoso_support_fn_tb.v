// Test-only harness: the combinational helpers in holoso_support_inline.vh are Verilog functions (the emitter splices
// them into each generated module; here we `include` the same source after declaring W), so they cannot be a cocotb
// toplevel directly. These thin wrapper modules expose them as instantiable modules with x/y ports for the HDL tests.

`default_nettype none

module holoso_ashiftc_tb #(parameter W = 24) (
    input  wire signed [W-1:0] x,
    input  wire signed [W-1:0] shamt,
    output wire signed [W-1:0] y
);
    localparam WEXP = 1;
    localparam WMAN = W - WEXP;
    `include "holoso_support_inline.vh"
    assign y = holoso_ashiftc(x, shamt);
endmodule

module holoso_fisfinite_tb #(parameter WEXP = 6, parameter WMAN = 18) (
    input  wire [WEXP+WMAN-1:0] x,
    output wire                 y
);
    localparam W = WEXP + WMAN;
    `include "holoso_support_inline.vh"
    assign y = holoso_fisfinite(x);
endmodule

module holoso_fisposinf_tb #(parameter WEXP = 6, parameter WMAN = 18) (
    input  wire [WEXP+WMAN-1:0] x,
    output wire                 y
);
    localparam W = WEXP + WMAN;
    `include "holoso_support_inline.vh"
    assign y = holoso_fisposinf(x);
endmodule

module holoso_fisneginf_tb #(parameter WEXP = 6, parameter WMAN = 18) (
    input  wire [WEXP+WMAN-1:0] x,
    output wire                 y
);
    localparam W = WEXP + WMAN;
    `include "holoso_support_inline.vh"
    assign y = holoso_fisneginf(x);
endmodule

module holoso_fsaturate_tb #(parameter WEXP = 6, parameter WMAN = 18) (
    input  wire [WEXP+WMAN-1:0] x,
    output wire [WEXP+WMAN-1:0] y
);
    localparam W = WEXP + WMAN;
    `include "holoso_support_inline.vh"
    assign y = holoso_fsaturate(x);
endmodule

module holoso_fsgnop_tb #(parameter WEXP = 6, parameter WMAN = 18) (
    input  wire [WEXP+WMAN-1:0] x,
    input  wire           [1:0] op,
    output wire [WEXP+WMAN-1:0] y
);
    localparam W = WEXP + WMAN;
    `include "holoso_support_inline.vh"
    assign y = holoso_fsgnop(x, op);
endmodule

`default_nettype wire

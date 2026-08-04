// Test-only harness: the combinational helpers in holoso_support_inline.vh are Verilog functions (the emitter splices
// them into each generated module; here we `include` the same source after declaring WFLT and WINT), so they cannot
// be a cocotb toplevel directly. These thin wrapper modules expose them as instantiable modules with x/y ports for
// the HDL tests.

`default_nettype none

// The spliced file defines the float helpers too, so even a shift-only harness must name a float format for them.
module holoso_ashiftc_tb #(parameter WINT = 24) (
    input  wire signed [WINT-1:0] x,
    input  wire signed [WINT-1:0] shamt,
    output wire signed [WINT-1:0] y
);
    localparam WEXP = 6;
    localparam WMAN = 18;
    localparam WFLT = WEXP + WMAN;
    `include "holoso_support_inline.vh"
    assign y = holoso_ashiftc(x, shamt);
endmodule

module holoso_fisfinite_tb #(parameter WEXP = 6, parameter WMAN = 18) (
    input  wire [WEXP+WMAN-1:0] x,
    output wire                 y
);
    localparam WFLT = WEXP + WMAN;
    localparam WINT = 24;  // unused here, but the spliced file's shift helper names it
    `include "holoso_support_inline.vh"
    assign y = holoso_fisfinite(x);
endmodule

module holoso_fisposinf_tb #(parameter WEXP = 6, parameter WMAN = 18) (
    input  wire [WEXP+WMAN-1:0] x,
    output wire                 y
);
    localparam WFLT = WEXP + WMAN;
    localparam WINT = 24;  // unused here, but the spliced file's shift helper names it
    `include "holoso_support_inline.vh"
    assign y = holoso_fisposinf(x);
endmodule

module holoso_fisneginf_tb #(parameter WEXP = 6, parameter WMAN = 18) (
    input  wire [WEXP+WMAN-1:0] x,
    output wire                 y
);
    localparam WFLT = WEXP + WMAN;
    localparam WINT = 24;  // unused here, but the spliced file's shift helper names it
    `include "holoso_support_inline.vh"
    assign y = holoso_fisneginf(x);
endmodule

module holoso_fsaturate_tb #(parameter WEXP = 6, parameter WMAN = 18) (
    input  wire [WEXP+WMAN-1:0] x,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFLT = WEXP + WMAN;
    localparam WINT = 24;  // unused here, but the spliced file's shift helper names it
    `include "holoso_support_inline.vh"
    assign y = holoso_fsaturate(x);
endmodule

module holoso_fsgnop_tb #(parameter WEXP = 6, parameter WMAN = 18) (
    input  wire [WEXP+WMAN-1:0] x,
    input  wire           [1:0] op,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFLT = WEXP + WMAN;
    localparam WINT = 24;  // unused here, but the spliced file's shift helper names it
    `include "holoso_support_inline.vh"
    assign y = holoso_fsgnop(x, op);
endmodule

`default_nettype wire

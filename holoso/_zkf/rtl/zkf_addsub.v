/// Streamed Zubax Kulibin float adder/subtractor.
/// y = a + b when op_sub == 0; y = a - b when op_sub == 1.

`default_nettype none

// The latency is the same as zkf_add
module zkf_addsub #(
    parameter WEXP            = 6,
    parameter WMAN            = 18,   // significand precision including the hidden bit
    parameter STAGE_INPUT     = 0,    // forwarded to zkf_add
    parameter STAGE_DECODE    = 0,    // forwarded to zkf_add
    parameter STAGE_ALIGN     = 0,    // forwarded to zkf_add
    parameter STAGE_NORMALIZE = 0,    // forwarded to zkf_add
    parameter STAGE_PACK      = 0,    // forwarded to zkf_add
    parameter STAGE_OUTPUT    = 0,    // forwarded to zkf_add
    parameter LATENCY         = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b,
    input wire                 op_sub,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFULL = WEXP + WMAN;

    // LATENCY is forwarded to zkf_add, which performs the latency self-check.
    zkf_add #(
        .WEXP(WEXP), .WMAN(WMAN),
        .STAGE_INPUT(STAGE_INPUT),
        .STAGE_DECODE(STAGE_DECODE), .STAGE_ALIGN(STAGE_ALIGN),
        .STAGE_NORMALIZE(STAGE_NORMALIZE),
        .STAGE_PACK(STAGE_PACK),
        .STAGE_OUTPUT(STAGE_OUTPUT),
        .LATENCY(LATENCY)
    ) u_add (
        .clk(clk),
        .rst(rst),
        .in_valid(in_valid),
        .a(a),
        .b({b[WFULL-1] ^ op_sub, b[WFULL-2:0]}),
        .out_valid(out_valid),
        .y(y)
    );
endmodule

`default_nettype wire

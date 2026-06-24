/// Streamed Zubax Kulibin float adder/subtractor.
/// y = a + b when op_sub == 0; y = a - b when op_sub == 1.

`default_nettype none

// The latency is the same as zkf_add
`define ZKF_ADDSUB_LATENCY (4 + STAGE_INPUT + STAGE_DECODE + STAGE_ALIGN + STAGE_NORMALIZE + STAGE_PACK + STAGE_OUTPUT)

module zkf_addsub #(
    parameter WEXP            = 6,    // exponent field width
    parameter WMAN            = 18,   // significand precision including the hidden bit
    parameter STAGE_INPUT     = 0,    // forwarded to zkf_add
    parameter STAGE_DECODE    = 0,    // forwarded to zkf_add
    parameter STAGE_ALIGN     = 0,    // forwarded to zkf_add
    parameter STAGE_NORMALIZE = 0,    // forwarded to zkf_add
    parameter STAGE_PACK      = 0,    // forwarded to zkf_add
    parameter STAGE_OUTPUT    = 0,    // forwarded to zkf_add
    parameter LATENCY         = `ZKF_ADDSUB_LATENCY   // must equal the register-stage count; checked below
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

    generate
        if (LATENCY != `ZKF_ADDSUB_LATENCY) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    // Forward LATENCY into zkf_add so a drift in zkf_add's own stage count breaks this wrapper's default build too.
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

`undef ZKF_ADDSUB_LATENCY
`default_nettype wire

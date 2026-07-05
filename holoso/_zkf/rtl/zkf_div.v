/// Streamed Zubax Kulibin float divider. The quotient is rounded by _zkf_pack; div0 is aligned with q/out_valid.
/// div0 reports that the divisor's exponent field is zero (i.e., the divisor encodes +0). It is
/// independent of the quotient: in particular div0 is also asserted for 0/0, where q = +0.
///
/// STAGE_INPUT=0: input combinational paths are exposed.
/// STAGE_INPUT=1: inputs are latched, the external module sees registers at the input (one extra cycle).
/// STAGE_INPUT>1: add extra dummy stages; helps in routing-congested designs (+STAGE_INPUT cycles).
///
/// STAGE_PACK=0: pack inputs are combinational (default).
/// STAGE_PACK=1: register pack inputs (forwarded to _zkf_pack.STAGE_INPUT) (+1 cycle).
///
/// STAGE_OUTPUT=0: q and div0 are combinational (default)
/// STAGE_OUTPUT=1: registered (one extra cycle).

`default_nettype none

module zkf_div #(
    parameter WEXP         = 6,
    parameter WMAN         = 18,   // significand precision including the hidden bit
    parameter STAGE_INPUT  = 0,    // number of input register stages (>=0); +STAGE_INPUT cycles
    parameter STAGE_PACK   = 0,    // 0 = comb pack inputs; 1 = register pack inputs (+1 cycle)
    parameter STAGE_OUTPUT = 0,    // 0 = combinational outputs; 1 = registered outputs (+1 cycle)
    parameter LATENCY      = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] q,
    output wire                 div0
);
    localparam WFULL         = WEXP + WMAN;
    localparam WEXP_UNBIASED = WEXP + 2;

    localparam LATENCY_REF = 2 + STAGE_INPUT + ((WMAN+2+((WMAN+2)%2))/2) + STAGE_PACK + STAGE_OUTPUT;
    generate
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    // Optional input register stage(s): latch the operands before any combinational logic (+STAGE_INPUT cycles).
    wire                in_valid_q;
    wire [2*WFULL-1:0]  pipe_out;
    zkf_pipe #(.W(2*WFULL), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in({b, a}), .out_valid(in_valid_q), .out(pipe_out)
    );
    wire [WFULL-1:0] a_q = pipe_out[WFULL-1:0];
    wire [WFULL-1:0] b_q = pipe_out[2*WFULL-1:WFULL];

    wire                            core_valid;
    wire                            core_sign;
    wire                            core_force_zero;
    wire                            core_force_inf;
    wire signed [WEXP_UNBIASED-1:0] core_exp_biased;
    // normalized quotient significand.
    wire                 [WMAN-1:0] core_significand;
    wire                            core_guard;
    wire                            core_round;
    wire                            core_sticky;
    wire                            core_div0;

    _zkf_div_core #(.WEXP(WEXP), .WMAN(WMAN)) u_core (
        .clk(clk),
        .rst(rst),
        .in_valid(in_valid_q),
        .a(a_q),
        .b(b_q),
        .out_valid(core_valid),
        .sign(core_sign),
        .force_zero(core_force_zero),
        .force_inf(core_force_inf),
        .exp_biased(core_exp_biased),
        .significand(core_significand),
        .guard(core_guard),
        .round(core_round),
        .sticky(core_sticky),
        .div0(core_div0),
        .exp_diff(),
        .raw(),
        .den(),
        .partial_rem()  // Partial remainder is not used in this module.
    );

    // The packer drives the external q/out_valid directly; STAGE_OUTPUT selects registered vs combinational output.
    _zkf_pack #(
        .WEXP(WEXP), .WMAN(WMAN), .EXP_IS_BIASED(1),
        .STAGE_INPUT(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_pack (
        .clk(clk),
        .rst(rst),
        .in_valid(core_valid),
        .sign(core_sign),
        .force_zero(core_force_zero),
        .force_inf(core_force_inf),
        .exp_unbiased(core_exp_biased),
        .significand(core_significand),
        .guard(core_guard),
        .round(core_round),
        .sticky(core_sticky),
        .out_valid(out_valid),
        .y(q)
    );

    // The delay line is a pure free-running datapath (no reset port). See reset policy.
    // STAGE_OUTPUT matches the packer so div0 stays aligned with q.
    _zkf_pack_delay #(
        .W(1), .STAGE_INPUT(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_pack_delay (.clk(clk), .x(core_div0), .y(div0));
endmodule

`default_nettype wire

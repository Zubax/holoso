/// Streamed Zubax Kulibin float multiplier.
///
/// STAGE_INPUT=0: operands feed the multiplier combinationally (default).
/// STAGE_INPUT=1: latch the inputs before any combinational logic, isolating them from upstream paths (+1 cycle).
///
/// STAGE_PRODUCT sets the pipeline depth of _zkf_pmul multiplier (1+STAGE_PRODUCT cycles). Refer to _zkf_pmul.
/// WMULTIPLIER is an optional hint of the native DSP tile argument width; forwaded to _zkf_pmul, refer there.
///
/// STAGE_PACK=0: pack inputs are combinational (default).
/// STAGE_PACK=1: register pack inputs (forwarded to _zkf_pack.STAGE_INPUT) (+1 cycle).
///
/// STAGE_OUTPUT=0: the result is combinational (default).
/// STAGE_OUTPUT=1: the result is registered; good if the module feeds long external combinational paths (+1 cycle).

`default_nettype none

module zkf_mul #(
    parameter WEXP          = 6,    // exponent field width
    parameter WMAN          = 18,   // significand precision including the hidden bit
    parameter WMULTIPLIER   = 0,    // forwarded to _zkf_pmul
    parameter STAGE_INPUT   = 0,
    parameter STAGE_PRODUCT = 0,    // forwarded to _zkf_pmul
    parameter STAGE_PACK    = 0,
    parameter STAGE_OUTPUT  = 0,
    parameter LATENCY       = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam LATENCY_REF = 1 + STAGE_INPUT + STAGE_PRODUCT + STAGE_PACK + STAGE_OUTPUT;
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wman
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if ((STAGE_INPUT != 0) && (STAGE_INPUT != 1)) begin : g_invalid_stage_input
            _zkf_invalid_stage_input u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    localparam WFRAC         = WMAN - 1;
    localparam WFULL         = WEXP + WMAN;
    localparam WMAG          = 2 * WMAN;
    localparam WEXP_UNBIASED = WEXP + 2;

    localparam [WEXP-1:0] EXP_BIAS = {1'b0, {WEXP-1{1'b1}}};
    localparam [WEXP-1:0] EXP_INF  = {WEXP{1'b1}};

    localparam signed [WEXP_UNBIASED-1:0] ZERO_EXT = {WEXP_UNBIASED{1'b0}};
    localparam signed [WEXP_UNBIASED-1:0] ONE_EXT  = {{(WEXP_UNBIASED-1){1'b0}}, 1'b1};

    // Optional input register stage: latch the operands before any combinational logic (+1 cycle when STAGE_INPUT=1).
    wire             in_valid_q;
    wire [WFULL-1:0] a_q;
    wire [WFULL-1:0] b_q;
    zkf_pipe #(.W(2*WFULL), .N(STAGE_INPUT ? 1 : 0)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in({b, a}),
        .out_valid(in_valid_q), .out({b_q, a_q})
    );

    // Operand decode/classification.
    wire             a_sign = a_q[WFULL-1];
    wire             b_sign = b_q[WFULL-1];
    wire [WEXP-1:0]  a_exp  = a_q[WFULL-2:WFRAC];
    wire [WEXP-1:0]  b_exp  = b_q[WFULL-2:WFRAC];
    wire [WFRAC-1:0] a_frac = a_q[WFRAC-1:0];
    wire [WFRAC-1:0] b_frac = b_q[WFRAC-1:0];

    wire            a_zero        = a_exp == {WEXP{1'b0}};
    wire            b_zero        = b_exp == {WEXP{1'b0}};
    wire            a_inf         = a_exp == EXP_INF;
    wire            b_inf         = b_exp == EXP_INF;
    wire            result_zero   = a_zero || b_zero;
    wire            result_inf    = !result_zero && (a_inf || b_inf);
    wire [WMAN-1:0] a_significand = {1'b1, a_frac};
    wire [WMAN-1:0] b_significand = {1'b1, b_frac};

    wire signed [WEXP_UNBIASED-1:0] a_exp_ext       = {{(WEXP_UNBIASED-WEXP){1'b0}}, a_exp};
    wire signed [WEXP_UNBIASED-1:0] b_exp_ext       = {{(WEXP_UNBIASED-WEXP){1'b0}}, b_exp};
    wire signed [WEXP_UNBIASED-1:0] bias_ext        = {{(WEXP_UNBIASED-WEXP){1'b0}}, EXP_BIAS};
    wire signed [WEXP_UNBIASED-1:0] exp_unbiased_in = a_exp_ext + b_exp_ext - (bias_ext <<< 1);

    wire pre_sign       = a_sign ^ b_sign;
    wire pre_force_zero = result_zero;
    wire pre_force_inf  = result_inf;

    // Shared multiplier: the significand product rides through _zkf_pmul (both operands unsigned), while the sign,
    // the unbiased exponent base, and the force-zero/force-inf controls travel in its sideband so they land registered
    // in lockstep with the product. The full product is kept (no sticky-tail trim): trimming moves the tail
    // OR-reduction onto the multiplier output and measurably hurts fmax by weakening retiming.
    localparam WSB_MUL = WEXP_UNBIASED + 3;
    wire [WSB_MUL-1:0] mul_sb_in = {pre_sign, exp_unbiased_in, pre_force_zero, pre_force_inf};
    wire [WSB_MUL-1:0] mul_sb_out;

    wire                            s1_valid;
    wire                 [WMAG-1:0] s1_mag;
    wire                            s1_sign              = mul_sb_out[WSB_MUL-1];
    wire signed [WEXP_UNBIASED-1:0] s1_exp_unbiased_base = $signed(mul_sb_out[WSB_MUL-2 -: WEXP_UNBIASED]);
    wire                            s1_force_zero        = mul_sb_out[1];
    wire                            s1_force_inf         = mul_sb_out[0];

    _zkf_pmul #(
        .WA(WMAN), .WB(WMAN), .A_SIGNED(0), .B_SIGNED(0),
        .WSB(WSB_MUL), .WMULTIPLIER(WMULTIPLIER), .STAGE_PRODUCT(STAGE_PRODUCT)
    ) u_pmul (
        .clk(clk), .rst(rst), .in_valid(in_valid_q), .sb_in(mul_sb_in),
        .a(a_significand), .b(b_significand),
        .out_valid(s1_valid), .sb_out(mul_sb_out), .p(s1_mag)
    );

    // A nonzero hidden-bit product has its leading one in one of the two most-significant product bits.
    // Keep the two overlapping sticky reductions separate: sharing s1_sticky_lo saved no resources and hurt fmax.
    wire                            s1_product_high   = s1_mag[WMAG-1];
    wire signed [WEXP_UNBIASED-1:0] s1_exp_adjust     = s1_product_high ? ONE_EXT : ZERO_EXT;
    wire signed [WEXP_UNBIASED-1:0] s1_exp_unbiased   = s1_exp_unbiased_base + s1_exp_adjust;
    wire                 [WMAN-1:0] s1_significand_hi = s1_mag[WMAG-1 -: WMAN];
    wire                 [WMAN-1:0] s1_significand_lo = s1_mag[WMAG-2 -: WMAN];
    wire                            s1_guard_hi       = s1_mag[WMAN-1];
    wire                            s1_round_hi       = s1_mag[WMAN-2];
    wire                            s1_guard_lo       = s1_mag[WMAN-2];
    wire                            s1_round_lo       = s1_mag[WMAN-3];
    wire                            s1_sticky_hi      = |s1_mag[WMAN-3:0];
    wire                            s1_sticky_lo      = |s1_mag[WMAN-4:0];

    _zkf_pack #(
        .WEXP(WEXP), .WMAN(WMAN),
        .STAGE_INPUT(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_pack (
        .clk(clk),
        .rst(rst),
        .in_valid(s1_valid),
        .sign(s1_sign),
        .force_zero(s1_force_zero),
        .force_inf(s1_force_inf),
        .exp_unbiased(s1_exp_unbiased),
        .significand(s1_product_high ? s1_significand_hi : s1_significand_lo),
        .guard(s1_product_high ? s1_guard_hi : s1_guard_lo),
        .round(s1_product_high ? s1_round_hi : s1_round_lo),
        .sticky(s1_product_high ? s1_sticky_hi : s1_sticky_lo),
        .out_valid(out_valid),
        .y(y)
    );
endmodule

`default_nettype wire

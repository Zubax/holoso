/// Pack a normalized unsigned significand into float with infinity and rounding to nearest.
/// The exact finite input value before rounding is: (-1)^sign * 1.significand_fraction * 2^exp_unbiased
///
/// Number of stages = STAGE_INPUT+STAGE_OUTPUT. Pure comb is possible, in which case clk/rst are ignored.
///
/// The significand input includes the hidden bit. The guard/round/sticky inputs carry the discarded tail bits.
/// force_zero and force_inf override the finite value; force_zero wins if both are asserted.
///
/// The output is canonical zero for zero or finite magnitudes below 0.5*MIN_NORMAL, signed MIN_NORMAL for finite
/// magnitudes at or above that boundary but below MIN_NORMAL, round-to-nearest ties-to-even for normal values, and
/// canonical signed infinity for exponent overflow. Subnormals are not generated.
///
/// STAGE_INPUT=0: the packer's combinational cone starts immediately at the input ports (default).
/// STAGE_INPUT=1: insert one register stage in front of the rounding/saturation cone. Useful when the caller wants
///     to isolate the packer's rounder from a wide upstream cone (e.g. a close-cancellation normalize). The
///     accompanying `_zkf_pack_delay` accepts the same parameter so any sideband payload can ride the same delay.
///
/// STAGE_OUTPUT=0: the output is combinational, zero cycle latency (default).
/// STAGE_OUTPUT=1: one register stage at the output.
///
/// EXP_IS_BIASED=0: The exp_unbiased port is unbiased and the bias is added here.
/// EXP_IS_BIASED=1: It already carries the signed biased exponent, so the bias add is skipped - for a caller that
///     folded the bias into its own exponent arithmetic to shorten its critical path (e.g. zkf_add:
///     large_exp - normalize_shift is already the biased exponent, avoiding a -BIAS/+BIAS round trip).
///
/// ASSUME_NO_OVERFLOW=0 (default): the biased exponent is range-checked and a value above the finite range maps to
///     canonical signed infinity.
/// ASSUME_NO_OVERFLOW=1: the caller guarantees the biased exponent stays within the finite range [0, EXP_MAX_FINITE]
///     for every valid input, so the overflow detector is pruned at elaboration. The force_inf path (still mapped to
///     infinity) and the zero / MIN_NORMAL underflow paths are unaffected, and a round-carry from EXP_MAX_FINITE to
///     EXP_INF still produces canonical infinity (it rides the rounding adder, not the detector). Used by
///     bounded-output transcendentals such as zkf_log2, whose result is always representable for finite x.

`default_nettype none

module _zkf_pack #(
    parameter WEXP          = 6,
    parameter WMAN          = 18,         // significand precision including the hidden bit
    parameter WEXP_UNBIASED = WEXP + 2,   // signed unbiased exponent width
    parameter EXP_IS_BIASED = 0,
    parameter ASSUME_NO_OVERFLOW = 0,     // 0 = normal behavior; 1 = caller guarantees no overflow, checks removed
    parameter STAGE_INPUT   = 0,          // 0 = combinational inputs (default); 1 = one register stage at the input
    parameter STAGE_OUTPUT  = 0           // 0 = combinational output (default); 1 = registered output (one stage)
)(
    input  wire clk,
    input  wire rst,

    input  wire                            in_valid,
    input  wire                            sign,
    input  wire                            force_zero,
    input  wire                            force_inf,
    input  wire signed [WEXP_UNBIASED-1:0] exp_unbiased,
    input  wire                 [WMAN-1:0] significand,
    input  wire                            guard,
    input  wire                            round,
    input  wire                            sticky,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wman
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if ((ASSUME_NO_OVERFLOW < 0) || (ASSUME_NO_OVERFLOW > 1)) begin : g_invalid_assume_no_overflow
            _zkf_invalid_assume_no_overflow_out_of_range u_invalid();
        end
        if ((STAGE_INPUT != 0) && (STAGE_INPUT != 1)) begin : g_invalid_stage_input
            _zkf_invalid_stage_input u_invalid();
        end
        if ((STAGE_OUTPUT != 0) && (STAGE_OUTPUT != 1)) begin : g_invalid_stage_output
            _zkf_invalid_stage_output u_invalid();
        end
    endgenerate

    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;
    localparam WEXP_BIASED_EXT = WEXP_UNBIASED + 1;

    localparam [WEXP-1:0] EXP_BIAS       = {1'b0, {WEXP-1{1'b1}}};
    localparam [WEXP-1:0] EXP_INF        = {WEXP{1'b1}};

    // Optional input register stage. When STAGE_INPUT=1, the input ports are captured here and the rest of the
    // packer's combinational cone runs from the registered copies; this isolates a wide upstream cone (e.g. a
    // close-cancellation normalize) from the rounder/saturation cone below. Reset clears only stream validity;
    // payload registers free-run per project policy (control-only reset).
    wire                            i_valid;
    wire                            i_sign;
    wire                            i_force_zero;
    wire                            i_force_inf;
    wire signed [WEXP_UNBIASED-1:0] i_exp_unbiased;
    wire                 [WMAN-1:0] i_significand;
    wire                            i_guard;
    wire                            i_round;
    wire                            i_sticky;
    generate
        if (STAGE_INPUT != 0) begin : g_in_reg
            reg                            valid_r;
            reg                            sign_r;
            reg                            force_zero_r;
            reg                            force_inf_r;
            reg signed [WEXP_UNBIASED-1:0] exp_r;
            reg                 [WMAN-1:0] sig_r;
            reg                            guard_r;
            reg                            round_r;
            reg                            sticky_r;
            always @(posedge clk) begin
                if (rst) valid_r <= 1'b0;
                else     valid_r <= in_valid;
                sign_r       <= sign;
                force_zero_r <= force_zero;
                force_inf_r  <= force_inf;
                exp_r        <= exp_unbiased;
                sig_r        <= significand;
                guard_r      <= guard;
                round_r      <= round;
                sticky_r     <= sticky;
            end
            assign i_valid        = valid_r;
            assign i_sign         = sign_r;
            assign i_force_zero   = force_zero_r;
            assign i_force_inf    = force_inf_r;
            assign i_exp_unbiased = exp_r;
            assign i_significand  = sig_r;
            assign i_guard        = guard_r;
            assign i_round        = round_r;
            assign i_sticky       = sticky_r;
        end else begin : g_in_comb
            assign i_valid        = in_valid;
            assign i_sign         = sign;
            assign i_force_zero   = force_zero;
            assign i_force_inf    = force_inf;
            assign i_exp_unbiased = exp_unbiased;
            assign i_significand  = significand;
            assign i_guard        = guard;
            assign i_round        = round;
            assign i_sticky       = sticky;
        end
    endgenerate

    // Input combinational exponent classification. Values exactly one exponent below the normal range are at or above
    // the zero/MIN_NORMAL midpoint, so they round directly to MIN_NORMAL. Lower exponents round to canonical zero.
    // bias_ext is a compile-time-constant bias widened with constant padding.
    wire signed [WEXP_BIASED_EXT-1:0] bias_ext         = {{(WEXP_BIASED_EXT-WEXP){1'b0}}, EXP_BIAS};
    wire signed [WEXP_BIASED_EXT-1:0] exp_unbiased_ext = {i_exp_unbiased[WEXP_UNBIASED-1], i_exp_unbiased};
    // EXP_IS_BIASED callers pass the signed biased exponent directly (already sign-extended by the wider field), so the
    // bias add is skipped; the parameter is constant so this is a compile-time select, not a runtime mux.
    wire signed [WEXP_BIASED_EXT-1:0] exp_biased_ext = EXP_IS_BIASED ? exp_unbiased_ext : (exp_unbiased_ext + bias_ext);
    wire                   [WEXP-1:0] exp_biased         = exp_biased_ext[WEXP-1:0];
    wire                              exp_underflow_zero = exp_biased_ext[WEXP_BIASED_EXT-1];
    wire                              exp_one_below_min  = ~|exp_biased_ext;
    wire                              exp_biased_high_nonzero;
    generate
        if (WEXP_UNBIASED > WEXP) begin : g_biased_overflow_wide
            assign exp_biased_high_nonzero = |exp_biased_ext[WEXP_UNBIASED-1:WEXP];
        end else begin : g_biased_overflow_min_width
            assign exp_biased_high_nonzero = 1'b0;
        end
    endgenerate
    // ASSUME_NO_OVERFLOW=1 forces this to a constant 0 and synthesis prunes unused nets.
    wire exp_overflow = (ASSUME_NO_OVERFLOW != 0) ? 1'b0
                                                  : (!exp_underflow_zero && (exp_biased_high_nonzero || (&exp_biased)));

    // Single combinational cone feeding one output register (this packer is one register stage). Rounding,
    // round-to-nearest ties-to-even, folds the increment into a single {exp_biased, significand} adder - the full
    // significand including the hidden bit - so a true significand carry-out (significand was all-ones) ripples
    // straight into the exponent without a separate incrementer, while a carry that only fills the hidden bit of a
    // denormalized input stays out of the exponent, matching the reference. A round-carry at exp_biased ==
    // EXP_MAX_FINITE lands the exponent on EXP_INF with fraction 0 - canonical infinity - on the normal path.
    localparam WEXPSIG = WEXP + WMAN;
    wire               round_increment = i_guard && (i_round || i_sticky || i_significand[0]);
    wire [WEXPSIG-1:0] expsig          = {exp_biased, i_significand};
    wire [WEXPSIG-1:0] expsig_rounded  = expsig + {{(WEXPSIG-1){1'b0}}, round_increment};
    wire    [WEXP-1:0] exp_rounded     = expsig_rounded[WEXPSIG-1 -: WEXP];
    wire   [WFRAC-1:0] frac_rounded    = expsig_rounded[WFRAC-1:0];
    wire               infinity        = i_force_inf || exp_overflow;

    // Result classification. force_zero wins over force_inf; a tiny finite magnitude exactly one exponent below the
    // normal range rounds to signed MIN_NORMAL, anything lower to canonical +0.
    wire result_zero       = i_force_zero || (!i_force_inf && exp_underflow_zero);
    wire result_infinity   = !result_zero && infinity;
    wire result_min_normal = !result_zero && !i_force_inf && exp_one_below_min;
    wire result_normal     = !result_zero && !result_infinity && !result_min_normal;

    // Canonicalize by masking instead of a full-width 4:1 output mux: the stored fraction is nonzero only for normal
    // results, so it collapses to an AND-mask; the exponent selects one of three small constants or the rounded
    // exponent; the sign is forced to 0 only for canonical +0. This keeps the wide fraction field off the mux tree.
    wire             out_sign = i_sign & ~result_zero;
    wire [WEXP-1:0]  out_exp  = result_zero       ? {WEXP{1'b0}} :
                                result_infinity   ? EXP_INF :
                                result_min_normal ? {{(WEXP-1){1'b0}}, 1'b1} :
                                                    exp_rounded;
    wire [WFRAC-1:0] out_frac = frac_rounded & {WFRAC{result_normal}};

    // Output stage.
    generate
        if (STAGE_OUTPUT != 0) begin : g_out_reg
            reg             out_valid_r;
            reg [WFULL-1:0] y_r;
            always @(posedge clk) begin
                if (rst) out_valid_r <= 1'b0;
                else     out_valid_r <= i_valid;
                y_r <= {out_sign, out_exp, out_frac};
            end
            assign out_valid = out_valid_r;
            assign y         = y_r;
        end else begin : g_out_comb
            assign out_valid = i_valid;
            assign y         = {out_sign, out_exp, out_frac};
        end
    endgenerate
endmodule

/// Delay a sideband payload through the same input + output stages as _zkf_pack: pass STAGE_INPUT / STAGE_OUTPUT
/// to match it. When changing the packer pipeline, update this one as well.
/// Total delay in cycles = STAGE_INPUT + STAGE_OUTPUT (combinational pass-through when both are 0).
module _zkf_pack_delay #(parameter W = 1, parameter STAGE_INPUT = 0, parameter STAGE_OUTPUT = 0)(
    input wire clk, input wire [W-1:0] x, output wire [W-1:0] y);
    zkf_pipe #(.W(W), .N(STAGE_INPUT + STAGE_OUTPUT)) u_pipe (
        .clk(clk), .rst(1'b0),
        .in_valid(1'b0), .in(x),
        .out_valid(), .out(y)
    );
endmodule

`default_nettype wire

/// Streamed round of a Zubax Kulibin float to an integer value in the same WEXP/WMAN format.
/// The rounding mode is selected per transaction by the 2-bit round_mode input:
///   0 = round to nearest integer, ties to even (the IEEE default)
///   1 = floor (toward -inf)
///   2 = ceil  (toward +inf)
///   3 = trunc (toward zero)
/// Tie round_mode to a constant to let synthesis constant-propagate and prune the unused mode logic.
///
/// Behaviour: rounding to an integer preserves the exponent except for a possible +1 carry, so the datapath clears the
/// fractional mantissa bits below the integer boundary (at bit p = (WFRAC+BIAS) - exp of the significand), adds a
/// mode-selected increment at that boundary, and absorbs the single-bit carry into the exponent. The already-rounded
/// significand is handed to _zkf_pack with guard/round/sticky = 0, so _zkf_pack performs only the bias add,
/// exponent-overflow -> canonical signed inf (reachable only in tiny formats whose top finite value still has a
/// fractional part), and zero/inf canonicalization. Results therefore stay bit-consistent with the rest of the library:
///   - +-inf passes through as canonical signed inf;
///   - zero and flushed subnormals (exp == 0) round to canonical +0;
///   - a zero-magnitude result canonicalizes to +0 regardless of sign (e.g. ceil(-0.3) -> +0);
///   - a rounded integer that does not fit the format overflows to signed inf.
///
/// STAGE_INPUT=1: Latch {a, round_mode} at the input (+1 cycle), shielding the rounder from upstream.
///
/// STAGE_DECODE=1: Register the decode + boundary-mask cone, splitting the rounder's variable-position mask generation
///                 from the guard/sticky reduction and increment adder (+1 cycle).
///
/// STAGE_PACK=1: Forwarded to _zkf_pack.STAGE_INPUT (+1 cycle).
///
/// STAGE_OUTPUT=1: Forwarded to _zkf_pack.STAGE_OUTPUT, registering the output (+1 cycle).

`default_nettype none

// With all stages disabled the module becomes combinational and the clk/rst are ignored.
module zkf_round #(
    parameter WEXP         = 6,
    parameter WMAN         = 18,
    parameter STAGE_INPUT  = 0,
    parameter STAGE_DECODE = 0,
    parameter STAGE_PACK   = 0,
    parameter STAGE_OUTPUT = 0,
    parameter LATENCY      = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire           [1:0] round_mode,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam LATENCY_REF = STAGE_INPUT + STAGE_DECODE + STAGE_PACK + STAGE_OUTPUT;
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        // Shift by WEXP >= 32 would overflow Verilog's integer constant arithmetic and yield tool-dependent values.
        if (WEXP >= 32) begin : g_invalid_wexp_too_wide
            _zkf_invalid_round_wexp_too_wide_unportable u_invalid();
        end
        if ((STAGE_INPUT != 0) && (STAGE_INPUT != 1)) begin : g_invalid_stage_input
            _zkf_invalid_stage_input u_invalid();
        end
        if ((STAGE_DECODE != 0) && (STAGE_DECODE != 1)) begin : g_invalid_stage_decode
            _zkf_invalid_stage_decode u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    localparam ROUND_NEAREST_EVEN = 2'd0;
    localparam ROUND_FLOOR        = 2'd1;
    localparam ROUND_CEIL         = 2'd2;
    localparam ROUND_TRUNC        = 2'd3;

    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;
    localparam WEU   = WEXP + 2;                  // signed unbiased exponent width handed to _zkf_pack

    // Compile-time constants. BIAS is the format bias; KK = WFRAC + BIAS is the input exponent at which the
    // value is exactly 2^WFRAC, i.e. the integer/fraction boundary sits at the significand LSB (one fractional
    // bit) -- exp >= KK means the value is already an integer, exp == BIAS-1 means |value| in [0.5, 1).
    localparam integer BIAS_INT = (1 << (WEXP - 1)) - 1;
    localparam integer KK_INT   = WFRAC + BIAS_INT;
    localparam         PW       = $clog2(WFRAC + 1);                // holds the boundary position pp in [0, WFRAC]
    localparam         KCW      = $clog2(KK_INT + 1);
    localparam         DW       = ((KCW > WEXP) ? KCW : WEXP) + 2;  // signed width for p = KK - exp (sign + headroom)

    // Bias as sized constants (an integer localparam cannot be part-selected portably; size it explicitly).
    localparam       [WEXP-1:0] BIAS_VEC    = BIAS_INT;
    localparam       [WEXP-1:0] BIAS_M1_VEC = BIAS_INT - 1;        // exp == BIAS-1 marks |value| in [0.5, 1)
    localparam signed [WEU-1:0] BIAS_EXT    = BIAS_INT;

    // -- Optional input register stage carrying {round_mode, a}. round_mode rides the rounder cone below.
    localparam IN_W = WFULL + 2;
    wire            in_valid_q;
    wire [IN_W-1:0] in_q;
    zkf_pipe #(.W(IN_W), .N(STAGE_INPUT ? 1 : 0)) u_input_pipe (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in({round_mode, a}),
        .out_valid(in_valid_q), .out(in_q)
    );
    wire [WFULL-1:0] a_q          = in_q[WFULL-1:0];
    wire       [1:0] round_mode_q = in_q[WFULL+1:WFULL];

    // ============================================================================================================
    // Stage 1 (combinational): decode + variable boundary position + mask generation. This is the cone the
    // STAGE_DECODE register splits off from the reduction/increment cone.
    // ============================================================================================================
    wire             sign_in    = a_q[WFULL-1];
    wire  [WEXP-1:0] exp_in     = a_q[WFULL-2:WFRAC];
    wire [WFRAC-1:0] frac_in    = a_q[WFRAC-1:0];
    wire             is_zero_s1 = ~|exp_in;
    wire             is_inf_s1  =  &exp_in;
    wire  [WMAN-1:0] sig_s1     = {1'b1, frac_in};                 // significand with the hidden bit in place

    // Boundary position. p = KK - exp; for |value| >= 1 (exp >= BIAS) this lies in [<=0, WFRAC], where <=0 means
    // there are no fractional bits (already an integer). The sub-one branch is handled separately, so pp is
    // clamped to [0, WFRAC] and only consumed there.
    localparam signed [DW-1:0] KK_S = KK_INT;
    wire signed [DW-1:0] exp_s = $signed({{(DW-WEXP){1'b0}}, exp_in});
    wire signed [DW-1:0] p_s   = KK_S - exp_s;
    wire                 already_int = (p_s <= 0);                 // exp >= KK: no fractional bits to round
    wire                 sub_one_s1  = exp_in < BIAS_VEC;          // |value| < 1: result is 0 or +-1
    wire        [PW-1:0] pp          = already_int ? {PW{1'b0}} : p_s[PW-1:0];

    wire [WMAN-1:0] bit_pp_s1    = {{(WMAN-1){1'b0}}, 1'b1} << pp; // one-hot at the boundary (1 << pp)
    wire [WMAN-1:0] frac_mask_s1 = bit_pp_s1 - {{(WMAN-1){1'b0}}, 1'b1};  // low pp bits (the fractional part)

    // Sub-one helpers folded into stage 1. The result's biased exponent is just exp_in (+carry, added in stage 2):
    // it is forwarded to _zkf_pack with EXP_IS_BIASED=1, so neither this module nor the packer round-trips through
    // the bias (no exp_in-BIAS subtractor here, no +BIAS adder in the packer).
    wire                  e_is_neg1_s1    = exp_in == BIAS_M1_VEC; // |value| in [0.5, 1): the 0.5 bit is the hidden bit
    wire                  frac_nonzero_s1 = |frac_in;

    // -- Optional STAGE_DECODE register. Reset clears only validity; payload free-runs (control-only reset).
    // Only sig, frac_mask and the one-hot bit_pp are registered; half_mask (= bit_pp >> 1) and below_mask
    // (= frac_mask ^ half_mask) are cheap re-derivations done in stage 2, so they need not occupy flops.
    // Packed (LSB-first): is_inf, is_zero, frac_nonzero, e_is_neg1, sub_one, round_mode[2], sign, exp_in[WEXP],
    // bit_pp, frac_mask, sig.
    // O_EXP = 8: the five 1-bit specials (is_inf,is_zero,frac_nonzero,e_is_neg1,sub_one) + round_mode (2) + sign (1).
    localparam O_EXP    = 8;
    localparam O_BITPP  = O_EXP + WEXP;
    localparam O_FRAC   = O_BITPP + WMAN;
    localparam O_SIG    = O_FRAC + WMAN;
    localparam DEC_W    = O_SIG + WMAN;
    wire             dec_valid;
    wire [DEC_W-1:0] dec_payload;
    wire [DEC_W-1:0] s1_payload = {sig_s1, frac_mask_s1, bit_pp_s1,
                                   exp_in, sign_in, round_mode_q, sub_one_s1, e_is_neg1_s1,
                                   frac_nonzero_s1, is_zero_s1, is_inf_s1};
    zkf_pipe #(.W(DEC_W), .N(STAGE_DECODE ? 1 : 0)) u_decode_pipe (
        .clk(clk), .rst(rst),
        .in_valid(in_valid_q), .in(s1_payload),
        .out_valid(dec_valid), .out(dec_payload)
    );
    // sig is sig_s1 carried through the decode pipe.
    wire [WMAN-1:0]       sig          = dec_payload[O_SIG   +: WMAN];
    wire [WMAN-1:0]       frac_mask    = dec_payload[O_FRAC  +: WMAN];
    wire [WMAN-1:0]       bit_pp       = dec_payload[O_BITPP +: WMAN];
    wire      [WEXP-1:0]  exp_biased_d = dec_payload[O_EXP   +: WEXP];   // the input biased exponent (carried through)
    wire                  sign_d       = dec_payload[7];
    wire            [1:0] round_mode_d = dec_payload[6:5];
    wire                  sub_one      = dec_payload[4];
    wire                  e_is_neg1    = dec_payload[3];
    wire                  frac_nonzero = dec_payload[2];
    wire                  is_zero      = dec_payload[1];
    wire                  is_inf       = dec_payload[0];
    // half_mask = bit_pp >> 1 and bit_pp = 1 << pp with pp <= WFRAC, so half_mask's highest possible set bit is
    // WFRAC-1; the MSB (bit WMAN-1 = WFRAC) is therefore never set.
    wire [WMAN-1:0]       half_mask    = bit_pp >> 1;              // bit pp-1, the 0.5 bit (re-derived, not registered)
    wire [WMAN-1:0]       below_mask   = frac_mask ^ half_mask;    // bits below the 0.5 bit (re-derived)

    // ============================================================================================================
    // Stage 2 (combinational): guard/sticky reduction, mode-selected increment, and the boundary add. The add is
    // computed speculatively (sig_cleared + 2^pp) so it runs in parallel with the reduction instead of waiting on
    // the increment decision -- this keeps the stage-2 cone short.
    // ============================================================================================================
    wire            guard_a     = |(sig & half_mask);
    wire            sticky_a    = |(sig & below_mask);
    wire            lsb_a       = |(sig & bit_pp);                 // integer LSB (parity for ties-to-even)
    wire [WMAN-1:0] sig_cleared = sig & ~frac_mask;
    wire [WMAN:0]   sig_plus    = {1'b0, sig_cleared} + {1'b0, bit_pp};   // sig_cleared + 2^pp (speculative)
    wire            carry_plus  = sig_plus[WMAN];

    wire inc_a = (round_mode_d == ROUND_NEAREST_EVEN) ? (guard_a & (sticky_a | lsb_a))   :
                 (round_mode_d == ROUND_FLOOR)        ? ( sign_d & (guard_a | sticky_a)) :
                 (round_mode_d == ROUND_CEIL)         ? (~sign_d & (guard_a | sticky_a)) : 1'b0;  // ROUND_TRUNC

    // Sub-one branch: the 0.5 bit is the hidden bit only when e == -1; otherwise the whole magnitude is sticky.
    wire guard_c  = e_is_neg1;
    wire sticky_c = e_is_neg1 ? frac_nonzero : 1'b1;
    wire inc_c = (round_mode_d == ROUND_NEAREST_EVEN) ? (guard_c & sticky_c)             :
                 (round_mode_d == ROUND_FLOOR)        ? ( sign_d & (guard_c | sticky_c)) :
                 (round_mode_d == ROUND_CEIL)         ? (~sign_d & (guard_c | sticky_c)) : 1'b0;  // ROUND_TRUNC

    wire [WMAN-1:0] sig_inc    = carry_plus ? {1'b1, {WFRAC{1'b0}}} : sig_plus[WMAN-1:0];
    wire [WMAN-1:0] sig_norm_a = inc_a ? sig_inc : sig_cleared;
    wire            carry      = inc_a & carry_plus;

    // -- Assemble the packer inputs. _zkf_pack receives an already-rounded normalized significand with no
    // guard/round/sticky and a pre-biased exponent (EXP_IS_BIASED=1), so it only canonicalizes specials and
    // detects overflow. The biased exponent is exp_in + carry; the sub-one branch is exactly +-1.0 (exp == BIAS).
    // The overflow case still rides this exponent: max_finite (exp == 2^WEXP-2) + carry == 2^WEXP-1 == EXP_INF.
    wire signed [WEU-1:0] exp_biased_base = $signed({{(WEU-WEXP){1'b0}}, exp_biased_d});
    wire signed [WEU-1:0] exp_biased_inc  = exp_biased_base + $signed({{(WEU-1){1'b0}}, 1'b1});
    wire signed [WEU-1:0] exp_biased_a    = carry ? exp_biased_inc : exp_biased_base;
    wire                  force_inf    = is_inf;
    wire                  force_zero   = is_zero | (sub_one & ~inc_c);
    wire [WMAN-1:0]       significand  = sub_one ? {1'b1, {WFRAC{1'b0}}} : sig_norm_a;
    wire signed [WEU-1:0] exp_biased   = sub_one ? BIAS_EXT : exp_biased_a;

    _zkf_pack #(
        .WEXP(WEXP), .WMAN(WMAN),
        .WEXP_UNBIASED(WEU),
        .EXP_IS_BIASED(1),
        .ASSUME_NO_OVERFLOW(0),
        .STAGE_INPUT(STAGE_PACK),
        .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_pack (
        .clk(clk), .rst(rst),
        .in_valid(dec_valid),
        .sign(sign_d),
        .force_zero(force_zero),
        .force_inf(force_inf),
        .exp_unbiased(exp_biased),
        .significand(significand),
        .guard(1'b0),
        .round(1'b0),
        .sticky(1'b0),
        .out_valid(out_valid),
        .y(y)
    );
endmodule

`default_nettype wire

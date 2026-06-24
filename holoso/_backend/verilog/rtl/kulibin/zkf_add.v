/// Streamed Zubax Kulibin float adder.
///
/// STAGE_INPUT=0: operands feed the datapath combinationally (default).
/// STAGE_INPUT=1: latch the inputs before any combinational logic, isolating them from upstream paths (+1 cycle).
///
/// STAGE_OUTPUT=0: the result is combinational, zero cycle latency at the output (default).
/// STAGE_OUTPUT=1: one register stage at the output (+1 cycle).
///
/// STAGE_DECODE=0: per-operand decode feeds the s0 capture combinationally.
/// STAGE_DECODE=1: registers the full decoded-operand bundle between the raw decode and the s0 capture (+1 cycle).
///
/// STAGE_ALIGN=0: single-cycle alignment shifter (radix-4 cascade combinational).
/// STAGE_ALIGN=1: registers one stage inside the alignment shifter, splitting the radix-4 cascade  (+1 cycle).
///
/// STAGE_NORMALIZE={0,1,2}: number of internal register barriers in the close-cancellation _zkf_normshift cascade
/// (direct forward to _zkf_normshift.STAGE_SPLIT). Adds STAGE_NORMALIZE cycles.
///
/// STAGE_PACK=0: pack inputs are combinational (default).
/// STAGE_PACK=1: register pack inputs (forwarded to _zkf_pack.STAGE_INPUT) (+1 cycle).

`default_nettype none

`define ZKF_ADD_LATENCY (4 + STAGE_INPUT + STAGE_DECODE + STAGE_ALIGN + STAGE_NORMALIZE + STAGE_PACK + STAGE_OUTPUT)

module zkf_add #(
    parameter WEXP            = 6,    // exponent field width
    parameter WMAN            = 18,   // significand precision including the hidden bit
    parameter STAGE_INPUT     = 0,    // 0 = combinational inputs; 1 = latch inputs before any logic (+1 cycle)
    parameter STAGE_DECODE    = 0,    // 0 = decode feeds s0 combinationally; 1 = registered decoded bundle (+1 cycle)
    parameter STAGE_ALIGN     = 0,    // 0 = single-cycle alignment; 1 = split alignment shifter (+1 cycle)
    parameter STAGE_NORMALIZE = 0,    // {0,1,2} internal close-cancel normshift barriers
    parameter STAGE_PACK      = 0,    // 0 = comb pack inputs; 1 = register pack inputs (+1 cycle)
    parameter STAGE_OUTPUT    = 0,    // 0 = combinational output; 1 = registered output (+1 cycle)
    parameter LATENCY         = `ZKF_ADD_LATENCY   // must equal the register-stage count; checked below
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wman
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        // STAGE_INPUT and STAGE_DECODE are realized locally as a single optional register each, so only {0,1} is
        // meaningful. STAGE_ALIGN / STAGE_NORMALIZE / STAGE_PACK / STAGE_OUTPUT forward to their owners
        // (_zkf_rshift_sticky / _zkf_normshift / _zkf_pack), which validate their own ranges.
        if ((STAGE_INPUT != 0) && (STAGE_INPUT != 1)) begin : g_invalid_stage_input
            _zkf_invalid_stage_input u_invalid();
        end
        if ((STAGE_DECODE != 0) && (STAGE_DECODE != 1)) begin : g_invalid_stage_decode
            _zkf_invalid_stage_decode u_invalid();
        end
        if (LATENCY != `ZKF_ADD_LATENCY) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    // Note of caution: merely replacing a named net with its expression at place-of-use may drastically affect
    // the synthesis outcome even though the circuit topology remains unchanged. All synthesis tools are unreliable.
    localparam WFRAC         = WMAN - 1;
    localparam WFULL         = WEXP + WMAN;
    localparam WGRS          = 3;
    localparam WEXT          = WMAN + WGRS;
    localparam WRAW          = WEXT + 1;
    localparam WINDEX        = $clog2(WRAW);
    localparam WSHIFT        = (WEXP > (WINDEX + 1)) ? WEXP : (WINDEX + 1);
    localparam WEXP_SIGNED   = WEXP + 1;
    localparam WSHIFT_SIGNED = WINDEX + 2;
    localparam WEXP_UNBIASED = (WEXP_SIGNED > WSHIFT_SIGNED) ? WEXP_SIGNED : WSHIFT_SIGNED;

    localparam [WINDEX-1:0] NORM_TOP = WMAN + 2;

    // -- Optional input register stage: latch the operands before any combinational logic.
    wire             in_valid_q;
    wire [WFULL-1:0] a_q;
    wire [WFULL-1:0] b_q;
    zkf_pipe #(.W(2*WFULL), .N(STAGE_INPUT ? 1 : 0)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in({b, a}),
        .out_valid(in_valid_q), .out({b_q, a_q})
    );

    // Operand decode/classification. Exponent-zero operands are zero regardless of sign/fraction payload.
    wire            a_sign        = a_q[WFULL-1];
    wire            b_sign        = b_q[WFULL-1];
    wire            raw_same_sign = ~(a_sign ^ b_sign);
    wire [WEXP-1:0] a_exp         = a_q[WFULL-2:WFRAC];
    wire [WEXP-1:0] b_exp         = b_q[WFULL-2:WFRAC];

    wire             raw_a_inf         = &a_exp;
    wire             raw_b_inf         = &b_exp;
    wire             a_finite          = (|a_exp) && !raw_a_inf;
    wire             b_finite          = (|b_exp) && !raw_b_inf;
    wire [WFRAC-1:0] a_fraction        = a_q[WFRAC-1:0];
    wire [WFRAC-1:0] b_fraction        = b_q[WFRAC-1:0];
    // hidden-bit MSB is structurally 1.
    wire [WMAN-1:0]  a_significand     = {1'b1, a_fraction};
    wire [WMAN-1:0]  b_significand     = {1'b1, b_fraction};

    // Masked exponent/significand keys: zero and non-finite operands contribute magnitude 0 to the datapath, so a
    // zero operand correctly leaves the other unchanged and a non-finite one is overridden by force_inf/force_zero.
    // The larger-magnitude operand selects between these keys below.
    wire [WEXP-1:0] raw_a_key_exp = a_finite ? a_exp : {WEXP{1'b0}};
    wire [WEXP-1:0] raw_b_key_exp = b_finite ? b_exp : {WEXP{1'b0}};
    wire [WMAN-1:0] raw_a_key_sig = a_finite ? a_significand : {WMAN{1'b0}};
    wire [WMAN-1:0] raw_b_key_sig = b_finite ? b_significand : {WMAN{1'b0}};

    // Full sign-magnitude order from a single unsigned compare of the {exponent, fraction} field (the operand word
    // minus its sign bit). For finite operands this is exactly the magnitude order: the exponent dominates and the
    // fraction breaks ties, so the larger-magnitude operand also has the larger-or-equal exponent (exp_diff >= 0
    // below). A zero input (exponent 0) sorts below every finite; a non-finite input's order is don't-care because
    // force_inf/force_zero overrides the datapath. This single compare replaces the former separate exponent compare
    // and significand compare, and selecting the larger-magnitude operand directly removes the s1 equal-exponent swap.
    wire raw_a_mag_ge_b_mag;
    _zkf_add_ge #(.W(WFULL-1)) u_mag_ge (.a(a_q[WFULL-2:0]), .b(b_q[WFULL-2:0]), .ge(raw_a_mag_ge_b_mag));

    // Decoded-operand bundle. When STAGE_DECODE=0 the d_* signals are combinational aliases of the raw
    // decoded wires above; when STAGE_DECODE!=0 they are registered, so the s0 capture below sees the
    // decode results one cycle later but its own combinational fanin (compare-mux + 8-bit subtract) starts
    // afresh from registered inputs. This removes the input-flop -> 8-bit compare -> mux -> subtract ->
    // s0 register chain that otherwise dominates timing at wide WMAN.
    wire            d_valid;
    wire            d_a_sign;
    wire            d_b_sign;
    wire            d_same_sign;
    wire            d_a_inf;
    wire            d_b_inf;
    wire [WEXP-1:0] d_a_key_exp;
    wire [WEXP-1:0] d_b_key_exp;
    wire [WMAN-1:0] d_a_key_sig;
    wire [WMAN-1:0] d_b_key_sig;
    wire            d_a_mag_ge_b_mag;

    generate
        if (STAGE_DECODE == 0) begin : g_no_decode_register
            assign d_valid           = in_valid_q;
            assign d_a_sign          = a_sign;
            assign d_b_sign          = b_sign;
            assign d_same_sign       = raw_same_sign;
            assign d_a_inf           = raw_a_inf;
            assign d_b_inf           = raw_b_inf;
            assign d_a_key_exp       = raw_a_key_exp;
            assign d_b_key_exp       = raw_b_key_exp;
            assign d_a_key_sig       = raw_a_key_sig;
            assign d_b_key_sig       = raw_b_key_sig;
            assign d_a_mag_ge_b_mag  = raw_a_mag_ge_b_mag;
        end else begin : g_decode_register
            reg            r_valid;
            reg            r_a_sign;
            reg            r_b_sign;
            reg            r_same_sign;
            reg            r_a_inf;
            reg            r_b_inf;
            reg [WEXP-1:0] r_a_key_exp;
            reg [WEXP-1:0] r_b_key_exp;
            reg [WMAN-1:0] r_a_key_sig;
            reg [WMAN-1:0] r_b_key_sig;
            reg            r_a_mag_ge_b_mag;
            always @(posedge clk) begin
                if (rst) r_valid <= 1'b0;
                else     r_valid <= in_valid_q;
                r_a_sign         <= a_sign;
                r_b_sign         <= b_sign;
                r_same_sign      <= raw_same_sign;
                r_a_inf          <= raw_a_inf;
                r_b_inf          <= raw_b_inf;
                r_a_key_exp      <= raw_a_key_exp;
                r_b_key_exp      <= raw_b_key_exp;
                r_a_key_sig      <= raw_a_key_sig;
                r_b_key_sig      <= raw_b_key_sig;
                r_a_mag_ge_b_mag <= raw_a_mag_ge_b_mag;
            end
            assign d_valid           = r_valid;
            assign d_a_sign          = r_a_sign;
            assign d_b_sign          = r_b_sign;
            assign d_same_sign       = r_same_sign;
            assign d_a_inf           = r_a_inf;
            assign d_b_inf           = r_b_inf;
            assign d_a_key_exp       = r_a_key_exp;
            assign d_b_key_exp       = r_b_key_exp;
            assign d_a_key_sig       = r_a_key_sig;
            assign d_b_key_sig       = r_b_key_sig;
            assign d_a_mag_ge_b_mag  = r_a_mag_ge_b_mag;
        end
    endgenerate

    // Combinational selections sourced from the decoded bundle. These feed the s0 capture below. The larger-magnitude
    // operand is the minuend ("large"); the smaller is aligned and subtracted ("small"). The finite result sign is the
    // larger-magnitude operand's sign (exact cancellation is forced to canonical +0 by the packer).
    wire finite_sign = d_a_mag_ge_b_mag ? d_a_sign : d_b_sign;
    wire inf_sign    = (d_a_inf & d_a_sign) | (d_b_inf & d_b_sign);

    wire [WEXP-1:0] large_exp     = d_a_mag_ge_b_mag ? d_a_key_exp : d_b_key_exp;
    wire [WEXP-1:0] small_exp     = d_a_mag_ge_b_mag ? d_b_key_exp : d_a_key_exp;
    wire [WMAN-1:0] large_sig_exp = d_a_mag_ge_b_mag ? d_a_key_sig : d_b_key_sig;
    wire [WMAN-1:0] small_sig_exp = d_a_mag_ge_b_mag ? d_b_key_sig : d_a_key_sig;

    // Stage 0: decoded/classified operands ordered by magnitude.
    reg                            s0_valid;
    reg                            s0_finite_sign;
    reg                            s0_inf_sign;
    reg                            s0_same_sign;
    reg                            s0_force_zero;
    reg                            s0_force_inf;
    // The carry/add-path biased exponent only ever holds the non-negative larger-operand exponent (and the +1 add
    // carry), bounded by EXP_INF, so it is stored at the native WEXP width with no sign/zero-extension headroom. Only
    // the sub-path correction can go negative, so it (and the packer feed) widen to the signed WEXP_UNBIASED below.
    reg                 [WEXP-1:0] s0_exp_biased;
    reg                 [WEXP-1:0] s0_exp_diff;
    reg                 [WMAN-1:0] s0_large_sig_exp;
    reg                 [WMAN-1:0] s0_small_sig_exp;

    // Alignment shifter. When STAGE_ALIGN != 0 the shifter inserts one pipeline register inside its radix-4 cascade,
    // so s0_small_aligned arrives one cycle later than the other s0_* signals; the s0b_* intermediate stage below
    // delays the sideband signals to match.
    wire [WEXT-1:0] s0_small_aligned;
    _zkf_rshift_sticky #(.W(WEXT), .WSHIFT(WSHIFT), .STAGE_SPLIT(STAGE_ALIGN)) u_align_small (
        .clk(clk),
        .x({s0_small_sig_exp, {WGRS{1'b0}}}),
        .shamt({{(WSHIFT-WEXP){1'b0}}, s0_exp_diff}),
        .y(s0_small_aligned)
    );

    // Intermediate stage s0b: when STAGE_ALIGN=0 it is a combinational alias of s0_*; when STAGE_ALIGN!=0 it is a
    // real register stage that delays the sideband signals by one cycle so they remain aligned with the
    // late-arriving s0_small_aligned.
    wire                            s0b_valid;
    wire                            s0b_finite_sign;
    wire                            s0b_inf_sign;
    wire                            s0b_same_sign;
    wire                            s0b_force_zero;
    wire                            s0b_force_inf;
    wire                 [WEXP-1:0] s0b_exp_biased;
    wire                 [WMAN-1:0] s0b_large_sig_exp;

    generate
        if (STAGE_ALIGN == 0) begin : g_no_align_register
            assign s0b_valid         = s0_valid;
            assign s0b_finite_sign   = s0_finite_sign;
            assign s0b_inf_sign      = s0_inf_sign;
            assign s0b_same_sign     = s0_same_sign;
            assign s0b_force_zero    = s0_force_zero;
            assign s0b_force_inf     = s0_force_inf;
            assign s0b_exp_biased    = s0_exp_biased;
            assign s0b_large_sig_exp = s0_large_sig_exp;
        end else begin : g_align_register
            reg                            r_valid;
            reg                            r_finite_sign;
            reg                            r_inf_sign;
            reg                            r_same_sign;
            reg                            r_force_zero;
            reg                            r_force_inf;
            reg                 [WEXP-1:0] r_exp_biased;
            reg                 [WMAN-1:0] r_large_sig_exp;
            always @(posedge clk) begin
                if (rst) r_valid <= 1'b0;
                else     r_valid <= s0_valid;
                r_finite_sign       <= s0_finite_sign;
                r_inf_sign          <= s0_inf_sign;
                r_same_sign         <= s0_same_sign;
                r_force_zero        <= s0_force_zero;
                r_force_inf         <= s0_force_inf;
                r_exp_biased        <= s0_exp_biased;
                r_large_sig_exp     <= s0_large_sig_exp;
            end
            assign s0b_valid         = r_valid;
            assign s0b_finite_sign   = r_finite_sign;
            assign s0b_inf_sign      = r_inf_sign;
            assign s0b_same_sign     = r_same_sign;
            assign s0b_force_zero    = r_force_zero;
            assign s0b_force_inf     = r_force_inf;
            assign s0b_exp_biased    = r_exp_biased;
            assign s0b_large_sig_exp = r_large_sig_exp;
        end
    endgenerate

    // Stage 1: registered aligned operands.
    reg                            s1_valid;
    reg                            s1_finite_sign;
    reg                            s1_inf_sign;
    reg                            s1_same_sign;
    reg                            s1_force_zero;
    reg                            s1_force_inf;
    reg                 [WEXP-1:0] s1_exp_biased;
    // the larger operand's extended significand carries the always-1 hidden bit
    // and fixed GRS pad in its upper bits.
    reg                 [WEXT-1:0] s1_large_ext_exp;
    reg                 [WEXT-1:0] s1_small_aligned;

    // The larger-magnitude operand is always the minuend, so no equal-exponent swap is needed: large is the adder's
    // a input, the aligned small operand is the b input. Effective subtraction complements b and adds a carry-in.
    // The adder operands' top bit is the constant carry pad and their upper bits carry the always-1 hidden
    // bit / fixed GRS positions.
    wire [WRAW-1:0] s1_adder_a     = {1'b0, s1_large_ext_exp};
    wire [WRAW-1:0] s1_adder_b_abs = {1'b0, s1_small_aligned};
    wire [WRAW-1:0] s1_adder_b     = s1_same_sign ? s1_adder_b_abs : ~s1_adder_b_abs;
    wire [WRAW-1:0] s1_raw_result  = s1_adder_a + s1_adder_b + {{(WRAW-1){1'b0}}, !s1_same_sign};
    wire            s1_result_sign = s1_force_inf ? s1_inf_sign : s1_finite_sign;

    // Stage 2: registered raw add/sub result.
    reg                            s2_valid;
    reg                            s2_sign;
    reg                            s2_same_sign;
    reg                            s2_force_zero;
    reg                            s2_force_inf;
    reg                 [WEXP-1:0] s2_exp_biased;
    reg                 [WRAW-1:0] s2_raw_result;

    // Same-sign addition never left-normalizes, so a jammed LSB remains sticky. For subtraction, close
    // cancellation only occurs with small exact alignment shifts; far cancellation cannot require a full-width
    // discarded tail, and the compact GRS representation supplies the packer with sufficient rounding state.
    wire                            s2_add_carry      = s2_raw_result[WRAW-1];
    wire             [WEXP-1:0]    s2_add_exp_biased = s2_exp_biased + {{(WEXP-1){1'b0}}, s2_add_carry};
    wire [WMAN-1:0] s2_add_significand = s2_add_carry ? s2_raw_result[WRAW-1 -: WMAN] : s2_raw_result[NORM_TOP -: WMAN];
    wire s2_add_guard  = s2_add_carry ?   s2_raw_result[WRAW-WMAN-1]    :   s2_raw_result[NORM_TOP-WMAN];
    wire s2_add_round  = s2_add_carry ?   s2_raw_result[WRAW-WMAN-2]    :   s2_raw_result[NORM_TOP-WMAN-1];
    wire s2_add_sticky = s2_add_carry ? (|s2_raw_result[WRAW-WMAN-3:0]) : (|s2_raw_result[NORM_TOP-WMAN-2:0]);

    // Add-path s2x catch-up: the s2 add-path bundle rides the sub-path normalizer's own sideband (u_sub_norm below,
    // STAGE_SPLIT=STAGE_NORMALIZE, STAGE_OUTPUT=0), so it is delayed by exactly the STAGE_NORMALIZE cycles the sub-path
    // spends inside the normshift and both paths reach the s3 register boundary aligned. STAGE_NORMALIZE=0 is a pure
    // passthrough. q_valid/q_out are driven by u_sub_norm's out_valid/sb_out below; the sideband free-runs (only valid
    // resets), matching the former zkf_pipe payload semantics.
    // Bundle: {sign, same_sign, force_zero, force_inf} + exp_biased + add_exp_biased + add_significand + GRS.
    localparam Q_W = 4 + 2*WEXP + WMAN + 3;
    wire [Q_W-1:0] s2_q_in  = {s2_sign, s2_same_sign, s2_force_zero, s2_force_inf,
                                s2_exp_biased, s2_add_exp_biased, s2_add_significand,
                                s2_add_guard, s2_add_round, s2_add_sticky};
    wire           q_valid;
    wire [Q_W-1:0] q_out;
    wire                q_sign              = q_out[Q_W-1];
    wire                q_same_sign         = q_out[Q_W-2];
    wire                q_force_zero        = q_out[Q_W-3];
    wire                q_force_inf         = q_out[Q_W-4];
    wire     [WEXP-1:0] q_exp_biased        = q_out[Q_W-5 -: WEXP];
    wire     [WEXP-1:0] q_add_exp_biased    = q_out[Q_W-5-WEXP -: WEXP];
    wire     [WMAN-1:0] q_add_significand   = q_out[Q_W-5-2*WEXP -: WMAN];
    wire                q_add_guard         = q_out[2];
    wire                q_add_round         = q_out[1];
    wire                q_add_sticky        = q_out[0];

    // The sub path's close-cancellation normalize lives in the s3 region below. STAGE_NORMALIZE directly forwards
    // to its STAGE_SPLIT (0/1/2 internal register barriers). For SN=0 the cascade is combinational and an explicit
    // s3-boundary register is added below so its zero/count/aligned outputs are aligned with the registered
    // add-path; for SN>=1 the normshift's internal register provides that alignment. A leading 1 above bit NORM_TOP
    // is impossible after a close-cancellation subtraction. Only the add path's exponent adjust is resolved in this
    // s2 cone; the sub path's exponent correction moves to the s3 cone where the normalize count becomes available.
    localparam NORM_TOP_INT = WMAN + 2;
    localparam NINPUT       = NORM_TOP_INT + 1;

    // Stage 3: registered add normalization and subtraction shift count.
    reg                            s3_valid;
    reg                            s3_sign;
    reg                            s3_same_sign;
    reg                            s3_force_zero;
    reg                            s3_force_inf;
    reg                 [WEXP-1:0] s3_exp_biased;       // base (large-operand) exponent, for the sub-path correction
    reg                 [WEXP-1:0] s3_add_exp_biased;   // add-path exponent, resolved in the s2 cone
    reg                 [WMAN-1:0] s3_add_significand;
    reg                            s3_add_guard;
    reg                            s3_add_round;
    reg                            s3_add_sticky;

    // Sub-path close-cancellation normalize: fused leading-zero count + left shift. STAGE_NORMALIZE forwards
    // directly to _zkf_normshift.STAGE_SPLIT (0/1/2 internal register barriers).
    wire                norm_sub_zero;
    wire   [WINDEX-1:0] norm_sub_shift;
    // Normalized magnitude; its slices feed the significand/GRS below.
    wire   [NINPUT-1:0] norm_sub_aligned;
    _zkf_normshift #(.W(NINPUT), .WSHAMT(WINDEX), .STAGE_SPLIT(STAGE_NORMALIZE), .WSB(Q_W)) u_sub_norm (
        .clk(clk), .rst(rst),
        .in_valid(s2_valid),
        .sb_in(s2_q_in),
        .x(s2_raw_result[NINPUT-1:0]),
        .out_valid(q_valid),
        .sb_out(q_out),
        .zero(norm_sub_zero),
        .count(norm_sub_shift),
        .y(norm_sub_aligned)
    );

    // s3-aligned sub-path signals. The sub-path is ALWAYS registered at the s3 boundary, even when STAGE_NORMALIZE
    // is 0 (normshift combinational), so the total sub-path delay from s2 is STAGE_NORMALIZE + 1 cycles, matching
    // the add-path's s2x (depth STAGE_NORMALIZE) + s3 register.
    reg                 s3_sub_zero;
    reg    [WINDEX-1:0] s3_sub_shift;
    reg    [NINPUT-1:0] s3_sub_aligned;
    always @(posedge clk) begin
        s3_sub_zero    <= norm_sub_zero;
        s3_sub_shift   <= norm_sub_shift;
        s3_sub_aligned <= norm_sub_aligned;
    end
    wire [WMAN-1:0] s3_sub_significand = s3_sub_aligned[NINPUT-1 -: WMAN];
    wire            s3_sub_guard       = s3_sub_aligned[2];
    wire            s3_sub_round       = s3_sub_aligned[1];
    wire            s3_sub_sticky      = s3_sub_aligned[0];

    // Sub-path exponent correction lands here because the normalize count is now produced in this cone.
    // zero-extension padding of the normalize shift amount (high bits constant 0).
    wire signed [WEXP_UNBIASED-1:0] s3_sub_shift_ext    = {{(WEXP_UNBIASED-WINDEX){1'b0}}, s3_sub_shift};
    // Widen the native-width base exponent to the signed WEXP_UNBIASED for the subtraction, which can go negative on a
    // close-cancellation underflow. The add-path exponent is non-negative, so it zero-extends into the same field.
    // zero-extension padding (high bits constant 0, like the shift extension above).
    wire signed [WEXP_UNBIASED-1:0] s3_exp_biased_ext  = {{(WEXP_UNBIASED-WEXP){1'b0}}, s3_exp_biased};
    wire signed [WEXP_UNBIASED-1:0] s3_sub_exp_biased  = s3_exp_biased_ext - s3_sub_shift_ext;
    wire signed [WEXP_UNBIASED-1:0] s3_pack_exp_biased =
        s3_same_sign ? {{(WEXP_UNBIASED-WEXP){1'b0}}, s3_add_exp_biased} : s3_sub_exp_biased;

    wire s3_finite_zero = s3_same_sign ? (~|{s3_add_significand, s3_add_guard, s3_add_round, s3_add_sticky})
                                       : s3_sub_zero;
    wire            s3_pack_force_zero  = s3_force_zero || (!s3_force_inf && s3_finite_zero);
    wire [WMAN-1:0] s3_pack_significand = s3_same_sign ? s3_add_significand  : s3_sub_significand;
    wire            s3_pack_guard       = s3_same_sign ? s3_add_guard        : s3_sub_guard;
    wire            s3_pack_round       = s3_same_sign ? s3_add_round        : s3_sub_round;
    wire            s3_pack_sticky      = s3_same_sign ? s3_add_sticky       : s3_sub_sticky;

    _zkf_pack#(
        .WEXP(WEXP), .WMAN(WMAN), .WEXP_UNBIASED(WEXP_UNBIASED), .EXP_IS_BIASED(1),
        .STAGE_INPUT(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_pack (
        .clk(clk),
        .rst(rst),
        .in_valid(s3_valid),
        .sign(s3_sign),
        .force_zero(s3_pack_force_zero),
        .force_inf(s3_force_inf),
        .exp_unbiased(s3_pack_exp_biased),
        .significand(s3_pack_significand),
        .guard(s3_pack_guard),
        .round(s3_pack_round),
        .sticky(s3_pack_sticky),
        .out_valid(out_valid),
        .y(y)
    );

    // Reset only stream validity. Payload registers intentionally free-run. s0b_valid is reset
    // inside the generate block above when STAGE_ALIGN!=0; this always block only manages s0/s1/s2/s3. s2x is its
    // own conditional generate block (above) and resets its valid there when STAGE_NORMALIZE==2.
    always @(posedge clk) begin
        if (rst) begin
            s0_valid <= 1'b0;
            s1_valid <= 1'b0;
            s2_valid <= 1'b0;
            s3_valid <= 1'b0;
        end else begin
            s0_valid <= d_valid;
            s1_valid <= s0b_valid;
            s2_valid <= s1_valid;
            s3_valid <= q_valid;
        end

        // Stage 0 capture: magnitude-ordered operands, exponent delta, and special-case controls.
        s0_finite_sign       <= finite_sign;
        s0_inf_sign          <= inf_sign;
        s0_same_sign         <= d_same_sign;
        s0_force_zero        <= d_a_inf && d_b_inf && !d_same_sign;
        s0_force_inf         <= d_a_inf || d_b_inf;
        // Carry the larger operand's biased exponent directly (no -BIAS here): the result's biased exponent is then
        // large_exp + add_carry (add path) or large_exp - normalize_shift (sub path), and _zkf_pack is told the value
        // is already biased (EXP_IS_BIASED). This folds away the former -BIAS/+BIAS round trip, removing pack's bias
        // add from the adder's exponent critical path.
        s0_exp_biased        <= large_exp;
        s0_exp_diff          <= large_exp - small_exp;
        s0_large_sig_exp     <= large_sig_exp;
        s0_small_sig_exp     <= small_sig_exp;

        // Stage 1 capture: aligned operands sourced from s0b_* (which is either s0_* directly or a
        // delayed copy of s0_*, depending on STAGE_ALIGN). The add/sub carry-chain is in the next stage.
        s1_finite_sign       <= s0b_finite_sign;
        s1_inf_sign          <= s0b_inf_sign;
        s1_same_sign         <= s0b_same_sign;
        s1_force_zero        <= s0b_force_zero;
        s1_force_inf         <= s0b_force_inf;
        s1_exp_biased        <= s0b_exp_biased;
        s1_large_ext_exp     <= {s0b_large_sig_exp, {WGRS{1'b0}}};
        s1_small_aligned     <= s0_small_aligned;

        // Stage 2 capture: the single carry-chain computes add or subtract by conditionally inverting the small
        // aligned operand and adding the carry-in.
        s2_sign              <= s1_result_sign;
        s2_same_sign         <= s1_same_sign;
        s2_force_zero        <= s1_force_zero;
        s2_force_inf         <= s1_force_inf;
        s2_exp_biased        <= s1_exp_biased;
        s2_raw_result        <= s1_raw_result;

        // Stage 3 capture: add-path normalization and subtract-path shift metadata. Inputs come from q_*, which is
        // the s2 add-path delayed by STAGE_NORMALIZE cycles through the s2x catch-up pipe, so it stays aligned
        // with the sub-path's normshift output at the s3 register boundary.
        s3_sign              <= q_sign;
        s3_same_sign         <= q_same_sign;
        s3_force_zero        <= q_force_zero;
        s3_force_inf         <= q_force_inf;
        s3_exp_biased        <= q_exp_biased;
        s3_add_exp_biased    <= q_add_exp_biased;
        s3_add_significand   <= q_add_significand;
        s3_add_guard         <= q_add_guard;
        s3_add_round         <= q_add_round;
        s3_add_sticky        <= q_add_sticky;
    end
endmodule

`undef ZKF_ADD_LATENCY


// Compare unsigned values through an explicit carry-chain-friendly subtraction; enables much better timings than
// the ordinary comparison operator. Related:
// https://stackoverflow.com/questions/60844496/does-subtraction-need-less-resource-than-comparison-symbol-in-verilog
module _zkf_add_ge #(parameter W = 18) (input wire [W-1:0] a, input wire [W-1:0] b, output wire ge);
    wire [W:0] diff = {1'b0, a} - {1'b0, b};
    assign ge = !diff[W];
endmodule


`default_nettype wire

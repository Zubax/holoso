/// Streamed float -> signed fixed-point reduction shared by zkf_to_int (FF=0) and zkf_exp2 (FF>0).
/// Decodes the float, computes the left/right shift amounts as parallel folded-constant subtractions, applies the
/// barrel shift, and exposes the unsigned post-shift magnitude with binary point at bit FF along with the sign and
/// the OOR/specials flags. The output is *not* yet two's-complement; the caller forms the signed value via a
/// {sign, mag} negate as needed (zkf_to_int negates after rounding; zkf_exp2 negates before splitting into i and f).
///
/// Register stages: STAGE_INPUT+2. Zero-bubble, throughput-1, no backpressure.
///
/// OOR_EXP_THRESHOLD: input-exponent value at or above which `oor` fires, in addition to the intrinsic
/// "magnitude won't fit in WI signed bits" predicate and `is_inf`. Default is (1 << WEXP) which sits above the
/// max value `exp_in` can take, so the extrinsic predicate resolves to constant 0 at elaboration; this is what
/// zkf_to_int wants. zkf_exp2 sets OOR_EXP_THRESHOLD = BIAS+WEXP-1 so the result's integer part is guaranteed
/// to fit in WEXP signed bits (the value's exponent is already out of representable range above that point).
///
/// Output semantics:
///
///   - mag[WI+FF-1:0]: unsigned magnitude scaled by 2^FF (binary point at bit FF). Don't-care when oor=1.
///
///   - guard: bit at virtual position -1 just below LSB of mag (the RTNE tie bit when the caller rounds at the bit-FF
///     boundary). Only nonzero when FF=0; structurally constant 0 when FF>0.
///
///   - lost_sticky: OR of all bits dropped below `guard` (the combined round|sticky bit). For FF=0 this comes from
///     the right shifter's own sticky collapse; for FF>0 the dropped bits sit further below mag's LSB.
///
///   - sign / is_inf / is_zero: decoded from the input float.
///
///   - oor: |a| won't fit in WI signed bits OR exp >= OOR_EXP_THRESHOLD OR input was +-inf. When asserted, mag and
///     guard/lost_sticky are don't-care; the caller routes to its saturation/force_inf/force_zero path.

`default_nettype none

module _zkf_to_fixpoint #(
    parameter WEXP                      = 6,    // exponent field width
    parameter WMAN                      = 18,   // significand precision including the hidden bit
    parameter WI                        = 8,    // signed integer-part bits of the result; mag fits in WI+FF when oor=0
    parameter FF                        = 0,    // fractional bits kept; binary point at bit FF of mag
    parameter STAGE_INPUT               = 0,    // 0 = expose combinational inputs; 1 = latch inputs (+1 cycle)
    parameter integer OOR_EXP_THRESHOLD = (1 << WEXP)   // default disables the extrinsic predicate at elaboration
) (
    input  wire clk,
    input  wire rst,

    input  wire                  in_valid,
    input  wire  [WEXP+WMAN-1:0] a,

    output wire                  out_valid,
    // mag is the wide fixed-point reduction carrier (sign-extended integer part above the fraction); its top bits are
    // structural sign-extension / headroom. Its meaningful bits feed the caller's split.
    output wire    [WI+FF-1:0]   mag,
    output wire                  guard,
    output wire                  lost_sticky,
    output wire                  sign,
    output wire                  is_inf,
    output wire                  is_zero,
    output wire                  oor
);
    generate
        if ((WEXP < 2) || (WMAN < 4) || (WI < 2)) begin : g_invalid_widths
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if ((STAGE_INPUT != 0) && (STAGE_INPUT != 1)) begin : g_invalid_stage_input
            _zkf_invalid_stage_input u_invalid();
        end
        // Shift by WEXP >= 31 overflows Verilog's 32-bit integer constant arithmetic and yields tool-dependent values.
        if (WEXP >= 31) begin : g_invalid_wexp_too_wide
            _zkf_invalid_to_fixpoint_wexp_too_wide_unportable u_invalid();
        end
    endgenerate

    localparam WFRAC      = WMAN - 1;
    localparam WFULL      = WEXP + WMAN;
    // Right-shifter low-bit padding: 2 for FF==0 (gives guard+sticky pads for RTNE rounding at the integer
    // boundary -- matches today's zkf_to_int instantiation), 1 for FF>0 (no kept guard pad; one padding bit is
    // still needed below the kept WMAN bits because _zkf_rshift_sticky's output bit 0 OR's the data bit at
    // position 0 together with the dropped sticky, and the caller wants those separate: f_right[0] is a real
    // magnitude bit of mag, and lost_sticky is the OR of bits dropped further below).
    localparam GUARD_PAD  = (FF == 0) ? 2 : 1;
    localparam WRSHIFTER  = WMAN + GUARD_PAD;           // right-shift container width
    // Largest useful left shift before the result is provably outside WI+FF magnitude bits.
    localparam LSH_MAX    = (WI + FF > WMAN) ? (WI + FF - WMAN) : 0;
    localparam WLSH       = (LSH_MAX > 0) ? $clog2(LSH_MAX + 1) : 1;
    localparam WLEFT      = WMAN + LSH_MAX;             // left-shifter output width
    // Largest useful right shift before everything becomes sticky; one GRS-pad above WMAN when FF==0, just WMAN
    // when FF>0 (no guard padding -- the fractional bits go into the kept mag instead).
    localparam RSH_MAX    = WRSHIFTER;
    localparam WRSH       = $clog2(RSH_MAX + 1);

    // Folded shift-magnitude / predicate thresholds in integer arithmetic so widths are sized from them, not from
    // WEXP alone. The left/right boundary moves by FF (so for FF=0 this is zkf_to_int's BIAS+WFRAC, and for FF>0 this
    // becomes BIAS+WFRAC-FF = zkf_exp2's BIAS-SHIFT_OFF = BIAS-13). LEFT_SHIFT_BASE can be negative at small
    // BIAS / large FF; the predicates below select the "always left shift" arm in that case.
    //   LEFT_SHIFT_BASE  : exp_in == this means value == 2^WFRAC * 2^(-FF) = 2^(WFRAC-FF); the boundary between
    //                      right and left shift. left_shift_full = exp_in - LEFT_SHIFT_BASE.
    //   MAG_OVER_BASE    : exp_in >= this means |value| >= 2^WI, so the magnitude overflows the WI+FF container and
    //                      the WI+FF truncation below would silently drop its high bits -- independent of shift
    //                      direction. oor is raised here so the caller saturates instead of emitting a truncated
    //                      magnitude. This is BIAS_INT+WI directly, NOT LEFT_SHIFT_BASE+LSH_MAX+1: the two coincide
    //                      while LSH_MAX>0, but when WI+FF <= WMAN (LSH_MAX clamped to 0) the latter sits too high
    //                      and would let over-range finite values through (e.g. to_int with WINT < WMAN). When
    //                      LSH_MAX>0 this is also the exp at which the left shifter would run past LSH_MAX, so it
    //                      still doubles as the lshamt clamp boundary.
    //   RIGHT_OVER_BASE  : exp_in < this means the right shift amount exceeds RSH_MAX, so the shifter would not
    //                      capture any useful bits and we clamp to RSH_MAX.
    localparam integer BIAS_INT         = (1 << (WEXP - 1)) - 1;
    localparam integer LEFT_SHIFT_BASE  = BIAS_INT + WFRAC - FF;
    localparam integer MAG_OVER_BASE    = BIAS_INT + WI;
    localparam integer RIGHT_OVER_BASE  = LEFT_SHIFT_BASE - RSH_MAX;
    // WEU sizes the two wide subtractions left_shift_full and right_shift_full.
    localparam integer MAX_EXP_IN       = (1 << WEXP) - 1;
    localparam integer ABS_LSB          = (LEFT_SHIFT_BASE >= 0) ? LEFT_SHIFT_BASE : -LEFT_SHIFT_BASE;
    localparam integer MAX_POS_DELTA    = MAX_EXP_IN - LEFT_SHIFT_BASE;     // = MAX_EXP_IN + |LSB| when LSB<0
    localparam integer MAX_ABS_DELTA    = (ABS_LSB > MAX_POS_DELTA) ? ABS_LSB : MAX_POS_DELTA;
    localparam integer WEU              = $clog2(MAX_ABS_DELTA + 1) + 1;

    // LEFT_SHIFT_BASE may be negative; the integer slice into WEU bits preserves the sign bits via Verilog's
    // built-in sign extension of `integer` constants.
    localparam signed [WEU-1:0] LEFT_SHIFT_BASE_EXT = LEFT_SHIFT_BASE[WEU-1:0];
    localparam signed [WEU-1:0] LEFT_SHIFT_OFFSET   = -LEFT_SHIFT_BASE_EXT;

    // Optional input register stage.
    wire             in_valid_q;
    wire [WFULL-1:0] a_q;
    zkf_pipe #(.W(WFULL), .N(STAGE_INPUT ? 1 : 0)) u_input_pipe (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in(a),
        .out_valid(in_valid_q), .out(a_q)
    );

    // -- Stage-1 cone: decode the float, derive the shift magnitudes, and resolve the predicates that pick between
    // the right- and left-shift branches (and decide oor). The barrel shifters themselves are intentionally placed
    // in the next cone so neither stage carries both a wide subtract and a wide variable shift.
    wire             sign_in    = a_q[WFULL-1];
    wire [WEXP-1:0]  exp_in     = a_q[WFULL-2:WFRAC];
    wire [WFRAC-1:0] frac_in    = a_q[WFRAC-1:0];
    wire             is_zero_in = ~|exp_in;
    wire             is_inf_in  =  &exp_in;
    // The hidden bit is implicit for normal values. For zero inputs the right shifter cannot always saturate enough
    // to wipe the hidden bit (BIAS + WFRAC can be less than RSH_MAX for small WEXP at FF=0), so explicitly zero the
    // significand here; the downstream pipeline then yields mag = 0 without a separate is_zero late-stage mux. The
    // zeroing is also harmless when FF>0 because the right shift already saturates the magnitude to zero in that
    // configuration (RSH_MAX = WMAN).
    wire [WMAN-1:0]  sig_in     = is_zero_in ? {WMAN{1'b0}} : {1'b1, frac_in};

    // Zero-extension padding of a non-negative exponent (high bits constant 0).
    wire signed [WEU-1:0] exp_in_ext = $signed({{(WEU-WEXP){1'b0}}, exp_in});

    // Two parallel folded-constant subtractions provide the shift magnitudes (only their low WLSH / WRSH bits are
    // consumed downstream). right_shift_full uses the positive constant directly rather than negating
    // left_shift_full, which would put both shift amounts on the same serial carry chain.
    wire signed [WEU-1:0] left_shift_full  = exp_in_ext + LEFT_SHIFT_OFFSET;
    wire signed [WEU-1:0] right_shift_full = LEFT_SHIFT_BASE_EXT - exp_in_ext;

    // Predicates as unsigned comparisons of exp_in against compile-time non-negative constants. Constants out of
    // exp_in's unsigned range resolve at elaboration and emit no runtime logic; in-range constants map to a shallow
    // LUT compare rather than a WEU-wide signed subtract, so the predicate does not chain a wide carry-chain stack onto
    // the critical path feeding the shifter mux.
    wire is_left_shift;
    wire mag_too_big;
    wire right_too_big;
    wire exp_oor_extrinsic;
    generate
        if (LEFT_SHIFT_BASE <= 0) begin : g_lshift_always
            assign is_left_shift = 1'b1;
        end else if (LEFT_SHIFT_BASE > MAX_EXP_IN) begin : g_lshift_never
            assign is_left_shift = 1'b0;
        end else begin : g_lshift_cmp
            assign is_left_shift = exp_in >= LEFT_SHIFT_BASE[WEXP-1:0];
        end

        if (MAG_OVER_BASE <= 0) begin : g_mover_always
            assign mag_too_big = 1'b1;
        end else if (MAG_OVER_BASE > MAX_EXP_IN) begin : g_mover_never
            assign mag_too_big = 1'b0;
        end else begin : g_mover_cmp
            assign mag_too_big = exp_in >= MAG_OVER_BASE[WEXP-1:0];
        end

        if (RIGHT_OVER_BASE <= 0) begin : g_rover_never
            assign right_too_big = 1'b0;
        end else if (RIGHT_OVER_BASE > MAX_EXP_IN) begin : g_rover_always
            assign right_too_big = 1'b1;
        end else begin : g_rover_cmp
            assign right_too_big = exp_in < RIGHT_OVER_BASE[WEXP-1:0];
        end

        // Extrinsic OOR (the caller-supplied threshold). Defaulted to (1<<WEXP) -> always-false / elision.
        if (OOR_EXP_THRESHOLD > MAX_EXP_IN) begin : g_oor_extrinsic_never
            assign exp_oor_extrinsic = 1'b0;
        end else if (OOR_EXP_THRESHOLD <= 0) begin : g_oor_extrinsic_always
            assign exp_oor_extrinsic = 1'b1;
        end else begin : g_oor_extrinsic_cmp
            assign exp_oor_extrinsic = exp_in >= OOR_EXP_THRESHOLD[WEXP-1:0];
        end
    endgenerate

    wire oor_in = is_inf_in | mag_too_big | exp_oor_extrinsic;

    // The left-shift clamp uses the combined oor so that mag stays in-container even when the extrinsic threshold
    // fires before mag_too_big does (zkf_exp2's case). lshamt / rshamt are don't-care for the non-selected
    // direction (the mux picks one), so each clamp only has to be correct in its own direction.
    wire [WLSH-1:0] lshamt_clamped = (is_left_shift && !oor_in) ? left_shift_full[WLSH-1:0] : {WLSH{1'b0}};
    wire [WRSH-1:0] rshamt_clamped = right_too_big ? RSH_MAX[WRSH-1:0] : right_shift_full[WRSH-1:0];

    // -- Stage 1: capture pre-shift state (decode + clamp). Reset only validity; payload free-runs.
    reg             s1_valid;
    reg             s1_sign;
    reg             s1_is_inf;
    reg             s1_is_zero;
    reg             s1_is_left_shift;
    reg             s1_oor;
    reg [WMAN-1:0]  s1_sig;
    reg [WRSH-1:0]  s1_rshamt;
    reg [WLSH-1:0]  s1_lshamt;   // registered lshamt_clamped

    // -- Stage 1 -> Stage 2 combinational: the heavy barrel shifters. The right-shift barrel folds the discarded
    // tail into a single sticky bit; the left-shift is exact (no GRS). The two branches are muxed by
    // s1_is_left_shift.
    //
    // Right shifter padding (constant generate-if, no runtime mux):
    //   FF==0: W = WMAN + 2; input is {sig, 2'b00}. Output [W-1:2] = mag, [1] = guard, [0] = sticky. This is
    //          byte-equivalent to the GRS layout that zkf_to_int consumes.
    //   FF>0 : W = WMAN + 1; input is {sig, 1'b0}.  Output [W-1:1] = mag low bits, [0] = sticky; guard is
    //          structurally 0. The one extra pad bit is required because _zkf_rshift_sticky OR's data bit 0
    //          together with the dropped sticky -- we need them separated for FF>0 (the data bit 0 lives in mag,
    //          the sticky is a separate sideband).
    wire [WRSHIFTER-1:0] rsh_out_pre;
    // rsh_in[0] is a structural pad (the {.., 1'b0} / {.., 2'b00} sticky-alignment bit, always 0); the upper bits just
    // re-present s1_sig.
    wire [WRSHIFTER-1:0] rsh_in;
    generate
        if (FF == 0) begin : g_rsh_in_grs
            assign rsh_in = {s1_sig, 2'b00};
        end else begin : g_rsh_in_no_grs
            assign rsh_in = {s1_sig, 1'b0};
        end
    endgenerate
    _zkf_rshift_sticky #(.W(WRSHIFTER), .WSHIFT(WRSH), .STAGE_SPLIT(0)) u_rshift (
        .clk(clk),
        .x(rsh_in),
        .shamt(s1_rshamt),
        .y(rsh_out_pre)
    );

    // Uniform extraction: the top WMAN bits of rsh_out_pre are always the mag bits; bit 0 is always the sticky;
    // guard is the second-from-bottom bit when FF==0, zero otherwise.
    wire [WMAN-1:0] rsh_mag_pre    = rsh_out_pre[WRSHIFTER-1 -: WMAN];
    wire            rsh_sticky_pre = rsh_out_pre[0];
    wire            rsh_guard_pre;
    generate
        if (FF == 0) begin : g_rsh_guard
            assign rsh_guard_pre = rsh_out_pre[1];
        end else begin : g_rsh_no_guard
            assign rsh_guard_pre = 1'b0;
        end
    endgenerate

    // Left shift: zero-extend the WMAN-bit significand and shift into the WLEFT-bit container. When LSH_MAX==0
    // (rare; would mean WI+FF <= WMAN), the left branch is a passthrough.
    wire [WLEFT-1:0] lsh_out_pre;
    generate
        if (LSH_MAX > 0) begin : g_lshift
            assign lsh_out_pre = {{LSH_MAX{1'b0}}, s1_sig} << s1_lshamt;
        end else begin : g_no_lshift
            assign lsh_out_pre = s1_sig;
        end
    endgenerate

    // Zero-extension of the shifted magnitude into the WI+FF working width; the high padding bits are
    // structurally constant.
    wire [WI+FF-1:0] mag_pre_rsh_in;
    wire [WI+FF-1:0] mag_pre_lsh_in;
    generate
        if (WI + FF > WMAN) begin : g_mag_pre_rsh_pad
            assign mag_pre_rsh_in = {{(WI+FF-WMAN){1'b0}}, rsh_mag_pre};
        end else begin : g_mag_pre_rsh_trunc
            assign mag_pre_rsh_in = rsh_mag_pre[WI+FF-1:0];
        end
        if (WI + FF > WLEFT) begin : g_mag_pre_lsh_pad
            assign mag_pre_lsh_in = {{(WI+FF-WLEFT){1'b0}}, lsh_out_pre};
        end else begin : g_mag_pre_lsh_trunc
            assign mag_pre_lsh_in = lsh_out_pre[WI+FF-1:0];
        end
    endgenerate
    wire [WI+FF-1:0] mag_pre_in     = s1_is_left_shift ? mag_pre_lsh_in : mag_pre_rsh_in;
    wire             guard_in       = s1_is_left_shift ? 1'b0 : rsh_guard_pre;
    wire             lost_sticky_in = s1_is_left_shift ? 1'b0 : rsh_sticky_pre;

    // -- Stage 2: capture post-shift state. Reset only validity; payload free-runs.
    reg             s2_valid;
    reg             s2_sign;
    reg             s2_is_inf;
    reg             s2_is_zero;
    reg             s2_oor;
    // WI+FF magnitude reg; the bits above the active range are
    // structurally constant.
    reg [WI+FF-1:0] s2_mag;
    reg             s2_guard;
    reg             s2_lost_sticky;

    always @(posedge clk) begin
        if (rst) begin
            s1_valid <= 1'b0;
            s2_valid <= 1'b0;
        end else begin
            s1_valid <= in_valid_q;
            s2_valid <= s1_valid;
        end
        s1_sign          <= sign_in;
        s1_is_inf        <= is_inf_in;
        s1_is_zero       <= is_zero_in;
        s1_is_left_shift <= is_left_shift;
        s1_oor           <= oor_in;
        s1_sig           <= sig_in;
        s1_rshamt        <= rshamt_clamped;
        s1_lshamt        <= lshamt_clamped;

        s2_sign        <= s1_sign;
        s2_is_inf      <= s1_is_inf;
        s2_is_zero     <= s1_is_zero;
        s2_oor         <= s1_oor;
        s2_mag         <= mag_pre_in;
        s2_guard       <= guard_in;
        s2_lost_sticky <= lost_sticky_in;
    end

    assign out_valid   = s2_valid;
    assign mag         = s2_mag;
    assign guard       = s2_guard;
    assign lost_sticky = s2_lost_sticky;
    assign sign        = s2_sign;
    assign is_inf      = s2_is_inf;
    assign is_zero     = s2_is_zero;
    assign oor         = s2_oor;
endmodule

`default_nettype wire

/// Streamed base-2 logarithm for the Zubax Kulibin float format: y = log2(x).
/// Zero-bubble, throughput-1, no backpressure.
/// Behavior:
///
///   log2(finite>0) = log2(x), round-to-nearest ties-to-even
///   log2(+inf)     = +inf
///   log2(+0)       = -inf, pole=1
///   log2(x<0)      = -inf, domain_error=1
///
/// Algorithm (symmetric argument reduction; mirrors tb/zkf_model.py::log2_reference bit-for-bit):
///
///  1. With x = m * 2^e (m = 1.frac in [1,2), e = exp-BIAS), log2(x) = e + log2(m). Re-center the mantissa into the
///     symmetric interval: if m >= sqrt(2) (significand sig >= THR = round(sqrt(2)*2^WFRAC)), halve m and increment e,
///     so the reduced mantissa m' in [sqrt(1/2), sqrt(2)) and log2(m') in [-1/2, 1/2). The reduced fraction f = m'-1 is
///     exact (Sterbenz). This removes the catastrophic x->1 cancellation of the old m in [1,2) reduction (where e=-1
///     and log2(m)->1 nearly cancel): now x->1 maps to e=0, f->0 -- the direct, cancellation-free path.
///
///  2. Two exact integer quantities are formed from the stored fraction (no irrational subtraction; the index
///     arithmetic is identical model<->RTL), at scale 2^-(WFRAC+1):
///       v = f + 1/2 in [0.207, 0.914)  -- UNSIGNED index coordinate (top K bits select the segment); WFRAC+1 = WMAN
///                                         bits. m < sqrt(2): v = 2^WFRAC + 2*frac;  m >= sqrt(2): v = frac.
///       f = v - 2^WFRAC                -- SIGNED combine operand (= the reduced fraction at scale 2^-(WFRAC+1)).
///                                         m < sqrt(2): f = 2*frac (>= 0);  m >= sqrt(2): f = frac - 2^WFRAC (< 0).
///
///  3. The pipelined per-WMAN table+polynomial core (selected by the generate-if) evaluates the smooth kernel
///     C(f) = log2(1+f)/f via the segmented truncating Horner (indexed by v) and returns the SIGNED product
///     log2(m') = f*C(f) as a fixed-point value at scale 2^-F2 (F2 = WFRAC+1+CF).
///
///  4. The signed fixed-point sum R = (e << F2) + log2(m') is renormalized and rounded by _zkf_fixed_to_float, which
///     owns the _zkf_normshift instance, the GRS extraction, the exp_unbiased arithmetic, the optional packer input
///     register, and the _zkf_pack output stage. Results are always representable for finite x, so no overflow path.
///
/// STAGE_INPUT=1 registers the raw input before decode.
/// STAGE_DECODE=1 splits classification/re-center across two registers; STAGE_DECODE=0 keeps one register.
/// The generated evaluator uses a registered ROM read followed by a mandatory fabric register before Horner;
/// non-arithmetic sideband payloads are aligned by plain delay pipes.
///
/// STAGE_PRODUCT selects product computation staging; see _zkf_pmul for details.
/// STAGE_PRODUCT_FINAL selects product computation staging for only the final f*C(f) multiply; defaults to
/// STAGE_PRODUCT.
/// WMULTIPLIER is an optional hint of the native DSP tile argument width; forwaded to _zkf_pmul, refer there.
/// STAGE_NORMALIZE={0,1,2} forwards directly to _zkf_normshift.STAGE_SPLIT.
/// STAGE_NORMALIZE_OUTPUT={0,1} forwards directly to _zkf_normshift.STAGE_OUTPUT.
/// STAGE_PACK={0,1} forwards to _zkf_pack.STAGE_INPUT (insulates rounder from normshift cone).
/// STAGE_OUTPUT={0,1} registers the output.

`default_nettype none

module zkf_log2 #(
    parameter WEXP                  = 6,    // exponent field width
    parameter WMAN                  = 18,   // significand precision including the hidden bit
    parameter WMULTIPLIER           = 0,    // forwarded to _zkf_pmul
    parameter STAGE_INPUT           = 0,    // 0: combinational inputs;   1: latch inputs before any logic (+1 stage)
    parameter STAGE_DECODE          = 0,    // 0: one decode/re-center register; 1: split across two registers
    parameter STAGE_PRODUCT         = 0,    // forwarded to the Horner _zkf_pmul instances
    parameter STAGE_PRODUCT_FINAL   = STAGE_PRODUCT,  // forwarded to the final f*C(f) _zkf_pmul only
    parameter STAGE_NORMALIZE       = 0,    // 0/1/2 internal normshift barriers (direct -> _zkf_normshift.STAGE_SPLIT)
    parameter STAGE_NORMALIZE_OUTPUT = 0,   // 0/1 register normshift outputs (direct -> _zkf_normshift.STAGE_OUTPUT)
    parameter STAGE_PACK            = 0,    // 0: comb pack input; 1: register pack input (insulates rounder)
    parameter STAGE_OUTPUT          = 0,    // 0: combinational outputs;     1: registered outputs, +1 stage
    parameter LATENCY               = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] x,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y,
    output wire                 domain_error,
    output wire                 pole
);
    localparam DEGREE = (((WMAN+18)/11)-1);
    localparam LATENCY_REF = STAGE_INPUT + STAGE_DECODE + 5 + STAGE_PRODUCT_FINAL + STAGE_NORMALIZE
                           + STAGE_NORMALIZE_OUTPUT + STAGE_PACK + DEGREE*(2+STAGE_PRODUCT) + STAGE_OUTPUT;
    generate
        // BIAS below uses an unsized integer shift on WEXP; WEXP >= 31 would overflow 32-bit integer constants.
        if ((WEXP < 2) || (WMAN < 4) || (WEXP >= 31)) begin : g_invalid_wman
            _zkf_invalid_wexp_or_wman u_invalid();
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

    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;
    // CF: MUST match the generator's GUARD_CF (float/zkf_transcendental.py). The symmetric reduction puts the reduced
    // fraction f at scale 2^-(WFRAC+1), so the combine scale gains one bit over the old reduction: F2 = WFRAC + 1 + CF.
    localparam CF     = WMAN + 12;
    localparam F2     = WFRAC + 1 + CF;         // fractional bits of log2(m') and of the R accumulator
    // For finite positive inputs, re-centering may increment e to 2^(WEXP-1), but then log2(m') < 0; otherwise
    // e <= 2^(WEXP-1)-1 and log2(m') < 1/2. Thus |e + log2(m')| < 2^(WEXP-1), so WEXP-1 unsigned integer magnitude
    // bits plus F2 fractional bits are sufficient. Specials are forced through the sideband and ignore this magnitude.
    localparam WNORM  = (WEXP - 1) + F2;        // finite-result magnitude width fed to the normalizer
    localparam WR     = WNORM + 1;              // signed R = (e << F2) + log2(m') width
    localparam WE     = WEXP + 1;               // signed e = exp - BIAS (+1 on re-center)
    // Signed unbiased result exponent in [-F2, WEXP-1]; also kept >= WEXP+2 because _zkf_pack requires its
    // exponent field to be at least WEXP+1 bits wide for its internal bias arithmetic.
    localparam WEU_RAW = $clog2(F2 + 1) + 2;
    localparam WEU     = (WEU_RAW > (WEXP + 2)) ? WEU_RAW : (WEXP + 2);
    localparam SBW     = WE + 4;                // delayed evaluator sideband: {e, is_special, special_sign, pole, de}

    localparam integer BIAS = (1 << (WEXP - 1)) - 1;

    // Re-center threshold THR = round(sqrt(2) * 2^WFRAC): re-center iff the WMAN-bit significand sig >= THR
    // (m >= sqrt2).
    // Computed at elaboration by an exact integer sqrt (the same value as the model's _trans_sqrt2_threshold and the
    // generator's log2_sqrt2_threshold): round(sqrt(S)) for S = 2^(2*WFRAC+1) is (floor(sqrt(4*S)) + 1) / 2, and
    // 4*S = 2^(2*WFRAC+3). The streaming digit-by-digit isqrt below is a fixed-bound constant function (no while loop)
    // so it elaborates portably; WMAN <= 53 keeps 2*WFRAC+3 <= 107 < 128.
    // Constant isqrt evaluated only at elaboration for THR; no runtime line/branch coverage.
    // verilator coverage_off
    function automatic [127:0] _zkf_isqrt128;
        input [127:0] n_in;
        reg [127:0] n, rem, root;
        integer i;
        begin
            n    = n_in;
            rem  = 0;
            root = 0;
            for (i = 0; i < 64; i = i + 1) begin
                root = root << 1;
                rem  = (rem << 2) | ((n >> 126) & 128'd3);   // stream the next two bits from the MSB
                n    = n << 2;
                if (rem > (root << 1)) begin                 // rem > 2*root  <=>  rem >= 2*root + 1
                    rem  = rem - ((root << 1) | 128'd1);
                    root = root + 1;
                end
            end
            _zkf_isqrt128 = root;
        end
    endfunction
    // verilator coverage_on
    localparam [WFRAC:0] THR = (_zkf_isqrt128(128'd1 << (2 * WFRAC + 3)) + 128'd1) >> 1;

    // -- Optional raw input register stage: with STAGE_INPUT=1, no decode/re-center logic sits on the input side.
    wire             in_valid_q;
    wire [WFULL-1:0] x_q;
    zkf_pipe #(.W(WFULL), .N(STAGE_INPUT ? 1 : 0)) u_input_pipe (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in(x),
        .out_valid(in_valid_q), .out(x_q)
    );

    // -- Decode, classify, and prepare for re-center.
    wire             sign_in = x_q[WFULL-1];
    wire [WEXP-1:0]  exp_in  = x_q[WFULL-2:WFRAC];
    wire [WFRAC-1:0] frac_in = x_q[WFRAC-1:0];
    wire             is_zero = ~|exp_in;
    wire             is_inf  =  &exp_in;
    // Special results are all +/-inf: +inf for +inf input; -inf for +0 (pole), negative finite, or -inf (domain).
    wire             is_special_pre   = is_inf | is_zero | sign_in;
    wire             special_sign_pre = is_zero | sign_in;          // 0 -> +inf, 1 -> -inf
    wire             pole_pre         = is_zero;
    wire             de_pre           = sign_in & ~is_zero;

    // Symmetric re-center. sig = {hidden 1, frac} is the WMAN-bit significand; re-center when m >= sqrt(2). v is the
    // unsigned index coordinate (WMAN bits) and f the signed combine operand (WMAN = WFRAC+1 bits), both formed exactly
    // from frac (no irrational subtraction), matching the model. v and f are carry-free concatenations: the +2^WFRAC
    // only sets bit WFRAC, which never collides with the low bits of 2*frac in the branch that selects them
    // (m < sqrt(2) keeps frac < 2^(WFRAC-1)). The two's-complement identity {1'b1, frac} (= sig) read as signed is
    // exactly frac - 2^WFRAC, the re-center branch's f.
    wire [WFRAC:0]   sig_in    = {1'b1, frac_in};                  // WMAN-bit significand, m = sig / 2^WFRAC
    wire             recenter  = sig_in >= THR;                    // m >= sqrt(2)

    // For special/noncanonical transactions, clamp v to the exact x=1 reduced argument (v=1/2, f=0) so compact log2
    // ROMs are never addressed outside their reachable segment span.
    wire [WMAN-1:0]       v_pre_raw_1 = recenter ? {1'b0, frac_in}                  // v = frac
                                                  : {1'b1, frac_in[WFRAC-2:0], 1'b0}; // v = 2^WFRAC + 2*frac
    wire signed [WE-1:0]  e_pre_1     = ($signed({1'b0, exp_in}) - $signed(BIAS[WE-1:0]))
                                           + $signed({{(WE-1){1'b0}}, recenter});
    wire [WMAN-1:0]       v_safe_1    = is_special_pre ? {1'b1, {WFRAC{1'b0}}} : v_pre_raw_1;
    wire signed [WE-1:0]  e_safe_1    = is_special_pre ? {WE{1'b0}} : e_pre_1;

    wire                  r0_valid;
    wire       [WMAN-1:0] r0_v;
    wire signed [WE-1:0]  r0_e;
    wire                  r0_is_special;
    wire                  r0_special_sign;
    wire                  r0_pole;
    wire                  r0_de;
    generate
        if (STAGE_DECODE == 0) begin : g_decode_one
            // One decode/re-center register. Reset only validity; payload free-runs.
            reg                  r_valid;
            reg       [WMAN-1:0] r_v;
            reg signed [WE-1:0]  r_e;
            reg                  r_is_special;
            reg                  r_special_sign;
            reg                  r_pole;
            reg                  r_de;
            always @(posedge clk) begin
                if (rst) r_valid <= 1'b0;
                else     r_valid <= in_valid_q;
                r_v            <= v_safe_1;
                r_e            <= e_safe_1;
                r_is_special   <= is_special_pre;
                r_special_sign <= special_sign_pre;
                r_pole         <= pole_pre;
                r_de           <= de_pre;
            end
            assign r0_valid       = r_valid;
            assign r0_v           = r_v;
            assign r0_e           = r_e;
            assign r0_is_special  = r_is_special;
            assign r0_special_sign = r_special_sign;
            assign r0_pole        = r_pole;
            assign r0_de          = r_de;
        end else begin : g_decode_two
            // Split raw decode/re-center predicate from final coordinate/exponent formation.
            reg              d0_valid;
            reg [WEXP-1:0]   d0_exp;
            reg [WFRAC-1:0]  d0_frac;
            reg              d0_recenter;
            reg              d0_is_special;
            reg              d0_special_sign;
            reg              d0_pole;
            reg              d0_de;
            always @(posedge clk) begin
                if (rst) d0_valid <= 1'b0;
                else     d0_valid <= in_valid_q;
                d0_exp          <= exp_in;
                d0_frac         <= frac_in;
                d0_recenter     <= recenter;
                d0_is_special   <= is_special_pre;
                d0_special_sign <= special_sign_pre;
                d0_pole         <= pole_pre;
                d0_de           <= de_pre;
            end

            wire [WMAN-1:0]      v_pre_raw = d0_recenter ? {1'b0, d0_frac}
                                                         : {1'b1, d0_frac[WFRAC-2:0], 1'b0};
            wire signed [WE-1:0] e_pre     = ($signed({1'b0, d0_exp}) - $signed(BIAS[WE-1:0]))
                                               + $signed({{(WE-1){1'b0}}, d0_recenter});
            wire [WMAN-1:0]      v_safe    = d0_is_special ? {1'b1, {WFRAC{1'b0}}} : v_pre_raw;
            wire signed [WE-1:0] e_safe    = d0_is_special ? {WE{1'b0}} : e_pre;

            reg                  r_valid;
            reg       [WMAN-1:0] r_v;
            reg signed [WE-1:0]  r_e;
            reg                  r_is_special;
            reg                  r_special_sign;
            reg                  r_pole;
            reg                  r_de;
            always @(posedge clk) begin
                if (rst) r_valid <= 1'b0;
                else     r_valid <= d0_valid;
                r_v            <= v_safe;
                r_e            <= e_safe;
                r_is_special   <= d0_is_special;
                r_special_sign <= d0_special_sign;
                r_pole         <= d0_pole;
                r_de           <= d0_de;
            end
            assign r0_valid       = r_valid;
            assign r0_v           = r_v;
            assign r0_e           = r_e;
            assign r0_is_special  = r_is_special;
            assign r0_special_sign = r_special_sign;
            assign r0_pole        = r_pole;
            assign r0_de          = r_de;
        end
    endgenerate

    // -- Pipelined evaluator: log2(m') = f*C(f). e and the special-case flags ride the sideband, aligned to the result.
    // The evaluator returns the magnitude |log2(m')| and its sign separately (the sign is folded into the reconstruction
    // add/subtract below), avoiding a standalone negate carry chain in the post-product cone.
    wire [SBW-1:0] sb_in_l = {r0_e, r0_is_special, r0_special_sign, r0_pole, r0_de};
    wire           ev_valid;
    wire [SBW-1:0] sb_out_l;
    wire [F2:0] l_mag;   // |log2(m')| = |f|*C(f) magnitude at scale 2^-F2 (F2+1 bits, unsigned)
    wire        l_neg;   // sign of log2(m') (= reduced f sign); 1 when m >= sqrt(2)
    // We pass the closed-form degree D below; the core asserts it matches the degree its ROM was fitted for (mirrors
    // the LATENCY parameter), so the Horner depth / latency cannot drift.
    // Intentional: unsupported in-range WMAN names missing _zkf_log2_m<WMAN>, prompting table generation.
    `define ZKF_LOG2_TABLE(W) end else if (WMAN == W) begin \
        _zkf_log2_m``W #( \
            .D(DEGREE), .WSB(SBW), \
            .WMULTIPLIER(WMULTIPLIER), \
            .STAGE_PRODUCT(STAGE_PRODUCT), .STAGE_PRODUCT_FINAL(STAGE_PRODUCT_FINAL) \
        ) u_eval ( \
            .clk(clk), .rst(rst), .in_valid(r0_valid), .sb_in(sb_in_l), .v(r0_v), \
            .out_valid(ev_valid), .sb_out(sb_out_l), .l_mag(l_mag), .l_neg(l_neg) \
        );
    // verilog_lint: waive-start generate-label  (macro-expanded selector blocks are intentionally unlabeled)
    generate
        if (1'b0) begin  // seed: the macro opens with "end else if", so every table line is uniform
        `ZKF_LOG2_TABLE(4)
        `ZKF_LOG2_TABLE(5)
        `ZKF_LOG2_TABLE(6)
        `ZKF_LOG2_TABLE(7)
        `ZKF_LOG2_TABLE(8)
        `ZKF_LOG2_TABLE(9)
        `ZKF_LOG2_TABLE(10)
        `ZKF_LOG2_TABLE(11)
        `ZKF_LOG2_TABLE(12)
        `ZKF_LOG2_TABLE(13)
        `ZKF_LOG2_TABLE(14)
        `ZKF_LOG2_TABLE(15)
        `ZKF_LOG2_TABLE(16)
        `ZKF_LOG2_TABLE(17)
        `ZKF_LOG2_TABLE(18)
        `ZKF_LOG2_TABLE(19)
        `ZKF_LOG2_TABLE(20)
        `ZKF_LOG2_TABLE(21)
        `ZKF_LOG2_TABLE(22)
        `ZKF_LOG2_TABLE(23)
        `ZKF_LOG2_TABLE(24)
        `ZKF_LOG2_TABLE(25)
        `ZKF_LOG2_TABLE(26)
        `ZKF_LOG2_TABLE(27)
        `ZKF_LOG2_TABLE(28)
        `ZKF_LOG2_TABLE(29)
        `ZKF_LOG2_TABLE(30)
        `ZKF_LOG2_TABLE(31)
        `ZKF_LOG2_TABLE(32)
        `ZKF_LOG2_TABLE(33)
        `ZKF_LOG2_TABLE(34)
        `ZKF_LOG2_TABLE(35)
        `ZKF_LOG2_TABLE(36)
        `ZKF_LOG2_TABLE(37)
        `ZKF_LOG2_TABLE(38)
        `ZKF_LOG2_TABLE(39)
        `ZKF_LOG2_TABLE(40)
        `ZKF_LOG2_TABLE(41)
        `ZKF_LOG2_TABLE(42)
        `ZKF_LOG2_TABLE(43)
        `ZKF_LOG2_TABLE(44)
        `ZKF_LOG2_TABLE(45)
        `ZKF_LOG2_TABLE(46)
        `ZKF_LOG2_TABLE(47)
        `ZKF_LOG2_TABLE(48)
        `ZKF_LOG2_TABLE(49)
        `ZKF_LOG2_TABLE(50)
        `ZKF_LOG2_TABLE(51)
        `ZKF_LOG2_TABLE(52)
        `ZKF_LOG2_TABLE(53)
        end else begin
            _zkf_invalid_unsupported_table_wman u_invalid();
        end
    endgenerate
    `undef ZKF_LOG2_TABLE
    // verilog_lint: waive-stop generate-label
    wire signed [WE-1:0] e_o       = sb_out_l[SBW-1 -: WE];
    wire                 e_special = sb_out_l[3];
    wire                 e_ssign   = sb_out_l[2];
    wire                 e_pole    = sb_out_l[1];
    wire                 e_de      = sb_out_l[0];

    // -- |R| = |e + log2(m')|, computed directly without a serial add -> result-sign -> wide-abs dependency chain.
    // For finite x, log2(m') in [-1/2, 1/2), so the fractional term cannot flip the sign once |e| >= 1: the result sign
    // is fixed by e alone (and falls back to l_neg only when e == 0), and the magnitude is |e| -/+ |log2(m')| with the
    // direction known up front. So the sign and the add/subtract direction are resolved from the small exponent/sign
    // flags -- off the critical path -- leaving a single wide add/subtract instead of an add feeding a sign-dependent
    // abs (two dependent wide carry chains). Latency-unchanged and bit-identical to taking |(e << F2) + log2(m')|.
    wire                 e_neg   = e_o[WE-1];
    wire                 e_zero  = ~|e_o;
    wire                 r_sign  = e_neg | (e_zero & l_neg);                       // sign(e + log2(m'))
    wire        [WE-1:0] e_mag   = e_neg ? (~e_o + {{(WE-1){1'b0}}, 1'b1}) : e_o;  // |e|, small WE-bit abs
    wire        [WR-1:0] e_sh    = {{(WR-WE){1'b0}}, e_mag} << F2;                 // |e| at scale 2^-F2
    wire        [WR-1:0] l_ext   = {{(WR-(F2+1)){1'b0}}, l_mag};                   // |log2(m')|, zero-extended
    // |e| and |log2(m')| add toward |R| when e and log2(m') share a sign, else subtract. e == 0 forces the add branch:
    // e_sh is 0 there, so the magnitude is exactly l_mag (the subtract branch would wrongly negate it).
    wire                 add_mag = e_zero | ~(e_neg ^ l_neg);
    wire        [WR-1:0] mag_full = add_mag ? (e_sh + l_ext) : (e_sh - l_ext);
    wire     [WNORM-1:0] mag      = mag_full[WNORM-1:0];

    // Resolve the final sign at the P1 input: when the evaluator flagged a special result (+/-inf), the resolved
    // sign is the special-case sign carried in the sideband; otherwise it is the sign of R = e + log2(m').
    wire resolved_sign = e_special ? e_ssign : r_sign;

    // -- Stage P1: register the magnitude, the resolved sign, and the special-case sideband ahead of the
    // _zkf_fixed_to_float helper. Reset only validity; payload free-runs.
    reg                  p1_valid;
    reg      [WNORM-1:0] p1_mag;
    reg                  p1_sign;
    reg                  p1_special;
    reg                  p1_pole;
    reg                  p1_de;
    always @(posedge clk) begin
        if (rst) p1_valid <= 1'b0;
        else     p1_valid <= ev_valid;
        p1_mag     <= mag;
        p1_sign    <= resolved_sign;
        p1_special <= e_special;
        p1_pole    <= e_pole;
        p1_de      <= e_de;
    end

    // -- Normalize, combine, and pack via the shared back-end. The helper owns the _zkf_normshift instance
    // (STAGE_SPLIT = STAGE_NORMALIZE, STAGE_OUTPUT = STAGE_NORMALIZE_OUTPUT), GRS extraction,
    // exp = EXP_OFFSET_LOG2 - shamt, the optional P2 pack-input register (STAGE_PACK forwarded to
    // _zkf_pack.STAGE_INPUT), and the _zkf_pack output. The pole / domain_error flags ride the WSB=2 sideband and
    // emerge in lockstep with y.
    // EXP_OFFSET carries the bias (EXP_IS_BIASED=1): folding +BIAS into this elaboration-time constant removes the
    // packer's runtime bias add from the output cone at no logic cost (exp = BIAS + (WNORM-1-F2) - shamt = biased).
    wire [1:0] sb_out_flags;
    localparam signed [WEU-1:0] EXP_OFFSET_LOG2 = BIAS + WNORM - 1 - F2;
    _zkf_fixed_to_float #(
        .WEXP(WEXP), .WMAN(WMAN),
        .WMAG(WNORM), .WEU(WEU),
        .EXP_IS_BIASED(1),
        .ASSUME_NO_OVERFLOW(1),  // log2(finite>0) is always representable, disable overflow detection circuit
        .WSB(2),
        .STAGE_NORMALIZE(STAGE_NORMALIZE),
        .STAGE_NORMALIZE_OUTPUT(STAGE_NORMALIZE_OUTPUT),
        .STAGE_PACK(STAGE_PACK),
        .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_fixed_to_float (
        .clk(clk), .rst(rst),
        .in_valid(p1_valid),
        .sign(p1_sign),
        .force_zero(1'b0),
        .force_inf(p1_special),
        .exp_offset(EXP_OFFSET_LOG2),
        .mag(p1_mag),
        .sb_in({p1_pole, p1_de}),
        .out_valid(out_valid),
        .y(y),
        .sb_out(sb_out_flags)
    );
    assign pole         = sb_out_flags[1];
    assign domain_error = sb_out_flags[0];
endmodule

`default_nettype wire

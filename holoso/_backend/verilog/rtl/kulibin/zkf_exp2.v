/// Streamed base-2 exponential for the Zubax Kulibin float format: y = 2**x.
/// Zero-bubble, throughput-1, no backpressure.
/// Behavior:
///
///   exp2(-inf)   = +0
///   exp2(+0)     = 1.0
///   exp2(finite) = 2**x, faithfully rounded (within 1 ULP)
///   exp2(+inf)   = +inf
///   tiny finite results follow the zero/MIN_NORMAL boundary rule; overflow maps to +inf
///
/// Algorithm:
///
///  1. Split x = i + f with i = floor(x) and f in [0,1) by shifting the significand by the exponent into a fixed-point.
///
///  2. Then 2**x = 2**f * 2**i, where 2**f in [1,2) is a normalized significand produced by the pipelined per-WMAN
///     table+polynomial core selected by the generate-if below.
///
///  3. The result is packed with exponent i via _zkf_pack, which applies overflow->inf and tiny/MIN_NORMAL boundary.
///
/// The reduction is split across register stages (shift-amount computation, barrel shift, negate) and the evaluator's
/// ROM read is followed by a mandatory fabric register, so no single stage carries both a wide carry chain and a
/// multiply.
///
/// STAGE_PRODUCT selects product computation staging; see _zkf_pmul.
/// WMULTIPLIER optionally hints the native DSP tile argument width; see _zkf_pmul.
/// STAGE_PACK={0,1} forwards to _zkf_pack.STAGE_INPUT, registering the packer's input cone (+1 cycle).
/// STAGE_OUTPUT={0,1} registers the output.

`default_nettype none

module zkf_exp2 #(
    parameter WEXP          = 6,    // exponent field width
    parameter WMAN          = 18,   // significand precision including the hidden bit
    parameter WMULTIPLIER   = 0,    // see _zkf_pmul
    parameter STAGE_INPUT   = 0,    // 0: combinational inputs;   1: latch inputs before any logic, +1 stage
    parameter STAGE_REDUCE  = 0,    // 0: direct fixed->ROM input; 1: register reduced i/f/flags, +1 stage
    parameter STAGE_PRODUCT = 0,    // see _zkf_pmul
    parameter STAGE_PACK    = 0,    // 0: comb pack input; 1: register pack input (+1 stage)
    parameter STAGE_OUTPUT  = 0,    // 0: combinational outputs;  1: registered outputs, +1 stage
    parameter LATENCY       = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] x,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam DEGREE = (((WMAN+18)/11)-1);
    localparam LATENCY_REF = STAGE_INPUT + STAGE_REDUCE + 4 + DEGREE*(2+STAGE_PRODUCT) + STAGE_PACK + STAGE_OUTPUT;
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wman
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        // BIAS / OOR_THRESHOLD constants below use unsized integer shifts on WEXP; WEXP >= 31 would overflow
        // Verilog's 32-bit integer constant arithmetic.
        if (WEXP >= 31) begin : g_invalid_wexp_too_wide
            _zkf_invalid_exp2_wexp_too_wide_unportable u_invalid();
        end
        if ((STAGE_INPUT != 0) && (STAGE_INPUT != 1)) begin : g_invalid_stage_input
            _zkf_invalid_stage_input u_invalid();
        end
        if ((STAGE_REDUCE != 0) && (STAGE_REDUCE != 1)) begin : g_invalid_stage_reduce
            _zkf_invalid_stage_reduce u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    localparam WFRAC = WMAN - 1;
    // FF: fraction bits kept for the reduced argument f. MUST equal the generator's GUARD_FF (zkf_transcendental.py).
    localparam FF   = WMAN + 12;
    localparam WEU  = WEXP;                // signed unbiased exponent fed to _zkf_pack
    localparam SBW  = WEU + 4;             // evaluator sideband: {i, force_inf, force_zero, is_zero, lost_sticky}

    localparam integer BIAS    = (1 << (WEXP - 1)) - 1;
    // |x| >= 2^(WEXP-1) is always out of range. exp >= OOR_THRESHOLD <=> e >= WEXP-1 (also true for +/-inf).
    localparam integer OOR_THRESHOLD = BIAS + WEXP - 1;

    // -- Float -> signed fixed-point reduction. _zkf_to_fixpoint owns the decode, the folded-constant shift
    // predicates (left/right shift selection, left/right overflow clamps), the radix-4 right shifter, the raw left
    // shifter, and the two internal register stages (S1 decode+clamps; S2 post-shift magnitude+sticky+specials).
    // WI=WEU and FF=WMAN+12 give the (i, f) layout we need; OOR_EXP_THRESHOLD=OOR_THRESHOLD makes the helper
    // saturate the magnitude before the integer part of x can exceed WEXP signed bits (the result exponent range).
    wire             rb_valid;
    // The mag bus's high (integer) bits feed i_full below and stay covered through r0_i; the low (fraction) bits
    // feed r0_f directly. rb_guard_unused is structurally zero for FF>0 and intentionally ignored.
    wire [WEU+FF-1:0] rb_mag;
    wire             rb_guard_unused;          // FF>0 -> structurally 0; not consumed
    // Lost-sticky reduction path: rb_lost_sticky asserts only when the float->fixed reduction drops nonzero low bits,
    // which needs a wide exponent (e well below -WMAN).
    wire             rb_lost_sticky;
    wire             rb_sign;
    wire             rb_is_inf_unused;         // folded into rb_oor by the helper
    wire             rb_is_zero;
    wire             rb_oor;
    _zkf_to_fixpoint #(
        .WEXP(WEXP), .WMAN(WMAN),
        .WI(WEU), .FF(FF),
        .STAGE_INPUT(STAGE_INPUT),
        .OOR_EXP_THRESHOLD(OOR_THRESHOLD)
    ) u_to_fixpoint (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(x),
        .out_valid(rb_valid),
        .mag(rb_mag),
        .guard(rb_guard_unused),
        .lost_sticky(rb_lost_sticky),
        .sign(rb_sign),
        .is_inf(rb_is_inf_unused),
        .is_zero(rb_is_zero),
        .oor(rb_oor)
    );
    wire _unused_exp2 = &{1'b0, rb_guard_unused, rb_is_inf_unused, 1'b0};

    // -- RB2 combinational: form the signed two's-complement value, then split into the signed integer part i and
    // the unsigned fraction f. Negating an unsigned magnitude gives correct signed floor semantics for negative x:
    // floor(-3.25) maps to -4 because the slice picks up the sign-extended integer bits. The OOR cases are routed
    // via rb_oor downstream (force_inf for positive overflow, force_zero for negative), so the magnitude and split
    // for those inputs are don't-care.
    wire signed [WEU+FF:0]   v_signed = rb_sign ? (~{1'b0, rb_mag} + {{(WEU+FF){1'b0}}, 1'b1}) : {1'b0, rb_mag};
    wire signed [WEU:0]      i_full   = v_signed[WEU+FF:FF];
    wire [FF-1:0]            f_bits   = v_signed[FF-1:0];
    wire signed [WEU-1:0]    i_clamped     = i_full[WEU-1:0];   // i fits in WEXP signed bits when oor=0
    wire                     force_inf_in  = rb_oor & ~rb_sign; // +inf / positive overflow
    wire                     force_zero_in = rb_oor &  rb_sign; // -inf / negative underflow

    wire                     eval_in_valid;
    wire signed [WEU-1:0]    eval_i;
    wire [FF-1:0]            eval_f;
    wire                     eval_force_inf;
    wire                     eval_force_zero;
    wire                     eval_is_zero;
    wire                     eval_lost_sticky;

    generate
        if (STAGE_REDUCE != 0) begin : g_reduce_stage
            reg                    r0_valid;
            reg signed [WEU-1:0]   r0_i;
            reg [FF-1:0]           r0_f;
            reg                    r0_force_inf;
            reg                    r0_force_zero;
            reg                    r0_is_zero;
            // Registered lost-sticky. At the per-PR STAGE_REDUCE=1 coverage format (w2m16, WEXP<=4) lost_sticky is
            // structurally 0, so this store constant-folds to a line Verilator 5.048 cannot mark covered.
            // verilator coverage_off
            reg                    r0_lost_sticky;
            // verilator coverage_on

            always @(posedge clk) begin
                if (rst) begin
                    r0_valid <= 1'b0;
                end else begin
                    r0_valid <= rb_valid;
                end

                r0_i           <= i_clamped;
                r0_f           <= f_bits;
                r0_force_inf   <= force_inf_in;
                r0_force_zero  <= force_zero_in;
                r0_is_zero     <= rb_is_zero;
                r0_lost_sticky <= rb_lost_sticky;
            end

            assign eval_in_valid    = r0_valid;
            assign eval_i           = r0_i;
            assign eval_f           = r0_f;
            assign eval_force_inf   = r0_force_inf;
            assign eval_force_zero  = r0_force_zero;
            assign eval_is_zero     = r0_is_zero;
            assign eval_lost_sticky = r0_lost_sticky;
        end else begin : g_reduce_direct
            assign eval_in_valid    = rb_valid;
            assign eval_i           = i_clamped;
            assign eval_f           = f_bits;
            assign eval_force_inf   = force_inf_in;
            assign eval_force_zero  = force_zero_in;
            assign eval_is_zero     = rb_is_zero;
            assign eval_lost_sticky = rb_lost_sticky;
        end
    endgenerate

    // -- Pipelined evaluator: 2**f significand + GRS. The sideband {i, force_inf, force_zero, is_zero, lost} is delayed
    // inside the generated evaluator by a plain pipe, aligned to the evaluator output.
    wire [SBW-1:0]  sb_in_e = {
        eval_i, eval_force_inf, eval_force_zero, eval_is_zero, eval_lost_sticky
    };
    wire            ev_valid;
    wire [SBW-1:0]  sb_out_e;   // bit 0 = lost; high bits sliced into e_i/e_finf/e_fzero/e_is_zero below
    wire [WMAN-1:0] eval_sig;
    wire            eval_guard;
    wire            eval_round;
    wire            eval_sticky;
    // The table+polynomial core is pre-generated per WMAN by zkf_transcendental.py as _zkf_exp2_m<WMAN>. We pass the
    // closed-form degree D below; the core asserts it equals the degree its ROM was fitted for (mirrors the LATENCY
    // parameter), so the Horner depth / latency cannot drift.
    // Intentional: unsupported in-range WMAN names missing _zkf_exp2_m<WMAN>, prompting table generation.
    `define ZKF_EXP2_TABLE(W) end else if (WMAN == W) begin \
        _zkf_exp2_m``W #( \
            .D(DEGREE), .WSB(SBW), \
            .WMULTIPLIER(WMULTIPLIER), .STAGE_PRODUCT(STAGE_PRODUCT) \
        ) u_eval ( \
            .clk(clk), .rst(rst), .in_valid(eval_in_valid), .sb_in(sb_in_e), .f(eval_f), \
            .out_valid(ev_valid), .sb_out(sb_out_e), .significand(eval_sig), \
            .guard(eval_guard), .round(eval_round), .sticky(eval_sticky));
    // verilog_lint: waive-start generate-label  (macro-expanded selector blocks are intentionally unlabeled)
    generate
        if (1'b0) begin  // seed: the macro opens with "end else if", so every table line is uniform
        `ZKF_EXP2_TABLE(4)
        `ZKF_EXP2_TABLE(5)
        `ZKF_EXP2_TABLE(6)
        `ZKF_EXP2_TABLE(7)
        `ZKF_EXP2_TABLE(8)
        `ZKF_EXP2_TABLE(9)
        `ZKF_EXP2_TABLE(10)
        `ZKF_EXP2_TABLE(11)
        `ZKF_EXP2_TABLE(12)
        `ZKF_EXP2_TABLE(13)
        `ZKF_EXP2_TABLE(14)
        `ZKF_EXP2_TABLE(15)
        `ZKF_EXP2_TABLE(16)
        `ZKF_EXP2_TABLE(17)
        `ZKF_EXP2_TABLE(18)
        `ZKF_EXP2_TABLE(19)
        `ZKF_EXP2_TABLE(20)
        `ZKF_EXP2_TABLE(21)
        `ZKF_EXP2_TABLE(22)
        `ZKF_EXP2_TABLE(23)
        `ZKF_EXP2_TABLE(24)
        `ZKF_EXP2_TABLE(25)
        `ZKF_EXP2_TABLE(26)
        `ZKF_EXP2_TABLE(27)
        `ZKF_EXP2_TABLE(28)
        `ZKF_EXP2_TABLE(29)
        `ZKF_EXP2_TABLE(30)
        `ZKF_EXP2_TABLE(31)
        `ZKF_EXP2_TABLE(32)
        `ZKF_EXP2_TABLE(33)
        `ZKF_EXP2_TABLE(34)
        `ZKF_EXP2_TABLE(35)
        `ZKF_EXP2_TABLE(36)
        `ZKF_EXP2_TABLE(37)
        `ZKF_EXP2_TABLE(38)
        `ZKF_EXP2_TABLE(39)
        `ZKF_EXP2_TABLE(40)
        `ZKF_EXP2_TABLE(41)
        `ZKF_EXP2_TABLE(42)
        `ZKF_EXP2_TABLE(43)
        `ZKF_EXP2_TABLE(44)
        `ZKF_EXP2_TABLE(45)
        `ZKF_EXP2_TABLE(46)
        `ZKF_EXP2_TABLE(47)
        `ZKF_EXP2_TABLE(48)
        `ZKF_EXP2_TABLE(49)
        `ZKF_EXP2_TABLE(50)
        `ZKF_EXP2_TABLE(51)
        `ZKF_EXP2_TABLE(52)
        `ZKF_EXP2_TABLE(53)
        end else begin
            _zkf_invalid_unsupported_table_wman u_invalid();
        end
    endgenerate
    `undef ZKF_EXP2_TABLE
    // verilog_lint: waive-stop generate-label
    wire signed [WEU-1:0] e_i        = sb_out_e[SBW-1 -: WEU];
    wire                  e_finf     = sb_out_e[3];
    wire                  e_fzero    = sb_out_e[2];
    wire                  e_is_zero  = sb_out_e[1];
    wire                  e_lost     = sb_out_e[0];   // lost-sticky; see rb_lost_sticky above

    // For x == +0, 2**0 = 1.0 (exp_unbiased 0, significand 1.0, no GRS); otherwise 2**f * 2**i.
    wire signed [WEU-1:0] pack_exp = e_is_zero ? {WEU{1'b0}} : e_i;
    wire [WMAN-1:0]       pack_sig = e_is_zero ? {1'b1, {WFRAC{1'b0}}} : eval_sig;
    wire                  pack_g   = e_is_zero ? 1'b0 : eval_guard;
    wire                  pack_r   = e_is_zero ? 1'b0 : eval_round;
    wire                  pack_s   = e_is_zero ? 1'b0 : (eval_sticky | e_lost);

    _zkf_pack #(
        .WEXP(WEXP), .WMAN(WMAN), .WEXP_UNBIASED(WEU),
        .STAGE_INPUT(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_pack (
        .clk(clk),
        .rst(rst),
        .in_valid(ev_valid),
        .sign(1'b0),
        .force_zero(e_fzero),
        .force_inf(e_finf),
        .exp_unbiased(pack_exp),
        .significand(pack_sig),
        .guard(pack_g),
        .round(pack_r),
        .sticky(pack_s),
        .out_valid(out_valid),
        .y(y)
    );
endmodule

`default_nettype wire

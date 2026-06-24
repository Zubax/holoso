/// Iterative (single-transaction) two-argument arctangent and vector magnitude:
///   theta = atan2(y, x) in turns, range (-0.5, 0.5];   mag = hypot(y, x) = sqrt(x*x + y*y)
///
/// This is NOT a throughput-1 pipeline. A transaction is accepted when `in_ready` is high; the module then runs for a
/// fixed data-invariant latency and holds `out_valid` with a stable result until `out_ready` accepts it.
/// LATENCY is accept-to-out_valid latency; with out_ready high, in_ready reasserts one cycle after retirement.
/// Faithful rounding: each finite output is within <= 1 ULP of the correctly-rounded result.
///
/// Behavior (no NaN, only +0; tiny negative results flush to +0):
///   theta = atan2(y, x) / (2*pi)   [turns]; mag = hypot(y, x). Axis/diagonal specials:
///     y=+0,x>0 -> +0 ; y=+0,x<0 -> 1/2 ; y>0,x=0 -> 1/4 ; y<0,x=0 -> -1/4 ; y=0,x=0 -> +0 (mag +0).
///     |y|=inf,x finite -> +-1/4 ; x=+inf,y finite -> +-0 -> +0 ; x=-inf,y finite -> +1/2 ;
///     (inf,inf) -> +-1/8 (x>0) / +-3/8 (x<0).  mag = +inf whenever any input is inf.
/// mag overflow is faithfully rounded, not hard-clamped: a finite-input hypot whose true value lands within 1 ULP of
/// the overflow threshold may round to max-finite rather than +inf (the <= 1 ULP bound above). +inf is never produced
/// for an in-range result, so the rounding only ever errs toward finite -- there is no spurious infinity.
///
/// Algorithm (vectoring mixed CORDIC; the engine is in _zkf_cordic):
///
///  1. Decode (y, x); order den=max(|x|,|y|), num=min; align num down to den's binade. The vector is seeded into the
///     engine pre-scaled by 1/4 (den in [0.25, 0.5)*2**XF) so the CORDIC magnitude growth x_K = gain*hypot stays
///     inside the shared engine's signed width WX (which sincos sizes for a ~1*2**XF rotated magnitude).
///     a0 = atan(num/den).
///
///  2. Run N vectoring iterations driving y -> 0: z_K ~= a0 (turns, 2**-ZF), x_K ~= gain*hypot, y_K ~= 0.
///
///  3. Finish the small angle with ONE division: a0 = z_K + (y_K/x_K)*INV_TAU (the vectoring analogue of the sincos
///     linear-rotation multiply). A small-ratio bypass handles the near-+x-axis corner where theta underflows the
///     fixed-turns accumulator: theta = (|y|/|x|)*INV_TAU computed directly as a float (single rounding).
///
///  4. Octant/quadrant unmap from (sign x, sign y, swap) places theta in (-0.5, 0.5]; the magnitude descales x_K by
///     1/gain (== the per-WMAN KINV) and carries the den binade. One shared _zkf_fixed_to_float back-end
///     renormalizes + rounds both results (magnitude then theta of the single in-flight transaction, time-multiplexed).
///
/// Arithmetic reuse: ONE folded radix-4 divider (the _zkf_div_core primitives _zkf_div_radix4_step/_zkf_div_raw_stage,
/// STEPS = ceil(XF/2) cycles, 2 quotient bits each) computes Q = floor(num*2**F/den) + sticky for BOTH the residual
/// (num=|y_K|, den=x_K) and the bypass (num=sig_y, den=sig_x); they are mutually exclusive per transaction. ONE shared
/// _zkf_pmul computes the magnitude product x_K*KINV (issued DURING the divide, so it costs no latency) and the
/// post-divide Q*INV_TAU (the residual correction, and the bypass theta -- single-rounded via the divide sticky).
///
/// Tuning knobs follow zkf_sincos. UNROLL100 is forwarded to _zkf_cordic. STAGE_INPUT latches the inputs (+1 cycle);
/// STAGE_PRODUCT / WMULTIPLIER tune the shared _zkf_pmul (depth 1+STAGE_PRODUCT; WMULTIPLIER sizes the DSP-tile grid);
/// STAGE_NORMALIZE / STAGE_PACK tune the one shared _zkf_fixed_to_float back-end.
/// STAGE_OUTPUT registers the public theta/mag/out_valid outputs.

`default_nettype none

`define ZKF_ATAN2_N         (((WMAN+1)/2)+1)
`define ZKF_ATAN2_XF        (((3*WMAN+1)/2)+6)
`define ZKF_ATAN2_XYCYC     ((`ZKF_ATAN2_N*100 + UNROLL100 - 1)/UNROLL100)
`define ZKF_ATAN2_STEPS     ((`ZKF_ATAN2_XF + 1)/2)
`define ZKF_ATAN2_DIVCYC    (`ZKF_ATAN2_STEPS + 1)
`define ZKF_ATAN2_BASE      8
`define ZKF_ATAN2_LATENCY \
    (`ZKF_ATAN2_BASE + STAGE_INPUT + `ZKF_ATAN2_XYCYC + `ZKF_ATAN2_DIVCYC + STAGE_PRODUCT \
        + STAGE_NORMALIZE + STAGE_PACK + STAGE_OUTPUT)

module zkf_atan2 #(
    parameter WEXP            = 6,      // exponent field width
    parameter WMAN            = 18,     // significand precision including the hidden bit
    parameter WMULTIPLIER     = 0,      // optional native DSP tile operand width for tighter optimization
    parameter UNROLL100       = 100,
    parameter STAGE_INPUT     = 0,
    parameter STAGE_PRODUCT   = 0,      // shared-multiplier pipeline depth (1+STAGE_PRODUCT cycles)
    parameter STAGE_NORMALIZE = 0,
    parameter STAGE_PACK      = 0,
    parameter STAGE_OUTPUT    = 0,
    parameter LATENCY         = `ZKF_ATAN2_LATENCY  // accept-to-out_valid cycles; checked below
) (
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 in_valid,    // start a transaction (sampled only when in_ready)
    output wire                 in_ready,    // high when idle and able to accept a transaction
    input  wire [WEXP+WMAN-1:0] y,           // first argument (numerator / opposite)
    input  wire [WEXP+WMAN-1:0] x,           // second argument (denominator / adjacent)

    output wire                 out_valid,   // result is ready; held until out_ready (back-pressure)
    input  wire                 out_ready,   // consumer accepts the result on a cycle where out_valid & out_ready
    output wire [WEXP+WMAN-1:0] theta,       // atan2(y, x) in turns, range (-0.5, 0.5]
    output wire [WEXP+WMAN-1:0] mag          // hypot(y, x)
);
    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;
    localparam WE    = WEXP + 1;
    // CORDIC geometry MUST match zkf_trig.py (n_atan2, GUARD_*, XF/ZF) and the per-WMAN _zkf_cordic_m tables.
    // N here is atan2's iteration count (GUARD_ITER_ATAN2 == 1); XF is atan2's x/y width (GUARD_XY == 6,
    // the shared-engine guard -- atan2's theta accuracy is XF-bound and sets this floor) and MUST equal the table's
    // baked WX-2 (the engine's x0/y0/xn/yn ports are sized by the table), so the engine and divider share one width.
    localparam integer GUARD_FF   = (12 > (WMAN / 2 + 2)) ? 12 : (WMAN / 2 + 2);
    localparam integer GUARD_ZF   = 6;
    localparam integer GUARD_Z    = 3;
    localparam integer GUARD_DIV  = 8;                  // residual/bypass quotient guard (mirrors zkf_trig GUARD_DIV)
    localparam integer FF   = WMAN + GUARD_FF;
    localparam integer WT   = FF - 2;
    localparam integer N    = `ZKF_ATAN2_N;
    localparam integer XF   = `ZKF_ATAN2_XF;            // x/y fractional scale
    localparam integer WX   = XF + 2;                   // signed x/y width (engine)
    localparam integer ZF   = WT + 2 + GUARD_ZF;        // angle (turns) fractional scale
    localparam integer WZ   = ZF + GUARD_Z;             // signed angle width (engine)
    // The shared _zkf_pmul multiplies x_K*KINV (MAG) and Q*INV_TAU (residual correction AND bypass theta). Both
    // constants arrive PRE-NARROWED from the per-WMAN table at their native fixed-point scales: kinv_mag at scale
    // 2**-KINV_S and inv_tau at scale 2**-INVTAU_S, each round-narrowed to WMAN+5 bits. Connecting these narrowed
    // operands directly keeps the multiply's `b` operand below one DSP column (4x4 -> 4x3 at WMAN=36) and the shared
    // back-end width WMAG minimal, and every dependent scaling derives its shift / exp-offset from the constant's own
    // scale -- "product-scale minus target-scale", no correction token. MUST match the table's INVTAU_S / KINV_S.
    localparam integer ITWB     = WMAN + 5;             // narrowed inv_tau width (== the table's inv_tau port width)
    localparam integer KINV_MAG = WMAN + 5;            // narrowed kinv_mag width (== the table's kinv_mag port width)
    localparam integer INVTAU_S = WMAN + 7;            // native scale of the narrowed inv_tau
    localparam integer KINV_S   = WMAN + 5;            // native scale of the narrowed kinv_mag
    localparam integer BIAS     = (1 << (WEXP - 1)) - 1;

    // Folded radix-4 divider geometry: STEPS cycles (2 quotient bits each) producing Q = floor(num*2**F/den), where
    // the F = 2*STEPS >= XF fractional bits give the residual its ~wman+(N-1) significant quotient bits. WDIV is the
    // operand/remainder width (residual divisor x_K < 2**(XF+1); the bypass divisor sig_x zero-extends in).
    localparam integer STEPS = `ZKF_ATAN2_STEPS;
    localparam integer F     = 2 * STEPS;               // quotient fractional bits
    localparam integer WDIV  = WX;                      // divider operand / remainder width
    // The residual quotient (|y_K| < x_K) is < 1 so its integer bit is structurally 0, but the BYPASS divides
    // SIGNIFICANDS sig_y/sig_x which can reach ~2 (sig_y<2**WMAN, sig_x>=2**(WMAN-1)), so its integer bit can be 1.
    // WQUO therefore keeps the integer bit (F fractional bits + 1 integer bit); the bypass seeds it via div_ibit.
    localparam integer WQUO  = F + 1;                   // quotient: integer bit (bypass) + F fractional bits
    localparam integer WCNT  = $clog2(STEPS + 1);

    // The shared _zkf_fixed_to_float back-end uses one magnitude width sized to the widest pre-narrowed product.
    // The MAG product drops the CORDIC x_K sign bit because finite vectoring outputs are strictly positive; the QT
    // product still keeps WQUO bits. With the WMAN+5-bit narrowed operands this is the minimal width (102 vs the
    // full-precision 2*XF+4 == 124 at WMAN=36).
    localparam integer WA_MAG   = WX - 1;
    localparam integer WA_MUL   = (WA_MAG > WQUO) ? WA_MAG : WQUO;
    localparam integer WMAG_MAG = WA_MAG + KINV_MAG;
    localparam integer WMAG_QT  = WQUO + ITWB;
    localparam integer WMAG_TH  = WZ + 2;
    localparam integer WMAG_AB  = (WMAG_MAG > WMAG_QT) ? WMAG_MAG : WMAG_QT;
    localparam integer WMAG     = (WMAG_AB > WMAG_TH) ? WMAG_AB : WMAG_TH;
    localparam integer WEU   = WEXP + $clog2(WMAG + 1) + 3;

    // The CORDIC sideband is only a dummy bit. Input-derived metadata is captured into hd_* for the whole single
    // in-flight transaction (busy blocks the next accept until output retirement), so the back-end can read it directly
    // when cd_done arms the divider instead of replicating it through every CORDIC stage.
    localparam integer WSB         = 1;

    // Octant-local turns constants (scale 2**-ZF) for the quadrant unmap, in a WZ+2 signed container.
    localparam signed [WZ+1:0] QUARTER = {{(WZ+2-(ZF-1)){1'b0}}, 1'b1, {(ZF-2){1'b0}}}; // 1/4 turn
    localparam signed [WZ+1:0] HALF    = {{(WZ+2-ZF){1'b0}}, 1'b1, {(ZF-1){1'b0}}};     // 1/2 turn

    generate
        if ((WEXP < 2) || (WMAN < 4) || (WEXP >= 31)) begin : g_invalid_wexp_or_wman
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if ((STAGE_INPUT != 0) && (STAGE_INPUT != 1)) begin : g_invalid_stage_input
            _zkf_invalid_stage_input u_invalid();
        end
        if ((STAGE_OUTPUT != 0) && (STAGE_OUTPUT != 1)) begin : g_invalid_stage_output
            _zkf_invalid_stage_output u_invalid();
        end
        if (LATENCY != `ZKF_ATAN2_LATENCY) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    // k/8 of a turn (k in {0:+0, 1:1/8, 2:1/4, 3:3/8, 4:+1/2}) with sign s, correctly rounded to ZKF. The half-turn
    // endpoint is canonical +1/2; signed -1/2 is normalized there. The constants are exact dyadics, but for very small
    // WEXP they can underflow the normal range -- so this mirrors
    // round_fraction_to_zkf / _zkf_pack: a normal result when the biased exponent is >= 1, else MIN_NORMAL when it is
    // exactly 0 (each k/8 is a normalized 1.f, so biased == 0 means the value is in [0.5*MIN_NORMAL, MIN_NORMAL)
    // and rounds up), else flush to +0. (For WEXP >= 4 all four constants are normal, so only the tiny WEXP {2,3}
    // ever reach the underflow branches.)
    //
    // Each octant's unbiased exponent (eunb) and fraction (f) are compile-time literals, so the biased exponent and
    // the >=1 / ==0 underflow resolution are ELABORATION-TIME constants -- precomputed here as the packed {exp, frac}
    // body (TURN8_K*) plus a +0 flag (TURN8_Z*). turn8() is then a pure 5-way select of a constant body with the
    // runtime sign ORed in: no runtime adder/comparator, so no carry chain on whatever cone evaluates it (it is
    // evaluated at the output stage). The bodies mirror the (eunb, f) table: 1/8 -> eunb -3; 1/4, 3/8 -> -2
    // (3/8 sets the top frac bit); 1/2 -> -1. TURN8_BODY(eunb, f) packs the normal field when biased>=1,
    // MIN_NORMAL when biased==0, else marks +0. A k/8 body, packed {exp[WEXP-1:0], frac[WFRAC-1:0]} into WFULL-1
    // bits, computed at elaboration from constants: biased exponent EB = EUNB + BIAS (EUNB the octant's literal
    // unbiased exponent), MANTHI = (1<<(WFRAC-1)) for 3/8 (which carries the leading fraction bit) else 0.
    // Normal when EB>=1: body = EB*2**WFRAC + MANTHI (no field overlap, so the multiply-add equals the
    // {exp, frac} concatenation). When EB==0: MIN_NORMAL (exp field == 1, frac 0). The body is unused when the value
    // underflows (EB<0), flagged by TURN8_Z* so turn8() returns +0 instead. TURN8_ONE is a SIZED (WFULL-1-bit)
    // constant 1, so `TURN8_ONE << WFRAC` is a WFULL-1-bit shift (== 2**WFRAC). An unsized `1 << WFRAC` is
    // self-determined to 32 bits and overflows to 0 for WFRAC >= 32 (WMAN >= 33: 36/48/53): a tool-dependent
    // elaboration hazard (some elaborators widen the literal, others fold it to 0) -- so size it explicitly.
    localparam [WFULL-2:0] TURN8_ONE = {{(WFULL-2){1'b0}}, 1'b1};
    `define TURN8_EB(EUNB)      ((EUNB) + BIAS)
    `define TURN8_BODY(EUNB, MANTHI) \
        ((`TURN8_EB(EUNB) >= 1) ? (`TURN8_EB(EUNB) * (TURN8_ONE << WFRAC) + (MANTHI)) \
                                : (TURN8_ONE << WFRAC))   /* EB==0 -> MIN_NORMAL: exp field 1, frac 0 (== 2**WFRAC) */
    `define TURN8_ISZERO(EUNB)  (`TURN8_EB(EUNB) < 0)
    localparam [WFULL-2:0] TURN8_K1  = `TURN8_BODY(-3, 0);                          // 1/8
    localparam [WFULL-2:0] TURN8_K2  = `TURN8_BODY(-2, 0);                          // 1/4
    localparam [WFULL-2:0] TURN8_K3  = `TURN8_BODY(-2, (TURN8_ONE << (WFRAC-1)));   // 3/8 (leading fraction bit set)
    localparam [WFULL-2:0] TURN8_K4  = `TURN8_BODY(-1, 0);                          // 1/2
    localparam [WFULL-2:0] TURN8_KNZ = `TURN8_BODY(0, 0);                           // eunb 0 (==1.0): unused k>=5
    localparam             TURN8_Z1  = `TURN8_ISZERO(-3);
    localparam             TURN8_Z2  = `TURN8_ISZERO(-2);
    localparam             TURN8_Z3  = `TURN8_ISZERO(-2);
    localparam             TURN8_Z4  = `TURN8_ISZERO(-1);
    `undef TURN8_EB
    `undef TURN8_BODY
    `undef TURN8_ISZERO
    // Pure 5-way select of the precomputed constant bodies; no runtime add/compare -> no carry chain on the evaluating
    // cone. k == 0 -> +0; k in 1..4 -> the octant body (or +0 when it underflowed, via TURN8_Z*); k == 4 forces the
    // sign clear because the half-turn endpoint is canonical +1/2; the unused k >= 5 codes mirror the former eunb == 0
    // fall-through (TURN8_KNZ, never +0 for WEXP >= 2).
    function automatic [WFULL-1:0] turn8;
        input               s;
        input [2:0]         k;
        reg [WFULL-2:0]     body;          // packed {exp, frac}, selected per k from the precomputed constants
        reg                 zero;          // result is +0 (k == 0, or the constant underflowed the normal range)
        begin
            case (k)
                3'd0:    begin body = {(WFULL-1){1'b0}}; zero = 1'b1; end   // +0
                3'd1:    begin body = TURN8_K1;  zero = TURN8_Z1; end
                3'd2:    begin body = TURN8_K2;  zero = TURN8_Z2; end
                3'd3:    begin body = TURN8_K3;  zero = TURN8_Z3; end
                3'd4:    begin body = TURN8_K4;  zero = TURN8_Z4; end
                // k>=5 never occurs (turn8 indices are 0..4); generate-completeness arm.
                // verilator coverage_off
                default: begin body = TURN8_KNZ; zero = 1'b0;     end       // unused k >= 5 (mirrors old eunb 0)
                // verilator coverage_on
            endcase
            turn8 = zero ? {WFULL{1'b0}} : {(k == 3'd4) ? 1'b0 : s, body};
        end
    endfunction

    // ================================================================================================================
    // Front-end: accept and decode ONCE; derive the seed, the bypass operands (sig_y, sig_x), and the special-case
    // results -- all functions of the inputs -- then capture that metadata into hd_* for the back-end.
    // ================================================================================================================
    reg  busy;
    wire accept = in_valid & in_ready;
    assign in_ready = ~busy;

    // Per-WMAN pre-narrowed constant outputs from the CORDIC table (driven by the engine instance below; used at the
    // back-end). inv_tau (turns scaling) and kinv_mag (magnitude descale) each arrive at WMAN+5 bits at their native
    // scale -- connected straight to the shared pmul's `b` operand, no re-narrowing here. atan2 leaves the table's
    // const2pi (sin/cos only) and full-precision kinv (the sin/cos seed) unconnected.
    wire [ITWB-1:0]      eng_inv_tau;
    wire [KINV_MAG-1:0]  eng_kinv_mag;

    wire             si_valid;
    wire [WFULL-1:0] si_y, si_x;
    generate
        if (STAGE_INPUT == 0) begin : g_in_comb
            assign si_valid = accept; assign si_y = y; assign si_x = x;
        end else begin : g_in_reg
            reg             r_v;
            reg [WFULL-1:0] r_y, r_x;
            always @(posedge clk) begin
                if (rst) r_v <= 1'b0; else r_v <= accept;
                r_y <= y; r_x <= x;
            end
            assign si_valid = r_v; assign si_y = r_y; assign si_x = r_x;
        end
    endgenerate

    // -- Stage F0/D0: the |x|-vs-|y| magnitude compare, split into half-width compares so the wide carry chain (the
    // front-end's closure-critical cone) is pipelined across the D0 register. The magnitude key is the input's low WK
    // bits (sign cleared): {biased exp, fraction}, whose unsigned ordering IS the magnitude ordering (the hidden bit
    // is implicit and equal). swap = |y| > |x| = keyy > keyx = hi_gt | (hi_eq & lo_gt).
    localparam integer WK  = WFULL - 1;                  // magnitude key width (biased exponent + fraction)
    localparam integer WKL = WK / 2;
    localparam integer WKH = WK - WKL;
    wire [WK-1:0]    f0_keyx = si_x[WK-1:0];
    wire [WK-1:0]    f0_keyy = si_y[WK-1:0];
    wire             f0_hi_gt = f0_keyy[WK-1 -: WKH] >  f0_keyx[WK-1 -: WKH];
    wire             f0_hi_eq = f0_keyy[WK-1 -: WKH] == f0_keyx[WK-1 -: WKH];
    wire             f0_lo_gt = f0_keyy[WKL-1:0]     >  f0_keyx[WKL-1:0];
    // The swap decision (|y| > |x|) is resolved HERE, behind the wide half-compares, and registered into D0 so it is a
    // ready register output in D1 -- one fewer LUT level (the OR/AND combine) on the D1 order/align/bypass cone,
    // which is the front-end's closure-critical path on the wide config.
    wire             f0_swap  = f0_hi_gt | (f0_hi_eq & f0_lo_gt);
    // Per-operand class flags (each a single-operand reduction of one input's exponent field) -- registered into D0
    // as a NARROW descriptor. The special-case theta (turn8) and magnitude are then built in D1 from these few
    // registered bits, so neither the turn8 carry chain nor a wide |x|/|y| select sits on the cone that reads the
    // wide inputs.
    wire             f0_sx = si_x[WFULL-1], f0_sy = si_y[WFULL-1];
    wire [WEXP-1:0]  f0_xe = si_x[WFULL-2:WFRAC], f0_ye = si_y[WFULL-2:WFRAC];
    wire             f0_xz = ~|f0_xe, f0_xi = &f0_xe, f0_yz = ~|f0_ye, f0_yi = &f0_ye;
    reg              d0_valid, d0_swap;
    reg [WFULL-1:0]  d0_y, d0_x;
    reg              d0_sx, d0_sy, d0_xz, d0_xi, d0_yz, d0_yi;          // narrow special-case descriptor
    always @(posedge clk) begin
        if (rst) d0_valid <= 1'b0; else d0_valid <= si_valid;
        d0_swap <= f0_swap;
        d0_y <= si_y; d0_x <= si_x;
        d0_sx <= f0_sx; d0_sy <= f0_sy;
        d0_xz <= f0_xz; d0_xi <= f0_xi; d0_yz <= f0_yz; d0_yi <= f0_yi;
    end

    // -- Stage D1: field-extract the (registered) operands, finish the compare, order den/num, derive the alignment
    // shift, the bypass operands, and the special-case results. The wide compare is already done (stage 0), so every
    // cone here is short.
    // Field-extract inline (the former _zkf_atan2_decode submodule was a thin per-operand decode whose sign/class
    // outputs were already covered by the D0 narrow descriptor d0_sx/d0_sy/d0_xz/...): the significand is just the
    // hidden bit prepended to the fraction, and the biased exponent is the raw field. The mag exponent (f1_eden) is
    // carried BIASED here -- the former decode's -BIAS unbias subtract is dropped because the only consumer, the
    // p2_mexp offset add, folds the +BIAS back in (so the two cancel and the packed result is bit-identical).
    wire [WMAN-1:0]  f1_sigx = {1'b1, d0_x[WFRAC-1:0]};
    wire [WMAN-1:0]  f1_sigy = {1'b1, d0_y[WFRAC-1:0]};
    wire             f1_special = d0_xz | d0_xi | d0_yz | d0_yi;         // any operand zero/inf (from the descriptor)
    wire             f1_swap    = d0_swap;                              // |y| > |x| (resolved + registered in D0)
    wire [WMAN-1:0]  f1_den_sig = f1_swap ? f1_sigy : f1_sigx;          // max(|x|,|y|) significand (the bypass divisor)
    wire [WMAN-1:0]  f1_num_sig = f1_swap ? f1_sigx : f1_sigy;          // min significand (the bypass dividend)
    // larger operand's BIASED exponent (for mag); zero-extended into the signed WE container (biased >= 0 always)
    wire signed [WE-1:0] f1_eden = $signed({1'b0, (f1_swap ? d0_y[WFULL-2:WFRAC] : d0_x[WFULL-2:WFRAC])});
    // alignment / bypass: shift_dn = |ey - ex| (>= 0 by construction). Computed straight from the BIASED exponent
    // fields (ye - xe == ey - ex) so the seed-critical chain is one subtract, NOT the decode's unbias subtract THEN
    // this one -- the unbiased ex/ey are only for f1_eden (the mag exponent, off the seed path). The clamp/threshold
    // compares use a NARROW signed width WCMP (enough for WX+XF and the exponent range, + sign), NOT 32 bits,
    // so the carry chains stay short. WCMP >= WSH+1, so the shamt slice is valid even when WE+1 < WSH (small WEXP).
    //
    // Timing: BOTH exponent-difference orderings -- xe-ye and ye-xe -- and their clamped shift amounts are formed in
    // parallel (each a narrow WCMP subtract + compare), then swap selects the final shamt. So swap drives only a narrow
    // late mux instead of feeding the subtract's operands, taking the WCMP carry chain off the swap->subtract->compare
    // cone that set the D1 critical path. The selected difference is |ey-ex| (swap=>ye>=xe, ~swap=>xe>=ye, each >= 0).
    // The bypass uses the ~swap ordering directly (xe-ye, valid since ~swap => xe>=ye). f1_eydiff (the SIGNED ey-ex,
    // needed only for the bypass exponent) is computed in parallel, off this critical cone.
    wire [WEXP-1:0] f1_xe = d0_x[WFULL-2:WFRAC];
    wire [WEXP-1:0] f1_ye = d0_y[WFULL-2:WFRAC];
    localparam integer WSH  = $clog2(WX + XF + 1);
    localparam integer WCMP = (((WE + 1) > (WSH + 1)) ? (WE + 1) : (WSH + 1)) + 1;
    localparam signed [WCMP-1:0] CLAMP_C = WX + XF;
    localparam signed [WCMP-1:0] TINY_C  = ZF - WMAN - GUARD_DIV;
    wire signed [WE:0]     f1_eydiff = $signed({2'b00, f1_ye}) - $signed({2'b00, f1_xe});   // ey-ex (biased)
    // Both orderings of |ey-ex| in an unsigned WCMP container; only the one selected by swap is nonnegative/meaningful.
    wire [WCMP-1:0]        f1_xe_c     = {{(WCMP - WEXP){1'b0}}, f1_xe};
    wire [WCMP-1:0]        f1_ye_c     = {{(WCMP - WEXP){1'b0}}, f1_ye};
    wire [WCMP-1:0]        f1_shift_xy = f1_xe_c - f1_ye_c;                  // ~swap ordering xe-ye (>= 0 when ~swap)
    wire [WCMP-1:0]        f1_shift_yx = f1_ye_c - f1_xe_c;                  //  swap ordering ye-xe (>= 0 when  swap)
    localparam [WSH-1:0] SHCLAMP = WX + XF;                                            // shamt clamp = datapath width
    wire [WSH-1:0] f1_shamt_xy = (f1_shift_xy > CLAMP_C) ? SHCLAMP : f1_shift_xy[WSH-1:0];
    wire [WSH-1:0] f1_shamt_yx = (f1_shift_yx > CLAMP_C) ? SHCLAMP : f1_shift_yx[WSH-1:0];
    wire [WSH-1:0] f1_shamt    = f1_swap ? f1_shamt_yx : f1_shamt_xy;                  // == clamp(|ey-ex|)
    wire           f1_bypass   = (~f1_swap) & (~d0_sx) & (~f1_special) & (f1_shift_xy > TINY_C);
    // Bypass initial remainder / integer bit, precomputed here (the bypass only fires when ~swap, where the ordered
    // significands ARE sig_y/sig_x), registered into D, then captured into hd_* at the seed boundary. Neither the
    // divider's one-shot arm cone nor the F2 seed cone carries this WMAN compare+subtract. One subtract yields both the
    // compare (borrow == sig_y < sig_x) and difference.
    // The bypass divides SIGNIFICANDS sig_y/sig_x (the exponent difference is applied later via the texp offset), and
    // sig_y can exceed sig_x (both are in [2**WFRAC, 2**WMAN)), so the integer quotient bit is genuinely 1 when
    // sig_y >= sig_x -- it is NOT structurally 0 (the small VALUE ratio |y|/|x| << 1 lives in the exponent, not here).
    wire [WMAN:0]    f1_byp_diff = {1'b0, f1_num_sig} - {1'b0, f1_den_sig};            // {borrow, sig_y - sig_x}
    wire             f1_byp_ibit = ~f1_byp_diff[WMAN];                                 // sig_y >= sig_x
    wire [WMAN-1:0]  f1_byp_irem = f1_byp_ibit ? f1_byp_diff[WMAN-1:0] : f1_num_sig;   // bypass initial remainder

    // Special-case theta / mag, built from narrow descriptors: turn8 reads only the 3-bit class code spk (no wide
    // operand read), and the magnitude REUSES the already-ordered denominator -- for the axis specials the nonzero
    // operand IS the denominator (den = max(|x|,|y|)), so |x|/|y| is {den_exp, den_sig} with no second wide mux.
    wire [2:0]       f1_spk = (d0_xi & d0_yi) ? (d0_sx ? 3'd3 : 3'd1)   // (inf,inf): 3/8 (x<0) or 1/8 (x>0)
                            : d0_yi           ? 3'd2                     // |y|=inf -> 1/4
                            : d0_xi           ? (d0_sx ? 3'd4 : 3'd0)    // x=-inf -> 1/2 ; x=+inf -> +0
                            : (d0_xz & d0_yz) ? 3'd0                     // atan2(0, 0) -> +0 (before the y=0 rule)
                            : d0_yz           ? (d0_sx ? 3'd4 : 3'd0)    // y=0, x!=0 -> 1/2 (x<0) or +0 (x>0)
                            : d0_xz           ? 3'd2                     // x=0, y!=0 -> 1/4
                            :                   3'd0;
    wire             f1_sp_sign  = d0_yz ? 1'b0 : d0_sy;
    // turn8(f1_sp_sign, f1_spk) is NOT assembled here -- the descriptor (spk, sp_sign) is carried narrow and turn8 is
    // evaluated at the output stage, keeping its biased-add carry chain off the front-end's D-register cone.
    wire [WEXP-1:0]  f1_den_exp  = f1_swap ? f1_ye : f1_xe;             // biased exponent of the larger operand
    wire [WFULL-1:0] f1_sp_mag   = (d0_xi | d0_yi) ? {1'b0, {WEXP{1'b1}}, {WFRAC{1'b0}}}        // +inf
                                 : (d0_xz & d0_yz) ? {WFULL{1'b0}}                               // +0
                                 :                   {1'b0, f1_den_exp, f1_den_sig[WFRAC-1:0]};  // |larger|

    // -- Stage D: register the ordered/aligned quantities. Splits the (now short) decode + den/num order cone from the
    // align/seed cone (the wide variable barrel shift below).
    reg                  d_valid;
    reg [WMAN-1:0]       d_den_sig, d_num_sig;
    reg [WSH-1:0]        d_shamt;
    reg                  d_swap, d_sx, d_sy, d_special, d_bypass;
    reg signed [WE-1:0]  d_eden;
    reg signed [WE:0]    d_eydiff;
    reg [WFULL-1:0]      d_sp_mag;
    reg [2:0]            d_spk;                                   // narrow special-case octant code (-> turn8 @ out)
    reg                  d_sp_sign;                               // narrow special-case theta sign (-> turn8 @ out)
    reg [WMAN-1:0]       d_byp_irem;                              // precomputed bypass initial remainder
    reg                  d_byp_ibit;                              // precomputed bypass integer quotient bit
    always @(posedge clk) begin
        if (rst) d_valid <= 1'b0; else d_valid <= d0_valid;
        d_den_sig <= f1_den_sig; d_num_sig <= f1_num_sig; d_shamt <= f1_shamt;
        d_swap <= f1_swap; d_sx <= d0_sx; d_sy <= d0_sy; d_special <= f1_special; d_bypass <= f1_bypass;
        d_eden <= f1_eden; d_eydiff <= f1_eydiff;
        d_spk <= f1_spk; d_sp_sign <= f1_sp_sign; d_sp_mag <= f1_sp_mag;
        d_byp_irem <= f1_byp_irem; d_byp_ibit <= f1_byp_ibit;
    end

    // Held input-derived metadata for the single in-flight transaction. The D-stage payload free-runs after d_valid,
    // while the CORDIC runs for many cycles, so capture the fields once at the seed boundary instead of carrying them
    // through every CORDIC stage.
    reg                  hd_swap, hd_sx, hd_sy, hd_special, hd_bypass;
    reg signed [WE-1:0]  hd_eden;
    reg signed [WE:0]    hd_eydiff;
    reg [WFULL-1:0]      hd_sp_mag;
    reg [2:0]            hd_spk;
    reg                  hd_sp_sign;
    reg [WMAN-1:0]       hd_den_sig;
    reg [WMAN-1:0]       hd_byp_irem;
    reg                  hd_byp_ibit;
    always @(posedge clk) begin
        if (d_valid) begin
            hd_den_sig  <= d_den_sig;
            hd_swap     <= d_swap;
            hd_sx       <= d_sx;
            hd_sy       <= d_sy;
            hd_special  <= d_special;
            hd_bypass   <= d_bypass;
            hd_eden     <= d_eden;
            hd_eydiff   <= d_eydiff;
            hd_spk      <= d_spk;
            hd_sp_sign  <= d_sp_sign;
            hd_sp_mag   <= d_sp_mag;
            hd_byp_irem <= d_byp_irem;
            hd_byp_ibit <= d_byp_ibit;
        end
    end

    // -- Stage F2: the engine seed (den pre-scaled by 1/4 to [0.25,0.5)*2**XF; num aligned down to den's binade by the
    // wide variable shift). The input-derived metadata is captured into hd_* alongside this seed; single-in-flight
    // operation keeps that bank stable until cd_done arms the divider. f2_num_up is the 1/4-pre-scaled numerator
    // significand before the alignment right-shift.
    // d_num_sig (WMAN bits, top at WFRAC) shifted up by (XF-WFRAC-2) tops out at bit XF-2, so it fits in
    // WX (== XF+2) bits with room to spare; the former WX+XF width carried XF dead high bits
    // (only the low WX were ever read). Sizing it to WX drops them.
    wire [WX-1:0]    f2_den_fix = {{(WX-WMAN){1'b0}}, d_den_sig} << (XF - WFRAC - 2);
    wire [WX-1:0]    f2_num_up  = {{(WX-WMAN){1'b0}}, d_num_sig} << (XF - WFRAC - 2);
    wire [WX-1:0]    f2_num_fix = f2_num_up >> d_shamt;
    reg                  f2_valid;
    reg signed [WX-1:0]  f2_x0, f2_y0;
    always @(posedge clk) begin
        if (rst) f2_valid <= 1'b0; else f2_valid <= d_valid;
        f2_x0 <= $signed({1'b0, f2_den_fix});
        f2_y0 <= $signed({1'b0, f2_num_fix});
    end
    wire eng_start = f2_valid;

    // ================================================================================================================
    // Vectoring CORDIC engine (MODE=1), per-WMAN table.
    // ================================================================================================================
    wire                 cd_done;
    wire [WSB-1:0]       cd_sb_unused;
    wire signed [WX-1:0] cd_xn, cd_yn;
    wire signed [WZ-1:0] cd_zn;
    // Vectoring is always lock-step (the engine's decoupled z-path requires MODE=0), so PARALLEL is hardwired to 0.
    `define ZKF_ATAN2_CORE(W) end else if (WMAN == W) begin : g_m``W \
        _zkf_cordic_m``W #( \
            .MODE(1), .UNROLL100(UNROLL100), .PARALLEL(0), .WSB(WSB), \
            .EXPECT_WMAN(WMAN), .EXPECT_N(N), .EXPECT_XF(XF), .EXPECT_WX(WX), .EXPECT_WT(WT), \
            .EXPECT_ZF(ZF), .EXPECT_WZ(WZ), \
            .EXPECT_INVTAU_W(ITWB), .EXPECT_INVTAU_S(INVTAU_S), \
            .EXPECT_KINV_MAG_W(KINV_MAG), .EXPECT_KINV_S(KINV_S) \
        ) u_cordic ( \
            .clk(clk), .rst(rst), .start(eng_start), .sb_in({WSB{1'b0}}), \
            .x0(f2_x0), .y0(f2_y0), .z0({WZ{1'b0}}), \
            .busy(), .done(cd_done), .z_done(), .sb_out(cd_sb_unused), \
            .xn(cd_xn), .yn(cd_yn), .zn(cd_zn), .const2pi(), .inv_tau(eng_inv_tau), .kinv_mag(eng_kinv_mag), .kinv());
    // Intentional: unsupported in-range WMAN names missing _zkf_cordic_m<WMAN>, prompting table generation.
    generate
        if (1'b0) begin : g_none
        `ZKF_ATAN2_CORE(11)
        `ZKF_ATAN2_CORE(12)
        `ZKF_ATAN2_CORE(13)
        `ZKF_ATAN2_CORE(14)
        `ZKF_ATAN2_CORE(15)
        `ZKF_ATAN2_CORE(16)
        `ZKF_ATAN2_CORE(17)
        `ZKF_ATAN2_CORE(18)
        `ZKF_ATAN2_CORE(19)
        `ZKF_ATAN2_CORE(20)
        `ZKF_ATAN2_CORE(21)
        `ZKF_ATAN2_CORE(22)
        `ZKF_ATAN2_CORE(23)
        `ZKF_ATAN2_CORE(24)
        `ZKF_ATAN2_CORE(25)
        `ZKF_ATAN2_CORE(26)
        `ZKF_ATAN2_CORE(27)
        `ZKF_ATAN2_CORE(28)
        `ZKF_ATAN2_CORE(29)
        `ZKF_ATAN2_CORE(30)
        `ZKF_ATAN2_CORE(31)
        `ZKF_ATAN2_CORE(32)
        `ZKF_ATAN2_CORE(33)
        `ZKF_ATAN2_CORE(34)
        `ZKF_ATAN2_CORE(35)
        `ZKF_ATAN2_CORE(36)
        `ZKF_ATAN2_CORE(37)
        `ZKF_ATAN2_CORE(38)
        `ZKF_ATAN2_CORE(39)
        `ZKF_ATAN2_CORE(40)
        `ZKF_ATAN2_CORE(41)
        `ZKF_ATAN2_CORE(42)
        `ZKF_ATAN2_CORE(43)
        `ZKF_ATAN2_CORE(44)
        `ZKF_ATAN2_CORE(45)
        `ZKF_ATAN2_CORE(46)
        `ZKF_ATAN2_CORE(47)
        `ZKF_ATAN2_CORE(48)
        `ZKF_ATAN2_CORE(49)
        `ZKF_ATAN2_CORE(50)
        `ZKF_ATAN2_CORE(51)
        `ZKF_ATAN2_CORE(52)
        `ZKF_ATAN2_CORE(53)
        end else begin : g_unsupported
            _zkf_invalid_wman_out_of_range u_invalid();
        end
    endgenerate
    `undef ZKF_ATAN2_CORE

    // ================================================================================================================
    // Back-end: read the held input-derived metadata, mux the divide operands, and run the folded radix-4 divider.
    // ================================================================================================================
    wire             be_bypass   = hd_bypass;
    wire             be_special  = hd_special;
    wire             be_sy       = hd_sy;
    wire             be_sx       = hd_sx;
    wire             be_swap     = hd_swap;
    wire signed [WE:0]   be_eydiff = hd_eydiff;
    wire signed [WE-1:0] be_eden   = hd_eden;
    wire [WMAN-1:0]  be_sigx     = hd_den_sig;
    wire [WMAN-1:0]  be_byp_irem = hd_byp_irem;   // precomputed bypass initial remainder (front-end)
    wire             be_byp_ibit = hd_byp_ibit;   // precomputed bypass integer quotient bit
    wire [WFULL-1:0] be_sp_mag   = hd_sp_mag;
    wire [2:0]       be_spk      = hd_spk;
    wire             be_sp_sign  = hd_sp_sign;

    // Operand mux: residual (|y_K|, x_K) or bypass (sig_y, sig_x). Q = floor(num*2**F/den) (fractional radix-4) + the
    // sticky from the final remainder. den is guarded against 0 (specials run on garbage, masked at the output).
    //
    // Timing: the initial remainder / integer bit feed the arm with NO carry chain. The residual remainder is just
    // |y_K| (one negate off cd_yn) and its integer bit is structurally 0 (|y_K| < x_K always).
    // The bypass remainder and integer bit are PRECOMPUTED at the front-end and captured into hd_* (slack-rich), so
    // neither the cd_yn negate nor the divisor mux sits behind a compare+subtract here.
    wire signed [WX-1:0] be_ykabs = cd_yn[WX-1] ? -cd_yn : cd_yn;                    // |y_K| (the only cd_yn-dep work)
    // The divisor is just selected here (bypass vs residual); 3*den is formed one cycle later in the divider's setup
    // state off the REGISTERED divisor (a reg->reg hop), keeping the wide 3*den carry chain off the engine-output arm
    // cone (cd_xn -> 3*den -> dv_den3 in one shot was the Diamond/Yosys limiter). The residual divisor is cd_xn
    // (the CORDIC magnitude x_K), which is > 0 for every finite transaction; a special transaction may leave
    // cd_xn == 0, but its divide output is discarded (the output mux selects the special-case theta/mag),
    // so no zero-guard is needed (a sim assertion below locks the invariant).
    // The bypass divisor sig_x has an implicit hidden bit so it too is nonzero.
    wire [WDIV-1:0]  res_den     = cd_xn[WDIV-1:0];
    wire [WDIV-1:0]  byp_den     = {{(WDIV-WMAN){1'b0}}, be_sigx};
    wire [WDIV-1:0]  div_den     = be_bypass ? byp_den  : res_den;       // selected divisor (3*den at setup)
    // Arm inputs: select the precomputed bypass remainder/bit or the residual |y_K| / structural 0 -- pure muxes.
    wire             div_ibit    = be_bypass & be_byp_ibit;
    wire [WDIV-1:0]  div_irem    = be_bypass ? {{(WDIV-WMAN){1'b0}}, be_byp_irem} : be_ykabs[WDIV-1:0];

    // -- Divide-setup register (arm) + the folded radix-4 divider (one reused _zkf_div_radix4_step, STEPS cycles).
    // The control/engine outputs registered here are held until the next transaction (one in flight),
    // so the post-divide multiply and unmap read them directly.
    reg                  dv_valid, dv_run;
    reg [WCNT-1:0]       dv_cnt;
    reg [WDIV-1:0]       dv_rem, dv_den;
    reg [WDIV+1:0]       dv_den3;
    reg [WQUO-1:0]       dv_quo;
    reg signed [WX-1:0]  dv_xn;
    reg signed [WZ-1:0]  dv_zn;
    reg                  dv_bypass, dv_yneg, dv_swap, dv_sx, dv_sy, dv_special;
    reg signed [WE:0]    dv_eydiff;
    reg signed [WE-1:0]  dv_eden;
    reg [WFULL-1:0]      dv_sp_mag;
    reg [2:0]            dv_spk;                                    // narrow special-case octant code (-> turn8 @ out)
    reg                  dv_sp_sign;                                // narrow special-case theta sign (-> turn8 @ out)
    // ONE reused stock _zkf_div_radix4_step (from _zkf_div_core, unchanged): one radix-4 digit (2 quotient bits) per
    // cycle, STEPS cycles, preceded by a one-cycle setup that forms dv_den3 = 3*dv_den off the REGISTERED divisor. The
    // setup keeps the wide 3*den carry chain on a reg->reg hop instead of chaining off the CORDIC x output on the
    // one-shot arm cone (which set the critical path otherwise). Costs one cycle (DIVCYC = STEPS + 1).
    wire [WDIV-1:0] step_rem_next;
    wire [1:0]      step_digit;
    _zkf_div_radix4_step #(.WMAN(WDIV)) u_step (
        .den(dv_den), .den3(dv_den3), .rem(dv_rem),
        .rem_next(step_rem_next), .digit(step_digit)
    );
    reg dv_setup;
    always @(posedge clk) begin
        if (rst) begin
            dv_valid <= 1'b0;
            dv_run   <= 1'b0;
            dv_setup <= 1'b0;
        end else begin
            dv_valid <= 1'b0;
            if (cd_done) begin                          // arm: load the selected divisor + initial remainder/bit
                dv_setup <= 1'b1;
                dv_run   <= 1'b0;
                dv_cnt   <= {WCNT{1'b0}};
                dv_rem   <= div_irem;
                dv_den   <= div_den;
                dv_quo   <= {{(WQUO-1){1'b0}}, div_ibit};
            end else if (dv_setup) begin                // setup: form 3*den off the REGISTERED divisor (reg->reg hop)
                dv_den3  <= {1'b0, dv_den, 1'b0} + {2'b00, dv_den};
                dv_setup <= 1'b0;
                dv_run   <= 1'b1;
            end else if (dv_run) begin                  // one radix-4 digit (2 quotient bits) per cycle
                dv_rem <= step_rem_next;
                dv_quo <= {dv_quo[WQUO-3:0], step_digit};
                dv_cnt <= dv_cnt + 1'b1;
                if (dv_cnt == (STEPS - 1)) begin
                    dv_run   <= 1'b0;
                    dv_valid <= 1'b1;                    // quotient (+ final remainder) ready
                end
            end
        end
    end

    // Sideband/datapath registers captured at arm time (reset-unconditional payload); the divisor regs dv_den/dv_den3
    // are loaded inside the divider FSM above.
    always @(posedge clk) begin
        if (cd_done) begin
            dv_xn       <= cd_xn;
            dv_zn       <= cd_zn;
            dv_bypass   <= be_bypass;
            dv_yneg     <= cd_yn[WX-1];
            dv_swap     <= be_swap;
            dv_sx       <= be_sx;
            dv_sy       <= be_sy;
            dv_special  <= be_special;
            dv_eydiff   <= be_eydiff;
            dv_eden     <= be_eden;
            dv_spk      <= be_spk;
            dv_sp_sign  <= be_sp_sign;
            dv_sp_mag   <= be_sp_mag;
        end
    end
    wire div_sticky = |dv_rem;                                   // remainder != 0 (jammed into the bypass mag sticky)

    // Sim-only invariant locked at arm time (cd_done): dropping the res_den zero-guard relies on the residual divisor
    // x_K being > 0 for every finite (non-special) transaction. Synthesis never sees this block; the bring-up tb and
    // the cocotb suites (icarus/verilator) run with SIMULATION=1, so it fires under exhaustive stimulus -- same
    // convention as the shared-back-end mutex assertion below. (Specials may leave cd_xn == 0, but their divide output
    // is discarded.)
`ifdef SIMULATION
    always @(posedge clk) begin
        if (!rst && cd_done && !be_special && cd_xn[WX-1])
            $fatal(1, "zkf_atan2: residual divisor cd_xn sign bit set for a non-special transaction");
        if (!rst && cd_done && !be_special && (cd_xn == {WX{1'b0}}))
            $fatal(1, "zkf_atan2: residual divisor cd_xn == 0 for a non-special transaction");
    end
`endif

    // ================================================================================================================
    // Shared multiplier: ONE _zkf_pmul time-shared for MAG = x_K*KINV (issued DURING the divide, overlapped) and
    // QT = Q*INV_TAU (issued when the divide completes; serves the residual correction AND the bypass theta). Both
    // products are unsigned (x_K, KINV, Q, INV_TAU all >= 0). The two issues are STEPS cycles apart (STEPS >= 10), so
    // they never collide in the 1+STAGE_PRODUCT-deep pipeline.
    // ================================================================================================================
    localparam [0:0] TAG_MAG = 1'b0, TAG_QT = 1'b1;
    // MAG is issued once on the first divide cycle (the cycle after cd_done arms the divide, when dv_xn is freshly
    // loaded and the pmul is idle). A registered one-cycle pulse off cd_done fires exactly once -- one pulse per
    // transaction, independent of the divider's internal cycle count (cd_done itself is a single-cycle pulse).
    reg              mag_issue;
    always @(posedge clk) begin
        if (rst) mag_issue <= 1'b0;
        else     mag_issue <= cd_done;
    end
    wire             qt_issue  = dv_valid;                            // divide done: Q ready
    wire             pmul_iv     = mag_issue | qt_issue;
    wire [0:0]       pmul_tag_in = qt_issue ? TAG_QT : TAG_MAG;
    // The `b` operand is the pre-narrowed table constant for the current product, connected directly (no re-narrowing):
    // eng_kinv_mag (scale 2**-KINV_S) for MAG, eng_inv_tau (scale 2**-INVTAU_S) for QT -- both already WMAN+5 bits.
    // The post-product scaling below derives its shift / exp-offset from those native scales, so no fold-back remains.
    wire [WA_MUL-1:0]  pmul_a = qt_issue ? {{(WA_MUL-WQUO){1'b0}}, dv_quo} : dv_xn[WA_MUL-1:0];
    wire [KINV_MAG-1:0] pmul_b = qt_issue ? eng_inv_tau : eng_kinv_mag;       // narrowed inv_tau (QT) / kinv_mag (MAG)
    wire             pmul_ov;
    wire [0:0]       pmul_tag_out;
    localparam integer WPMUL = WA_MUL + KINV_MAG;
    wire [WPMUL-1:0] pmul_p_raw;                                           // widest shared-mult product
    wire [WMAG-1:0]  pmul_p = {{(WMAG-WPMUL){1'b0}}, pmul_p_raw};          // pad only if theta magnitude dominates
    _zkf_pmul #(
        .WA(WA_MUL), .WB(KINV_MAG), .A_SIGNED(0), .B_SIGNED(0), .WSB(1),
        .WMULTIPLIER(WMULTIPLIER), .STAGE_PRODUCT(STAGE_PRODUCT)
    ) u_pmul (
        .clk(clk), .rst(rst), .in_valid(pmul_iv), .sb_in(pmul_tag_in),
        .a(pmul_a), .b(pmul_b), .out_valid(pmul_ov), .sb_out(pmul_tag_out), .p(pmul_p_raw)
    );
    wire mag_ov = pmul_ov && (pmul_tag_out == TAG_MAG);        // MAG product valid -> issue it to the shared back-end
    wire qt_ov  = pmul_ov && (pmul_tag_out == TAG_QT);         // QT product valid -> form theta this cycle

    // ================================================================================================================
    // Post-divide, split into two register stages so the wide signed unmap add does not chain behind the shared
    // multiplier on the critical path (a single ~WZ-wide add at WMAN=36 already fills a clock once the engine->divider
    // placement spread is added):
    //   P2 (this stage): captures the LATE product-derived correction terms -- res_delta (a constant shift of the QT
    //       product, so the pmul_p->P2 hop is wires only) and the EARLY unmap base un_base = unmap_const +/- z_K (from
    //       registered dv_*) -- plus the bypass jam and every dv_*-derived back-end input, pre-forming texp/mexp.
    //   B2 (next stage): performs the ONE remaining signed add un_tmag = un_base +/- res_delta and assembles the packer
    //       inputs. So the P2->B2 cone is exactly one add; the pmul_p->P2 cone is the shift wires. +1 latency cycle
    //       (folded into `ZKF_ATAN2_BASE / atan2_latency()).
    // The special-case descriptor {special, sp_sign, spk, sp_mag} is NOT re-registered through P2/B2:
    // it is read DIRECTLY from the held dv_* regs at the output (latched at cd_done, held for the whole single
    // in-flight transaction -- earlier and longer-lived than any p2_*/b2_* copy), so only the numeric theta payload
    // flows here.
    // The unmap algebra (bit-exact to the former single-stage form): un_tmag = unmap_const +/- res_a0, res_a0 =
    // z_K +/- res_delta, with res_a0 negated iff swap^sx and res_delta negated iff y_K<0. Folding gives un_base =
    // unmap_const +/- z_K (negate z_K iff swap^sx) and res_delta subtracted iff (swap^sx) ^ y_K<0.
    wire [WMAG-1:0]      qt_full   = pmul_p[WMAG-1:0];                 // Q * inv_tau (WA_MUL+KINV_MAG, padded to WMAG)
    // qt_full = Q * inv_tau is at scale 2**-(F + INVTAU_S); the right-shift to the angle scale 2**-ZF is the difference
    // F + INVTAU_S - ZF (>= 0 for every supported WMAN). inv_tau is the pre-narrowed operand, so no fold-back.
    wire [WZ+1:0]        res_delta = qt_full >> (F + INVTAU_S - ZF);
    wire signed [WZ+1:0] zn_ext    = $signed(dv_zn);                           // sign-extend z_K (can be slightly < 0)
    wire signed [WZ+1:0] unmap_const = dv_swap ? QUARTER : (dv_sx ? HALF : {(WZ+2){1'b0}});
    wire                 un_neg_a0   = dv_swap ^ dv_sx;                         // res_a0 (hence z_K) negated in unmap
    wire signed [WZ+1:0] un_base     = un_neg_a0 ? (unmap_const - zn_ext) : (unmap_const + zn_ext);  // early (no delta)
    wire                 un_sub_delta = un_neg_a0 ^ dv_yneg;                    // subtract res_delta when set
    wire [WMAG-1:0]      byp_tmag  = qt_full | {{(WMAG-1){1'b0}}, div_sticky}; // jam the divide sticky into the pack

    // -- Stage P2: register the correction operands + forward the THETA back-end inputs (one add still pending for B2).
    // The magnitude no longer flows through P2/B2: it is issued to the SHARED back-end early (the moment its product
    // returns, see mag_be_issue below), so p2/b2 carry only the theta-path payload.
    reg                  p2_valid, p2_bypass, p2_sub_delta;
    reg signed [WZ+1:0]  p2_un_base;
    reg [WZ+1:0]         p2_res_delta;
    reg [WMAG-1:0]       p2_byp_tmag;
    reg signed [WEU-1:0] p2_texp;
    reg                  p2_tsign;
    always @(posedge clk) begin
        if (rst) p2_valid <= 1'b0; else p2_valid <= qt_ov;
        p2_bypass    <= dv_bypass;
        p2_sub_delta <= un_sub_delta;
        p2_un_base   <= un_base;
        p2_res_delta <= res_delta;
        p2_byp_tmag  <= byp_tmag;
        // texp differs by path: bypass value = Q * 2**(ey-ex-F-INVTAU_S); residual value = theta_mag * 2**-ZF.
        // The format BIAS is folded in HERE so the shared _zkf_fixed_to_float back-end runs EXP_IS_BIASED=1
        // and the packer skips its bias add -- removing that adder from the post-normalize pack cone.
        // pre_exp = (offset + BIAS) - norm_count is the biased exponent; bit-exact (WEU has the range).
        // The bypass value = Q * 2**(ey-ex-F-INVTAU_S): Q at 2**-F, inv_tau at 2**-INVTAU_S, so the product is
        // at scale 2**-(F+INVTAU_S).
        // The residual value = theta_mag * 2**-ZF. No fold-back -- inv_tau is pre-narrowed.
        p2_texp      <= (dv_bypass ? ((WMAG - 1) - F - INVTAU_S + dv_eydiff) : ((WMAG - 1) - ZF)) + BIAS;
        p2_tsign     <= dv_sy;
    end
    wire signed [WZ+1:0] p2_un_tmag = p2_sub_delta ? (p2_un_base - $signed(p2_res_delta))
                                                   : (p2_un_base + $signed(p2_res_delta));

    // -- Stage B2: the one pending unmap add + the theta packer-input assembly (the magnitude path is separate now).
    reg              b2_valid;
    reg [WMAG-1:0]   b2_tmag;
    reg signed [WEU-1:0] b2_texp;
    reg              b2_tsign;
    always @(posedge clk) begin
        if (rst) b2_valid <= 1'b0; else b2_valid <= p2_valid;
        b2_tmag     <= p2_bypass ? p2_byp_tmag : {{(WMAG-(WZ+2)){1'b0}}, p2_un_tmag[WZ+1:0]};
        b2_texp     <= p2_texp;
        b2_tsign    <= p2_tsign;
    end

    // ================================================================================================================
    // ONE shared _zkf_fixed_to_float back-end, time-multiplexed over the magnitude then theta of a single transaction.
    // The MAG product (x_K*kinv_mag) returns from the shared _zkf_pmul ~STEPS cycles before theta is ready (the divide
    // is still running), so MAG is issued to the back-end the moment its product returns (mag_be_issue == mag_ov); its
    // packed result emerges and is LATCHED into mag_num_r well before theta. THETA is then issued when b2_valid fires,
    // exactly as the former theta back-end was, so it emerges on the SAME output cycle as before -- the share is
    // latency-flat. The two issues are mutually exclusive (>= STEPS+2 cycles apart) and tagged (the only thing the f2f
    // sideband carries -- WSB2 == 1) so the output stage knows which result a cycle carries. The special-case
    // descriptor {special, sp_sign, spk, sp_mag} does NOT ride the f2f delay: it is read directly from the
    // held dv_* regs at the output (single-in-flight keeps them stable through the THETA emergence), and drives the
    // special-case override for BOTH outputs (the engine/divide ran on garbage during specials); turn8(sp_sign, spk)
    // is assembled AFTER the back-end (off every register cone) so its biased-add carry lands with huge slack.
    // ================================================================================================================
    // Magnitude exponent, formed at MAG-issue from the cd_done-latched den binade (dv_eden, BIASED). value = M *
    // 2**(e_den + 2 - (XF+KINV_S)): M = x_K*kinv_mag at scale 2**-(XF+KINV_S), the 1/4 pre-scale undone (+2). No +BIAS
    // (dv_eden is already biased), EXP_IS_BIASED-packed -- bit-identical to the former p2_mexp.
    localparam signed [WEU-1:0] MAG_EXP_OFFS = (WMAG - 1) - (XF + KINV_S) + 2;
    wire signed [WEU-1:0] mag_be_exp = $signed({{(WEU-WE){dv_eden[WE-1]}}, dv_eden}) + MAG_EXP_OFFS;
    wire             mag_be_issue = mag_ov;                 // MAG product just returned -> issue magnitude
    wire             share_iv     = mag_be_issue | b2_valid;
    wire             share_is_th  = b2_valid;               // tag: 1 = theta (mutually exclusive with mag_be_issue)
    // Sim-only invariant: the shared back-end is mutually exclusive -- the MAG f2f-issue (mag_be_issue) and the THETA
    // f2f-issue (b2_valid) never fire on the same cycle (they are >= STEPS+2 cycles apart by construction).
    // The test suite must be run with SIMULATION=1.
`ifdef SIMULATION
    always @(posedge clk) begin
        if (!rst && mag_be_issue && b2_valid)
            $fatal(1, "zkf_atan2: shared back-end collision -- mag_be_issue and b2_valid both high");
    end
`endif
    // Input mux: (mag, exp_offset, sign) = (x_K*kinv_mag product, mag_be_exp, 0) for MAG vs
    // (b2_tmag, b2_texp, b2_tsign) for theta. Only the TAG (MAG vs THETA) needs the f2f back-end's delay alignment;
    // the special descriptor {special, sp_sign, spk, sp_mag} is read DIRECTLY from the held dv_* regs at the
    // output (see below), so it is not carried through the f2f pipe -- shrinking the sideband from WFULL+5 to a single
    // bit.
    wire [WMAG-1:0]       share_mag = share_is_th ? b2_tmag  : pmul_p;
    wire signed [WEU-1:0] share_exp = share_is_th ? b2_texp  : mag_be_exp;
    wire                  share_sgn = share_is_th ? b2_tsign : 1'b0;
    localparam integer WSB2 = 1;                        // {tag} only (the special descriptor is read direct, below)
    wire [WSB2-1:0]  share_sb = share_is_th;
    wire             be_ov;
    wire [WFULL-1:0] be_num;
    wire [WSB2-1:0]  be_sbo;
    _zkf_fixed_to_float #(
        .WEXP(WEXP), .WMAN(WMAN), .WMAG(WMAG), .WEU(WEU),
        .EXP_IS_BIASED(1), .ASSUME_NO_OVERFLOW(0), .WSB(WSB2),
        .STAGE_NORMALIZE(STAGE_NORMALIZE), .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(0)
    ) u_f2f (
        .clk(clk), .rst(rst),
        .in_valid(share_iv), .sign(share_sgn), .force_zero(1'b0), .force_inf(1'b0),
        .exp_offset(share_exp), .mag(share_mag), .sb_in(share_sb),
        .out_valid(be_ov), .y(be_num), .sb_out(be_sbo)
    );

    wire             out_tag      = be_sbo[WSB2-1];         // 1 => this cycle carries theta
    // The special-case descriptor (special / sp_sign / spk / sp_mag) is read DIRECTLY from the held dv_*
    // regs rather than from the f2f-delayed sideband: the module runs a single transaction in flight (in_ready is held
    // low while busy, which clears no earlier than the THETA emergence below), so dv_* still hold THIS transaction's
    // descriptor at the THETA output cycle -- they are latched at cd_done and held until retire (busy clears only at
    // out_valid & out_ready, i.e. no earlier than the THETA emergence), strictly outliving the former b2_* copies, and
    // no later transaction's cd_done can have overwritten them. This drives the override for BOTH outputs
    // (the f2f numeric output is garbage for specials, since the engine/divide ran on it).
    // The redundant p2_*/b2_* copies are therefore dropped.
    wire             out_special  = dv_special;
    wire             out_sp_sign  = dv_sp_sign;
    wire [2:0]       out_spk      = dv_spk;
    wire [WFULL-1:0] out_sp_mag   = dv_sp_mag;
    // Latch the magnitude's packed numeric result when the MAG pass emerges (it returns before theta); held until the
    // theta pass emerges, when both outputs are paired. Pure datapath: only sampled in lockstep with be_valid (reset).
    reg [WFULL-1:0]  mag_num_r;
    always @(posedge clk) begin
        if (be_ov && !out_tag) mag_num_r <= be_num;
    end
    // Deferred special-case theta: turn8's biased-exponent add runs here, off every register cone. Equivalent to the
    // previous front-end turn8 (the inputs are unchanged). The special override (from the held dv_* descriptor) selects
    // the exact special-case theta/mag for BOTH outputs.
    // Special-case theta is turn8 of a per-special-case octant/sign code: it selects among a fixed set of constant
    // turn bodies (0, +/-1/8, +/-1/4, +/-1/2).
    wire [WFULL-1:0] out_sp_theta = turn8(out_sp_sign, out_spk);
    wire             be_valid = be_ov & out_tag;            // theta emergence == the (unchanged) output-valid cycle
    // Canonicalize the generic half-turn. Near the negative-x axis (finite x<0, |y| -> 0) the generic magnitude rounds
    // to 1/2 turn with sign = sy, packing the out-of-range -1/2. The documented range is the half-open (-0.5, +0.5],
    // so -1/2 folds to the canonical +1/2 -- the convention turn8 applies at k==4 and the atan2(.,x=-inf) path uses.
    // Two facts keep this off the critical output cone: (a) the fold flips ONLY the sign bit (-1/2 and +1/2 share the
    // same magnitude), so the wide magnitude datapath into the registered output stays a plain mux; (b) the generic
    // magnitude is <= 1/2, so its exponent field equals the 1/2-turn exponent ONLY for the exact 1/2-turn (any smaller
    // magnitude has a strictly smaller exponent) -- a WEXP-wide exponent compare suffices, no full-width equality.
    // turn8(0,4) is the config-correct +1/2 body (computed identically to the live k==4 special path; its TURN8_Z4
    // underflow branch is unreachable for the legal WEXP>=2), so its exponent field is the correct reference.
    wire [WFULL-1:0] half_pos     = turn8(1'b0, 3'd4);
    wire             be_neg_half  = be_num[WFULL-1] & (be_num[WFULL-2:WFRAC] == half_pos[WFULL-2:WFRAC]);
    wire [WFULL-1:0] be_num_canon = {be_num[WFULL-1] & ~be_neg_half, be_num[WFULL-2:0]};
    wire [WFULL-1:0] be_theta = out_special ? out_sp_theta : be_num_canon;
    wire [WFULL-1:0] be_mag_o = out_special ? out_sp_mag   : mag_num_r;

    // ================================================================================================================
    // Output handshake with back-pressure. One transaction in flight; the result waits for out_ready. With out_ready
    // high, in_ready reasserts on the cycle after out_valid is retired; LATENCY is not the initiation interval.
    // STAGE_OUTPUT selects WHERE the paired output is held:
    //   0: combinational output -- be_* is presented on its valid cycle; a hold register catches it while out_ready is
    //      low (no added output-register latency when out_ready is high).
    //   1: a hard output register drives theta/mag/out_valid DIRECTLY (no combinational logic after it); it captures
    //      the result and holds it until out_ready (+1 cycle).
    // ================================================================================================================
    generate
        if (STAGE_OUTPUT == 0) begin : g_out_comb
            reg              pending;
            reg [WFULL-1:0]  hold_theta, hold_mag;
            always @(posedge clk) begin
                if (rst) pending <= 1'b0;
                else if (be_valid & ~out_ready) begin
                    pending    <= 1'b1;
                    hold_theta <= be_theta; hold_mag <= be_mag_o;
                end else if (pending & out_ready) begin
                    pending    <= 1'b0;
                end
            end
            assign out_valid = be_valid | pending;
            assign theta     = pending ? hold_theta : be_theta;
            assign mag       = pending ? hold_mag   : be_mag_o;
        end else begin : g_out_reg
            reg              r_valid;
            reg [WFULL-1:0]  r_theta, r_mag;
            always @(posedge clk) begin
                if (rst)            r_valid <= 1'b0;
                else if (be_valid)  r_valid <= 1'b1;
                else if (out_ready) r_valid <= 1'b0;
                if (be_valid) begin r_theta <= be_theta; r_mag <= be_mag_o; end
            end
            assign out_valid = r_valid;
            assign theta     = r_theta;
            assign mag       = r_mag;
        end
    endgenerate

    always @(posedge clk) begin
        if (rst)                        busy <= 1'b0;
        else if (accept)                busy <= 1'b1;
        else if (out_valid & out_ready) busy <= 1'b0;
    end

endmodule

`undef ZKF_ATAN2_LATENCY
`undef ZKF_ATAN2_BASE
`undef ZKF_ATAN2_STEPS
`undef ZKF_ATAN2_XYCYC
`undef ZKF_ATAN2_XF
`undef ZKF_ATAN2_N
`default_nettype wire

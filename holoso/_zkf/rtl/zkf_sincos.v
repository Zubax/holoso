/// Iterative (single-transaction) sine and cosine of a phase in turns (with the reduced quadrant exposed):
///   sin = sin(2*pi*x), cos = cos(2*pi*x)
///
/// This is NOT a throughput-1 pipeline. A transaction is accepted when `in_ready` is high; the module then runs for a
/// fixed data-invariant latency and holds `out_valid` with a stable result until `out_ready` accepts it.
/// LATENCY is accept-to-out_valid latency; with out_ready high, in_ready reasserts one cycle after retirement.
/// Faithful rounding: each finite output is within <= 1 ULP of the correctly-rounded result.
/// Behavior:
///
///   x finite : sin = sin(2*pi*x), cos = cos(2*pi*x); quadrant = floor(frac(x)*4) mod 4
///              Exact boundaries take the upper quadrant: 0->0, 1/4->1, 1/2->2, 3/4->3); exact-zero outputs are +0.
///   x = +inf : sin = +inf, cos = +inf, quadrant = 0.
///   x = -inf : sin = -inf, cos = -inf, quadrant = 0.
///
/// Algorithm (mixed CORDIC; the engine is in _zkf_cordic):
///
///  1. Reduce x mod 1 to the FF-bit fraction frac(x) (FF = WMAN + GUARD_FF): top 2 bits = |x| quadrant, low WT = FF-2
///     bits = quadrant-local coordinate t (local angle (pi/2)*t). Reduce |x| and use the sin sign flip / quadrant
///     reflection for x < 0. Fold the half-quadrant symmetry to bring the local angle theta' into [0, pi/4].
///
///  2. Run K CORDIC rotation iterations (rotation mode) on the folded engine to get (cos theta', sin theta') ~ at
///     scale 2**-XF and the small residual angle z_K.
///
///  3. Finish with ONE linear rotation by the residual: sin = y_K + x_K*phi, cos = x_K - y_K*phi, phi = 2*pi*z_K. The
///     correction multiplies (the operator's only ones) use the per-WMAN 2*pi constant. Tiny / below-TSA angles take
///     the linear small-angle bypass (sin = 2*pi*theta'_turns, cos = 1) from the same 2*pi constant.
///
///  4. Unmap the octant and |x| quadrant; one shared _zkf_fixed_to_float back-end renormalizes and rounds sin first,
///     then cos one cycle later. The packed sin is held until cos emerges so the outputs remain paired.
///
/// Tuning knobs:
///
/// WMULTIPLIER: Optional DSP multiplier argument width hint if wide multiplication is used.
///     By default (zero), the multiplication module will split arguments symmetrically, which is not always optimal.
///     If nonzero, the multiplier derives the minimal (possibly asymmetric) slice grid so each slice fits a
///     WMULTIPLIER-bit tile. This can significantly reduce DSP usage and may improve f_max.
///
/// UNROLL100={50,100,200,...}: CORDIC iterations per engine cycle x100. Values <100 split operations across cycles.
///     Choose the maximum value that closes timings. It is forwarded to _zkf_cordic; refer there for details.
///
/// STAGE_INPUT={0,1}: Latch the input x before the decode, isolating it from upstream (+1 cycle).
///
/// STAGE_PRODUCT: Pipeline depth of the shared correction multiply and its default symmetric split.
///     See _zkf_pmul for details. Value = latency cycle cost.
///
/// STAGE_NORMALIZE: Internal normshift barriers in the shared _zkf_fixed_to_float. Value = latency cycle cost.
///
/// STAGE_PACK={0,1}: Register the _zkf_pack input (insulates the rounder from the normshift cone) (+1 cycle).
///
/// STAGE_OUTPUT={0,1}: Register the results ahead of the out_ready hold (+1 cycle).

`default_nettype none

module zkf_sincos #(
    parameter WEXP            = 6,
    parameter WMAN            = 18,     // significand precision including the hidden bit
    parameter WMULTIPLIER     = 0,
    parameter UNROLL100       = 100,    // choose maximum value that closes timings
    parameter STAGE_INPUT     = 0,
    parameter STAGE_PRODUCT   = 0,
    parameter STAGE_NORMALIZE = 0,
    parameter STAGE_PACK      = 0,
    parameter STAGE_OUTPUT    = 0,
    parameter PARALLEL        = (UNROLL100 < 100) ? 1 : 0,  // Testing-only knob, DO NOT override.
    parameter LATENCY         = 0
) (
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 in_valid,    // start a transaction (sampled only when in_ready)
    output wire                 in_ready,    // high when idle and able to accept a transaction
    input  wire [WEXP+WMAN-1:0] x,

    output wire                 out_valid,   // result is ready; held until out_ready (back-pressure)
    input  wire                 out_ready,   // consumer accepts the result on a cycle where out_valid & out_ready
    output wire [WEXP+WMAN-1:0] sin,
    output wire [WEXP+WMAN-1:0] cos,
    output wire [1:0]           quadrant
);
    localparam integer K      = ((WMAN + 1) / 2) + 1;          // CORDIC iterations
    localparam integer XYCYC  = (K * 100 + UNROLL100 - 1) / UNROLL100;
    localparam integer ZGAP   = XYCYC - K;
    localparam integer PMUL_L = 1 + STAGE_PRODUCT;
    localparam integer SAVED  = (PARALLEL == 0) ? 0 : ((ZGAP < PMUL_L) ? ZGAP : PMUL_L);
    localparam LATENCY_REF =
        11 + (2 * STAGE_PRODUCT) + XYCYC - SAVED + STAGE_INPUT + STAGE_NORMALIZE + STAGE_PACK + STAGE_OUTPUT;
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
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;
    localparam integer GUARD_FF   = (12 > (WMAN / 2 + 2)) ? 12 : (WMAN / 2 + 2);
    localparam integer GUARD_XY   = 6;
    localparam integer GUARD_ZF   = 6;
    localparam integer GUARD_Z    = 3;
    localparam integer FF   = WMAN + GUARD_FF;          // reduced-fraction width: frac(x) at scale 2**-FF
    localparam integer WT   = FF - 2;                   // quadrant-local coordinate width (top 2 bits = quadrant)
    localparam integer XF   = ((3 * WMAN + 1) / 2) + GUARD_XY;  // x/y fractional scale
    localparam integer WX   = XF + 2;                   // signed x/y width
    // Native scale of the pre-narrowed const2pi from the per-WMAN _zkf_cordic_m table (MUST match its CONST2PI_S):
    // the table rounds 2*pi to WMAN+5 bits and emits it at scale 2**-CONST2PI_S, so const2pi is rounded there.
    localparam integer CONST2PI_S = WMAN + 2;
    // Two wide-datapath register stages are always present: an octant-fold register splits the WT-wide octant-fold
    // negate feeding the engine seed, and a merge-B3 register splits the WMAG-wide quadrant/octant magnitude muxing.
    // Both are latency-only (the back-end is event-driven off the engine `done`).
    localparam integer ZG   = GUARD_ZF;                 // angle fractional bits past the coordinate's WT+2
    localparam integer ZF   = WT + 2 + GUARD_ZF;        // angle accumulator fractional scale
    localparam integer WZ   = ZF + GUARD_Z;             // signed angle width
    // The shared correction multiply takes the per-WMAN const2pi on `a` for BOTH the PHI product (const2pi*z_K) and
    // the BYP/small-angle product (const2pi*operand). const2pi arrives PRE-NARROWED from the table at its native scale:
    // its top WMAN+5 bits (round-to-nearest) at scale 2**-CONST2PI_S (CONST2PI_S = XF - the bits dropped in narrowing).
    // That already-narrowed WMAN+5-bit operand is what the grid sees, so the shared DSP column count is the narrowed
    // one, and every consumer derives its shift / exp-offset from CONST2PI_S directly --
    // product-scale minus target-scale. The magnitude container and shared _zkf_fixed_to_float back-end are sized on
    // this narrowed CWB, so they shrink with it.
    localparam integer CWB  = WMAN + 5;                 // narrowed 2*pi width (== the table's const2pi port width)
    localparam integer TSA_BITS = (WT + 2) - ((WMAN + 1) / 2) - 3;  // small-angle handoff: t' < 2**TSA_BITS
    localparam integer WMAG = CWB + WT + 1;             // uniform magnitude width (small-angle full product is widest)
    // The CORDIC corrections and the exact +1 live at scale 2**-XF; the bypass/tiny/TSA magnitudes are const2pi
    // products at scale 2**-CONST2PI_S. Each path reads its magnitude back with the matching exp_offset
    // (field-width minus scale).
    localparam integer EONE_XF = WMAG - 1 - XF;          // exp_offset reading a 2**-XF-scaled magnitude back as itself
    localparam integer EONE_S  = WMAG - 1 - CONST2PI_S;  // exp_offset reading a 2**-CONST2PI_S-scaled magnitude
    localparam integer WOP  = (WMAN > (TSA_BITS + 1)) ? WMAN : (TSA_BITS + 1);  // small-angle operand width
    // Correction-multiply operand widths: phi keeps WPHI top bits of (const2pi*z_K)>>zf,
    // x_K/y_K keep WXC top bits -- so each correction multiply is small (~18 bits).
    localparam integer PHI_NAT      = XF - K + 2;
    localparam integer WPHI         = (WMAN + 6 < PHI_NAT) ? (WMAN + 6) : ((PHI_NAT > 2) ? PHI_NAT : 2);
    localparam integer PHI_TRUNC    = (PHI_NAT - WPHI > 0) ? (PHI_NAT - WPHI) : 0;
    localparam integer PHI_S        = XF - PHI_TRUNC;     // scale of the narrowed phi (phi == ~2*pi*z_K at 2**-PHI_S)
    localparam integer WXC          = WMAN + 6;
    localparam integer XK_TRUNC     = WX - WXC;
    localparam integer CORR_SHIFT   = XF - XK_TRUNC - PHI_TRUNC;

    localparam integer BIAS     = (1 << (WEXP - 1)) - 1;
    localparam integer SH_BASE  = BIAS - GUARD_FF - 1;
    localparam integer WE       = WEXP + 1;              // signed unbiased exponent e = exp - BIAS
    localparam integer WEU_RAW  = $clog2(WMAG + 1) + 2;
    localparam integer WEU      = (WEU_RAW > (WEXP + 3)) ? WEU_RAW : (WEXP + 3);
    // Engine sideband (carried through the CORDIC): {e, quad_abs, oct_flip, tzero, tiny, sa_sel, is_inf, sign}.
    // The bypass operand is NOT carried here -- const2pi*operand is computed early (during the CORDIC) and latched,
    // so the wide operand stays out of the engine's per-stage sideband registers.
    localparam integer WSB      = WE + 8;

    localparam integer WLSH               = $clog2(FF + 1);
    localparam integer WSH                = $clog2((1 << (WEXP - 1)) + GUARD_FF + 2) + 2;
    localparam signed [WSH-1:0] SH_BASE_S = SH_BASE[WSH-1:0];
    localparam signed [WSH:0]   SH_HI_S   = SH_BASE + FF;   // SH_BASE + FF, the upper clamp bound, as signed constant

    // ============================================================================================================
    // Front-end: accept a transaction, decode + reduce |x| mod 1, octant-fold, and start the folded CORDIC engine.
    // ============================================================================================================
    reg    busy;                              // a transaction is in flight (engine or back-end)
    wire   accept = in_valid & in_ready;
    assign in_ready = ~busy;

    // STAGE_INPUT: latch the input x at accept, isolating the decode from the upstream input path.
    wire             si_valid;
    wire [WFULL-1:0] si_x;
    generate
        if (STAGE_INPUT == 0) begin : g_sin_comb
            assign si_valid = accept; assign si_x = x;
        end else begin : g_sin_reg
            reg             r_si_valid;
            reg [WFULL-1:0] r_si_x;
            always @(posedge clk) begin
                if (rst) r_si_valid <= 1'b0;
                else     r_si_valid <= accept;
                r_si_x <= x;
            end
            assign si_valid = r_si_valid; assign si_x = r_si_x;
        end
    endgenerate

    // Decode (combinational): the raw fields, the shift-amount subtract sh = exp - SH_BASE, and the two clamp bounds.
    // The bounds are PARALLEL comparisons of exp against constants (e_lo = exp < SH_BASE i.e. sh < 0; e_hi = exp >
    // SH_BASE+FF i.e. sh > FF), so the shift-amount mux below is NOT a second carry chain in series with the subtract.
    wire [WEXP-1:0]       e0   = si_x[WFULL-2:WFRAC];
    wire signed [WSH-1:0] sh0  = $signed({{(WSH-WEXP){1'b0}}, e0}) - SH_BASE_S;
    wire                  e_lo = $signed({1'b0, e0}) < SH_BASE_S;   // exp < SH_BASE  (sh < 0: tiny -> lshamt 0)
    wire                  e_hi = $signed({1'b0, e0}) > SH_HI_S;     // exp > SH_BASE+FF (sh > FF: clamp lshamt = FF)
    wire                  d_sign    = si_x[WFULL-1];
    wire [WEXP-1:0]       d_exp     = e0;
    wire [WFRAC-1:0]      d_frac    = si_x[WFRAC-1:0];
    wire                  d_is_left = ~e_lo;
    wire                  d_valid   = si_valid;

    // Stage R1: latch + decode + clamped shift amount. Registered so the wide barrel shift is its own stage.
    reg              r1_valid;                           // a freshly accepted transaction is in R1
    reg              r1_sign, r1_is_inf, r1_is_left;
    reg [WMAN-1:0]   r1_sig;
    reg [WLSH-1:0]   r1_lshamt;
    reg signed [WE-1:0] r1_e;
    wire             is_zero = ~|d_exp;
    wire             is_inf  =  &d_exp;
    wire [WMAN-1:0]  sig_in  = is_zero ? {WMAN{1'b0}} : {1'b1, d_frac};
    wire signed [WE-1:0]  e_in   = $signed({1'b0, d_exp}) - $signed(BIAS[WE-1:0]);
    // Shift amount clamped to [0, FF]: the e_lo / e_hi range tests are precomputed (parallel to the subtract), so this
    // is just a 3:1 mux selecting 0, FF, or the low bits of sh -- no compare in series with the subtract.
    wire [WLSH-1:0]  lshamt  = e_lo ? {WLSH{1'b0}} : e_hi ? FF[WLSH-1:0] : sh0[WLSH-1:0];
    always @(posedge clk) begin
        if (rst) r1_valid <= 1'b0;
        else     r1_valid <= d_valid;
        r1_sign    <= d_sign;
        r1_is_inf  <= is_inf;
        r1_is_left <= d_is_left;
        r1_sig     <= sig_in;
        r1_lshamt  <= lshamt;
        r1_e       <= e_in;
    end

    // Stage R2 combinational: the wide barrel shift -> quadrant / in-octant coordinate. The shift is the long cone,
    // so its outputs are registered (R2) before the octant fold. The exact-zero reduction is delayed until the fold
    // stage; otherwise the variable-shift mux and wide OR-reduction share one timing cone.
    wire [FF-1:0] frac_pos  = {{(FF-WMAN){1'b0}}, r1_sig} << r1_lshamt;
    wire [1:0]    quad_abs  = r1_is_inf ? 2'b00 : frac_pos[FF-1:FF-2];
    wire [WT-1:0] t_abs     = frac_pos[WT-1:0];
    wire          tiny_c    = ~r1_is_left;

    // Stage R2 register: hold the barrel-shift result so the octant fold below is a fresh combinational stage.
    reg              r2_valid;
    reg              r2_sign, r2_is_inf, r2_tiny;
    reg [1:0]        r2_quad;
    reg [WT-1:0]     r2_t;
    reg [WMAN-1:0]   r2_sig;
    reg signed [WE-1:0] r2_e;
    always @(posedge clk) begin
        if (rst) r2_valid <= 1'b0;
        else     r2_valid <= r1_valid;
        r2_sign   <= r1_sign;
        r2_is_inf <= r1_is_inf;
        r2_tiny   <= tiny_c;
        r2_quad   <= quad_abs;
        r2_t      <= t_abs;
        r2_sig    <= r1_sig;
        r2_e      <= r1_e;
    end

    // Octant fold (combinational from R2): the pi/2-complement 2**WT - r2_t is a WT-wide two's-complement negate
    // (the +1 is needed for relative accuracy of the small reflected sine near the fold boundary), whose carry chain
    // is the dominant front-end cone on wide datapaths.
    wire          oct_flip_c = (~r2_tiny) & (r2_t > {1'b1, {(WT-1){1'b0}}});
    wire [WT-1:0] tp_w_c     = oct_flip_c ? (~r2_t + 1'b1) : r2_t;
    wire          tzero_c    = ~|r2_t;

    // Fold register stage: hold the folded coordinate so the WT-wide negate above is its own stage,
    // isolated from the seed-pack + engine-latch cone below.
    wire                 f_valid, f_octflip, f_tiny, f_tzero, f_sign, f_inf;
    wire [WT-1:0]        f_tpw;
    wire [WMAN-1:0]      f_sig;
    wire [1:0]           f_quad;
    wire signed [WE-1:0] f_e;
    reg                  fr_valid, fr_octflip, fr_tiny, fr_tzero, fr_sign, fr_inf;
    reg [WT-1:0]         fr_tpw;
    reg [WMAN-1:0]       fr_sig;
    reg [1:0]            fr_quad;
    reg signed [WE-1:0]  fr_e;
    always @(posedge clk) begin
        if (rst) fr_valid <= 1'b0;
        else     fr_valid <= r2_valid;
        fr_octflip <= oct_flip_c; fr_tpw <= tp_w_c; fr_tiny <= r2_tiny; fr_tzero <= tzero_c;
        fr_sign <= r2_sign; fr_inf <= r2_is_inf; fr_sig <= r2_sig; fr_quad <= r2_quad; fr_e <= r2_e;
    end
    assign f_valid = fr_valid; assign f_octflip = fr_octflip; assign f_tpw = fr_tpw;
    assign f_tiny = fr_tiny; assign f_tzero = fr_tzero; assign f_sign = fr_sign; assign f_inf = fr_inf;
    assign f_sig = fr_sig; assign f_quad = fr_quad; assign f_e = fr_e;

    // Seed pack (combinational from the fold stage): small-angle select, bypass operand, seed angle z0, sideband.
    // Small-angle handoff when the octant-local coordinate is below 2**TSA_BITS, i.e. its bits at TSA_BITS and above
    // are all zero. Written as a slice reduction rather than `f_tpw < (1 << TSA_BITS)`: the unsized `1 << TSA_BITS`
    // collapses to 0 once TSA_BITS >= 32 (WMAN >= 34, e.g. the synthesized WMAN=36) under standard constant sizing,
    // which would silently disable the handoff -- and tools differ on whether context widens it. The slice form is
    // exact and width-independent on every tool.
    wire          sa_sel   = f_tiny | f_tzero | (~|f_tpw[WT-1:TSA_BITS]);
    wire [WOP-1:0] operand = f_tiny ? {{(WOP-WMAN){1'b0}}, f_sig} : f_tpw[WOP-1:0];
    wire signed [WZ-1:0] z0 = $signed({{(WZ-WT-ZG){1'b0}}, f_tpw, {ZG{1'b0}}});
    wire [WSB-1:0] sb_red = {f_e, f_quad, f_octflip, f_tzero, f_tiny, sa_sel, f_inf, f_sign};
    wire eng_start = f_valid;

    // ============================================================================================================
    // Folded CORDIC engine (rotation mode) selected per WMAN. start = eng_start; carries the sideband to done.
    // ============================================================================================================
    wire               cd_done, cd_zdone;
    wire [WSB-1:0]     cd_sb;
    wire signed [WX-1:0] cd_xn;
    wire signed [WX-1:0] cd_yn;
    wire signed [WZ-1:0] cd_zn;
    wire [CWB-1:0]     const2pi;
    // Intentional: unsupported in-range WMAN names missing _zkf_cordic_m<WMAN>, prompting table generation.
    `define ZKF_SINCOS_CORE(W) end else if (WMAN == W) begin : g_m``W \
        _zkf_cordic_m``W #( \
            .MODE(0), .UNROLL100(UNROLL100), .PARALLEL(PARALLEL), .WSB(WSB), \
            .EXPECT_WMAN(WMAN), .EXPECT_N(K), .EXPECT_XF(XF), .EXPECT_WX(WX), .EXPECT_WT(WT), \
            .EXPECT_ZF(ZF), .EXPECT_WZ(WZ), \
            .EXPECT_CONST2PI_W(CWB), .EXPECT_CONST2PI_S(CONST2PI_S) \
        ) u_cordic ( \
            .clk(clk), .rst(rst), .start(eng_start), .sb_in(sb_red), \
            .x0({WX{1'b0}}), .y0({WX{1'b0}}), .z0(z0), \
            .busy(), .done(cd_done), .z_done(cd_zdone), .sb_out(cd_sb), \
            .xn(cd_xn), .yn(cd_yn), .zn(cd_zn), .const2pi(const2pi), \
            .inv_tau(), .kinv_mag(), .kinv());
    generate
        if (1'b0) begin : g_none
        `ZKF_SINCOS_CORE(11)
        `ZKF_SINCOS_CORE(12)
        `ZKF_SINCOS_CORE(13)
        `ZKF_SINCOS_CORE(14)
        `ZKF_SINCOS_CORE(15)
        `ZKF_SINCOS_CORE(16)
        `ZKF_SINCOS_CORE(17)
        `ZKF_SINCOS_CORE(18)
        `ZKF_SINCOS_CORE(19)
        `ZKF_SINCOS_CORE(20)
        `ZKF_SINCOS_CORE(21)
        `ZKF_SINCOS_CORE(22)
        `ZKF_SINCOS_CORE(23)
        `ZKF_SINCOS_CORE(24)
        `ZKF_SINCOS_CORE(25)
        `ZKF_SINCOS_CORE(26)
        `ZKF_SINCOS_CORE(27)
        `ZKF_SINCOS_CORE(28)
        `ZKF_SINCOS_CORE(29)
        `ZKF_SINCOS_CORE(30)
        `ZKF_SINCOS_CORE(31)
        `ZKF_SINCOS_CORE(32)
        `ZKF_SINCOS_CORE(33)
        `ZKF_SINCOS_CORE(34)
        `ZKF_SINCOS_CORE(35)
        `ZKF_SINCOS_CORE(36)
        `ZKF_SINCOS_CORE(37)
        `ZKF_SINCOS_CORE(38)
        `ZKF_SINCOS_CORE(39)
        `ZKF_SINCOS_CORE(40)
        `ZKF_SINCOS_CORE(41)
        `ZKF_SINCOS_CORE(42)
        `ZKF_SINCOS_CORE(43)
        `ZKF_SINCOS_CORE(44)
        `ZKF_SINCOS_CORE(45)
        `ZKF_SINCOS_CORE(46)
        `ZKF_SINCOS_CORE(47)
        `ZKF_SINCOS_CORE(48)
        `ZKF_SINCOS_CORE(49)
        `ZKF_SINCOS_CORE(50)
        `ZKF_SINCOS_CORE(51)
        `ZKF_SINCOS_CORE(52)
        `ZKF_SINCOS_CORE(53)
        end else begin : g_unsupported
            _zkf_invalid_wman_out_of_range u_invalid();
        end
    endgenerate
    `undef ZKF_SINCOS_CORE

    // ============================================================================================================
    // Back-end: one SHARED, pipelined multiplier time-shared over four products, told apart by a 2-bit sideband tag:
    //   BYP : const2pi * operand     -- the small-angle bypass magnitude. operand is known pre-CORDIC, so BYP is issued
    //         during the CORDIC (the multiply is otherwise idle then) and latched; the wide operand need not ride the
    //         engine sideband. The pipelined multiply keeps it in flight independently of the later products.
    //   PHI : tprod = const2pi * z_K -- the residual-angle product; issued at cd_done (z_K only then exists).
    //   S,C : corr_s = x_K * phi, corr_c = y_K * phi,  phi = tprod >> ((CONST2PI_S+ZF)-PHI_S) narrowed (~2*pi*z_K)
    // PHI is serial (S and C both need phi), but S and C are mutually independent and are issued back-to-back into the
    // pipelined multiply (II=1), overlapping in the pipe; the tag routes each result. The last product (C) folds
    // straight into sin = y_K + corr_s / cos = x_K - corr_c the cycle it returns; the octant-local magnitudes then go
    // to the merge.
    // WCP sizes the multiply's B operand to hold both the narrowed residual z_K and the bypass operand (WOP wide).
    localparam integer WCP = ((WOP > (ZF - K)) ? WOP : (ZF - K)) + 2;   // residual / bypass-operand width on B
    // a holds the pre-narrowed const2pi (CWB == WMAN+5 bits) for the const products and the sign-extended x_K/y_K
    // (WXC bits) for the corrections. CWB == WXC-1, so WA collapses to WXC.
    localparam integer WA  = ((CWB + 1) > WXC) ? (CWB + 1) : WXC;       // shared-multiply operand a
    localparam integer WB  = (WCP > WPHI) ? WCP : WPHI;                 // shared-multiply operand b
    localparam integer WP  = WA + WB;                                   // shared-multiply product
    localparam integer P_IDLE = 0, P_PHI = 1, P_SC = 2;
    localparam [1:0] BYP_TAG = 2'd0, PHI_TAG = 2'd1, S_TAG = 2'd2, C_TAG = 2'd3;  // multiply sideband product tags

    reg [2:0]            mphase;
    reg [1:0]            sc_iss;                // S/C issue step: 0 -> issue S, 1 -> issue C, 2 -> done issuing
    reg signed [WX-1:0]  e_xn;
    reg signed [WX-1:0]  e_yn;
    reg [WSB-1:0]        e_sb;
    reg signed [WP-1:0]  tprod_r;        // PHI product const2pi*z_K; phi = tprod_r >> ((CONST2PI_S+ZF)-PHI_S)
    reg signed [WP-1:0]  corr_s_r;       // S product (x_K*phi) registered; C product is consumed straight into b2_cos
    reg [WMAG-1:0]       bypass_mag_r;   // const2pi*operand, computed during the CORDIC (small-angle bypass magnitude)
    reg                  phi_seen;       // this transaction's PHI product (tprod_r) has returned -- skip the P_PHI wait

    wire signed [WCP-1:0] cphi_op   = $signed(cd_zn[WCP-1:0]);          // narrowed CORDIC residual z_K (corr. angle)
    // The PHI product const2pi*z_K is at scale 2**-(CONST2PI_S+ZF); narrow phi to its top WPHI bits at scale
    // 2**-PHI_S by a single right-shift (CONST2PI_S+ZF) - PHI_S. const2pi is already the narrowed WMAN+5-bit operand,
    // so this is a plain product-scale-minus-target-scale shift -- no correction token.
    wire signed [WPHI-1:0] phi      = tprod_r >>> ((CONST2PI_S + ZF) - PHI_S);  // ~2*pi*z_K narrowed to scale 2**-PHI_S
    wire signed [WXC-1:0]  xc       = e_xn >>> XK_TRUNC;
    wire signed [WXC-1:0]  yc       = e_yn >>> XK_TRUNC;
    // Shared-multiply operand select. a = the pre-narrowed const2pi for the const products (BYP during the CORDIC,
    // PHI at cd_zdone); x_K / y_K for the S / C corrections. b = bypass operand, residual z_K, or phi.
    wire                  issue_c   = (mphase == P_SC) && (sc_iss == 2'd1);
    wire signed [WA-1:0]  mul_a_sel = (mphase != P_SC) ? $signed({{(WA-CWB){1'b0}}, const2pi})
                                    : issue_c          ? $signed({{(WA-WXC){yc[WXC-1]}}, yc})
                                    :                    $signed({{(WA-WXC){xc[WXC-1]}}, xc});
    // In P_IDLE the multiply issues BYP at eng_start (b = bypass operand) and PHI at cd_zdone (b = the residual z_K
    // read straight off the engine output cd_zn -- these two events are on distinct cycles); S/C present phi in P_SC.
    // cd_zdone leads cd_done in the decoupled engine (==cd_done in lock-step), so PHI multiply overlaps CORDIC tail.
    wire signed [WB-1:0]  mul_b_sel = (mphase != P_IDLE) ? $signed({{(WB-WPHI){phi[WPHI-1]}}, phi})
                                    : cd_zdone           ? $signed({{(WB-WCP){cphi_op[WCP-1]}}, cphi_op})
                                    :                      $signed({{(WB-WOP){1'b0}}, operand});
    wire [1:0]            mul_sb_in = (mphase != P_IDLE) ? (issue_c ? C_TAG : S_TAG)
                                    : cd_zdone           ? PHI_TAG : BYP_TAG;
    wire signed [WP-1:0]   pmul_p;             // exact product of the current phase's operands (shared multiply)

    // One multiplier time-shared over the products. Up to two products are in flight at once (BYP overlaps the CORDIC;
    // S/C are pipelined two-deep); the 2-bit sideband tag (sb_in -> sb_out) carried in step with each product routes
    // its result. BYP and PHI are issued combinationally (at eng_start and cd_zdone, both in P_IDLE, off the engine
    // outputs); S and C are issued by the FSM in P_SC.
    wire       mul_valid;
    wire [1:0] mul_sb_out;
    wire       mul_in_valid = ((mphase == P_IDLE) && (eng_start | cd_zdone)) | ((mphase == P_SC) && (sc_iss < 2'd2));
    _zkf_pmul #(
        .WA(WA), .WB(WB), .A_SIGNED(1), .B_SIGNED(1), .WSB(2), .WMULTIPLIER(WMULTIPLIER), .STAGE_PRODUCT(STAGE_PRODUCT)
    ) u_pmul (
        .clk(clk), .rst(rst), .in_valid(mul_in_valid), .sb_in(mul_sb_in),
        .a(mul_a_sel), .b(mul_b_sel),
        .out_valid(mul_valid), .sb_out(mul_sb_out), .p(pmul_p)
    );

    reg signed [WX-1:0]  b2_sin, b2_cos;
    reg [WMAG-1:0]       b2_sa;
    reg [WSB-1:0]        b2_sb;
    reg                  b2_valid;
    always @(posedge clk) begin
        if (rst) begin
            mphase    <= P_IDLE[2:0];
            sc_iss    <= 2'd0;
            b2_valid  <= 1'b0;
            phi_seen  <= 1'b0;
        end else begin
            b2_valid  <= 1'b0;
            // Track whether this transaction's PHI product has returned. Armed at accept (far ahead of the CORDIC) and
            // set when PHI_TAG comes back -- which is DURING the CORDIC when decoupled (cd_zdone leads cd_done), so by
            // the time cd_done arrives phi is already in tprod_r and the back-end skips the P_PHI wait.
            if (accept) phi_seen <= 1'b0;
            else if (mul_valid && (mul_sb_out == PHI_TAG)) phi_seen <= 1'b1;
            if (mul_valid) begin
                case (mul_sb_out)
                    BYP_TAG: bypass_mag_r <= pmul_p;   // sign-extend the non-negative WP-bit product to WMAG (WMAG>WP)
                    PHI_TAG: tprod_r      <= pmul_p;
                    S_TAG:   corr_s_r     <= pmul_p;
                    default: ;                                   // C_TAG: consumed directly into b2_cos below
                endcase
            end
            case (mphase)
                P_IDLE[2:0]: if (cd_done) begin                  // x_K/y_K valid now; PHI was issued earlier (cd_zdone)
                    e_xn <= cd_xn; e_yn <= cd_yn; e_sb <= cd_sb;
                    if (phi_seen) begin                          // decoupled: phi already in tprod_r -> straight to S/C
                        mphase <= P_SC[2:0];
                        sc_iss <= 2'd0;
                    end else begin                               // lock-step: PHI issued this cycle, wait for it
                        mphase <= P_PHI[2:0];
                    end
                end
                // P_PHI is entered only in lock-step, where the PHI product is already valid this cycle, so the guard
                // is always true; the implicit "still waiting" else is unreachable.
                // verilator coverage_off
                P_PHI[2:0]: if (mul_valid && (mul_sb_out == PHI_TAG)) begin   // phi captured above; begin S/C
                    mphase <= P_SC[2:0];
                    sc_iss <= 2'd0;
                end
                // verilator coverage_on
                P_SC[2:0]: begin
                    if (sc_iss < 2'd2) sc_iss <= sc_iss + 2'd1;  // issue S (sc_iss 0), then C (sc_iss 1), next cycle
                    if (mul_valid && (mul_sb_out == C_TAG)) begin   // C is last: form sin/cos now, no PACK cycle
                        b2_sin   <= e_yn + (corr_s_r >>> CORR_SHIFT);   // y_K + x_K*phi  (corr_s registered)
                        b2_cos   <= e_xn - (pmul_p   >>> CORR_SHIFT);   // x_K - y_K*phi  (C product, live this cycle)
                        b2_sa    <= bypass_mag_r;           // const2pi*operand, computed early during the CORDIC
                        b2_sb    <= e_sb;
                        b2_valid <= 1'b1;
                        mphase   <= P_IDLE[2:0];
                    end
                end
                // FSM uses P_IDLE/P_PHI/P_SC only; the default is a generate-completeness safety arm.
                // verilator coverage_off
                default: mphase <= P_IDLE[2:0];
                // verilator coverage_on
            endcase
        end
    end

    // Merge: octant + quadrant unmap, signs, exp_offset; then the shared _zkf_fixed_to_float back-end.
    wire signed [WE-1:0] e_o     = $signed(b2_sb[WSB-1 -: WE]);
    wire [1:0]           quad_o  = b2_sb[7 -: 2];
    wire                 oct_o   = b2_sb[5];
    wire                 tzero_o = b2_sb[4];
    wire                 tiny_o  = b2_sb[3];
    wire                 sa_o    = b2_sb[2];
    wire                 inf_o   = b2_sb[1];
    wire                 sign_o  = b2_sb[0];

    // Each magnitude is read back with the exp_offset matching ITS native scale: the corr/one path (b2_sin/b2_cos and
    // the exact +1) lives at scale 2**-XF -> EONE_XF; the bypass/tiny/TSA magnitude is the const2pi product at scale
    // 2**-CONST2PI_S -> EONE_S, then minus the angle's own scale. No correction token: const2pi is already narrowed.
    localparam signed [WEU-1:0] EONE_XF_S    = EONE_XF;            // O(1) cos / +1 / corr path (scale 2**-XF)
    localparam signed [WEU-1:0] EONE_S_WFRAC = EONE_S - WFRAC;     // tiny bypass: const2pi*|sig| at 2**-CONST2PI_S
    localparam signed [WEU-1:0] EONE_S_ZFT   = EONE_S - (WT + 2);  // TSA bypass: const2pi*t' at 2**-CONST2PI_S
    wire signed [WEU-1:0] e_ext = $signed({{(WEU-WE){e_o[WE-1]}}, e_o});
    wire [WMAG-1:0] sin_tp_mag = sa_o ? b2_sa                                   : {{(WMAG-XF-1){1'b0}}, b2_sin[XF:0]};
    wire [WMAG-1:0] cos_tp_mag = sa_o ? {{(WMAG-XF-1){1'b0}}, 1'b1, {XF{1'b0}}} : {{(WMAG-XF-1){1'b0}}, b2_cos[XF:0]};
    wire signed [WEU-1:0] sin_tp_exp = !sa_o  ? EONE_XF_S
                                     : tiny_o ? (e_ext + EONE_S_WFRAC)
                                     :          EONE_S_ZFT;

    wire [WMAG-1:0]       sin_loc_mag = oct_o ? cos_tp_mag : sin_tp_mag;
    wire signed [WEU-1:0] sin_loc_exp = oct_o ? EONE_XF_S  : sin_tp_exp;
    wire [WMAG-1:0]       cos_loc_mag = oct_o ? sin_tp_mag : cos_tp_mag;
    wire signed [WEU-1:0] cos_loc_exp = oct_o ? sin_tp_exp : EONE_XF_S;
    wire [WMAG-1:0]       sin_mag = quad_o[0] ? cos_loc_mag : sin_loc_mag;
    wire signed [WEU-1:0] sin_exp = quad_o[0] ? cos_loc_exp : sin_loc_exp;
    wire [WMAG-1:0]       cos_mag = quad_o[0] ? sin_loc_mag : cos_loc_mag;
    wire signed [WEU-1:0] cos_exp = quad_o[0] ? sin_loc_exp : cos_loc_exp;
    wire sin_sgn = inf_o ? sign_o : (quad_o[1] ^ sign_o);
    wire cos_sgn = inf_o ? sign_o : (quad_o[1] ^ quad_o[0]);
    wire [1:0] quad_out = inf_o   ? 2'b00
                        : ~sign_o ? quad_o
                        : tzero_o ? (2'd0 - quad_o)
                        :           (2'd3 - quad_o);

    // Stage B3: register one merged payload at a time so the wide octant + quadrant magnitude muxing (WMAG-bit 4:1
    // trees) is its own stage. SIN is captured from the b2 state first; COS is captured one cycle later from the same
    // stable b2 state (single transaction in flight). This preserves the f2f issue timing while avoiding a second
    // WMAG-wide payload register bank.
    reg                  sh_valid, sh_is_cos, sh_inf, sh_sgn;
    reg [WMAG-1:0]       sh_mag;
    reg signed [WEU-1:0] sh_exp;
    reg [1:0]            sh_quad;
    always @(posedge clk) begin
        if (rst)                         sh_valid <= 1'b0;
        else if (b2_valid)               sh_valid <= 1'b1;
        else if (sh_valid && !sh_is_cos) sh_valid <= 1'b1;
        else                             sh_valid <= 1'b0;

        if (b2_valid) begin
            sh_is_cos <= 1'b0;
            sh_inf    <= inf_o;
            sh_sgn    <= sin_sgn;
            sh_mag    <= sin_mag;
            sh_exp    <= sin_exp;
            sh_quad   <= quad_out;
        end else if (sh_valid && !sh_is_cos) begin
            sh_is_cos <= 1'b1;
            sh_inf    <= inf_o;
            sh_sgn    <= cos_sgn;
            sh_mag    <= cos_mag;
            sh_exp    <= cos_exp;
            sh_quad   <= 2'b00;
        end
    end

    // One shared _zkf_fixed_to_float back-end. The registered B3 payload issues SIN, then COS one cycle later. The
    // back-end output is tagged; packed SIN and quadrant are latched when the SIN pass emerges, then paired with the
    // packed COS when the COS pass emerges one cycle later.
    localparam integer WSB2 = 3;                         // {is_cos, quadrant}; quadrant is meaningful on the SIN pass
    wire [WSB2-1:0]       sh_sb     = {sh_is_cos, sh_quad};

`ifdef SIMULATION
    always @(posedge clk) begin
        if (!rst && b2_valid && sh_valid)
            $fatal(1, "zkf_sincos: shared back-end collision -- new payload before prior pair issued");
    end
`endif

    wire             be_ov;
    wire [WFULL-1:0] be_num;
    wire [WSB2-1:0]  be_sbo;
    _zkf_fixed_to_float #(
        .WEXP(WEXP), .WMAN(WMAN), .WMAG(WMAG), .WEU(WEU),
        .EXP_IS_BIASED(0), .ASSUME_NO_OVERFLOW(1), .WSB(WSB2),
        .STAGE_NORMALIZE(STAGE_NORMALIZE), .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(0)
    ) u_f2f (
        .clk(clk), .rst(rst),
        .in_valid(sh_valid), .sign(sh_sgn), .force_zero(1'b0), .force_inf(sh_inf),
        .exp_offset(sh_exp), .mag(sh_mag), .sb_in(sh_sb),
        .out_valid(be_ov), .y(be_num), .sb_out(be_sbo)
    );

    wire             be_is_cos   = be_sbo[WSB2-1];
    wire [1:0]       be_quad_sin = be_sbo[1:0];
    reg [WFULL-1:0]  sin_num_r;
    reg [1:0]        quad_r;
    always @(posedge clk) begin
        if (be_ov && !be_is_cos) begin
            sin_num_r <= be_num;
            quad_r    <= be_quad_sin;
        end
    end

    wire             be_valid = be_ov & be_is_cos;
    wire [WFULL-1:0] be_sin   = sin_num_r;
    wire [WFULL-1:0] be_cos   = be_num;
    wire [1:0]       be_quad  = quad_r;

    // Output handshake with back-pressure. Only one transaction is ever in flight (busy stalls the engine until the
    // finished result is taken), so the result simply waits for out_ready. STAGE_OUTPUT selects WHERE it is held:
    //   0: combinational output -- be_* is presented on its valid cycle; a hold register catches it while out_ready is
    //      low (no added output-register latency when out_ready is high).
    //   1: a hard output register drives sin/cos/quadrant DIRECTLY (no combinational logic after it); it captures the
    //      result and holds it until out_ready (+1 cycle). This is the clean version a downstream stage registers off.
    generate
        if (STAGE_OUTPUT == 0) begin : g_out_comb
            reg              pending;
            reg [WFULL-1:0]  hold_sin, hold_cos;
            reg [1:0]        hold_quad;
            always @(posedge clk) begin
                if (rst) pending <= 1'b0;
                else if (be_valid & ~out_ready) begin
                    pending  <= 1'b1;
                    hold_sin <= be_sin; hold_cos <= be_cos; hold_quad <= be_quad;
                end else if (pending & out_ready) begin
                    pending  <= 1'b0;
                end
            end
            assign out_valid = be_valid | pending;
            assign sin       = pending ? hold_sin : be_sin;
            assign cos       = pending ? hold_cos : be_cos;
            assign quadrant  = pending ? hold_quad : be_quad;
        end else begin : g_out_reg
            reg              r_valid;
            reg [WFULL-1:0]  r_sin, r_cos;
            reg [1:0]        r_quad;
            always @(posedge clk) begin
                if (rst)            r_valid <= 1'b0;
                else if (be_valid)  r_valid <= 1'b1;     // result captured -> output valid next cycle
                else if (out_ready) r_valid <= 1'b0;     // consumed -> clear (single transaction: no new result yet)
                if (be_valid) begin r_sin <= be_sin; r_cos <= be_cos; r_quad <= be_quad; end
            end
            assign out_valid = r_valid;
            assign sin       = r_sin;
            assign cos       = r_cos;
            assign quadrant  = r_quad;
        end
    endgenerate

    // busy: set on accept, cleared when the consumer takes the result (out_valid & out_ready). One transaction in
    // flight at a time, so the engine/back-end stall while a finished result waits for out_ready.
    always @(posedge clk) begin
        if (rst)                        busy <= 1'b0;
        else if (accept)                busy <= 1'b1;
        else if (out_valid & out_ready) busy <= 1'b0;
    end

    // Intentionally-partial nets: the engine residual angle cd_zn drives the linear correction only through its low
    // bits (cd_zn[WCP-1:0]), and the octant-local magnitudes b2_sin/b2_cos are read only as their low XF+1 bits, so the
    // upper bits never reach an output. Reduction-xor the full vectors so the leftover bits read as used.
    wire _unused = ^{cd_zn, b2_sin, b2_cos};
endmodule

`default_nettype wire

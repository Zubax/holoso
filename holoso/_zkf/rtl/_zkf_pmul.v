/// Shared chip-agnostic pipelined integer multiply: p = a * b, exact, with per-operand signedness.
///
/// The operands are split into a GA x GB grid of near-equal slices; each slice product is a plain `*` the synthesis
/// tool maps to however many DSP tiles the target device has. The register between the slice products and their sum
/// breaks the timing-critical fabric cascade, so wide multiplies close timing on any part.
///
/// A_SIGNED / B_SIGNED mark each operand signed (1) or unsigned (0); the caller passes its NATIVE width.
/// The output `p` is the raw WA+WB product bits; the caller interprets them as signed or unsigned. Behavior:
///
///     Fully-unsigned (both 0): An unsigned slice grid -- full WMULTIPLIER-bit slices, unsigned products/sum (an
///                              18-bit unsigned operand at WMULTIPLIER=18 is ONE tile, no wasted bit).
///
///     Signed or mixed: A signed slice grid -- each operand's top slice is sign/zero-extended per its flag, lower
///                      slices are zero-prepended (a signed*unsigned slice product must carry the unsigned operand
///                      non-negative-signed, costing the sign bit, hence P = WMULTIPLIER-1). Both grids reconstruct
///                      the exact product.
///
/// STAGE_PRODUCT sets the pipeline depth, or the number of register stages (latency).
/// WMULTIPLIER chooses the grid construction (no effect on the latency), and applies only when the operands are
/// actually split (STAGE_PRODUCT >= 2, otherwise ignored):
///
///   WMULTIPLIER=0: symmetric split GA = GB = {native, native, 2, 3, 3}[STAGE_PRODUCT].
///   WMULTIPLIER>0: Each slice fits a WMULTIPLIER-bit signed/unsigned tile; the caller states its DSP width once;
///                  the core derives the minimal -- possibly asymmetric -- grid, e.g. 66x41 -> 4x3.
///                  P = WMULTIPLIER for the fully-unsigned grid, WMULTIPLIER-1 when a signed operand is present.
///
/// Register stages (latency) = 1+STAGE_PRODUCT:
///
///   STAGE_PRODUCT=0: single behavioural `a*b` -> output reg.
///   STAGE_PRODUCT=1: operand-capture reg, then the SAME native `a*b` -> output reg (enables better auto-tiling).
///   STAGE_PRODUCT=2: operand-capture reg, manual 2x2 split (one registered partial-product reduction).
///   STAGE_PRODUCT=3: operand-capture reg, manual 3x3 split (row-sum register, then the column-sum register).
///   STAGE_PRODUCT=4: same 3x3 grid, but the final column sum is pipelined into two stages (pairwise, then sum).

`default_nettype none

module _zkf_pmul #(
    parameter WA            = 16,  // operand a width  (illustrative default; the module works for any width)
    parameter WB            = 16,
    parameter A_SIGNED      = 1,   // 1 = a is signed (top slice sign-extended); 0 = unsigned
    parameter B_SIGNED      = 1,   // 1 = b is signed; 0 = unsigned
    parameter WSB           = 1,   // sideband carried alongside the pipeline
    parameter WMULTIPLIER   = 0,
    parameter STAGE_PRODUCT = 0
) (
    input  wire             clk,
    input  wire             rst,        // resets only the valid pipe (control); datapath regs free-run
    input  wire             in_valid,
    input  wire   [WSB-1:0] sb_in,
    input  wire    [WA-1:0] a,
    input  wire    [WB-1:0] b,
    output wire             out_valid,
    output wire   [WSB-1:0] sb_out,
    output wire [WA+WB-1:0] p           // exact a*b (raw bits; caller assigns signedness)
);
    localparam integer WP  = WA + WB;
    localparam integer SYM = (STAGE_PRODUCT >= 3) ? 3 : 2;
    // Signedness-aware slice payload: a signed*unsigned slice product must carry the unsigned operand as
    // non-negative-signed (+1 bit, an unavoidable behavioural cost), so only the fully-unsigned mode uses the full
    // WMULTIPLIER payload. P = 1 when WMULTIPLIER = 0 keeps the (then-unused) ceil division well-defined.
    localparam integer SIGNED_PATH = ((A_SIGNED != 0) || (B_SIGNED != 0)) ? 1 : 0;
    localparam integer P           = (WMULTIPLIER == 0) ? 1 : (SIGNED_PATH ? (WMULTIPLIER - 1) : WMULTIPLIER);
    localparam integer GA          = (WMULTIPLIER == 0) ? SYM : ((WA + P - 1) / P);   // ceil(WA/P)
    localparam integer GB          = (WMULTIPLIER == 0) ? SYM : ((WB + P - 1) / P);   // ceil(WB/P)

    generate
        if ((STAGE_PRODUCT < 0) || (STAGE_PRODUCT > 4)) begin : g_invalid_stage_product
            _zkf_invalid_stage_product_out_of_range u_invalid();
        end
        if ((WMULTIPLIER != 0) && (WMULTIPLIER < 8)) begin : g_invalid_wmultiplier
            _zkf_invalid_pmul_wmultiplier u_invalid();   // 0 (symmetric) or a sensible tile width (>= 8)
        end
        if ((STAGE_PRODUCT >= 2) && ((GA < 1) || (GB < 1) || (WA < GA) || (WB < GB))) begin : g_invalid_split
            _zkf_invalid_pmul_split u_invalid();   // need >= 1 bit per slice
        end
    endgenerate

    // Even split helpers: slice k of an N-way split of a w-bit operand. Low slices (small k) absorb the remainder,
    // so widths are ceil(w/N) down to floor(w/N) and sum to w; offsets are the running prefix sum.
    // Constant functions evaluated only at elaboration (slice widths/offsets); no runtime line coverage.
    // verilator coverage_off
    function automatic integer slc_w(input integer w, input integer n, input integer k);
        slc_w = (w + n - 1 - k) / n;
    endfunction
    function automatic integer slc_off(input integer w, input integer n, input integer k);
        integer i;
        begin
            slc_off = 0;
            for (i = 0; i < k; i = i + 1) slc_off = slc_off + ((w + n - 1 - i) / n);
        end
    endfunction
    // verilator coverage_on

    // Operand-capture stage (STAGE_PRODUCT >= 1): register the free-floating operands right before the product so
    // the placer can sit a latch at the DSP inputs.
    wire [WA-1:0]  c_a;
    wire [WB-1:0]  c_b;
    wire           c_v;
    wire [WSB-1:0] c_sb;
    generate
        if (STAGE_PRODUCT != 0) begin : g_capture
            reg [WA-1:0]  x_a;
            reg [WB-1:0]  x_b;
            reg           x_v;
            reg [WSB-1:0] x_sb;
            always @(posedge clk) begin
                if (rst) x_v <= 1'b0;
                else     x_v <= in_valid;
                x_a  <= a;
                x_b  <= b;
                x_sb <= sb_in;
            end
            assign c_a = x_a; assign c_b = x_b; assign c_v = x_v; assign c_sb = x_sb;
        end else begin : g_no_capture
            assign c_a = a; assign c_b = b; assign c_v = in_valid; assign c_sb = sb_in;
        end
    endgenerate

    generate
        if (STAGE_PRODUCT <= 1) begin : g_single
            wire [WP-1:0] prod;
            if ((A_SIGNED != 0) && (B_SIGNED != 0)) begin : g_ss
                assign prod = $signed(c_a) * $signed(c_b);
            end else if ((A_SIGNED == 0) && (B_SIGNED == 0)) begin : g_uu
                assign prod = c_a * c_b;
            end else if (A_SIGNED != 0) begin : g_su
                assign prod = $signed(c_a) * $signed({1'b0, c_b});
            end else begin : g_us
                assign prod = $signed({1'b0, c_a}) * $signed(c_b);
            end
            reg [WP-1:0]  r_p;
            reg           r_v;
            reg [WSB-1:0] r_sb;
            always @(posedge clk) begin
                if (rst) r_v <= 1'b0;
                else     r_v <= c_v;
                r_p  <= prod;
                r_sb <= c_sb;
            end
            assign p = r_p; assign out_valid = r_v; assign sb_out = r_sb;
        end else if (SIGNED_PATH != 0) begin : g_signed
            localparam integer MAXSA = ((WA + GA - 1) / GA) + 1;  // widest a-slice + sign bit
            localparam integer MAXSB = ((WB + GB - 1) / GB) + 1;
            localparam integer WSP   = MAXSA + MAXSB;             // signed slice-product width
            wire signed [MAXSA-1:0] a_s [0:GA-1];
            wire signed [MAXSB-1:0] b_s [0:GB-1];
            wire signed [WSP-1:0]   pp  [0:GA*GB-1];
            genvar gi, gj;
            for (gi = 0; gi < GA; gi = gi + 1) begin : g_aslice
                localparam integer WI = slc_w(WA, GA, gi);
                localparam integer OI = slc_off(WA, GA, gi);
                if ((gi == GA - 1) && (A_SIGNED != 0)) assign a_s[gi] = $signed(c_a[OI +: WI]);
                else                                   assign a_s[gi] = $signed({1'b0, c_a[OI +: WI]});
            end
            for (gj = 0; gj < GB; gj = gj + 1) begin : g_bslice
                localparam integer WJ = slc_w(WB, GB, gj);
                localparam integer OJ = slc_off(WB, GB, gj);
                if ((gj == GB - 1) && (B_SIGNED != 0)) assign b_s[gj] = $signed(c_b[OJ +: WJ]);
                else                                   assign b_s[gj] = $signed({1'b0, c_b[OJ +: WJ]});
            end
            for (gi = 0; gi < GA; gi = gi + 1) begin : g_pp_row
                for (gj = 0; gj < GB; gj = gj + 1) begin : g_pp_col
                    assign pp[gi*GB + gj] = a_s[gi] * b_s[gj];
                end
            end

            // -- Register the GA*GB slice products. --
            reg signed [WSP-1:0] m_pp [0:GA*GB-1];
            reg           m_v;
            reg [WSB-1:0] m_sb;
            integer mi;
            always @(posedge clk) begin
                if (rst) m_v <= 1'b0;
                else     m_v <= c_v;
                for (mi = 0; mi < GA*GB; mi = mi + 1) m_pp[mi] <= pp[mi];
                m_sb <= c_sb;
            end

            if (STAGE_PRODUCT == 2) begin : g_flat
                wire signed [WP-1:0] term [0:GA*GB-1];
                genvar ti, tj;
                for (ti = 0; ti < GA; ti = ti + 1) begin : g_term_row
                    for (tj = 0; tj < GB; tj = tj + 1) begin : g_term_col
                        localparam integer SH = slc_off(WA, GA, ti) + slc_off(WB, GB, tj);
                        assign term[ti*GB + tj] = $signed(m_pp[ti*GB + tj]) <<< SH;
                    end
                end
                reg signed [WP-1:0] csum;
                integer ci;
                always @* begin
                    csum = {WP{1'b0}};
                    for (ci = 0; ci < GA*GB; ci = ci + 1) csum = csum + term[ci];
                end
                reg [WP-1:0]  r_p;
                reg           r_v;
                reg [WSB-1:0] r_sb;
                always @(posedge clk) begin
                    if (rst) r_v <= 1'b0;
                    else     r_v <= m_v;
                    r_p  <= csum;
                    r_sb <= m_sb;
                end
                assign p = r_p; assign out_valid = r_v; assign sb_out = r_sb;
            end else if (STAGE_PRODUCT == 3) begin : g_rows
                wire signed [WP-1:0] brow [0:GA*GB-1];
                genvar ri, rj;
                for (ri = 0; ri < GA; ri = ri + 1) begin : g_brow_row
                    for (rj = 0; rj < GB; rj = rj + 1) begin : g_brow_col
                        localparam integer OJ = slc_off(WB, GB, rj);
                        assign brow[ri*GB + rj] = $signed(m_pp[ri*GB + rj]) <<< OJ;
                    end
                end
                reg signed [WP-1:0] rowc [0:GA-1];
                integer ri2, rj2;
                always @* begin
                    for (ri2 = 0; ri2 < GA; ri2 = ri2 + 1) begin
                        rowc[ri2] = {WP{1'b0}};
                        for (rj2 = 0; rj2 < GB; rj2 = rj2 + 1) rowc[ri2] = rowc[ri2] + brow[ri2*GB + rj2];
                    end
                end
                reg signed [WP-1:0] s_row [0:GA-1];
                reg           s_v;
                reg [WSB-1:0] s_sb;
                integer si;
                always @(posedge clk) begin
                    if (rst) s_v <= 1'b0;
                    else     s_v <= m_v;
                    for (si = 0; si < GA; si = si + 1) s_row[si] <= rowc[si];
                    s_sb <= m_sb;
                end
                wire signed [WP-1:0] arow [0:GA-1];
                genvar ai;
                for (ai = 0; ai < GA; ai = ai + 1) begin : g_arow
                    localparam integer OI = slc_off(WA, GA, ai);
                    assign arow[ai] = s_row[ai] <<< OI;
                end
                reg signed [WP-1:0] csum;
                integer ci;
                always @* begin
                    csum = {WP{1'b0}};
                    for (ci = 0; ci < GA; ci = ci + 1) csum = csum + arow[ci];
                end
                reg [WP-1:0]  r_p;
                reg           r_v;
                reg [WSB-1:0] r_sb;
                always @(posedge clk) begin
                    if (rst) r_v <= 1'b0;
                    else     r_v <= s_v;
                    r_p  <= csum;
                    r_sb <= s_sb;
                end
                assign p = r_p; assign out_valid = r_v; assign sb_out = r_sb;
            end else if (STAGE_PRODUCT == 4) begin : g_rows2
                // Same GA x GB signed grid as g_rows, but the final GA-way column sum is split across two register
                // stages: pairwise partial sums (s_col), then their sum. Halves the reduction adder depth for very
                // wide accumulators where the single-stage column sum is the limiter.
                wire signed [WP-1:0] brow [0:GA*GB-1];
                genvar zi, zj;
                for (zi = 0; zi < GA; zi = zi + 1) begin : g_brow_row
                    for (zj = 0; zj < GB; zj = zj + 1) begin : g_brow_col
                        localparam integer OJ = slc_off(WB, GB, zj);
                        assign brow[zi*GB + zj] = $signed(m_pp[zi*GB + zj]) <<< OJ;
                    end
                end
                reg signed [WP-1:0] rowc [0:GA-1];
                integer zr, zc;
                always @* begin
                    for (zr = 0; zr < GA; zr = zr + 1) begin
                        rowc[zr] = {WP{1'b0}};
                        for (zc = 0; zc < GB; zc = zc + 1) rowc[zr] = rowc[zr] + brow[zr*GB + zc];
                    end
                end
                reg signed [WP-1:0] s_row [0:GA-1];
                reg           s_v;
                reg [WSB-1:0] s_sb;
                integer zs;
                always @(posedge clk) begin
                    if (rst) s_v <= 1'b0;
                    else     s_v <= m_v;
                    for (zs = 0; zs < GA; zs = zs + 1) s_row[zs] <= rowc[zs];
                    s_sb <= m_sb;
                end
                wire signed [WP-1:0] arow [0:GA-1];
                wire signed [WP-1:0] psum [0:((GA+1)/2)-1];
                genvar yi;
                for (yi = 0; yi < GA; yi = yi + 1) begin : g_arow
                    localparam integer OI = slc_off(WA, GA, yi);
                    assign arow[yi] = s_row[yi] <<< OI;
                end
                localparam integer NH = (GA + 1) / 2;     // pairwise partial sums of the GA shifted rows
                genvar hi;
                for (hi = 0; hi < NH; hi = hi + 1) begin : g_psum
                    if (2*hi + 1 < GA) begin : g_pair assign psum[hi] = arow[2*hi] + arow[2*hi + 1]; end
                    else               begin : g_lone assign psum[hi] = arow[2*hi];                  end
                end
                reg signed [WP-1:0] s_col [0:NH-1];
                reg           t_v;
                reg [WSB-1:0] t_sb;
                integer zt;
                always @(posedge clk) begin
                    if (rst) t_v <= 1'b0;
                    else     t_v <= s_v;
                    for (zt = 0; zt < NH; zt = zt + 1) s_col[zt] <= psum[zt];
                    t_sb <= s_sb;
                end
                reg signed [WP-1:0] csum;
                integer zk;
                always @* begin
                    csum = {WP{1'b0}};
                    for (zk = 0; zk < NH; zk = zk + 1) csum = csum + s_col[zk];
                end
                reg [WP-1:0]  r_p;
                reg           r_v;
                reg [WSB-1:0] r_sb;
                always @(posedge clk) begin
                    if (rst) r_v <= 1'b0;
                    else     r_v <= t_v;
                    r_p  <= csum;
                    r_sb <= t_sb;
                end
                assign p = r_p; assign out_valid = r_v; assign sb_out = r_sb;
            end else begin : g_invalid_grid_stage
                _zkf_invalid_pmul_grid_stage u_invalid();
            end
        end else begin : g_unsigned
            // Fully-unsigned GA x GB slice grid (A_SIGNED = B_SIGNED = 0): full-width unsigned slices (no sign bit),
            // unsigned slice products, unsigned shift-aligned sum. A WMULTIPLIER-bit slice fills a whole unsigned tile.
            localparam integer MAXSA = (WA + GA - 1) / GA;   // widest a-slice (no sign bit)
            localparam integer MAXSB = (WB + GB - 1) / GB;
            localparam integer WSP   = MAXSA + MAXSB;
            wire [MAXSA-1:0] a_u [0:GA-1];
            wire [MAXSB-1:0] b_u [0:GB-1];
            wire [WSP-1:0]   pp  [0:GA*GB-1];
            genvar gi, gj;
            for (gi = 0; gi < GA; gi = gi + 1) begin : g_aslice
                localparam integer WI = slc_w(WA, GA, gi);
                localparam integer OI = slc_off(WA, GA, gi);
                assign a_u[gi] = c_a[OI +: WI];
            end
            for (gj = 0; gj < GB; gj = gj + 1) begin : g_bslice
                localparam integer WJ = slc_w(WB, GB, gj);
                localparam integer OJ = slc_off(WB, GB, gj);
                assign b_u[gj] = c_b[OJ +: WJ];
            end
            for (gi = 0; gi < GA; gi = gi + 1) begin : g_pp_row
                for (gj = 0; gj < GB; gj = gj + 1) begin : g_pp_col
                    assign pp[gi*GB + gj] = a_u[gi] * b_u[gj];
                end
            end
            // -- Register the GA*GB slice products. --
            reg [WSP-1:0] m_pp [0:GA*GB-1];
            reg           m_v;
            reg [WSB-1:0] m_sb;
            integer mi;
            always @(posedge clk) begin
                if (rst) m_v <= 1'b0;
                else     m_v <= c_v;
                for (mi = 0; mi < GA*GB; mi = mi + 1) m_pp[mi] <= pp[mi];
                m_sb <= c_sb;
            end

            if (STAGE_PRODUCT == 2) begin : g_flat
                wire [WP-1:0] term [0:GA*GB-1];
                genvar ti, tj;
                for (ti = 0; ti < GA; ti = ti + 1) begin : g_term_row
                    for (tj = 0; tj < GB; tj = tj + 1) begin : g_term_col
                        localparam integer SH = slc_off(WA, GA, ti) + slc_off(WB, GB, tj);
                        assign term[ti*GB + tj] = m_pp[ti*GB + tj] << SH;
                    end
                end
                reg [WP-1:0] csum;
                integer ci;
                always @* begin
                    csum = {WP{1'b0}};
                    for (ci = 0; ci < GA*GB; ci = ci + 1) csum = csum + term[ci];
                end
                reg [WP-1:0]  r_p;
                reg           r_v;
                reg [WSB-1:0] r_sb;
                always @(posedge clk) begin
                    if (rst) r_v <= 1'b0;
                    else     r_v <= m_v;
                    r_p  <= csum;
                    r_sb <= m_sb;
                end
                assign p = r_p; assign out_valid = r_v; assign sb_out = r_sb;
            end else if (STAGE_PRODUCT == 3) begin : g_rows
                wire [WP-1:0] brow [0:GA*GB-1];
                genvar ri, rj;
                for (ri = 0; ri < GA; ri = ri + 1) begin : g_brow_row
                    for (rj = 0; rj < GB; rj = rj + 1) begin : g_brow_col
                        localparam integer OJ = slc_off(WB, GB, rj);
                        assign brow[ri*GB + rj] = m_pp[ri*GB + rj] << OJ;
                    end
                end
                reg [WP-1:0] rowc [0:GA-1];
                integer ri2, rj2;
                always @* begin
                    for (ri2 = 0; ri2 < GA; ri2 = ri2 + 1) begin
                        rowc[ri2] = {WP{1'b0}};
                        for (rj2 = 0; rj2 < GB; rj2 = rj2 + 1) rowc[ri2] = rowc[ri2] + brow[ri2*GB + rj2];
                    end
                end
                reg [WP-1:0] s_row [0:GA-1];
                reg           s_v;
                reg [WSB-1:0] s_sb;
                integer si;
                always @(posedge clk) begin
                    if (rst) s_v <= 1'b0;
                    else     s_v <= m_v;
                    for (si = 0; si < GA; si = si + 1) s_row[si] <= rowc[si];
                    s_sb <= m_sb;
                end
                wire [WP-1:0] arow [0:GA-1];
                genvar ai;
                for (ai = 0; ai < GA; ai = ai + 1) begin : g_arow
                    localparam integer OI = slc_off(WA, GA, ai);
                    assign arow[ai] = s_row[ai] << OI;
                end
                reg [WP-1:0] csum;
                integer ci;
                always @* begin
                    csum = {WP{1'b0}};
                    for (ci = 0; ci < GA; ci = ci + 1) csum = csum + arow[ci];
                end
                reg [WP-1:0]  r_p;
                reg           r_v;
                reg [WSB-1:0] r_sb;
                always @(posedge clk) begin
                    if (rst) r_v <= 1'b0;
                    else     r_v <= s_v;
                    r_p  <= csum;
                    r_sb <= s_sb;
                end
                assign p = r_p; assign out_valid = r_v; assign sb_out = r_sb;
            end else if (STAGE_PRODUCT == 4) begin : g_rows2
                // Fully-unsigned counterpart of the signed g_rows2: same GA x GB grid as g_rows, with the final
                // GA-way column sum split into a registered pairwise partial-sum stage followed by their sum.
                wire [WP-1:0] brow [0:GA*GB-1];
                genvar zi, zj;
                for (zi = 0; zi < GA; zi = zi + 1) begin : g_brow_row
                    for (zj = 0; zj < GB; zj = zj + 1) begin : g_brow_col
                        localparam integer OJ = slc_off(WB, GB, zj);
                        assign brow[zi*GB + zj] = m_pp[zi*GB + zj] << OJ;
                    end
                end
                reg [WP-1:0] rowc [0:GA-1];
                integer zr, zc;
                always @* begin
                    for (zr = 0; zr < GA; zr = zr + 1) begin
                        rowc[zr] = {WP{1'b0}};
                        for (zc = 0; zc < GB; zc = zc + 1) rowc[zr] = rowc[zr] + brow[zr*GB + zc];
                    end
                end
                reg [WP-1:0]  s_row [0:GA-1];
                reg           s_v;
                reg [WSB-1:0] s_sb;
                integer zs;
                always @(posedge clk) begin
                    if (rst) s_v <= 1'b0;
                    else     s_v <= m_v;
                    for (zs = 0; zs < GA; zs = zs + 1) s_row[zs] <= rowc[zs];
                    s_sb <= m_sb;
                end
                wire [WP-1:0] arow [0:GA-1];
                wire [WP-1:0] psum [0:((GA+1)/2)-1];
                genvar yi;
                for (yi = 0; yi < GA; yi = yi + 1) begin : g_arow
                    localparam integer OI = slc_off(WA, GA, yi);
                    assign arow[yi] = s_row[yi] << OI;
                end
                localparam integer NH = (GA + 1) / 2;     // pairwise partial sums of the GA shifted rows
                genvar hi;
                for (hi = 0; hi < NH; hi = hi + 1) begin : g_psum
                    if (2*hi + 1 < GA) begin : g_pair assign psum[hi] = arow[2*hi] + arow[2*hi + 1]; end
                    else               begin : g_lone assign psum[hi] = arow[2*hi];                  end
                end
                reg [WP-1:0]  s_col [0:NH-1];
                reg           t_v;
                reg [WSB-1:0] t_sb;
                integer zt;
                always @(posedge clk) begin
                    if (rst) t_v <= 1'b0;
                    else     t_v <= s_v;
                    for (zt = 0; zt < NH; zt = zt + 1) s_col[zt] <= psum[zt];
                    t_sb <= s_sb;
                end
                reg [WP-1:0] csum;
                integer zk;
                always @* begin
                    csum = {WP{1'b0}};
                    for (zk = 0; zk < NH; zk = zk + 1) csum = csum + s_col[zk];
                end
                reg [WP-1:0]  r_p;
                reg           r_v;
                reg [WSB-1:0] r_sb;
                always @(posedge clk) begin
                    if (rst) r_v <= 1'b0;
                    else     r_v <= t_v;
                    r_p  <= csum;
                    r_sb <= t_sb;
                end
                assign p = r_p; assign out_valid = r_v; assign sb_out = r_sb;
            end else begin : g_invalid_grid_stage
                _zkf_invalid_pmul_grid_stage u_invalid();
            end
        end
    endgenerate
endmodule

`default_nettype wire

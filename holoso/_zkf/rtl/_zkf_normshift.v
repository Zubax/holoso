/// Leading-zero-normalizing left shifter: brings the most significant set bit of `x` up to the MSB (bit W-1) and
/// reports how far it had to shift. Outputs:
///   - `zero`  : asserted iff x == 0.
///   - `count` : (W-1) - position_of_leading_one, i.e. the applied left-shift amount. Don't-care when `zero`.
///   - `y`     : x << count, the normalized vector (leading 1 at bit W-1 for nonzero x).
///
/// This fuses what used to be a separate leading-one detector (_zkf_lod) followed by a separate barrel shift: the
/// same mux cascade that tests-and-shifts also yields the count as the concatenation of its per-level digits, roughly
/// halving the multiplexer footprint of the count-then-shift pair. Modelled on FloPoCo's Normalizer_Z. Used by the
/// adder's close-cancellation normalization and by the integer-to-float magnitude normalization.
///
/// Implementation: radix-4 cascade, processed largest-shift first, so the depth is half the radix-2 equivalent (which
/// matters for timing closure at wide W). Level k (weight G = 4^k) inspects the top G, 2G and 3G bits; the number of
/// all-zero leading G-groups (0..3) is the radix-4 digit, the data is shifted left by digit*G, and the digit is the
/// count's base-4 place k. Groups/shifts beyond W are clamped at elaboration (only ever selected for x == 0, whose
/// count is don't-care). Walking high-to-low brings the leading one to the MSB and assembles count = {digits}.
///
/// WSHAMT defaults to clog2(W); callers needing a wider count (downstream widths sized off a different bound) may
/// request WSHAMT > clog2(W) and the count is zero-padded. The internal radix-4 count is 2*ceil(clog2(W)/2) bits and
/// is truncated to WSHAMT; this is lossless because count <= W-1 < 2^clog2(W) <= 2^WSHAMT for every nonzero input.
///
/// STAGE_SPLIT=0: pure combinational, single-cycle (clk unused).
///
/// STAGE_SPLIT=1: one register barrier in the middle of the cascade; y/count/zero appear one cycle after x, and the
/// consumer must add a matching cycle to its pipeline. The early (pre-barrier) digits and zero are delayed one cycle
/// so the whole count and zero stay aligned with the late digits and the shifted output.
///
/// STAGE_SPLIT=2: two register barriers; the first sits right after the top (widest) radix-4 level (where the wide
/// leading-zero OR-reductions and the largest barrel-shift muxes live) and the second at the existing midpoint. The
/// pre-barrier digits/zero get the extra cycle of delay so the whole count and zero stay aligned. Useful at wide W
/// where one barrier still leaves the top two levels in the same combinational stage (a substantial f_max gain at wide
/// WMAN in zkf_log2; the same path zkf_fma uses for its close-cancellation normalize when STAGE_NORMALIZE=2). Requires
/// NL4 >= 3 so the two barriers do not coincide.
///
/// STAGE_OUTPUT=1: register the fully aligned y/count/zero outputs after the cascade (+1 cycle). This is useful when
/// the consumer needs a clean boundary after normalization without changing the internal split geometry.
///
/// Streaming sideband: in_valid/out_valid and the generic sb_in/sb_out bus are delayed by exactly the module's output
/// latency (STAGE_SPLIT + STAGE_OUTPUT) so they land aligned with y/count/zero. This lets callers thread their control
/// payload through the normalizer instead of a parallel zkf_pipe. Reset clears only out_valid; sb and the datapath
/// free-run per the project reset policy. Callers that don't need the sideband set WSB=1 and stub sb_in with a constant.

`default_nettype none

module _zkf_normshift #(
    parameter W            = 16,
    parameter WSHAMT       = $clog2(W),
    parameter STAGE_SPLIT  = 0,
    parameter STAGE_OUTPUT = 0,
    parameter WSB          = 1      // sideband passthrough bit width
) (
    input  wire              clk,   // unused when STAGE_SPLIT == 0 && STAGE_OUTPUT == 0
    input  wire              rst,   // resets only out_valid; unused when purely combinational
    input  wire              in_valid,
    input  wire    [WSB-1:0] sb_in,
    input  wire      [W-1:0] x,
    output wire              out_valid,
    output wire    [WSB-1:0] sb_out,
    output wire              zero,
    output wire [WSHAMT-1:0] count,
    output wire      [W-1:0] y
);
    localparam NL2         = $clog2(W);     // radix-2 levels that would be needed
    localparam NL4         = (NL2 + 1) / 2; // radix-4 levels (two count bits each)
    localparam CNTW        = 2 * NL4;       // internal count width
    // Register barrier sits after radix-4 level SPLIT_AFTER (only used when STAGE_SPLIT != 0). The front (pre-barrier)
    // levels do the larger shifts with the wider zero-detect OR-reductions; the consumer's back cone also carries the
    // count assembly and exponent arithmetic. (NL4-1)/2 keeps the front from one level too deep at wide W.
    localparam SPLIT_AFTER = (NL4 - 1) / 2;

    generate
        if ((STAGE_SPLIT < 0) || (STAGE_SPLIT > 2)) begin : g_invalid_stage_split
            _zkf_invalid_stage_split_out_of_range u_invalid();
        end
        if ((STAGE_SPLIT == 2) && (NL4 < 3)) begin : g_invalid_stage_split2_too_narrow
            _zkf_invalid_stage_split2_needs_wider_w u_invalid();
        end
        if ((STAGE_OUTPUT != 0) && (STAGE_OUTPUT != 1)) begin : g_invalid_stage_output
            _zkf_invalid_stage_output u_invalid();
        end
    endgenerate

    // data[s] is the input to radix-4 level s; data[0] = x, data[NL4] = normalized output.
    wire [W-1:0]    data     [0:NL4];
    wire [W-1:0]    data_pre [0:NL4];   // index 0 unused
    wire [CNTW-1:0] dig_pre;            // combinational per-level radix-4 digits, digit k at bits [2k+1:2k]
    assign data[0] = x;

    genvar s;
    generate
        for (s = 0; s < NL4; s = s + 1) begin : g_lvl
            localparam integer K = NL4 - 1 - s;      // this level resolves count digit K (weight 4^K)
            localparam integer G = 1 << (2 * K);     // 4^K

            // Zero-detect each leading G-group. The top group (G) always fits (G < W). A wider group (2G or 3G >= W)
            // can be all-zero only for x == 0, whose count is don't-care (force_zero downstream), so tie that detect
            // off instead of emitting a redundant full-width OR-reduction - this is what shortens the critical path on
            // the widest top levels, where 2G/3G spill past the MSB.
            wire z1 = ~|data[s][W-1 -: G];
            wire z2;
            wire z3;
            if (2 * G < W) begin : g_z2  assign z2 = ~|data[s][W-1 -: (2 * G)]; end
            else           begin : g_z2z assign z2 = 1'b0;                      end
            if (3 * G < W) begin : g_z3  assign z3 = ~|data[s][W-1 -: (3 * G)]; end
            else           begin : g_z3z assign z3 = 1'b0;                      end

            // Radix-4 count digit: number of leading all-zero G-groups (z3 => z2 => z1), 0..3.
            assign dig_pre[2*K +: 2] = z1 ? (z2 ? (z3 ? 2'd3 : 2'd2) : 2'd1) : 2'd0;

            // Shift candidates. The shift mux is driven directly by the zero-detects in priority order, so it does
            // not wait on the digit encoding.
            // G < W always holds (G is the radix-4 group shift, < W for every supported W), so sh1's overshoot
            // else-arm is unreachable (sh2/sh3's do fire at the smallest W).
            // verilator coverage_off
            wire [W-1:0] sh1 = (G     < W) ? (data[s] << G)     : {W{1'b0}};
            // verilator coverage_on
            wire [W-1:0] sh2 = (2 * G < W) ? (data[s] << (2*G)) : {W{1'b0}};
            wire [W-1:0] sh3 = (3 * G < W) ? (data[s] << (3*G)) : {W{1'b0}};
            assign data_pre[s+1] = z3 ? sh3 : (z2 ? sh2 : (z1 ? sh1 : data[s]));
        end
    endgenerate

    // Wire data[s] from the per-level combinational outputs; insert the register barrier at index SPLIT_AFTER+1 when
    // STAGE_SPLIT != 0, breaking the cascade into two clock periods. STAGE_SPLIT=2 adds a SECOND barrier at index 1
    // (right after the wide top level), captured by the separate `g_split_top` arm so the original `g_split` /
    // `g_pass` decision for STAGE_SPLIT=0/1 stays bit-for-bit identical to the pre-modification version.
    genvar t;
    generate
        for (t = 1; t <= NL4; t = t + 1) begin : g_data
            // Original STAGE_SPLIT=1 arm with the original `(SS != 0) && (t == SPLIT_AFTER + 1)` condition is narrowed
            // to STAGE_SPLIT==1 so its elaboration is bit-for-bit unchanged (the value the condition resolves to for
            // SS=1 is identical, but Yosys is sensitive to the expression form). STAGE_SPLIT=2 arms are appended
            // after, sharing the `g_split` label so the existing label stays the only one taken for SS=1.
            if ((STAGE_SPLIT == 1) && (t == SPLIT_AFTER + 1)) begin : g_split
                reg [W-1:0] data_r;
                always @(posedge clk) data_r <= data_pre[t];
                assign data[t] = data_r;
            end else if ((STAGE_SPLIT == 2) && (t == SPLIT_AFTER + 1)) begin : g_split
                reg [W-1:0] data_r;
                always @(posedge clk) data_r <= data_pre[t];
                assign data[t] = data_r;
            end else if ((STAGE_SPLIT == 2) && (t == 1)) begin : g_split
                reg [W-1:0] data_r;
                always @(posedge clk) data_r <= data_pre[t];
                assign data[t] = data_r;
            end else begin : g_pass
                assign data[t] = data_pre[t];
            end
        end
    endgenerate

    wire [W-1:0] y_aligned = data[NL4];

    // Count assembly. The digit for level K (= NL4-1-k) is computed at iteration s = k from data[s]; it has crossed
    // every barrier whose position is <= s. The aligned count must wait for the slowest digit, so digit k is delayed
    // by (STAGE_SPLIT - crossings(s)) cycles. zero is computed off data[0] and waits the full STAGE_SPLIT cycles.
    wire [CNTW-1:0] cnt;
    genvar k;
    generate
        for (k = 0; k < NL4; k = k + 1) begin : g_count
            localparam integer S = NL4 - 1 - k;  // data level whose zero-detect produced this digit
            // First arm narrowed from `(SS != 0) && ...` to `(SS == 1) && ...` so the SS=1 elaboration is bit-for-bit
            // unchanged; SS=2 arms appended after. SS=2's top digit (k = NL4-1) needs two cycles of delay and lands in
            // g_count_delay2; SS=2's mid digits share the original g_count_delay label.
            if ((STAGE_SPLIT == 1) && (k >= (NL4 - 1 - SPLIT_AFTER))) begin : g_count_delay
                reg [1:0] dig_r;
                always @(posedge clk) dig_r <= dig_pre[2*k +: 2];
                assign cnt[2*k +: 2] = dig_r;
            end else if ((STAGE_SPLIT == 2) && (k == NL4 - 1)) begin : g_count_delay2
                reg [1:0] dig_r1, dig_r2;
                always @(posedge clk) begin
                    dig_r1 <= dig_pre[2*k +: 2];
                    dig_r2 <= dig_r1;
                end
                assign cnt[2*k +: 2] = dig_r2;
            end else if ((STAGE_SPLIT == 2) && (k >= (NL4 - 1 - SPLIT_AFTER))) begin : g_count_delay
                reg [1:0] dig_r;
                always @(posedge clk) dig_r <= dig_pre[2*k +: 2];
                assign cnt[2*k +: 2] = dig_r;
            end else begin : g_count_pass
                assign cnt[2*k +: 2] = dig_pre[2*k +: 2];
            end
        end
    endgenerate

    wire zero_pre = ~|x;
    wire zero_aligned;
    generate
        if (STAGE_SPLIT == 1) begin : g_zero_delay
            reg zero_r;
            always @(posedge clk) zero_r <= zero_pre;
            assign zero_aligned = zero_r;
        end else if (STAGE_SPLIT == 2) begin : g_zero_delay2
            reg zero_r1, zero_r2;
            always @(posedge clk) begin
                zero_r1 <= zero_pre;
                zero_r2 <= zero_r1;
            end
            assign zero_aligned = zero_r2;
        end else begin : g_zero_pass
            assign zero_aligned = zero_pre;
        end
    endgenerate

    wire [WSHAMT-1:0] count_aligned;
    generate
        if (WSHAMT > CNTW) begin : g_pad
            assign count_aligned = {{(WSHAMT-CNTW){1'b0}}, cnt};
        end else begin : g_no_pad
            assign count_aligned = cnt[WSHAMT-1:0];
        end
    endgenerate

    // Sideband and valid ride the same STAGE_SPLIT-deep delay as `zero` (sampled off the inputs in the same cycle as
    // x), so after the optional output register they are delayed by exactly STAGE_SPLIT + STAGE_OUTPUT cycles. sb is
    // pure datapath (free-runs); valid is the only reset-bearing register, matching the project reset policy.
    wire [WSB-1:0] sb_aligned;
    wire           valid_aligned;
    generate
        if (STAGE_SPLIT == 1) begin : g_sb_delay
            reg [WSB-1:0] sb_r;
            reg           valid_r;
            always @(posedge clk) sb_r <= sb_in;
            always @(posedge clk) begin
                if (rst) valid_r <= 1'b0;
                else     valid_r <= in_valid;
            end
            assign sb_aligned    = sb_r;
            assign valid_aligned = valid_r;
        end else if (STAGE_SPLIT == 2) begin : g_sb_delay2
            reg [WSB-1:0] sb_r1, sb_r2;
            reg           valid_r1, valid_r2;
            always @(posedge clk) begin
                sb_r1 <= sb_in;
                sb_r2 <= sb_r1;
            end
            always @(posedge clk) begin
                if (rst) begin valid_r1 <= 1'b0; valid_r2 <= 1'b0; end
                else     begin valid_r1 <= in_valid; valid_r2 <= valid_r1; end
            end
            assign sb_aligned    = sb_r2;
            assign valid_aligned = valid_r2;
        end else begin : g_sb_pass
            assign sb_aligned    = sb_in;
            assign valid_aligned = in_valid;
        end
    endgenerate

    generate
        if (STAGE_OUTPUT) begin : g_output_reg
            reg              zero_r;
            reg [WSHAMT-1:0] count_r;
            reg      [W-1:0] y_r;
            reg  [WSB-1:0]   sb_r;
            reg              valid_r;
            always @(posedge clk) begin
                zero_r  <= zero_aligned;
                count_r <= count_aligned;
                y_r     <= y_aligned;
                sb_r    <= sb_aligned;
            end
            always @(posedge clk) begin
                if (rst) valid_r <= 1'b0;
                else     valid_r <= valid_aligned;
            end
            assign zero      = zero_r;
            assign count     = count_r;
            assign y         = y_r;
            assign sb_out    = sb_r;
            assign out_valid = valid_r;
        end else begin : g_output_pass
            assign zero      = zero_aligned;
            assign count     = count_aligned;
            assign y         = y_aligned;
            assign sb_out    = sb_aligned;
            assign out_valid = valid_aligned;
        end
    endgenerate
endmodule

`default_nettype wire

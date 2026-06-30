/// Streamed signed-magnitude fixed-point -> normalized float.
/// Internalizes the _zkf_normshift + sideband delay + pack-input combine + _zkf_pack pipeline. The caller forms the
/// unsigned magnitude (and registers it), resolves the special-case sign and force_inf ahead of the helper, and passes
/// generic sideband through sb_in / sb_out for outputs that don't fit the y / valid channels (e.g., zkf_log2's pole
/// and domain_error flags).
///
/// Register stages = STAGE_NORMALIZE+STAGE_NORMALIZE_OUTPUT+STAGE_PACK+STAGE_OUTPUT
///
/// Zero-bubble, throughput-1, no backpressure. Reset clears only the control signals.
///
/// The result exponent (unbiased/biased depending on EXP_IS_BIASED) is computed as
/// exp = exp_offset - normshift_count.
///
/// EXP_IS_BIASED: 0 = `exp` is unbiased (helper passes it to _zkf_pack so the packer adds the bias);
///                1 = `exp` is already biased (the caller folded the bias into exp_offset to skip the packer's
///                    bias add -- used by zkf_from_int with exp_offset = WX-1+BIAS).
///
/// ASSUME_NO_OVERFLOW: forwarded to _zkf_pack. 0 = detect exponent overflow -> infinity (default); 1 = the caller
///                guarantees the result exponent is always in range, so the packer's overflow detector is pruned
///                (e.g. zkf_log2, whose result is always representable for finite x). force_inf and the underflow
///                paths are unaffected.
///
/// Callers that don't need the sideband should set its width WSB=1 and stub with a constant.

`default_nettype none

module _zkf_fixed_to_float #(
    parameter WEXP                   = 6,
    parameter WMAN                   = 18,  // significand precision including the hidden bit
    parameter WMAG                   = 64,  // width of the unsigned magnitude fed to the normalizer
    parameter WEU                    = 8,   // internal signed exponent width, also passed to _zkf_pack as WEXP_UNBIASED
    parameter EXP_IS_BIASED          = 0,
    parameter ASSUME_NO_OVERFLOW     = 0,   // forwarded to _zkf_pack; 1 prunes overflow detect
    parameter WSB                    = 1,   // generic sideband width carried alongside the pipeline
    parameter STAGE_NORMALIZE        = 0,   // {0,1,2} direct forward to _zkf_normshift.STAGE_SPLIT
    parameter STAGE_NORMALIZE_OUTPUT = 0,   // {0,1} direct forward to _zkf_normshift.STAGE_OUTPUT
    parameter STAGE_PACK             = 0,   // {0,1} direct forward to _zkf_pack.STAGE_INPUT
    parameter STAGE_OUTPUT           = 0    // {0,1} direct forward to _zkf_pack.STAGE_OUTPUT
) (
    input  wire clk,
    input  wire rst,

    input  wire                  in_valid,
    input  wire                  sign,
    input  wire                  force_zero,
    input  wire                  force_inf,
    input  wire signed [WEU-1:0] exp_offset,
    input  wire       [WMAG-1:0] mag,
    input  wire        [WSB-1:0] sb_in,

    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y,
    output wire       [WSB-1:0] sb_out
);
    localparam WIDX = $clog2(WMAG);
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wman
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if (WMAG < WMAN + 3) begin : g_invalid_wmag
            _zkf_invalid_fixed_to_float_wmag_too_narrow_for_grs u_invalid();
        end
        if (WEU < WEXP + 2) begin : g_invalid_weu
            _zkf_invalid_fixed_to_float_weu_too_narrow u_invalid();
        end
        if (WEU < WIDX + 1) begin : g_invalid_weu_count
            _zkf_invalid_fixed_to_float_weu_too_narrow_for_count u_invalid();
        end
    endgenerate

    localparam WCOUNT_PAD = (WEU > WIDX) ? (WEU - WIDX) : 1;

    // -- Normalize the magnitude. STAGE_NORMALIZE/STAGE_NORMALIZE_OUTPUT forward to _zkf_normshift.STAGE_SPLIT/
    // STAGE_OUTPUT. norm_count is the left-shift amount; norm_zero asserts when mag == 0; norm_aligned has the leading
    // 1 at bit WMAG-1 for nonzero input. The control/sideband bundle {sign, force_zero, force_inf, exp_offset, sb_in}
    // rides the normalizer's own sideband, delayed by exactly STAGE_NORMALIZE + STAGE_NORMALIZE_OUTPUT cycles so it
    // lands aligned with norm_aligned -- no parallel zkf_pipe. The normalizer resets only out_valid; sb free-runs.
    localparam PIPE_W = 3 + WEU + WSB;
    wire              norm_zero;
    wire [WIDX-1:0]   norm_count;
    wire [WMAG-1:0]   norm_aligned;
    wire              sb_valid;
    wire [PIPE_W-1:0] sb_pipe_out;
    _zkf_normshift #(
        .W(WMAG), .STAGE_SPLIT(STAGE_NORMALIZE), .STAGE_OUTPUT(STAGE_NORMALIZE_OUTPUT), .WSB(PIPE_W)
    ) u_norm (
        .clk(clk), .rst(rst),
        .in_valid(in_valid),
        .sb_in({sign, force_zero, force_inf, exp_offset, sb_in}),
        .x(mag),
        .out_valid(sb_valid),
        .sb_out(sb_pipe_out),
        .zero(norm_zero),
        .count(norm_count),
        .y(norm_aligned)
    );
    wire                    sign_d       = sb_pipe_out[PIPE_W-1];
    wire                    force_zero_d = sb_pipe_out[PIPE_W-2];
    wire                    force_inf_d  = sb_pipe_out[PIPE_W-3];
    wire signed [WEU-1:0]   exp_offset_d = sb_pipe_out[WSB +: WEU];
    wire [WSB-1:0]          sb_d         = sb_pipe_out[WSB-1:0];

    // The optional post-normalize boundary is owned by _zkf_normshift.STAGE_OUTPUT. The control/sideband pipe above
    // matches that latency, so GRS/exponent combine can consume the aligned normalizer outputs directly.
    wire                  c_valid        = sb_valid;
    wire                  c_sign         = sign_d;
    wire                  c_force_inf    = force_inf_d;
    wire signed [WEU-1:0] c_exp_offset   = exp_offset_d;
    wire [WSB-1:0]        c_sb           = sb_d;
    wire                  c_norm_zero    = norm_zero;
    wire [WIDX-1:0]       c_norm_count   = norm_count;
    wire [WMAG-1:0]       c_norm_aligned = norm_aligned;

    // -- Pack-input combine (combinational). Slicing follows zkf_from_int's pattern exactly: the leading WMAN bits
    // of the aligned bus carry the significand (hidden bit included), the next bit is guard, then round, and the OR
    // of the rest is sticky.
    wire [WMAN-1:0]  pre_sig    =  c_norm_aligned[WMAG-1 -: WMAN];
    wire             pre_guard  =  c_norm_aligned[WMAG-WMAN-1];
    wire             pre_round  =  c_norm_aligned[WMAG-WMAN-2];
    wire             pre_sticky = |c_norm_aligned[WMAG-WMAN-3:0];

    // exp = exp_offset - norm_count, as a signed WEU-bit value. For zkf_from_int (EXP_IS_BIASED=1) this is
    // the biased exponent EXP_BIASED_TOP - shamt and is forwarded to _zkf_pack with the bias-add disabled; for
    // zkf_log2 and remquo it is the unbiased exponent.
    wire        [WEU-1:0] norm_count_ext = {{WCOUNT_PAD{1'b0}}, c_norm_count};
    wire signed [WEU-1:0] pre_exp        = c_exp_offset - $signed(norm_count_ext);

    wire pre_force_inf  = c_force_inf;
    wire pre_force_zero = force_zero_d || (~c_force_inf & c_norm_zero);

    // -- The packer owns its optional input register (STAGE_INPUT=STAGE_PACK) and its optional output
    // register (STAGE_OUTPUT). When STAGE_PACK=1, the rounder is insulated from the wide normshift output
    // cone at the cost of one extra cycle of latency; this is what zkf_log2 needs at WMAN=53.
    _zkf_pack #(
        .WEXP(WEXP), .WMAN(WMAN),
        .WEXP_UNBIASED(WEU),
        .EXP_IS_BIASED(EXP_IS_BIASED),
        .ASSUME_NO_OVERFLOW(ASSUME_NO_OVERFLOW),
        .STAGE_INPUT(STAGE_PACK),
        .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_pack (
        .clk(clk),
        .rst(rst),
        .in_valid(c_valid),
        .sign(c_sign),
        .force_zero(pre_force_zero),
        .force_inf(pre_force_inf),
        .exp_unbiased(pre_exp),
        .significand(pre_sig),
        .guard(pre_guard),
        .round(pre_round),
        .sticky(pre_sticky),
        .out_valid(out_valid),
        .y(y)
    );

    // -- Forward sideband through the packer's input + output stages so sb_out lands with out_valid. Pure datapath:
    // the delay free-runs with no reset; sb_out is only sampled in lockstep with out_valid, which is reset.
    _zkf_pack_delay #(.W(WSB), .STAGE_INPUT(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT)) u_sb_pack_delay (
        .clk(clk), .x(c_sb), .y(sb_out)
    );
endmodule

`default_nettype wire

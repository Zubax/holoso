/// Streamed cast between two Zubax Kulibin float formats.
/// If no stages are enabled, the module behaves combinationally; clk, rst are ignored.
///
/// STAGE_INPUT=0: input combinational paths are exposed.
/// STAGE_INPUT=1: inputs are latched, the external module sees registers at the input (one extra cycle).
///
/// STAGE_OUTPUT=0: outputs are combinational (default)
/// STAGE_OUTPUT=1: registered (one extra cycle).
///
/// Behaviour:
///   Widening both (WMAN_OUT >= WMAN_IN, WEXP_OUT >= WEXP_IN): exact result, no rounding, fast path.
///   Narrowing WMAN (WMAN_OUT < WMAN_IN): round-to-nearest, ties-to-even on the discarded fraction bits.
///   Narrowing WEXP (WEXP_OUT < WEXP_IN): output overflow maps to signed inf;
///                                        tiny finite outputs use the zero/MIN_NORMAL boundary rule.
///   Zero (exp_in == 0): canonicalises to +0 in the output format.
///   Infinity (exp_in == all-ones): canonicalises to signed infinity in the output format.

`default_nettype none

module zkf_resize #(
    parameter WEXP_IN      = 6,
    parameter WMAN_IN      = 18,
    parameter WEXP_OUT     = 5,
    parameter WMAN_OUT     = 11,
    parameter STAGE_INPUT  = 0,
    parameter STAGE_OUTPUT = 0,
    parameter LATENCY      = 0
) (
    input wire clk,
    input wire rst,

    input wire                       in_valid,
    input wire [WEXP_IN+WMAN_IN-1:0] a,

    output wire                         out_valid,
    output wire [WEXP_OUT+WMAN_OUT-1:0] y
);
    localparam LATENCY_REF = STAGE_INPUT + STAGE_OUTPUT;
    generate
        if ((WEXP_IN < 2) || (WMAN_IN < 4) || (WEXP_OUT < 2) || (WMAN_OUT < 4)) begin : g_invalid
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

    localparam WFRAC_IN  = WMAN_IN  - 1;
    localparam WFRAC_OUT = WMAN_OUT - 1;
    localparam WFULL_IN  = WEXP_IN  + WMAN_IN;
    localparam WFULL_OUT = WEXP_OUT + WMAN_OUT;

    // Optional input register stage.
    wire                in_valid_q;
    wire [WFULL_IN-1:0] a_q;
    zkf_pipe #(.W(WFULL_IN), .N(STAGE_INPUT ? 1 : 0)) u_input_pipe (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in(a),
        .out_valid(in_valid_q), .out(a_q)
    );

    // Decode under the input format. Canonicalisation of zero (frac/sign ignored when exp == 0) and
    // signed infinity (frac ignored when exp == all_ones) happens here.
    wire                sign_in = a_q[WFULL_IN-1];
    wire  [WEXP_IN-1:0] exp_in  = a_q[WFULL_IN-2:WFRAC_IN];
    wire [WFRAC_IN-1:0] frac_in = a_q[WFRAC_IN-1:0];
    wire                is_zero = ~|exp_in;
    wire                is_inf  =  &exp_in;

    // IN_BIAS as a sized vector (top bit 0, lower WEXP_IN-1 bits all 1 = 2^(WEXP_IN-1) - 1). Used by the
    // narrowing (pack) path; the widen path re-derives the equivalent constant as IN_BIAS_WIDENED below.
    localparam [WEXP_IN-1:0] IN_BIAS = {1'b0, {(WEXP_IN-1){1'b1}}};

    generate
        if ((WMAN_OUT >= WMAN_IN) && (WEXP_OUT >= WEXP_IN)) begin : g_widen_only
            // Fast path: output format strictly covers the input range. No rounding (significand is padded with zeros),
            // no overflow (output exp range is at least as wide), no underflow
            // (input exp_unbiased is always >= input min, which the output also covers).
            localparam integer FRAC_PAD = WMAN_OUT - WMAN_IN;

            // Bias offset = OUT_BIAS - IN_BIAS, non-negative because WEXP_OUT >= WEXP_IN.
            // Constructed via sized vectors to stay portable for any WEXP values.
            localparam [WEXP_OUT-1:0] OUT_BIAS         = {1'b0, {(WEXP_OUT-1){1'b1}}};
            localparam [WEXP_OUT-1:0] IN_BIAS_WIDENED  = {{(WEXP_OUT-WEXP_IN+1){1'b0}}, {(WEXP_IN-1){1'b1}}};
            localparam [WEXP_OUT-1:0] BIAS_OFFSET      = OUT_BIAS - IN_BIAS_WIDENED;

            // Re-biased exponent for the normal case. exp_in fits in WEXP_IN unsigned bits and BIAS_OFFSET is at most
            // 2^(WEXP_OUT-1), so the sum <= 2^WEXP_OUT-1 and never reaches the all-1 inf encoding for normal inputs.
            wire [WEXP_OUT-1:0]  exp_in_widened = {{(WEXP_OUT-WEXP_IN){1'b0}}, exp_in};
            wire [WEXP_OUT-1:0]  exp_widened    = exp_in_widened + BIAS_OFFSET;

            // Fraction widening: same width is a passthrough; wider output zero-pads on the LSB side.
            // Decided at elaboration so only one branch exists in the netlist.
            wire [WFRAC_OUT-1:0] frac_widened;
            if (FRAC_PAD == 0) begin : g_same_man
                assign frac_widened = frac_in;
            end else begin : g_pad_man
                assign frac_widened = {frac_in, {FRAC_PAD{1'b0}}};
            end

            // Final encoding: zero collapses to canonical +0 (wins over inf); infinity becomes canonical signed
            // infinity; normal values use the re-biased exponent and padded fraction. A ternary (not a case) so the
            // same expression feeds both the registered and combinational STAGE_OUTPUT branches below.
            wire [WFULL_OUT-1:0] y_widen = is_zero ? {WFULL_OUT{1'b0}}
                                         : is_inf  ? {sign_in, {WEXP_OUT{1'b1}}, {WFRAC_OUT{1'b0}}}
                                         :           {sign_in, exp_widened, frac_widened};
            if (STAGE_OUTPUT != 0) begin : g_owr
                reg                 s_valid;
                reg [WFULL_OUT-1:0] s_y;
                always @(posedge clk) begin
                    if (rst) s_valid <= 1'b0;
                    else     s_valid <= in_valid_q;
                    s_y <= y_widen;
                end
                assign out_valid = s_valid;
                assign y         = s_y;
            end else begin : g_owc
                assign out_valid = in_valid_q;
                assign y         = y_widen;
            end
        end else begin : g_pack
            // Slow path: at least one dimension narrows, so rounding and/or overflow detection are needed and
            // _zkf_pack handles them (its STAGE_OUTPUT sets registered vs combinational). Output-side accumulator
            // width for the unbiased exponent. Must hold the input format's full signed exp_unbiased range
            // (WEXP_IN + 1 signed bits) and also _zkf_pack's internal range requirement of at least WEXP_OUT + 2
            // signed bits.
            localparam WEU_PACK_MIN = WEXP_OUT + 2;
            localparam WEU_IN_MIN   = WEXP_IN  + 1;
            localparam WEU          = (WEU_PACK_MIN > WEU_IN_MIN) ? WEU_PACK_MIN : WEU_IN_MIN;

            localparam signed [WEU-1:0] IN_BIAS_EXT  = $signed({{(WEU-WEXP_IN){1'b0}}, IN_BIAS});
            wire       signed [WEU-1:0] exp_unbiased = $signed({{(WEU-WEXP_IN){1'b0}}, exp_in}) - IN_BIAS_EXT;
            wire [WMAN_IN-1:0]          sig_in       = {1'b1, frac_in};

            wire [WMAN_OUT-1:0] significand_out;
            wire                guard_out;
            wire                round_out;
            wire                sticky_out;

            if (WMAN_OUT >= WMAN_IN) begin : g_widen
                // Exact: copy the input significand and pad the new low bits with zeros.
                // Reached when WEXP_OUT < WEXP_IN (otherwise the fast path would have taken over).
                localparam integer PAD = WMAN_OUT - WMAN_IN;
                if (PAD == 0) begin : g_same_width
                    assign significand_out = sig_in;
                end else begin : g_zero_pad
                    assign significand_out = {sig_in, {PAD{1'b0}}};
                end
                assign guard_out  = 1'b0;
                assign round_out  = 1'b0;
                assign sticky_out = 1'b0;
            end else begin : g_narrow
                localparam integer DROP = WMAN_IN - WMAN_OUT;
                assign significand_out = sig_in[WMAN_IN-1 -: WMAN_OUT];
                assign guard_out       = sig_in[DROP - 1];
                if (DROP >= 2) begin : g_round_real
                    assign round_out = sig_in[DROP - 2];
                end else begin : g_round_zero
                    assign round_out = 1'b0;
                end
                if (DROP >= 3) begin : g_sticky_real
                    assign sticky_out = |sig_in[DROP - 3 : 0];
                end else begin : g_sticky_zero
                    assign sticky_out = 1'b0;
                end
            end

            _zkf_pack #(.WEXP(WEXP_OUT), .WMAN(WMAN_OUT), .WEXP_UNBIASED(WEU), .STAGE_OUTPUT(STAGE_OUTPUT)) u_pack (
                .clk(clk),
                .rst(rst),
                .in_valid(in_valid_q),
                .sign(sign_in),
                .force_zero(is_zero),
                .force_inf(is_inf),
                .exp_unbiased(exp_unbiased),
                .significand(significand_out),
                .guard(guard_out),
                .round(round_out),
                .sticky(sticky_out),
                .out_valid(out_valid),
                .y(y)
            );
        end
    endgenerate
endmodule

`default_nettype wire

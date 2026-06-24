/// Internal quotient generator for Zubax Kulibin float division.
///
/// The quotient bits are produced by an unrolled radix-4 restoring divider.
/// The final partial remainder is exposed as well since it is a byproduct that is occasionally useful.
/// The div0 output reports that the divisor's exponent field is zero (i.e., the divisor encodes +0).
/// It is independent of the result; in particular it is asserted for 0/0 even though the quotient is +0.
///
/// Register stages: 2+((WMAN+2+((WMAN+2)%2))/2).
/// The inputs are not latched but the outputs are. Throughput is one sample per cycle.

`default_nettype none

module _zkf_div_core #(
    parameter WEXP          = 6,      // exponent field width
    parameter WMAN          = 18,     // significand precision including the hidden bit
    parameter QFRAC_BASE    = WMAN + 2,
    parameter QFRAC         = QFRAC_BASE + (QFRAC_BASE % 2),
    parameter WEXP_UNBIASED = WEXP + 2
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire [WEXP+WMAN-1:0] b,

    output reg                            out_valid,
    output reg                            sign,
    output reg                            force_zero,
    output reg                            force_inf,
    output reg signed [WEXP_UNBIASED-1:0] exp_unbiased,
    output reg                 [WMAN-1:0] significand,
    output reg                            guard,
    output reg                            round,
    output reg                            sticky,
    output reg                            div0,
    output reg signed [WEXP_UNBIASED-1:0] exp_diff,
    output reg                 [QFRAC:0]  raw,
    // Final delayed divisor significand, exposed for consumers that derive residuals from the quotient stream.
    output reg                 [WMAN-1:0] den,
    output reg                 [WMAN-1:0] partial_rem
);
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wman
            _zkf_invalid_wexp_or_wman u_invalid();
        end
    endgenerate

    localparam WFRAC         = WMAN - 1;
    localparam WFULL         = WEXP + WMAN;
    localparam QSTAGES       = QFRAC / 2;
    localparam QRAW          = QFRAC + 1;
    localparam WREM4         = WMAN + 2;
    localparam QTRI          = (QSTAGES + 1) * (QSTAGES + 1);
    localparam TAIL_HI_WIDTH = QFRAC - WMAN - 1;
    localparam TAIL_LO_WIDTH = QFRAC - WMAN - 2;

    // QRAW contains one integer quotient bit followed by QFRAC fractional bits.
    localparam          [WEXP-1:0] EXP_INF         = {WEXP{1'b1}};
    localparam signed [WEXP_UNBIASED-1:0] ZERO_EXT = {WEXP_UNBIASED{1'b0}};
    localparam signed [WEXP_UNBIASED-1:0] ONE_EXT  = {{(WEXP_UNBIASED-1){1'b0}}, 1'b1};

    wire             a_sign = a[WFULL-1];
    wire             b_sign = b[WFULL-1];
    wire  [WEXP-1:0] a_exp  = a[WFULL-2:WFRAC];
    wire  [WEXP-1:0] b_exp  = b[WFULL-2:WFRAC];
    wire [WFRAC-1:0] a_frac = a[WFRAC-1:0];
    wire [WFRAC-1:0] b_frac = b[WFRAC-1:0];

    // Decode and canonicalize special cases into force-zero/force-infinity controls for _zkf_pack.
    // The quotient core still free-runs for all encodings; special cases override the packed result.
    wire            a_zero        = a_exp == {WEXP{1'b0}};
    wire            b_zero        = b_exp == {WEXP{1'b0}};
    wire            a_inf         = a_exp == EXP_INF;
    wire            b_inf         = b_exp == EXP_INF;
    wire            result_zero   = a_zero || b_inf;
    wire            result_inf    = !a_zero && !b_inf && (b_zero || a_inf);
    wire            result_sign   = b_zero ? a_sign : (a_sign ^ b_sign);
    wire [WMAN-1:0] a_significand = {1'b1, a_frac};
    wire [WMAN-1:0] b_significand = {1'b1, b_frac};

    // Emit the integer quotient bit before the radix-4 stages. Since both significands are in [1, 2),
    // this bit is the only possible integer part, and the initial remainder is strictly below the denominator.
    wire             initial_bit  = a_significand >= b_significand;
    wire  [WMAN-1:0] initial_rem  = initial_bit ? (a_significand - b_significand) : a_significand;
    wire [WREM4-1:0] initial_den3 = {1'b0, b_significand, 1'b0} + {2'b00, b_significand};  // x3

    // Zero-extension padding of non-negative exponents.
    wire signed [WEXP_UNBIASED-1:0] a_exp_ext = {{(WEXP_UNBIASED-WEXP){1'b0}}, a_exp};
    wire signed [WEXP_UNBIASED-1:0] b_exp_ext = {{(WEXP_UNBIASED-WEXP){1'b0}}, b_exp};
    wire signed [WEXP_UNBIASED-1:0] decoded_exp_unbiased = a_exp_ext - b_exp_ext;

    // Stage zero keeps input decode/classification and the first radix-4 digit off the same path. It also
    // precomputes 3*den once; later stages only form cheap 1*den and 2*den wires locally.
    // Each later pipeline stage resolves one radix-4 quotient digit. The quotient prefix is stored in a
    // triangular chain: stage i holds only the 1+2*i bits known by that point, not a full QRAW-wide word.
    reg                            r_valid        [0:QSTAGES];
    // Sideband delay lines aligned with the radix pipeline; keeping them scalar lets tools optimize bits freely.
    reg                            r_sign         [0:QSTAGES];
    reg signed [WEXP_UNBIASED-1:0] r_exp_unbiased [0:QSTAGES];
    reg                            r_force_zero   [0:QSTAGES];
    reg                            r_force_inf    [0:QSTAGES];
    reg                            r_div0         [0:QSTAGES];
    // Carry den and 3*den through the stages to keep each digit resolver free of repeated WMAN-wide adders.
    // This costs FFs, but it preserves the short per-stage subtract/compare structure that sets divider timing.
    reg                 [WMAN-1:0] r_den          [0:QSTAGES];
    reg                [WREM4-1:0] r_den3         [0:QSTAGES];
    reg                 [WMAN-1:0] r_rem          [0:QSTAGES];
    reg                            r_raw0;

    wire [QTRI-1:0] raw_tri;
    wire [QRAW-1:0] final_raw = raw_tri[(QSTAGES * QSTAGES) +: QRAW];

    assign raw_tri[0] = r_raw0;

    always @(posedge clk) begin
        if (rst) begin
            r_valid[0] <= 1'b0;
        end else begin
            r_valid[0] <= in_valid;
        end
        r_sign[0]         <= result_sign;
        r_exp_unbiased[0] <= decoded_exp_unbiased;
        r_force_zero[0]   <= result_zero;
        r_force_inf[0]    <= result_inf;
        r_div0[0]         <= b_zero;
        r_den[0]          <= b_significand;
        r_den3[0]         <= initial_den3;
        r_rem[0]          <= initial_rem;
        r_raw0            <= initial_bit;
    end

    genvar i_stage;
    generate
        for (i_stage = 1; i_stage <= QSTAGES; i_stage = i_stage + 1) begin : g_stage
            localparam WIN     = (2 * i_stage) - 1;
            localparam WOUT    = WIN + 2;
            localparam IN_OFF  = (i_stage - 1) * (i_stage - 1);
            localparam OUT_OFF = i_stage * i_stage;

            wire [WMAN-1:0] rem_next;
            wire      [1:0] digit;

            _zkf_div_radix4_step #(.WMAN(WMAN)) u_step (
                .den(r_den[i_stage-1]),
                .den3(r_den3[i_stage-1]),
                .rem(r_rem[i_stage-1]),
                .rem_next(rem_next),
                .digit(digit)
            );
            _zkf_div_raw_stage #(.WIN(WIN)) u_raw (
                .clk(clk),
                .raw_prefix(raw_tri[IN_OFF +: WIN]),
                .digit(digit),
                .raw_next(raw_tri[OUT_OFF +: WOUT])
            );

            // Reset only validity; payload registers intentionally free-run.
            always @(posedge clk) begin
                if (rst) begin
                    r_valid[i_stage] <= 1'b0;
                end else begin
                    r_valid[i_stage] <= r_valid[i_stage-1];
                end
                r_sign[i_stage]         <= r_sign[i_stage-1];
                r_exp_unbiased[i_stage] <= r_exp_unbiased[i_stage-1];
                r_force_zero[i_stage]   <= r_force_zero[i_stage-1];
                r_force_inf[i_stage]    <= r_force_inf[i_stage-1];
                r_div0[i_stage]         <= r_div0[i_stage-1];
                r_den[i_stage]          <= r_den[i_stage-1];
                r_den3[i_stage]         <= r_den3[i_stage-1];
                r_rem[i_stage]          <= rem_next;
            end
        end
    endgenerate

    // This is initial_bit delayed through raw_tri: radix-4 stages append LSBs only, so it cannot be demoted.
    wire            final_high           = final_raw[QFRAC];
    wire            final_rem_sticky     = |r_rem[QSTAGES];
    wire [WMAN-1:0] final_significand_hi = final_raw[QFRAC -: WMAN];
    wire [WMAN-1:0] final_significand_lo = final_raw[QFRAC-1 -: WMAN];
    wire            final_guard_hi       = final_raw[QFRAC-WMAN];
    wire            final_guard_lo       = final_raw[QFRAC-WMAN-1];
    wire            final_round_hi       = final_raw[QFRAC-WMAN-1];
    wire            final_round_lo       = final_raw[QFRAC-WMAN-2];
    wire            final_tail_hi;
    wire            final_tail_lo;
    wire            final_sticky_hi = final_tail_hi || final_rem_sticky;
    wire            final_sticky_lo = final_tail_lo || final_rem_sticky;
    // Moving this into stage zero makes initial_bit's significand compare feed the exponent sideband path.
    wire signed [WEXP_UNBIASED-1:0] final_exp_adjust = final_high ? ZERO_EXT : ONE_EXT;

    generate
        if (TAIL_HI_WIDTH > 0) begin : g_final_tail_hi
            assign final_tail_hi = |final_raw[TAIL_HI_WIDTH-1:0];
        end else begin : g_no_final_tail_hi
            assign final_tail_hi = 1'b0;
        end

        if (TAIL_LO_WIDTH > 0) begin : g_final_tail_lo
            assign final_tail_lo = |final_raw[TAIL_LO_WIDTH-1:0];
        end else begin : g_no_final_tail_lo
            assign final_tail_lo = 1'b0;
        end
    endgenerate

    // The GRS muxes normalize according to the delayed initial_bit above; no radix-4 stage can change that bit.
    // Final output stage closes the quotient-prefix/sticky/exponent combinational paths at the module boundary.
    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
        end else begin
            out_valid <= r_valid[QSTAGES];
        end
        sign         <= r_sign[QSTAGES];
        force_zero   <= r_force_zero[QSTAGES];
        force_inf    <= r_force_inf[QSTAGES];
        exp_unbiased <= r_exp_unbiased[QSTAGES] - final_exp_adjust;
        significand  <= final_high ? final_significand_hi : final_significand_lo;
        guard        <= final_high ? final_guard_hi : final_guard_lo;
        round        <= final_high ? final_round_hi : final_round_lo;
        sticky       <= final_high ? final_sticky_hi : final_sticky_lo;
        div0         <= r_div0[QSTAGES];
        exp_diff     <= r_exp_unbiased[QSTAGES];
        raw          <= final_raw;
        den          <= r_den[QSTAGES];
        partial_rem  <= r_rem[QSTAGES];
    end
endmodule


// Register one more radix-4 quotient digit into the narrow prefix known at this stage.
module _zkf_div_raw_stage#(parameter WIN = 1) (
    input wire           clk,
    input wire [WIN-1:0] raw_prefix,
    input wire     [1:0] digit,
    output reg [WIN+1:0] raw_next
);
    always @(posedge clk) begin
        raw_next <= {raw_prefix, digit};
    end
endmodule


// Resolve one radix-4 quotient digit using parallel candidate subtracts.
module _zkf_div_radix4_step#(parameter WMAN = 18) (
    input wire [WMAN-1:0] den,
    input wire [WMAN+1:0] den3,
    input wire [WMAN-1:0] rem,

    output wire [WMAN-1:0] rem_next,
    output wire      [1:0] digit
);
    localparam WREM4 = WMAN + 2;
    localparam WDIFF = WREM4 + 1;

    wire [WREM4-1:0] den1 = {2'b00, den};
    wire [WREM4-1:0] den2 = {1'b0, den, 1'b0};
    wire [WREM4-1:0] rem4 = {rem, 2'b00};

    wire [WDIFF-1:0] diff1 = {1'b0, rem4} - {1'b0, den1};
    wire [WDIFF-1:0] diff2 = {1'b0, rem4} - {1'b0, den2};
    wire [WDIFF-1:0] diff3 = {1'b0, rem4} - {1'b0, den3};
    wire             ge1   = !diff1[WREM4];
    wire             ge2   = !diff2[WREM4];
    wire             ge3   = !diff3[WREM4];

    assign digit[1] = ge2;
    assign digit[0] = ge3 || (ge1 && !ge2);
    assign rem_next = ge3 ? diff3[WMAN-1:0] :
                      ge2 ? diff2[WMAN-1:0] :
                      ge1 ? diff1[WMAN-1:0] :
                            rem4[WMAN-1:0];
endmodule

`default_nettype wire

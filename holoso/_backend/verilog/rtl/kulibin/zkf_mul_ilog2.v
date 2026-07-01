/// Power-of-two multiplier: y = a * 2^k, where k is a signed integer (ldexp/scalbn).
/// This is far cheaper than full multiplication (zkf_mul) or division (zkf_div) because the significand is preserved
/// bit-for-bit and only the biased exponent is shifted by k, and no rounding is required -- the operation is exact
/// in the format's normal range.
///
/// k is a signed value WK bits wide. Any k is legal: shifts that push the result past the format's range simply
/// saturate to signed infinity (overflow) or flush to zero (underflow), exactly as ldexp would. The default width
/// spans the entire useful range; widen WK if k is driven from a wider computed value.
///
/// STAGE_INPUT=0: operand and k feed the decode combinationally (default).
/// STAGE_INPUT=1: latch {a, k} before any combinational logic, isolating them from upstream paths (+1 cycle).
/// STAGE_INPUT>1: add extra dummy stages; helps in routing-congested designs (+STAGE_INPUT cycles).
///
/// STAGE_DECODE=0: single-cycle combinational decode + output mux (no intermediate register).
/// STAGE_DECODE=1: registers the decoded sign / new_exp / frac / classification predicates before the output mux.
///     Splits the long route from input port to output register (the dominant delay path at wide WMAN on
///     placement-sensitive tools). Costs one extra pipeline cycle.

`default_nettype none

module zkf_mul_ilog2 #(
    parameter         WEXP         = 6,
    parameter         WMAN         = 18,        // significand precision including the hidden bit
    parameter         WK           = WEXP + 1,  // width of the signed exponent shift k; default spans the useful range
    parameter         STAGE_INPUT  = 0,         // number of input register stages (>=0); +STAGE_INPUT cycles
    parameter         STAGE_DECODE = 0,         // 0 = single-cycle; 1 = register decoded signals (+1 cycle)
    parameter         LATENCY      = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,
    input wire signed [WK-1:0] k,

    output reg                 out_valid,
    output reg [WEXP+WMAN-1:0] y           // y = a * 2^k
);
    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;
    // Signed accumulator wide enough to hold a_exp (0 .. 2^WEXP-1) + k (any signed WK value) without wrapping.
    localparam WACC  = ((WEXP > WK) ? WEXP : WK) + 2;

    localparam [WEXP-1:0] EXP_INF = {WEXP{1'b1}};   // = 2^WEXP-1; new_exp >= EXP_INF is the overflow boundary

    localparam LATENCY_REF = 1 + STAGE_INPUT + STAGE_DECODE;
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wm
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if (WK < 1) begin : g_invalid_wk
            _zkf_invalid_mul_ilog2_wk u_invalid();
        end
        if ((STAGE_DECODE != 0) && (STAGE_DECODE != 1)) begin : g_invalid_stage_decode
            _zkf_invalid_stage_decode u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    // Optional input register stage: latch the operand and k together before any combinational logic.
    wire             in_valid_q;
    wire [WFULL-1:0] a_q;
    wire [WK-1:0]    k_q;
    zkf_pipe #(.W(WFULL + WK), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in({k, a}),
        .out_valid(in_valid_q), .out({k_q, a_q})
    );

    // Decode and classify.
    wire             a_sign    = a_q[WFULL-1];
    wire [WEXP-1:0]  a_exp     = a_q[WFULL-2:WFRAC];
    wire [WFRAC-1:0] a_frac    = a_q[WFRAC-1:0];
    wire             a_zero    = ~|a_exp;
    wire             a_inf     =  &a_exp;
    wire             a_special = a_zero || a_inf;   // zero/inf pass through regardless of k, with absolute priority

    // Result biased exponent a_exp + k, in a wide signed accumulator so any |k| lands outside the range without
    // wrapping. Both boundaries stay live (k's sign is unknown at elaboration). The !a_special gate is what the
    // constant module gets free from its |K| bound: without it, e.g. inf*2^(-big) drives new_exp < 0 and would be
    // misclassified as underflow-to-zero. It keeps ldexp(+-inf,k)=+-inf and ldexp(+-0,k)=+-0.
    wire signed [WACC-1:0] new_exp_acc = $signed({1'b0, a_exp}) + $signed(k_q);
    // Overflow via the sign bit of (new_exp_acc - EXP_INF), not a `>=` comparator: logically identical, but this form
    // (mirroring zkf_mul_ilog2_const) maps to a short carry chain that synthesizers handle predictably, whereas the
    // wide comparator was observed to map poorly on some tool/mapper combinations (bloated area, weaker timing).
    wire signed [WACC-1:0] of_acc = new_exp_acc - $signed({{(WACC-WEXP){1'b0}}, EXP_INF});
    wire overflow   = !a_special && ~of_acc[WACC-1];        // new_exp_acc >= EXP_INF
    wire underflow  = !a_special &&  new_exp_acc[WACC-1];   // < 0
    wire min_normal = !a_special && ~|new_exp_acc;          // == 0 rounds up to MIN_NORMAL (>= 0.5*MIN_NORMAL)

    wire result_is_zero       = a_zero || underflow;
    wire result_is_inf        = a_inf  || overflow;
    wire result_is_min_normal = min_normal;

    // Normal-output exponent: low WEXP bits of (a_exp + k). The truncating slice is fine because the normal-output mux
    // is suppressed whenever result_is_zero or result_is_inf is asserted.
    wire [WEXP-1:0] new_exp = new_exp_acc[WEXP-1:0];

    // Output candidate forms. Canonicalisation is implicit: zero has sign/frac cleared, infinity has frac cleared.
    wire [WFULL-1:0] y_inf_w        = {a_sign, EXP_INF, {WFRAC{1'b0}}};
    wire [WFULL-1:0] y_min_normal_w = {a_sign, {{(WEXP-1){1'b0}}, 1'b1}, {WFRAC{1'b0}}};
    wire [WFULL-1:0] y_normal_w     = {a_sign, new_exp, a_frac};

    // result_is_zero takes priority over result_is_inf (mutually exclusive for a normal input; the overlapping rows
    // are listed so the priority is explicit and the case is full).
    // Reset only stream validity. Payload register intentionally free-runs (project Reset strategy).
    generate
        if (STAGE_DECODE == 0) begin : g_decode_combinational
            always @(posedge clk) begin
                if (rst) begin
                    out_valid <= 1'b0;
                end else begin
                    out_valid <= in_valid_q;
                end
                case ({result_is_zero, result_is_inf, result_is_min_normal})
                    3'b100, 3'b101, 3'b110, 3'b111: y <= {WFULL{1'b0}};
                    3'b010, 3'b011:                 y <= y_inf_w;
                    3'b001:                         y <= y_min_normal_w;
                    default:                        y <= y_normal_w;
                endcase
            end
        end else begin : g_decode_registered
            // Intermediate stage: register decoded signals so the output mux sees registered inputs,
            // breaking the long input-to-output route into two short hops.
            reg             r_in_valid;
            reg             r_sign;
            reg             r_result_is_zero;
            reg             r_result_is_inf;
            reg             r_result_is_min_normal;
            reg [WEXP-1:0]  r_new_exp;
            reg [WFRAC-1:0] r_frac;
            always @(posedge clk) begin
                if (rst) begin
                    r_in_valid <= 1'b0;
                end else begin
                    r_in_valid <= in_valid_q;
                end
                r_sign                 <= a_sign;
                r_result_is_zero       <= result_is_zero;
                r_result_is_inf        <= result_is_inf;
                r_result_is_min_normal <= result_is_min_normal;
                r_new_exp              <= new_exp;
                r_frac                 <= a_frac;
            end

            wire [WFULL-1:0] r_y_inf_w        = {r_sign, EXP_INF, {WFRAC{1'b0}}};
            wire [WFULL-1:0] r_y_min_normal_w = {r_sign, {{(WEXP-1){1'b0}}, 1'b1}, {WFRAC{1'b0}}};
            wire [WFULL-1:0] r_y_normal_w     = {r_sign, r_new_exp, r_frac};

            always @(posedge clk) begin
                if (rst) begin
                    out_valid <= 1'b0;
                end else begin
                    out_valid <= r_in_valid;
                end
                case ({r_result_is_zero, r_result_is_inf, r_result_is_min_normal})
                    3'b100, 3'b101, 3'b110, 3'b111: y <= {WFULL{1'b0}};
                    3'b010, 3'b011:                 y <= r_y_inf_w;
                    3'b001:                         y <= r_y_min_normal_w;
                    default:                        y <= r_y_normal_w;
                endcase
            end
        end
    endgenerate
endmodule

`default_nettype wire

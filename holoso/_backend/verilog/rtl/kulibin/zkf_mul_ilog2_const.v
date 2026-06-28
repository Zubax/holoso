/// Constant-power-of-two multiplier: y = a * 2^K, where K is a compile-time signed integer parameter.
/// This is far cheaper than full multiplication (zkf_mul) or division (zkf_div) because the mantissa is preserved
/// bit-for-bit and only the biased exponent is incremented by K. Special inputs (zero, signed infinity) are
/// canonicalized at the output. No rounding is required: the operation is exact in the format's normal range.
///
/// Elaboration fails when K is so extreme that every normal input either overflows to signed infinity or underflows
/// to zero, since the module is then provably useless. Concretely, K must satisfy -EXP_MAX_FINITE <= K < EXP_MAX_FINITE
/// where EXP_MAX_FINITE = 2**WEXP-2. This bound preserves at least one exponent value that maps to a normal output.
///
/// STAGE_INPUT=0: operand feeds the decode combinationally (default).
/// STAGE_INPUT=1: latch the input before any combinational logic, isolating it from upstream paths (+1 cycle).
/// STAGE_INPUT>1: add extra dummy stages; helps in routing-congested designs (+STAGE_INPUT cycles).
///
/// STAGE_DECODE=0: single-cycle combinational decode + output mux (no intermediate register).
/// STAGE_DECODE=1: registers the decoded sign / new_exp / frac / classification predicates before the output mux.
///     Splits the long route from input port to output register (the dominant delay path at wide WMAN on
///     placement-sensitive tools). Costs one extra pipeline cycle.

`default_nettype none

module zkf_mul_ilog2_const #(
    parameter         WEXP         = 6,
    parameter         WMAN         = 18,    // significand precision including the hidden bit
    parameter integer K            = 0,     // signed integer exponent shift: y = a * 2^K
    parameter         STAGE_INPUT  = 0,     // number of input register stages (>=0); +STAGE_INPUT cycles
    parameter         STAGE_DECODE = 0,     // 0 = single-cycle; 1 = register decoded signals (+1 cycle)
    parameter         LATENCY      = 0
) (
    input wire clk,
    input wire rst,

    input wire                 in_valid,
    input wire [WEXP+WMAN-1:0] a,

    output reg                 out_valid,
    output reg [WEXP+WMAN-1:0] y           // y = a * 2^K, where -(2**WEXP-2) <= K < (2**WEXP-2)
);
    localparam WFRAC    = WMAN - 1;
    localparam WFULL    = WEXP + WMAN;
    localparam WEXP_EXT = WEXP + 2;     // signed accumulator wide enough for a_exp + K at any allowed K

    // -- Optional input register stage: latch the operand before any combinational logic.
    wire             in_valid_q;
    wire [WFULL-1:0] a_q;
    zkf_pipe #(.W(WFULL), .N(STAGE_INPUT)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in(a),
        .out_valid(in_valid_q), .out(a_q)
    );

    localparam [WEXP-1:0] EXP_INF = {WEXP{1'b1}};

    // Signed integer bounds for the always-overflow / always-underflow guards. Using `integer` keeps both the
    // positive and negative limits in 32-bit signed arithmetic so that the elaboration-time comparisons against K
    // are unambiguous across Verilog tools.
    localparam integer K_LIMIT_OVERFLOW  =   (1 << WEXP) - 2;
    localparam integer K_LIMIT_UNDERFLOW = -((1 << WEXP) - 2);

    localparam LATENCY_REF = 1 + STAGE_INPUT + STAGE_DECODE;
    generate
        if ((WEXP < 2) || (WMAN < 4)) begin : g_invalid_wm
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        if ((STAGE_DECODE != 0) && (STAGE_DECODE != 1)) begin : g_invalid_stage_decode
            _zkf_invalid_stage_decode u_invalid();
        end
        // K is an integer parameter, and the K bound checks below use 32-bit integer arithmetic.
        if (WEXP >= 31) begin : g_invalid_wexp_too_wide
            _zkf_invalid_mul_ilog2_const_wexp_too_wide_unportable u_invalid();
        end
        // K = EXP_MAX_FINITE forces every normal input to overflow (new_biased_exp >= EXP_INF for old_biased_exp >= 1).
        if (K >= K_LIMIT_OVERFLOW) begin : g_invalid_k_always_overflow
            _zkf_invalid_mul_ilog2_const_k_always_overflow u_invalid();
        end
        // K < -EXP_MAX_FINITE forces zero underflow for every normal input.
        if (K < K_LIMIT_UNDERFLOW) begin : g_invalid_k_always_underflow
            _zkf_invalid_mul_ilog2_const_k_always_underflow u_invalid();
        end
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    // Decode and classify.
    wire             a_sign = a_q[WFULL-1];
    wire [WEXP-1:0]  a_exp  = a_q[WFULL-2:WFRAC];
    wire [WFRAC-1:0] a_frac = a_q[WFRAC-1:0];
    wire             a_zero = ~|a_exp;
    wire             a_inf  =  &a_exp;

    // The biased exponent of the result is a_exp + K. Because K is a parameter, its sign chooses which boundary can
    // be reached: positive shifts can only overflow, negative shifts can only underflow or round to MIN_NORMAL.
    // Splitting the cases lets synthesis remove the unused boundary logic instead of building generic add/compare
    // cones and then proving most of them constant.
    wire overflow;
    wire underflow;
    wire min_normal;
    generate
        if (STAGE_DECODE == 0) begin : g_boundary_single_cycle
            localparam signed [WEXP_EXT-1:0] K_OF_OFFSET  = K - ((1 << WEXP) - 2) - 1;
            localparam signed [WEXP_EXT-1:0] K_EXP_OFFSET = K;
            wire signed [WEXP_EXT-1:0] of_acc      = $signed({{(WEXP_EXT-WEXP){1'b0}}, a_exp}) + K_OF_OFFSET;
            wire signed [WEXP_EXT-1:0] new_exp_acc = $signed({{(WEXP_EXT-WEXP){1'b0}}, a_exp}) + K_EXP_OFFSET;
            assign overflow   = ~of_acc[WEXP_EXT-1];
            assign underflow  =  new_exp_acc[WEXP_EXT-1];
            assign min_normal = ~|new_exp_acc;
        end else if (K > 0) begin : g_boundary_positive_shift
            localparam signed [WEXP_EXT-1:0] K_OF_OFFSET = K - ((1 << WEXP) - 2) - 1;
            wire signed [WEXP_EXT-1:0] of_acc = $signed({{(WEXP_EXT-WEXP){1'b0}}, a_exp}) + K_OF_OFFSET;
            assign overflow   = ~of_acc[WEXP_EXT-1];
            assign underflow  = 1'b0;
            assign min_normal = 1'b0;
        end else if (K < 0) begin : g_boundary_negative_shift
            localparam [WEXP-1:0] EXP_MIN_NORMAL_THRESHOLD = -K;
            assign overflow   = 1'b0;
            assign underflow  = a_exp < EXP_MIN_NORMAL_THRESHOLD;
            assign min_normal = a_exp == EXP_MIN_NORMAL_THRESHOLD;
        end else begin : g_boundary_zero_shift
            assign overflow   = 1'b0;
            assign underflow  = 1'b0;
            assign min_normal = 1'b0;
        end
    endgenerate
    wire result_is_zero       = a_zero || underflow;
    wire result_is_inf        = a_inf  || overflow;
    wire result_is_min_normal = !a_zero && !a_inf && min_normal;

    // Normal-output exponent: low WEXP bits of (a_exp + K). The truncating add wraps, which is fine because the
    // normal-output mux is suppressed whenever result_is_zero or result_is_inf is asserted.
    wire [WEXP-1:0] new_exp = a_exp + K[WEXP-1:0];

    // Output candidate forms. Canonicalisation is implicit: zero has sign/frac cleared, infinity has frac cleared.
    wire [WFULL-1:0] y_inf_w        = {a_sign, EXP_INF, {WFRAC{1'b0}}};
    wire [WFULL-1:0] y_min_normal_w = {a_sign, {{(WEXP-1){1'b0}}, 1'b1}, {WFRAC{1'b0}}};
    wire [WFULL-1:0] y_normal_w     = {a_sign, new_exp, a_frac};

    // result_is_zero takes priority over result_is_inf. For valid K the two flags are mutually exclusive,
    // but the overlapping rows are listed so the priority is explicit and the case is full.
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
            reg                 r_in_valid;
            reg                 r_sign;
            reg                 r_result_is_zero;
            reg                 r_result_is_inf;
            reg                 r_result_is_min_normal;
            reg [WEXP-1:0]      r_new_exp;
            reg [WFRAC-1:0]     r_frac;
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

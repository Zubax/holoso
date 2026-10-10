// FLOATING POINT OPERATORS
//
// Using Zubax Kulibin float (ZKF) -- an IEEE 754-like format optimized for control systems and DSP applications.
// ZKF technically has no negative zero, but it is not an error to produce it -- all operators ignore the sign bit
// when the magnitude is zero.
//
// Parameters: WEXP -- exponent bit width, WMAN -- mantissa/significand bit width (incl. hidden bit).
// The total width is WFULL=WEXP+WMAN (the significand MSb is absent but there is also the sign bit, like IEEE 754).
//
// Streaming wrappers require a LATENCY parameter, which is forwarded to the wrapped Kulibin operator for checking.
// Operator-specific parameters are forwarded to the corresponding operator.

`timescale 1ns/1ps

// Forward the macro hooks to ZKF. The user can disable/override this by explicitly defining the ZKF macro hooks.
// This is used to fine-tune synthesis for the ZKF operators, such as to guide ROM inference or avoid resource sharing.
`ifdef HOLOSO_ATTRIBUTE_KEEP
`ifndef ZKF_ATTRIBUTE_KEEP
`define ZKF_ATTRIBUTE_KEEP `HOLOSO_ATTRIBUTE_KEEP
`endif
`endif
`ifdef HOLOSO_ATTRIBUTE_ROM
`ifndef ZKF_ATTRIBUTE_ROM
`define ZKF_ATTRIBUTE_ROM `HOLOSO_ATTRIBUTE_ROM
`endif
`endif

// Combinational floating-point sign conditioner; to be used at the inputs of arithmetic operators.
// Sign conditioning is a trivial and/xor single-bit gate enabling free computation of abs/neg.
// Conditional inputs can be tied off to constants, in which case the corresponding circuits are optimized away.
// Function:
//      y =     +x      if op=0
//      y =     -x      if op=1
//      y = +abs(x)     if op=2
//      y = -abs(x)     if op=3
module holoso_fsgnop#(parameter WFULL = 24) (input wire [WFULL-1:0] x, input wire [1:0] op, output wire [WFULL-1:0] y);
    wire   op_abs = op[1];
    wire   op_neg = op[0];
    wire   s_in   = x[WFULL-1];
    wire   s_out  = (s_in & ~op_abs) ^ op_neg;
    assign y      = { s_out, x[WFULL-2:0] };
endmodule

// Floating point adder/subtractor with sign conditioning:  y = sgnop(a) + sgnop(b)
// E.g., subtraction: y=a+(-b); magnitude difference: y=abs(a)-abs(b), ...
module holoso_fadd#(parameter WEXP = 6, parameter WMAN = 18,
                    parameter STAGE_INPUT = 0, parameter STAGE_DECODE = 0, parameter STAGE_ALIGN = 0,
                    parameter STAGE_NORMALIZE = 0, parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0,
                    parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] b_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] b1;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a), .op(a_sgnop), .y(a1));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_b (.x(b), .op(b_sgnop), .y(b1));
    zkf_add#(.WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT),
             .STAGE_DECODE(STAGE_DECODE), .STAGE_ALIGN(STAGE_ALIGN), .STAGE_NORMALIZE(STAGE_NORMALIZE),
             .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT), .LATENCY(LATENCY)) u_add (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a1), .b(b1),
        .out_valid(out_valid), .y(y)
    );
endmodule

// Floating point multiplier with sign conditioning: y = sgnop(a) * sgnop(b)
// Caution: STAGE_PRODUCT is almost never a good idea unless WMAN is wider than DSP multiplier input widths.
module holoso_fmul#(parameter WEXP = 6, parameter WMAN = 18, parameter WMULTIPLIER = 0,
                    parameter STAGE_INPUT = 0, parameter STAGE_PRODUCT = 0,
                    parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0,
                    parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] b_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] b1;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a), .op(a_sgnop), .y(a1));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_b (.x(b), .op(b_sgnop), .y(b1));
    zkf_mul#(.WEXP(WEXP), .WMAN(WMAN), .WMULTIPLIER(WMULTIPLIER), .STAGE_INPUT(STAGE_INPUT),
             .STAGE_PRODUCT(STAGE_PRODUCT), .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT),
             .LATENCY(LATENCY)) u_mul (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a1), .b(b1),
        .out_valid(out_valid), .y(y)
    );
endmodule

// Floating point fused multiply-add with sign conditioning:  y = sgnop(a)*sgnop(b) + sgnop(c)
// The product is kept full-width and rounded once together with c (a single rounding, unlike a multiply then add).
module holoso_ffma#(parameter WEXP = 6, parameter WMAN = 18,
                    parameter WMULTIPLIER = 0, parameter STAGE_INPUT = 0, parameter STAGE_PRODUCT = 0,
                    parameter STAGE_DECODE = 0, parameter STAGE_ALIGN = 0, parameter STAGE_NORMALIZE = 0,
                    parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] b_sgnop,
    input  wire           [1:0] c_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    input  wire [WEXP+WMAN-1:0] c,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] b1;
    wire [WFULL-1:0] c1;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a), .op(a_sgnop), .y(a1));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_b (.x(b), .op(b_sgnop), .y(b1));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_c (.x(c), .op(c_sgnop), .y(c1));
    zkf_fma#(.WEXP(WEXP), .WMAN(WMAN), .WMULTIPLIER(WMULTIPLIER), .STAGE_INPUT(STAGE_INPUT),
             .STAGE_PRODUCT(STAGE_PRODUCT), .STAGE_DECODE(STAGE_DECODE), .STAGE_ALIGN(STAGE_ALIGN),
             .STAGE_NORMALIZE(STAGE_NORMALIZE), .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT),
             .LATENCY(LATENCY)) u_fma (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a1), .b(b1), .c(c1),
        .out_valid(out_valid), .y(y)
    );
endmodule

// Exponent extraction: y = ilog2(a), the count the scaler below multiplies by.
// The result is sign-invariant so no sign conditioning is needed.
module holoso_filog2#(parameter WEXP = 6, parameter WMAN = 18, parameter WINT = WEXP + WMAN,
                      parameter STAGE_INPUT = 0, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire [WEXP+WMAN-1:0] a,
    output wire                 out_valid,
    output wire signed [WINT-1:0] y
);
    zkf_ilog2#(.WEXP(WEXP), .WMAN(WMAN), .WINT(WINT), .STAGE_INPUT(STAGE_INPUT), .LATENCY(LATENCY)) u_ilog2 (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a),
        .out_valid(out_valid), .y(y),
        .zero(), .infinity(), .negative()
    );
endmodule

// Power-of-two scaler with sign conditioning: y = sgnop(a) * 2^k.
// Every representable k is legal; large values collapse to infinities or zero.
// Scaling is exact while the result remains normal and finite.
// Zero and infinity retain their class regardless of k.
module holoso_fmul_ilog2#(parameter WEXP = 6, parameter WMAN = 18, parameter WINT = WEXP + WMAN,
                          parameter STAGE_INPUT = 0, parameter STAGE_DECODE = 0,
                          parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    input  wire signed [WINT-1:0] k,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a), .op(a_sgnop), .y(a1));
    zkf_mul_ilog2#(.WEXP(WEXP), .WMAN(WMAN), .WK(WINT), .STAGE_INPUT(STAGE_INPUT),
                   .STAGE_DECODE(STAGE_DECODE), .LATENCY(LATENCY)) u_mul_ilog2 (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a1), .k(k),
        .out_valid(out_valid), .y(y)
    );
endmodule

// Floating point divider and square root with sign conditioning, the operation chosen per transaction, or fixed by MODE
// (0=division, 1=sqrt), which drops the other operation's datapath and ignores `op_sqrt`:
//      op_sqrt=0:  y = sgnop(a) / sgnop(b)
//      op_sqrt=1:  y = sqrt(sgnop(a)); b is ignored
// Both are correctly rounded (nearest, ties to even) and take the same latency. error is asserted alongside out_valid
// when the conditioned divisor is zero (y is then an infinity of the dividend's sign, or zero for a zero dividend),
// or when the conditioned radicand is negative (a negative zero is not; y is then -inf).
module holoso_fdivsqrt#(parameter WEXP = 6, parameter WMAN = 18,
                        parameter STAGE_INPUT = 0, parameter STAGE_DECODE = 0,
                        parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0,
                        parameter integer MODE = 2, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire                 op_sqrt,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] b_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y,
    output wire                 error
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] b1;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a), .op(a_sgnop), .y(a1));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_b (.x(b), .op(b_sgnop), .y(b1));
    zkf_divsqrt#(.WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT), .STAGE_DECODE(STAGE_DECODE),
                 .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT), .MODE(MODE), .LATENCY(LATENCY)) u_divsqrt (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .op_sqrt(op_sqrt), .a(a1), .b(b1),
        .out_valid(out_valid), .y(y), .error(error)
    );
endmodule

// Floating point min/max sorter with sign conditioning:
//      min = min(sgnop(a), sgnop(b))
//      max = max(sgnop(a), sgnop(b))
// Useful for e.g. sort-by-absolute-value.
module holoso_fsort#(parameter WEXP = 6, parameter WMAN = 18, parameter integer STAGE_INPUT = 0,
                     parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] b_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] min,
    output wire [WEXP+WMAN-1:0] max
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] b1;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a), .op(a_sgnop), .y(a1));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_b (.x(b), .op(b_sgnop), .y(b1));
    zkf_sort#(.WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT), .LATENCY(LATENCY)) u_sort (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a1), .b(b1),
        .out_valid(out_valid), .min(min), .max(max)
    );
endmodule

// Fixed-latency facade over the handshaked, non-throughput-1 zkf_cordic (one transaction in flight), its mode chosen
// per transaction, or fixed by MODE (0 rotation, 1 vectoring), which drops the other mode's datapath and ignores
// `vectoring` and the other mode's latency parameter:
//      vectoring=0:  r0 = sin(2*pi*sgnop(a)), r1 = cos(2*pi*sgnop(a)); b is ignored
//      vectoring=1:  r0 = atan2(sgnop(a), sgnop(b)) in turns, r1 = hypot(sgnop(a), sgnop(b))
// The scheduler spaces the issues by each transaction's initiation interval, its mode's LATENCY+1, so the core is
// idle at issue, out_ready is tied high, and the result is captured on its out_valid cycle -- the same static-schedule
// contract as the pipelined wrappers.
module holoso_fcordic#(parameter WEXP = 6, parameter WMAN = 18, parameter WMULTIPLIER = 0,
                       parameter integer UNROLL100 = 100,
                       parameter integer STAGE_INPUT = 0, parameter integer STAGE_PRODUCT = 0,
                       parameter integer STAGE_NORMALIZE = 0, parameter integer STAGE_PACK = 0,
                       parameter integer STAGE_OUTPUT = 0, parameter integer MODE = 2,
                       parameter integer LATENCY_ROTATION = 0, parameter integer LATENCY_VECTORING = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire                 vectoring,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] b_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] r0,
    output wire [WEXP+WMAN-1:0] r1
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] b1;
    wire             core_in_ready;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a), .op(a_sgnop), .y(a1));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_b (.x(b), .op(b_sgnop), .y(b1));
    zkf_cordic#(.WEXP(WEXP), .WMAN(WMAN), .WMULTIPLIER(WMULTIPLIER), .UNROLL100(UNROLL100),
                .STAGE_INPUT(STAGE_INPUT), .STAGE_PRODUCT(STAGE_PRODUCT), .STAGE_NORMALIZE(STAGE_NORMALIZE),
                .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT), .MODE(MODE),
                .LATENCY_ROTATION(LATENCY_ROTATION), .LATENCY_VECTORING(LATENCY_VECTORING)) u_cordic (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_ready(core_in_ready), .vectoring(vectoring), .a(a1), .b(b1),
        .out_valid(out_valid), .out_ready(1'b1), .r0(r0), .r1(r1), .quadrant()
    );
`ifdef SIMULATION
    always @(posedge clk) begin
        if (!rst && in_valid && !core_in_ready)
            $fatal(1, "holoso_fcordic over-issued: in_valid while busy (initiation_interval too small)");
    end
`endif
endmodule

// Floating point comparator with sign conditioning:
//      (a_gt_b, a_eq_b, a_lt_b) = compare(sgnop(a), sgnop(b))
// Outputs are mutually-exclusive one-hot flags.
module holoso_fcmp#(parameter WEXP = 6, parameter WMAN = 18, parameter integer STAGE_INPUT = 0,
                    parameter integer STAGE_OUTPUT = 0, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] b_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    output wire                 out_valid,
    output wire                 a_gt_b,
    output wire                 a_eq_b,
    output wire                 a_lt_b
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] b1;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a), .op(a_sgnop), .y(a1));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_b (.x(b), .op(b_sgnop), .y(b1));
    zkf_cmp#(.WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT), .STAGE_OUTPUT(STAGE_OUTPUT),
             .LATENCY(LATENCY)) u_cmp (
        .clk(clk), .rst(rst), .in_valid(in_valid), .a(a1), .b(b1),
        .out_valid(out_valid), .a_gt_b(a_gt_b), .a_eq_b(a_eq_b), .a_lt_b(a_lt_b));
endmodule

// Base-two exponential with sign conditioning:  y = 2 ** sgnop(a)
module holoso_fexp2#(parameter WEXP = 6, parameter WMAN = 18, parameter WMULTIPLIER = 0,
                     parameter STAGE_INPUT = 0, parameter STAGE_REDUCE = 0, parameter STAGE_PRODUCT = 0,
                     parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0,
                     parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a), .op(a_sgnop), .y(a1));
    zkf_exp2#(.WEXP(WEXP), .WMAN(WMAN), .WMULTIPLIER(WMULTIPLIER),
              .STAGE_INPUT(STAGE_INPUT), .STAGE_REDUCE(STAGE_REDUCE), .STAGE_PRODUCT(STAGE_PRODUCT),
              .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT), .LATENCY(LATENCY)) u_exp2 (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .x(a1),
        .out_valid(out_valid), .y(y)
    );
endmodule

// Base-two logarithm with sign conditioning:  y = log2(sgnop(a))
// domain_error is asserted alongside out_valid when the conditioned operand is negative; pole when it is zero. y is
// -inf in both cases.
module holoso_flog2#(parameter WEXP = 6, parameter WMAN = 18, parameter WMULTIPLIER = 0,
                     parameter STAGE_INPUT = 0, parameter STAGE_DECODE = 0, parameter STAGE_PRODUCT = 0,
                     parameter STAGE_PRODUCT_FINAL = 0, parameter STAGE_NORMALIZE = 0,
                     parameter STAGE_NORMALIZE_OUTPUT = 0, parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0,
                     parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y,
    output wire                 domain_error,
    output wire                 pole
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a), .op(a_sgnop), .y(a1));
    zkf_log2#(.WEXP(WEXP), .WMAN(WMAN), .WMULTIPLIER(WMULTIPLIER),
              .STAGE_INPUT(STAGE_INPUT), .STAGE_DECODE(STAGE_DECODE), .STAGE_PRODUCT(STAGE_PRODUCT),
              .STAGE_PRODUCT_FINAL(STAGE_PRODUCT_FINAL), .STAGE_NORMALIZE(STAGE_NORMALIZE),
              .STAGE_NORMALIZE_OUTPUT(STAGE_NORMALIZE_OUTPUT), .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT),
              .LATENCY(LATENCY)) u_log2 (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .x(a1),
        .out_valid(out_valid), .y(y), .domain_error(domain_error), .pole(pole)
    );
endmodule

// Floating point round-to-integer with sign conditioning, answered twice on the same cycle:
//      y_float = round(sgnop(a), round_mode) as a float of the same format
//      y_int   = the same value as a signed integer
// round_mode selects the rounding per transaction (matches zkf_rint): 0=nearest-even, 1=floor, 2=ceil, 3=trunc.
// y_int saturates, which is normal behavior, not an error: values above 2^(WINT-1)-1 become that maximum and values
// below -2^(WINT-1) that minimum, infinities included. An output left unconnected costs no logic.
module holoso_frint#(parameter WEXP = 6, parameter WMAN = 18, parameter WINT = WEXP + WMAN,
                     parameter STAGE_INPUT = 0, parameter STAGE_SHIFT = 0,
                     parameter STAGE_ROUND = 0, parameter STAGE_OUTPUT = 0,
                     parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] round_mode,
    input  wire [WEXP+WMAN-1:0] a,
    output wire                   out_valid,
    output wire [WEXP+WMAN-1:0]   y_float,
    output wire signed [WINT-1:0] y_int
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a), .op(a_sgnop), .y(a1));
    zkf_rint#(.WEXP(WEXP), .WMAN(WMAN), .WINT(WINT), .STAGE_INPUT(STAGE_INPUT), .STAGE_SHIFT(STAGE_SHIFT),
              .STAGE_ROUND(STAGE_ROUND), .STAGE_OUTPUT(STAGE_OUTPUT), .LATENCY(LATENCY)) u_rint (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a1), .round_mode(round_mode),
        .out_valid(out_valid), .y_float(y_float), .y_int(y_int)
    );
endmodule

// Signed-integer-to-float conversion.
// Conversion is round-to-nearest, ties-to-even; values outside the finite float range become signed infinity.
module holoso_ffromint#(parameter WEXP = 6, parameter WMAN = 18, parameter WINT = WEXP + WMAN,
                        parameter STAGE_INPUT = 0, parameter STAGE_NORMALIZE = 0,
                        parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0,
                        parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                   in_valid,
    input  wire signed [WINT-1:0] a,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    zkf_from_int#(.WEXP(WEXP), .WMAN(WMAN), .WINT(WINT), .STAGE_INPUT(STAGE_INPUT),
                  .STAGE_NORMALIZE(STAGE_NORMALIZE), .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT),
                  .LATENCY(LATENCY)) u_from_int (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a),
        .out_valid(out_valid), .y(y)
    );
endmodule

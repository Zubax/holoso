`timescale 1ns/1ps

// =====================================================================================================================
// FLOATING POINT BASIC OPERATORS
//
// Using Zubax Kulibin float (ZKF) -- an IEEE 754-like format with simplifications: no NaN, no subnormals.
// ZKF technically has no negative zero, but it is not an error to produce it -- all operators ignore the sign bit
// when the magnitude is zero.
//
// Parameters: WEXP -- exponent bit width, WMAN -- mantissa/significand bit width (incl. hidden bit).
// The total width is WFULL=WEXP+WMAN (the significand MSb is absent but there is also the sign bit, like IEEE 754).
//
// Streaming wrappers require a LATENCY parameter, which is forwarded to the wrapped Kulibin operator for checking.
// Stage parameters are forwarded as-is; refer to the corresponding Kulibin operator source for their timing details.

// Combinational floating-point sign conditioner operator; to be used at the inputs/outputs of arithmetic operators.
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

// Floating point adder/subtractor with sign conditioning:  y = sgnop(sgnop(a) + sgnop(b))
// E.g., subtraction: y=a+(-b); negative absolute difference: y=-abs(a-b), magnitude difference: y=abs(a)-abs(b), ...
// The inputs are sampled once at in_valid and are not required to remain stable during operation.
module holoso_fadd#(parameter WEXP = 6, parameter WMAN = 18,
                    parameter STAGE_INPUT = 0, parameter STAGE_DECODE = 0, parameter STAGE_ALIGN = 0,
                    parameter STAGE_NORMALIZE = 0, parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0,
                    parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] b_sgnop,
    input  wire           [1:0] y_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] b1;
    wire [WFULL-1:0] y1;
    wire       [1:0] y_sgnop_q;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a), .op(a_sgnop), .y(a1));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_b (.x(b), .op(b_sgnop), .y(b1));
    zkf_pipe#(.W(2), .N(LATENCY)) u_y_sgnop_pipe (.clk(clk), .rst(rst), .in_valid(in_valid), .in(y_sgnop),
                                                  .out_valid(), .out(y_sgnop_q));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_y (.x(y1), .op(y_sgnop_q), .y(y));
    zkf_add#(.WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT),
             .STAGE_DECODE(STAGE_DECODE), .STAGE_ALIGN(STAGE_ALIGN), .STAGE_NORMALIZE(STAGE_NORMALIZE),
             .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT), .LATENCY(LATENCY)) u_add (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a1), .b(b1),
        .out_valid(out_valid), .y(y1)
    );
endmodule

// Floating point multiplier with sign conditioning: y = sgnop(sgnop(a) * sgnop(b))
// The inputs are sampled once at in_valid and are not required to remain stable during operation.
// Caution: STAGE_PRODUCT is almost never a good idea unless WMAN is wider than DSP multiplier input widths.
module holoso_fmul#(parameter WEXP = 6, parameter WMAN = 18, parameter STAGE_INPUT = 0,
                    parameter STAGE_PRODUCT = 0, parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0,
                    parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] b_sgnop,
    input  wire           [1:0] y_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] b1;
    wire [WFULL-1:0] y1;
    wire       [1:0] y_sgnop_q;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a), .op(a_sgnop), .y(a1));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_b (.x(b), .op(b_sgnop), .y(b1));
    zkf_pipe#(.W(2), .N(LATENCY)) u_y_sgnop_pipe (.clk(clk), .rst(rst), .in_valid(in_valid), .in(y_sgnop),
                                                  .out_valid(), .out(y_sgnop_q));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_y (.x(y1), .op(y_sgnop_q), .y(y));
    zkf_mul#(.WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT),
             .STAGE_PRODUCT(STAGE_PRODUCT), .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT),
             .LATENCY(LATENCY)) u_mul (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a1), .b(b1),
        .out_valid(out_valid), .y(y1)
    );
endmodule

// Floating point fused multiply-add with sign conditioning:  y = sgnop(sgnop(a)*sgnop(b) + sgnop(c))
// The product is kept full-width and rounded once together with c (a single rounding, unlike a multiply then add).
// The inputs are sampled once at in_valid and are not required to remain stable during operation.
module holoso_ffma#(parameter WEXP = 6, parameter WMAN = 18,
                    parameter STAGE_INPUT = 0, parameter STAGE_PRODUCT = 0, parameter STAGE_DECODE = 0,
                    parameter STAGE_ALIGN = 0, parameter STAGE_NORMALIZE = 0, parameter STAGE_PACK = 0,
                    parameter STAGE_OUTPUT = 0, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] b_sgnop,
    input  wire           [1:0] c_sgnop,
    input  wire           [1:0] y_sgnop,
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
    wire [WFULL-1:0] y1;
    wire       [1:0] y_sgnop_q;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a), .op(a_sgnop), .y(a1));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_b (.x(b), .op(b_sgnop), .y(b1));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_c (.x(c), .op(c_sgnop), .y(c1));
    zkf_pipe#(.W(2), .N(LATENCY)) u_y_sgnop_pipe (.clk(clk), .rst(rst), .in_valid(in_valid), .in(y_sgnop),
                                                  .out_valid(), .out(y_sgnop_q));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_y (.x(y1), .op(y_sgnop_q), .y(y));
    zkf_fma#(.WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT), .STAGE_PRODUCT(STAGE_PRODUCT),
             .STAGE_DECODE(STAGE_DECODE), .STAGE_ALIGN(STAGE_ALIGN), .STAGE_NORMALIZE(STAGE_NORMALIZE),
             .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT), .LATENCY(LATENCY)) u_fma (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a1), .b(b1), .c(c1),
        .out_valid(out_valid), .y(y1)
    );
endmodule

// Constant-power-of-two scaler with sign conditioning:  y = sgnop(sgnop(a) * 2^K)
// K is a signed integer exponent shift; -2^(WEXP-1) < K < 2^(WEXP-1). The scaling is exact.
// The inputs are sampled once at in_valid and are not required to remain stable during operation.
module holoso_fmul_ilog2_const#(parameter WEXP = 6, parameter WMAN = 18, parameter integer K = 0,
                                parameter STAGE_INPUT = 0, parameter STAGE_DECODE = 0,
                                parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] y_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] y1;
    wire       [1:0] y_sgnop_q;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a),  .op(a_sgnop), .y(a1));
    zkf_pipe#(.W(2), .N(LATENCY)) u_y_sgnop_pipe (.clk(clk), .rst(rst), .in_valid(in_valid), .in(y_sgnop),
                                                  .out_valid(), .out(y_sgnop_q));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_y (.x(y1), .op(y_sgnop_q), .y(y));
    zkf_mul_ilog2_const#(.WEXP(WEXP), .WMAN(WMAN), .K(K),
                         .STAGE_INPUT(STAGE_INPUT), .STAGE_DECODE(STAGE_DECODE), .LATENCY(LATENCY)) u_mul_ilog2_const (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a1),
        .out_valid(out_valid), .y(y1)
    );
endmodule

// Floating point round-to-integer with sign conditioning:  y = sgnop(round(sgnop(a), round_mode))
// round_mode selects the mode per transaction: 0 nearest-even, 1 floor, 2 ceil, 3 trunc (the zkf_round encoding).
// The input is sampled once at in_valid and is not required to remain stable during operation.
module holoso_fround#(parameter WEXP = 6, parameter WMAN = 18,
                      parameter STAGE_INPUT = 0, parameter STAGE_DECODE = 0,
                      parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0,
                      parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] round_mode,
    input  wire           [1:0] y_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] y1;
    wire       [1:0] y_sgnop_q;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a), .op(a_sgnop), .y(a1));
    zkf_pipe#(.W(2), .N(LATENCY)) u_y_sgnop_pipe (.clk(clk), .rst(rst), .in_valid(in_valid), .in(y_sgnop),
                                                  .out_valid(), .out(y_sgnop_q));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_y (.x(y1), .op(y_sgnop_q), .y(y));
    zkf_round#(.WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT), .STAGE_DECODE(STAGE_DECODE),
               .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT), .LATENCY(LATENCY)) u_round (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a1), .round_mode(round_mode),
        .out_valid(out_valid), .y(y1)
    );
endmodule

// Floating point divider with sign conditioning: y = sgnop(sgnop(a) / sgnop(b))
// div0 is asserted alongside out_valid when the divisor is (positive) zero; the value of y is then unspecified.
// The quotient is rounded.
// The inputs are sampled once at in_valid and are not required to remain stable during operation.
module holoso_fdiv#(parameter WEXP = 6, parameter WMAN = 18,
                    parameter STAGE_INPUT = 0, parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0,
                    parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] b_sgnop,
    input  wire           [1:0] y_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y,
    output wire                 div0
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] b1;
    wire [WFULL-1:0] y1;
    wire       [1:0] y_sgnop_q;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a),  .op(a_sgnop), .y(a1));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_b (.x(b),  .op(b_sgnop), .y(b1));
    zkf_pipe#(.W(2), .N(LATENCY)) u_y_sgnop_pipe (.clk(clk), .rst(rst), .in_valid(in_valid), .in(y_sgnop),
                                                  .out_valid(), .out(y_sgnop_q));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_y (.x(y1), .op(y_sgnop_q), .y(y));
    zkf_div#(.WEXP(WEXP), .WMAN(WMAN),
             .STAGE_INPUT(STAGE_INPUT), .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT),
             .LATENCY(LATENCY)) u_div (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a1), .b(b1),
        .out_valid(out_valid), .q(y1), .div0(div0)
    );
endmodule

// Floating point min/max sorter with sign conditioning on inputs and outputs:
//      min = sgnop(min(sgnop(a), sgnop(b)))
//      max = sgnop(max(sgnop(a), sgnop(b)))
// Useful for e.g. sort-by-absolute-value or producing sign-flipped extrema.
module holoso_fsort#(parameter WEXP = 6, parameter WMAN = 18, parameter integer STAGE_INPUT = 0,
                     parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] b_sgnop,
    input  wire           [1:0] min_sgnop,
    input  wire           [1:0] max_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] min,
    output wire [WEXP+WMAN-1:0] max
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] b1;
    wire [WFULL-1:0] min1;
    wire [WFULL-1:0] max1;
    wire       [3:0] out_sgnop_q;
    wire       [1:0] min_sgnop_q = out_sgnop_q[1:0];
    wire       [1:0] max_sgnop_q = out_sgnop_q[3:2];
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a   (.x(a),    .op(a_sgnop),   .y(a1));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_b   (.x(b),    .op(b_sgnop),   .y(b1));
    zkf_pipe#(.W(4), .N(LATENCY)) u_out_sgnop_pipe (.clk(clk), .rst(rst), .in_valid(in_valid),
                                                    .in({max_sgnop, min_sgnop}), .out_valid(), .out(out_sgnop_q));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_min (.x(min1), .op(min_sgnop_q), .y(min));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_max (.x(max1), .op(max_sgnop_q), .y(max));
    zkf_sort#(.WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT), .LATENCY(LATENCY)) u_sort (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .a(a1), .b(b1),
        .out_valid(out_valid), .min(min1), .max(max1)
    );
endmodule

// Fixed-latency facade over the handshaked, non-throughput-1 zkf_sincos CORDIC (one transaction in flight): the
// scheduler spaces issues by initiation_interval = LATENCY+1, so the core is idle at issue, out_ready is tied high,
// and the result is captured on its out_valid cycle -- the same static-schedule contract as the pipelined wrappers.
module holoso_fsincos#(parameter WEXP = 6, parameter WMAN = 18, parameter integer UNROLL100 = 100,
                       parameter integer STAGE_INPUT = 0, parameter integer STAGE_PRODUCT = 0,
                       parameter integer STAGE_NORMALIZE = 0, parameter integer STAGE_PACK = 0,
                       parameter integer STAGE_OUTPUT = 0, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] sin_sgnop,
    input  wire           [1:0] cos_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] sin,
    output wire [WEXP+WMAN-1:0] cos
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] sin1;
    wire [WFULL-1:0] cos1;
    wire       [3:0] out_sgnop_q;
    wire             core_in_ready;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a   (.x(a),    .op(a_sgnop),           .y(a1));
    zkf_pipe#(.W(4), .N(LATENCY)) u_out_sgnop_pipe (.clk(clk), .rst(rst), .in_valid(in_valid),
                                                    .in({cos_sgnop, sin_sgnop}), .out_valid(), .out(out_sgnop_q));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_sin (.x(sin1), .op(out_sgnop_q[1:0]), .y(sin));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_cos (.x(cos1), .op(out_sgnop_q[3:2]), .y(cos));
    zkf_sincos#(.WEXP(WEXP), .WMAN(WMAN), .UNROLL100(UNROLL100), .STAGE_INPUT(STAGE_INPUT),
                .STAGE_PRODUCT(STAGE_PRODUCT), .STAGE_NORMALIZE(STAGE_NORMALIZE), .STAGE_PACK(STAGE_PACK),
                .STAGE_OUTPUT(STAGE_OUTPUT), .LATENCY(LATENCY)) u_sincos (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_ready(core_in_ready), .x(a1),
        .out_valid(out_valid), .out_ready(1'b1), .sin(sin1), .cos(cos1), .quadrant()
    );
`ifdef SIMULATION
    always @(posedge clk) begin
        if (!rst && in_valid && !core_in_ready)
            $fatal(1, "holoso_fsincos over-issued: in_valid while busy (initiation_interval too small)");
    end
`endif
endmodule

// Fixed-latency facade over the handshaked zkf_atan2 CORDIC (see holoso_fsincos). Operand a is y, b is x; outputs
// theta in turns and mag = hypot(y, x).
module holoso_fatan2#(parameter WEXP = 6, parameter WMAN = 18, parameter integer UNROLL100 = 100,
                      parameter integer STAGE_INPUT = 0, parameter integer STAGE_PRODUCT = 0,
                      parameter integer STAGE_NORMALIZE = 0, parameter integer STAGE_PACK = 0,
                      parameter integer STAGE_OUTPUT = 0, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] b_sgnop,
    input  wire           [1:0] theta_sgnop,
    input  wire           [1:0] mag_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    input  wire [WEXP+WMAN-1:0] b,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] theta,
    output wire [WEXP+WMAN-1:0] mag
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] b1;
    wire [WFULL-1:0] theta1;
    wire [WFULL-1:0] mag1;
    wire       [3:0] out_sgnop_q;
    wire             core_in_ready;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a     (.x(a), .op(a_sgnop), .y(a1));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_b     (.x(b), .op(b_sgnop), .y(b1));
    zkf_pipe#(.W(4), .N(LATENCY)) u_out_sgnop_pipe (.clk(clk), .rst(rst), .in_valid(in_valid),
                                                    .in({mag_sgnop, theta_sgnop}), .out_valid(), .out(out_sgnop_q));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_theta (.x(theta1), .op(out_sgnop_q[1:0]), .y(theta));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_mag   (.x(mag1),   .op(out_sgnop_q[3:2]), .y(mag));
    zkf_atan2#(.WEXP(WEXP), .WMAN(WMAN), .UNROLL100(UNROLL100), .STAGE_INPUT(STAGE_INPUT),
               .STAGE_PRODUCT(STAGE_PRODUCT), .STAGE_NORMALIZE(STAGE_NORMALIZE), .STAGE_PACK(STAGE_PACK),
               .STAGE_OUTPUT(STAGE_OUTPUT), .LATENCY(LATENCY)) u_atan2 (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_ready(core_in_ready), .y(a1), .x(b1),
        .out_valid(out_valid), .out_ready(1'b1), .theta(theta1), .mag(mag1)
    );
`ifdef SIMULATION
    always @(posedge clk) begin
        if (!rst && in_valid && !core_in_ready)
            $fatal(1, "holoso_fatan2 over-issued: in_valid while busy (initiation_interval too small)");
    end
`endif
endmodule

// Floating point comparator with sign conditioning on inputs only:
//      (a_gt_b, a_eq_b, a_lt_b) = compare(sgnop(a), sgnop(b))
// Outputs are mutually-exclusive one-hot flags.
module holoso_fcmp#(parameter WEXP = 6, parameter WMAN = 18, parameter integer STAGE_INPUT = 0,
                    parameter integer LATENCY = 0) (
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
    zkf_cmp#(.WEXP(WEXP), .WMAN(WMAN), .STAGE_INPUT(STAGE_INPUT), .LATENCY(LATENCY)) u_cmp (
        .clk(clk), .rst(rst), .in_valid(in_valid), .a(a1), .b(b1),
        .out_valid(out_valid), .a_gt_b(a_gt_b), .a_eq_b(a_eq_b), .a_lt_b(a_lt_b));
endmodule

// Base-two exponential with sign conditioning:  y = sgnop(2 ** sgnop(a))
// The input is sampled once at in_valid and is not required to remain stable during operation.
module holoso_fexp2#(parameter WEXP = 6, parameter WMAN = 18,
                     parameter STAGE_INPUT = 0, parameter STAGE_REDUCE = 0, parameter STAGE_PRODUCT = 0,
                     parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0,
                     parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] y_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] y1;
    wire       [1:0] y_sgnop_q;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a),  .op(a_sgnop), .y(a1));
    zkf_pipe#(.W(2), .N(LATENCY)) u_y_sgnop_pipe (.clk(clk), .rst(rst), .in_valid(in_valid), .in(y_sgnop),
                                                  .out_valid(), .out(y_sgnop_q));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_y (.x(y1), .op(y_sgnop_q), .y(y));
    zkf_exp2#(.WEXP(WEXP), .WMAN(WMAN),
              .STAGE_INPUT(STAGE_INPUT), .STAGE_REDUCE(STAGE_REDUCE), .STAGE_PRODUCT(STAGE_PRODUCT),
              .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT), .LATENCY(LATENCY)) u_exp2 (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .x(a1),
        .out_valid(out_valid), .y(y1)
    );
endmodule

// Base-two logarithm with sign conditioning:  y = sgnop(log2(sgnop(a)))
// domain_error is asserted alongside out_valid when the conditioned operand is negative; pole when it is zero. y is
// -inf in both cases. The input is sampled once at in_valid and is not required to remain stable during operation.
module holoso_flog2#(parameter WEXP = 6, parameter WMAN = 18,
                     parameter STAGE_INPUT = 0, parameter STAGE_DECODE = 0, parameter STAGE_PRODUCT = 0,
                     parameter STAGE_PRODUCT_FINAL = 0, parameter STAGE_NORMALIZE = 0,
                     parameter STAGE_NORMALIZE_OUTPUT = 0, parameter STAGE_PACK = 0, parameter STAGE_OUTPUT = 0,
                     parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire                 in_valid,
    input  wire           [1:0] a_sgnop,
    input  wire           [1:0] y_sgnop,
    input  wire [WEXP+WMAN-1:0] a,
    output wire                 out_valid,
    output wire [WEXP+WMAN-1:0] y,
    output wire                 domain_error,
    output wire                 pole
);
    localparam WFULL = WEXP + WMAN;
    wire [WFULL-1:0] a1;
    wire [WFULL-1:0] y1;
    wire       [1:0] y_sgnop_q;
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_a (.x(a),  .op(a_sgnop), .y(a1));
    zkf_pipe#(.W(2), .N(LATENCY)) u_y_sgnop_pipe (.clk(clk), .rst(rst), .in_valid(in_valid), .in(y_sgnop),
                                                  .out_valid(), .out(y_sgnop_q));
    holoso_fsgnop#(.WFULL(WFULL)) u_sgnop_y (.x(y1), .op(y_sgnop_q), .y(y));
    zkf_log2#(.WEXP(WEXP), .WMAN(WMAN),
              .STAGE_INPUT(STAGE_INPUT), .STAGE_DECODE(STAGE_DECODE), .STAGE_PRODUCT(STAGE_PRODUCT),
              .STAGE_PRODUCT_FINAL(STAGE_PRODUCT_FINAL), .STAGE_NORMALIZE(STAGE_NORMALIZE),
              .STAGE_NORMALIZE_OUTPUT(STAGE_NORMALIZE_OUTPUT), .STAGE_PACK(STAGE_PACK), .STAGE_OUTPUT(STAGE_OUTPUT),
              .LATENCY(LATENCY)) u_log2 (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .x(a1),
        .out_valid(out_valid), .y(y1), .domain_error(domain_error), .pole(pole)
    );
endmodule

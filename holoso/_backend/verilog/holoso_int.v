// SIGNED INTEGER OPERATORS
//
// Every operator has a mandatory input and output latches, exposing no combinational circuits outside.
//
//  Module          | Operation                                     |Latency| Inputs    | Outputs
//  ----------------|-----------------------------------------------|-------|-----------|---------------------------
//  holoso_iadds    | Signed addition, saturated                    | 2     | a, b      | y, saturated
//  holoso_isubs    | Signed subtraction, saturated                 | 2     | a, b      | y, saturated
//  holoso_imuls    | Signed multiplication, saturated              | 2..6  | a, b      | y, saturated
//  holoso_idivs    | Signed division and modulo, saturated         | 2+W/2 | num, den  | quo, rem, saturated, div0
//  holoso_iabss    | Absolute value, saturated                     | 2     | x         | y, saturated
//  holoso_ashift   | Arith. shift by runtime amount left+/right-   | 2     | x, shamt  | y
//  holoso_ashiftc  | Like holoso_ashift but by a constant          | 0     | x, shamt  | (inline comb function)
//  holoso_icmp     | Signed comparison                             | 2     | a, b      | a_gt_b, a_eq_b, a_lt_b

`timescale 1ns/1ps

// Signed integer adder with saturation.
module holoso_iadds#(parameter W = 44, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire signed [W-1:0] a,
    input  wire signed [W-1:0] b,
    output reg out_valid,
    output reg signed [W-1:0] y,
    output reg saturated
);
    localparam integer LATENCY_REF = 2;
    localparam signed [W-1:0] MIN = {1'b1, {(W-1){1'b0}}};
    localparam signed [W-1:0] MAX = {1'b0, {(W-1){1'b1}}};
    generate
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _holoso_invalid_integer_latency u_invalid();
        end
    endgenerate

    reg signed [W-1:0] a_q;
    reg signed [W-1:0] b_q;
    reg input_valid_q;
    wire [W:0] sum_ext = {1'b0, a_q} + {1'b0, b_q};
    wire carry_into_sign = sum_ext[W-1] ^ a_q[W-1] ^ b_q[W-1];
    wire overflow = carry_into_sign ^ sum_ext[W];

    always @(posedge clk) begin
        a_q <= a;
        b_q <= b;
        y <= overflow ? (a_q[W-1] ? MIN : MAX) : sum_ext[W-1:0];
        saturated <= overflow;
        if (rst) begin
            input_valid_q <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            input_valid_q <= in_valid;
            out_valid <= input_valid_q;
        end
    end
endmodule

// Signed integer subtractor with saturation.
module holoso_isubs#(parameter W = 44, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire signed [W-1:0] a,
    input  wire signed [W-1:0] b,
    output reg out_valid,
    output reg signed [W-1:0] y,
    output reg saturated
);
    localparam integer LATENCY_REF = 2;
    localparam signed [W-1:0] MIN = {1'b1, {(W-1){1'b0}}};
    localparam signed [W-1:0] MAX = {1'b0, {(W-1){1'b1}}};
    generate
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _holoso_invalid_integer_latency u_invalid();
        end
    endgenerate

    reg signed [W-1:0] a_q;
    reg signed [W-1:0] b_q;
    reg input_valid_q;
    wire signed [W-1:0] diff = a_q - b_q;
    wire overflow = (a_q[W-1] ^ b_q[W-1]) & (diff[W-1] ^ a_q[W-1]);

    always @(posedge clk) begin
        a_q <= a;
        b_q <= b;
        y <= overflow ? (a_q[W-1] ? MIN : MAX) : diff;
        saturated <= overflow;
        if (rst) begin
            input_valid_q <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            input_valid_q <= in_valid;
            out_valid <= input_valid_q;
        end
    end
endmodule

// Signed integer multiplier with saturation. Inputs and outputs are always registered; the internals are configurable.
// LATENCY = 2 + STAGE_PRODUCT
// STAGE_PRODUCT=0: native multiplication without additional registers;
// STAGE_PRODUCT=1: native multiplication with a dedicated product result stage (DSP output latch);
// STAGE_PRODUCT=2: native multiplication with operand capture and product result stages (DSP registered on both ends);
// STAGE_PRODUCT=3: STAGE_PRODUCT=2 plus registered 2x2 split products and reduction;
// STAGE_PRODUCT=4: STAGE_PRODUCT=2 plus registered 3x3 split products, row reduction, and final reduction.
module holoso_imuls #(parameter W = 44, parameter integer STAGE_PRODUCT = 0, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire signed [W-1:0] a,
    input  wire signed [W-1:0] b,
    output reg out_valid,
    output reg signed [W-1:0] y,
    output reg saturated
);
    localparam integer WP = 2 * W;
    localparam integer LATENCY_REF = 2 + STAGE_PRODUCT;
    localparam signed [W-1:0] MIN = {1'b1, {(W-1){1'b0}}};
    localparam signed [W-1:0] MAX = {1'b0, {(W-1){1'b1}}};
    generate
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _holoso_invalid_integer_latency u_invalid();
        end
    endgenerate

    reg signed [W-1:0] input_a;
    reg signed [W-1:0] input_b;
    reg input_valid_q;
    wire signed [WP-1:0] product;
    wire product_valid;
    generate
        if (STAGE_PRODUCT == 0) begin : g_sp0
            _holoso_imuls_sp0#(.W(W)) u_product (
                .clk(clk), .rst(rst), .in_valid(input_valid_q), .a(input_a), .b(input_b),
                .out_valid(product_valid), .product(product)
            );
        end else if (STAGE_PRODUCT == 1) begin : g_sp1
            _holoso_imuls_sp1#(.W(W)) u_product (
                .clk(clk), .rst(rst), .in_valid(input_valid_q), .a(input_a), .b(input_b),
                .out_valid(product_valid), .product(product)
            );
        end else if (STAGE_PRODUCT == 2) begin : g_sp2
            _holoso_imuls_sp2#(.W(W)) u_product (
                .clk(clk), .rst(rst), .in_valid(input_valid_q), .a(input_a), .b(input_b),
                .out_valid(product_valid), .product(product)
            );
        end else if (STAGE_PRODUCT == 3) begin : g_sp3
            _holoso_imuls_sp3#(.W(W)) u_product (
                .clk(clk), .rst(rst), .in_valid(input_valid_q), .a(input_a), .b(input_b),
                .out_valid(product_valid), .product(product)
            );
        end else if (STAGE_PRODUCT == 4) begin : g_sp4
            _holoso_imuls_sp4#(.W(W)) u_product (
                .clk(clk), .rst(rst), .in_valid(input_valid_q), .a(input_a), .b(input_b),
                .out_valid(product_valid), .product(product)
            );
        end else begin : g_invalid_stage_product
            _holoso_invalid_imuls_stage_product u_invalid();
        end
    endgenerate

    wire overflow = |(product[WP-1:W] ^ {W{product[W-1]}});
    always @(posedge clk) begin
        input_a <= a;
        input_b <= b;
        y <= overflow ? (product[WP-1] ? MIN : MAX) : product[W-1:0];
        saturated <= overflow;
        if (rst) begin
            input_valid_q <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            input_valid_q <= in_valid;
            out_valid <= product_valid;
        end
    end
endmodule

module _holoso_imuls_sp0#(parameter W = 44) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire signed [W-1:0] a,
    input  wire signed [W-1:0] b,
    output wire out_valid,
    output wire signed [2*W-1:0] product
);
    assign out_valid = in_valid;
    assign product = a * b;
endmodule

module _holoso_imuls_sp1#(parameter W = 44) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire signed [W-1:0] a,
    input  wire signed [W-1:0] b,
    output reg out_valid,
    output reg signed [2*W-1:0] product
);
    always @(posedge clk) begin
        product <= a * b;
        if (rst) out_valid <= 1'b0;
        else     out_valid <= in_valid;
    end
endmodule

module _holoso_imuls_sp2#(parameter W = 44) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire signed [W-1:0] a,
    input  wire signed [W-1:0] b,
    output reg out_valid,
    output reg signed [2*W-1:0] product
);
    reg signed [W-1:0] a_q;
    reg signed [W-1:0] b_q;
    reg input_valid_q;
    always @(posedge clk) begin
        a_q <= a;
        b_q <= b;
        product <= a_q * b_q;
        if (rst) begin
            input_valid_q <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            input_valid_q <= in_valid;
            out_valid <= input_valid_q;
        end
    end
endmodule

module _holoso_imuls_sp3#(parameter W = 44) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire signed [W-1:0] a,
    input  wire signed [W-1:0] b,
    output reg out_valid,
    output reg signed [2*W-1:0] product
);
    localparam integer WP = 2 * W;
    localparam integer SW = (W + 1) / 2;
    localparam integer EW = 2 * SW;
    localparam integer WS = SW + 1;
    localparam integer WSP = 2 * WS;

    reg signed [W-1:0] a_q;
    reg signed [W-1:0] b_q;
    reg input_valid_q;
    wire signed [EW-1:0] a_ext = {{(EW-W){a_q[W-1]}}, a_q};
    wire signed [EW-1:0] b_ext = {{(EW-W){b_q[W-1]}}, b_q};
    wire signed [WS-1:0] a_slice [0:1];
    wire signed [WS-1:0] b_slice [0:1];
    assign a_slice[0] = $signed({1'b0, a_ext[0 +: SW]});
    assign a_slice[1] = $signed(a_ext[SW +: SW]);
    assign b_slice[0] = $signed({1'b0, b_ext[0 +: SW]});
    assign b_slice[1] = $signed(b_ext[SW +: SW]);

    wire signed [WSP-1:0] partial [0:3];
    assign partial[0] = a_slice[0] * b_slice[0];
    assign partial[1] = a_slice[0] * b_slice[1];
    assign partial[2] = a_slice[1] * b_slice[0];
    assign partial[3] = a_slice[1] * b_slice[1];
    reg signed [WSP-1:0] partial_q [0:3];
    reg partial_valid_q;
    wire signed [WP-1:0] term [0:3];
    assign term[0] = partial_q[0];
    assign term[1] = $signed(partial_q[1]) <<< SW;
    assign term[2] = $signed(partial_q[2]) <<< SW;
    assign term[3] = $signed(partial_q[3]) <<< (2 * SW);

    integer i;
    always @(posedge clk) begin
        a_q <= a;
        b_q <= b;
        for (i = 0; i < 4; i = i + 1) partial_q[i] <= partial[i];
        product <= term[0] + term[1] + term[2] + term[3];
        if (rst) begin
            input_valid_q <= 1'b0;
            partial_valid_q <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            input_valid_q <= in_valid;
            partial_valid_q <= input_valid_q;
            out_valid <= partial_valid_q;
        end
    end
endmodule

module _holoso_imuls_sp4#(parameter W = 44) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire signed [W-1:0] a,
    input  wire signed [W-1:0] b,
    output reg out_valid,
    output reg signed [2*W-1:0] product
);
    localparam integer WP = 2 * W;
    localparam integer SW = (W + 2) / 3;
    localparam integer EW = 3 * SW;
    localparam integer WS = SW + 1;
    localparam integer WSP = 2 * WS;

    reg signed [W-1:0] a_q;
    reg signed [W-1:0] b_q;
    reg input_valid_q;
    wire signed [EW-1:0] a_ext = {{(EW-W){a_q[W-1]}}, a_q};
    wire signed [EW-1:0] b_ext = {{(EW-W){b_q[W-1]}}, b_q};
    wire signed [WS-1:0] a_slice [0:2];
    wire signed [WS-1:0] b_slice [0:2];
    assign a_slice[0] = $signed({1'b0, a_ext[0 +: SW]});
    assign a_slice[1] = $signed({1'b0, a_ext[SW +: SW]});
    assign a_slice[2] = $signed(a_ext[2*SW +: SW]);
    assign b_slice[0] = $signed({1'b0, b_ext[0 +: SW]});
    assign b_slice[1] = $signed({1'b0, b_ext[SW +: SW]});
    assign b_slice[2] = $signed(b_ext[2*SW +: SW]);

    wire signed [WSP-1:0] partial [0:8];
    genvar ai, bi;
    generate
        for (ai = 0; ai < 3; ai = ai + 1) begin : g_partial_row
            for (bi = 0; bi < 3; bi = bi + 1) begin : g_partial_column
                assign partial[ai*3 + bi] = a_slice[ai] * b_slice[bi];
            end
        end
    endgenerate
    reg signed [WSP-1:0] partial_q [0:8];
    reg partial_valid_q;
    wire signed [WP-1:0] row_term [0:8];
    genvar ri, rj;
    generate
        for (ri = 0; ri < 3; ri = ri + 1) begin : g_row_term_row
            for (rj = 0; rj < 3; rj = rj + 1) begin : g_row_term_column
                assign row_term[ri*3 + rj] = $signed(partial_q[ri*3 + rj]) <<< (rj * SW);
            end
        end
    endgenerate
    wire signed [WP-1:0] row_sum [0:2];
    assign row_sum[0] = row_term[0] + row_term[1] + row_term[2];
    assign row_sum[1] = row_term[3] + row_term[4] + row_term[5];
    assign row_sum[2] = row_term[6] + row_term[7] + row_term[8];
    reg signed [WP-1:0] row_sum_q [0:2];
    reg row_valid_q;
    wire signed [WP-1:0] column [0:2];
    assign column[0] = row_sum_q[0];
    assign column[1] = row_sum_q[1] <<< SW;
    assign column[2] = row_sum_q[2] <<< (2 * SW);
    wire signed [WP-1:0] sum_xor = column[0] ^ column[1] ^ column[2];
    wire signed [WP-1:0] sum_carry = ((column[0] & column[1]) | (column[0] & column[2]) | (column[1] & column[2])) << 1;
    wire signed [WP-1:0] sum = sum_xor + sum_carry;

    integer i;
    always @(posedge clk) begin
        a_q <= a;
        b_q <= b;
        for (i = 0; i < 9; i = i + 1) partial_q[i] <= partial[i];
        for (i = 0; i < 3; i = i + 1) row_sum_q[i] <= row_sum[i];
        product <= sum;
        if (rst) begin
            input_valid_q <= 1'b0;
            partial_valid_q <= 1'b0;
            row_valid_q <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            input_valid_q <= in_valid;
            partial_valid_q <= input_valid_q;
            row_valid_q <= partial_valid_q;
            out_valid <= row_valid_q;
        end
    end
endmodule

// Signed saturating division with Python floor or truncation-toward-zero quotient semantics.
// Division by zero returns MIN for a negative numerator and MAX otherwise, preserves the numerator as remainder,
// and asserts outputs div0 and saturated.
// LATENCY = 2 + ceil(W/2)
module holoso_idivs #(parameter W = 44, parameter integer QUOTIENT_FLOOR = 1, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire signed [W-1:0] num,
    input  wire signed [W-1:0] den,
    output reg out_valid,
    output reg signed [W-1:0] quo,
    output reg signed [W-1:0] rem,
    output reg saturated,
    output reg div0
);
    localparam integer NSTEPS = (W + 1) / 2;
    localparam integer WPAD = 2 * NSTEPS;
    localparam integer WDIV = W + WPAD;
    localparam integer LATENCY_REF = 2 + NSTEPS;
    localparam signed [W-1:0] MIN = {1'b1, {(W-1){1'b0}}};
    localparam signed [W-1:0] MAX = {1'b0, {(W-1){1'b1}}};

    generate
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _holoso_invalid_integer_latency u_invalid();
        end
    endgenerate

    wire [W-1:0] num_magnitude = num[W-1] ? -num : num;
    wire [W-1:0] den_magnitude = den[W-1] ? -den : den;
    wire [W+1:0] den_magnitude3 = {1'b0, den_magnitude, 1'b0} + {2'b00, den_magnitude};
    wire input_div0 = den == {W{1'b0}};
    wire input_overflow = (num == MIN) && (den == {W{1'b1}});

    reg [NSTEPS:0] valid_q;
    reg [WDIV-1:0] work_q [0:NSTEPS];
    reg [W-1:0] den_q [0:NSTEPS];
    reg [W+1:0] den3_q [0:NSTEPS];
    reg num_negative_q [0:NSTEPS];
    reg den_negative_q [0:NSTEPS];
    reg div0_q [0:NSTEPS];
    reg overflow_q [0:NSTEPS];
    wire [WDIV-1:0] step_work [1:NSTEPS];

    genvar i_stage;
    generate
        for (i_stage = 1; i_stage <= NSTEPS; i_stage = i_stage + 1) begin : g_stage
            wire [W-1:0] remainder_next;
            wire [1:0] digit;
            wire [WPAD-1:0] quotient_work_next = (work_q[i_stage-1][WPAD-1:0] << 2) | digit;
            assign step_work[i_stage] = {remainder_next, quotient_work_next};
            _holoso_idiv_radix4_step #(.W(W)) u_step (
                .den(den_q[i_stage-1]),
                .den3(den3_q[i_stage-1]),
                .partial_rem(work_q[i_stage-1][WDIV-1:WPAD]),
                .next_bits(work_q[i_stage-1][WPAD-1 -: 2]),
                .rem_next(remainder_next),
                .digit(digit)
            );
        end
    endgenerate

    wire [W-1:0] quotient_magnitude = work_q[NSTEPS][W-1:0];
    wire [W-1:0] remainder_magnitude = work_q[NSTEPS][WDIV-1:WPAD];
    wire signs_differ = num_negative_q[NSTEPS] ^ den_negative_q[NSTEPS];
    wire remainder_nonzero = |remainder_magnitude;
    wire signed [W-1:0] corrected_quo;
    wire signed [W-1:0] corrected_rem;

    generate
        if (QUOTIENT_FLOOR == 1) begin : g_floor
            reg [W-1:0] floor_quo;
            wire [W-1:0] negative_quo = -quotient_magnitude;
            wire [W-1:0] unequal_rem = den_negative_q[NSTEPS] ?
                (remainder_magnitude - den_q[NSTEPS]) : (den_q[NSTEPS] - remainder_magnitude);
            wire [W-1:0] unequal_rem_or_zero = remainder_nonzero ? unequal_rem : {W{1'b0}};
            wire [W-1:0] equal_rem = den_negative_q[NSTEPS] ? -remainder_magnitude : remainder_magnitude;
            always @* begin
                case ({signs_differ, remainder_nonzero})
                    2'b10: floor_quo = negative_quo;
                    2'b11: floor_quo = ~quotient_magnitude;
                    default: floor_quo = quotient_magnitude;
                endcase
            end
            assign corrected_quo = floor_quo;
            assign corrected_rem = signs_differ ? unequal_rem_or_zero : equal_rem;
        end else if (QUOTIENT_FLOOR == 0) begin : g_truncate
            assign corrected_quo = signs_differ ? -quotient_magnitude : quotient_magnitude;
            assign corrected_rem = num_negative_q[NSTEPS] ? -remainder_magnitude : remainder_magnitude;
        end else begin : g_invalid_quotient_mode
            _holoso_invalid_idivs_quotient_mode u_invalid();
            assign corrected_quo = {W{1'bx}};
            assign corrected_rem = {W{1'bx}};
        end
    endgenerate

    integer i;
    always @(posedge clk) begin
        work_q[0] <= {{WPAD{1'b0}}, num_magnitude};
        den_q[0] <= den_magnitude;
        den3_q[0] <= den_magnitude3;
        num_negative_q[0] <= num[W-1];
        den_negative_q[0] <= den[W-1];
        div0_q[0] <= input_div0;
        overflow_q[0] <= input_overflow;
        for (i = 1; i <= NSTEPS; i = i + 1) begin
            work_q[i] <= step_work[i];
            den_q[i] <= den_q[i-1];
            den3_q[i] <= den3_q[i-1];
            num_negative_q[i] <= num_negative_q[i-1];
            den_negative_q[i] <= den_negative_q[i-1];
            div0_q[i] <= div0_q[i-1];
            overflow_q[i] <= overflow_q[i-1];
        end
        if (div0_q[NSTEPS]) begin
            quo <= num_negative_q[NSTEPS] ? MIN : MAX;
            rem <= num_negative_q[NSTEPS] ? -remainder_magnitude : remainder_magnitude;
        end else if (overflow_q[NSTEPS]) begin
            quo <= MAX;
            rem <= {W{1'b0}};
        end else begin
            quo <= corrected_quo;
            rem <= corrected_rem;
        end
        saturated <= div0_q[NSTEPS] | overflow_q[NSTEPS];
        div0 <= div0_q[NSTEPS];
        if (rst) begin
            valid_q <= {(NSTEPS+1){1'b0}};
            out_valid <= 1'b0;
        end else begin
            valid_q <= {valid_q[NSTEPS-1:0], in_valid};
            out_valid <= valid_q[NSTEPS];
        end
    end
endmodule

module _holoso_idiv_radix4_step #(parameter W = 44) (
    input  wire [W-1:0] den,
    input  wire [W+1:0] den3,
    input  wire [W-1:0] partial_rem,
    input  wire [1:0] next_bits,
    output reg [W-1:0] rem_next,
    output reg [1:0] digit
);
    localparam integer WCANDIDATE = W + 2;
    localparam integer WDIFF = WCANDIDATE + 1;

    wire [WCANDIDATE-1:0] den1 = {2'b00, den};
    wire [WCANDIDATE-1:0] den2 = {1'b0, den, 1'b0};
    wire [WCANDIDATE-1:0] candidate = {partial_rem, next_bits};
    wire [WDIFF-1:0] diff1 = {1'b0, candidate} - {1'b0, den1};
    wire [WDIFF-1:0] diff2 = {1'b0, candidate} - {1'b0, den2};
    wire [WDIFF-1:0] diff3 = {1'b0, candidate} - {1'b0, den3};
    wire ge1 = !diff1[WCANDIDATE];
    wire ge2 = !diff2[WCANDIDATE];
    wire ge3 = !diff3[WCANDIDATE];

    always @* begin
        casez ({ge3, ge2, ge1})
            3'b1??: begin
                rem_next = diff3[W-1:0];
                digit = 2'd3;
            end
            3'b01?: begin
                rem_next = diff2[W-1:0];
                digit = 2'd2;
            end
            3'b001: begin
                rem_next = diff1[W-1:0];
                digit = 2'd1;
            end
            default: begin
                rem_next = candidate[W-1:0];
                digit = 2'd0;
            end
        endcase
    end
endmodule

// Signed integer absolute value with saturation: the edge case -(2**(W-1)) is mapped to (2**(W-1)-1) with saturated=1.
module holoso_iabss#(parameter W = 44, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire signed [W-1:0] x,
    output reg out_valid,
    output reg signed [W-1:0] y,
    output reg saturated
);
    localparam integer LATENCY_REF = 2;
    localparam signed [W-1:0] MIN = {1'b1, {(W-1){1'b0}}};
    localparam signed [W-1:0] MAX = {1'b0, {(W-1){1'b1}}};
    generate
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _holoso_invalid_integer_latency u_invalid();
        end
    endgenerate

    reg signed [W-1:0] x_q;
    reg input_valid_q;
    wire signed [W-1:0] neg = -x_q;
    wire clamp = x_q == MIN;

    always @(posedge clk) begin
        x_q <= x;
        y <= x_q[W-1] ? (clamp ? MAX : neg) : x_q;
        saturated <= clamp;
        if (rst) begin
            input_valid_q <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            input_valid_q <= in_valid;
            out_valid <= input_valid_q;
        end
    end
endmodule

// Signed integer barrel shifter: shift left if shamt>0, shift right if shamt<0.
// For constant shamt use ordinary inline combinational shift expression instead.
module holoso_ashift#(parameter W = 44, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire signed [W-1:0] x,
    input  wire signed [W-1:0] shamt,
    output reg out_valid,
    output reg signed [W-1:0] y
);
    localparam integer LATENCY_REF = 2;
    localparam integer SW = $clog2(W);
    localparam integer PW = $clog2(SW);
    localparam [SW:0] W_AMOUNT = W;
    generate
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _holoso_invalid_integer_latency u_invalid();
        end
    endgenerate

    reg signed [W-1:0] x_q;
    reg signed [W-1:0] shamt_q;
    reg input_valid_q;
    wire [SW-1:0] shamt_narrow = shamt_q[SW-1:0];
    wire [SW-1:0] right_prefix [0:PW];
    assign right_prefix[0] = shamt_narrow;
    generate
        genvar i;
        for (i = 0; i < PW; i = i + 1) begin : g_right_prefix
            assign right_prefix[i+1] = right_prefix[i] | (right_prefix[i] << (1 << i));
        end
    endgenerate
    wire [SW-1:0] right_amount = ~shamt_narrow ^ ~(right_prefix[PW] << 1);
    wire [SW:0] left_amount_ext = {1'b0, shamt_narrow};
    wire [SW:0] right_amount_ext = {1'b0, right_amount};
    wire left_large = (|shamt_q[W-1:SW]) | (left_amount_ext >= W_AMOUNT);
    wire right_large = (~&shamt_q[W-1:SW]) | (~|shamt_narrow) | (right_amount_ext >= W_AMOUNT);
    wire signed [W-1:0] shifted_left = x_q << shamt_narrow;
    wire signed [W-1:0] shifted_right = x_q >>> right_amount;

    always @(posedge clk) begin
        x_q <= x;
        shamt_q <= shamt;
        casez ({shamt_q[W-1], right_large, left_large})
            3'b0?1: y <= {W{1'b0}};
            3'b0?0: y <= shifted_left;
            3'b11?: y <= {W{x_q[W-1]}};
            3'b10?: y <= shifted_right;
            default: y <= {W{1'bx}};
        endcase
        if (rst) begin
            input_valid_q <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            input_valid_q <= in_valid;
            out_valid <= input_valid_q;
        end
    end
endmodule

// Signed integer comparator.
module holoso_icmp#(parameter W = 44, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire signed [W-1:0] a,
    input  wire signed [W-1:0] b,
    output reg out_valid,
    output reg a_gt_b,
    output reg a_eq_b,
    output reg a_lt_b
);
    localparam integer LATENCY_REF = 2;
    generate
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _holoso_invalid_integer_latency u_invalid();
        end
    endgenerate

    reg signed [W-1:0] a_q;
    reg signed [W-1:0] b_q;
    reg input_valid_q;
    wire signed [W:0] diff = $signed({a_q[W-1], a_q}) - $signed({b_q[W-1], b_q});
    wire less = diff[W];
    wire equal = a_q == b_q;

    always @(posedge clk) begin
        a_q <= a;
        b_q <= b;
        a_gt_b <= ~less & ~equal;
        a_eq_b <= equal;
        a_lt_b <= less;
        if (rst) begin
            input_valid_q <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            input_valid_q <= in_valid;
            out_valid <= input_valid_q;
        end
    end
endmodule

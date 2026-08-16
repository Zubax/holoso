// SIGNED INTEGER OPERATORS
//
// Every operator has a mandatory input and output latches, exposing no combinational circuits outside.
//
//  Module          | Operation                             |Latency| Inputs    | Outputs
//  ----------------|---------------------------------------|-------|-----------|---------------------------
//  holoso_iadds    | Signed addition, saturated            | 2     | a, b      | y, saturated
//  holoso_isubs    | Signed subtraction, saturated         | 2     | a, b      | y, saturated
//  holoso_imuls    | Signed multiplication, saturated      | 2..6  | a, b      | y, saturated
//  holoso_idivs    | Signed division and modulo, saturated | 3+W/2 | num, den  | quo, rem, saturated, div0
//  holoso_iabss    | Absolute value, saturated             | 2     | x         | y, saturated
//  holoso_ishl     | Arith. shift, left+/right-            | 2     | x, shamt  | shft, prod, saturated
//  holoso_ishr     | Arith. shift, right+/left-            | 2     | x, shamt  | shft
//  holoso_icmp     | Signed comparison                     | 2     | a, b      | a_gt_b, a_eq_b, a_lt_b
//  holoso_ipopcnt  | Population count of the magnitude     | 2     | x         | y

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
// LATENCY = 3 + ceil(W/2)
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
    localparam integer LATENCY_REF = 3 + NSTEPS;
    localparam signed [W-1:0] MIN = {1'b1, {(W-1){1'b0}}};
    localparam signed [W-1:0] MAX = {1'b0, {(W-1){1'b1}}};

    generate
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _holoso_invalid_integer_latency u_invalid();
        end
    endgenerate

    reg signed [W-1:0] input_num;
    reg signed [W-1:0] input_den;
    wire [W-1:0] num_magnitude = input_num[W-1] ? -input_num : input_num;
    wire [W-1:0] den_magnitude = input_den[W-1] ? -input_den : input_den;
    wire [W+1:0] den_magnitude3 = {1'b0, den_magnitude, 1'b0} + {2'b00, den_magnitude};
    wire input_div0 = input_den == {W{1'b0}};
    wire input_overflow = (input_num == MIN) && (input_den == {W{1'b1}});

    reg [NSTEPS+1:0] valid_q;
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
        input_num <= num;
        input_den <= den;
        work_q[0] <= {{WPAD{1'b0}}, num_magnitude};
        den_q[0] <= den_magnitude;
        den3_q[0] <= den_magnitude3;
        num_negative_q[0] <= input_num[W-1];
        den_negative_q[0] <= input_den[W-1];
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
            valid_q <= {(NSTEPS+2){1'b0}};
            out_valid <= 1'b0;
        end else begin
            valid_q <= {valid_q[NSTEPS:0], in_valid};
            out_valid <= valid_q[NSTEPS+1];
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
// A left shift can overflow, and both readings of that event are emitted at once: `shft` is the raw bit shift that
// lets the high bits fall off the word, while `prod` is the saturating multiplication by a power of two that clamps
// to the representable range and reports the clamp on `saturated`.
// Right shifts cannot overflow, so there the two results agree and the flag stays low.
module holoso_ishl#(parameter W = 44, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire signed [W-1:0] x,
    input  wire signed [W-1:0] shamt,
    output reg out_valid,
    output reg signed [W-1:0] shft,
    output reg signed [W-1:0] prod,
    output reg saturated
);
    localparam integer LATENCY_REF = 2;
    localparam integer SW = $clog2(W);
    localparam integer PW = $clog2(SW);
    localparam integer GROUP_BITS = 2;
    localparam integer GROUP = 1 << GROUP_BITS;
    // Sized by the index range rather than by the data, so no shift amount can walk the group select off the end;
    // the groups past the magnitude are constant zero and fold away.
    localparam integer NGROUPS = ((1 << SW) + GROUP - 1) / GROUP;
    localparam [SW:0] W_AMOUNT = W;
    localparam signed [W-1:0] MIN = {1'b1, {(W-1){1'b0}}};
    localparam signed [W-1:0] MAX = {1'b0, {(W-1){1'b1}}};
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
    genvar i;
    generate
        for (i = 0; i < PW; i = i + 1) begin : g_right_prefix
            assign right_prefix[i+1] = right_prefix[i] | (right_prefix[i] << (1 << i));
        end
    endgenerate
    wire [SW-1:0] right_amount = shamt_narrow ^ (right_prefix[PW] << 1);
    wire [SW:0] left_amount_ext = {1'b0, shamt_narrow};
    // The left flag keeps its explicit test because `left_overflow` below reads it to decide the saturation flag,
    // where a count past the word overflows for every operand but zero -- which the magnitude cannot see. The right
    // flag only steers the mux, so it drops the test it used to carry, `| ({1'b0, right_amount} >= W_AMOUNT)`: a
    // count filling the whole word vacates every bit position, and IEEE 1364-2005 section 5.1.12 fills each one
    // with the sign for `>>>` on a signed result, which is the word the fill arm below would have written anyway.
    wire left_large = (|shamt_q[W-1:SW]) | (left_amount_ext >= W_AMOUNT);
    wire right_large = (~&shamt_q[W-1:SW]) | (~|shamt_narrow);
    wire signed [W-1:0] shifted_left = x_q << shamt_narrow;
    wire signed [W-1:0] shifted_right = x_q >>> right_amount;

    // A left shift by s is exact iff the top s+1 bits of x already equal its sign, so overflow is the question of
    // whether any of the s bits the shift pushes out differs from the sign. Reversing the magnitude turns "the top s
    // bits" into "the low s bits", and splitting s into a group index and an offset inside the group lets the two
    // halves of the test resolve in parallel: whole groups below the boundary are reduced without consulting s at
    // all, and only the straddled group needs a bit-level mask. Masking the whole word instead, which reads more
    // directly, stacks a decode and a word-wide reduction behind the shift amount and fails the timing closure.
    // A shift past the word is exact only for zero, which the magnitude alone cannot tell from -1, hence the branch.
    wire [W-2:0] magnitude = x_q[W-2:0] ^ {(W-1){x_q[W-1]}};
    wire [NGROUPS*GROUP-1:0] lost_order;
    wire [NGROUPS-1:0] group_any;
    generate
        for (i = 0; i < NGROUPS * GROUP; i = i + 1) begin : g_lost_order
            if (i < W - 1) begin : g_bit
                assign lost_order[i] = magnitude[W-2-i];
            end else begin : g_pad
                assign lost_order[i] = 1'b0;
            end
        end
        for (i = 0; i < NGROUPS; i = i + 1) begin : g_group_any
            assign group_any[i] = |lost_order[i*GROUP +: GROUP];
        end
    endgenerate
    wire [SW-1:0] group_index = shamt_narrow >> GROUP_BITS;
    wire [GROUP_BITS-1:0] group_offset = shamt_narrow;  // truncation is the intent; a slice would not fit W = 2
    wire [NGROUPS-1:0] whole_groups = ~({NGROUPS{1'b1}} << group_index);
    wire [GROUP-1:0] straddled_group = lost_order[group_index*GROUP +: GROUP];
    wire [GROUP-1:0] straddled_mask = ~({GROUP{1'b1}} << group_offset);
    wire left_overflow = left_large ? (|x_q) : (|(group_any & whole_groups) | |(straddled_group & straddled_mask));
    wire clamp = ~shamt_q[W-1] & left_overflow;

    // Zero fill and sign fill are the same uniform word once the direction is known, so folding the two range flags
    // into one select leaves every `prod` bit a six-input function of three selects, the operand sign and the two
    // shifter bits: one slice rank on a 4-LUT fabric.
    wire unshifted = shamt_q[W-1] ? right_large : left_large;
    wire fill = shamt_q[W-1] & x_q[W-1];

    always @(posedge clk) begin
        x_q <= x;
        shamt_q <= shamt;
        casez ({unshifted, shamt_q[W-1]})
            2'b1?: shft <= {W{fill}};
            2'b00: shft <= shifted_left;
            2'b01: shft <= shifted_right;
        endcase
        casez ({clamp, unshifted, shamt_q[W-1]})
            3'b1??: prod <= x_q[W-1] ? MIN : MAX;
            3'b01?: prod <= {W{fill}};
            3'b000: prod <= shifted_left;
            3'b001: prod <= shifted_right;
        endcase
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

// Signed integer barrel shifter, the mirror of holoso_ishl: shift right if shamt>0, shift left if shamt<0.
// Neither direction can rail here -- a right shift cannot overflow and the left one is the raw bit shift.
module holoso_ishr#(parameter W = 44, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire signed [W-1:0] x,
    input  wire signed [W-1:0] shamt,
    output reg out_valid,
    output reg signed [W-1:0] shft
);
    localparam integer LATENCY_REF = 2;
    localparam integer SW = $clog2(W);
    localparam integer PW = $clog2(SW);
    generate
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _holoso_invalid_integer_latency u_invalid();
        end
    endgenerate

    reg signed [W-1:0] x_q;
    reg signed [W-1:0] shamt_q;
    reg input_valid_q;
    wire [SW-1:0] shamt_narrow = shamt_q[SW-1:0];
    wire [SW-1:0] left_prefix [0:PW];
    assign left_prefix[0] = shamt_narrow;
    genvar i;
    generate
        for (i = 0; i < PW; i = i + 1) begin : g_left_prefix
            assign left_prefix[i+1] = left_prefix[i] | (left_prefix[i] << (1 << i));
        end
    endgenerate
    wire [SW-1:0] left_amount = shamt_narrow ^ (left_prefix[PW] << 1);
    // Neither flag reaches an overflow test here, so both drop the count-past-the-word test they used to carry,
    // `| ({1'b0, shamt_narrow} >= W_AMOUNT)` and `| ({1'b0, left_amount} >= W_AMOUNT)`: such a count vacates every
    // bit position, and IEEE 1364-2005 section 5.1.12 fills each one with the sign for `>>>` on a signed result and
    // with zero for `<<`, which is exactly the word the fill arm below would have written.
    wire right_large = |shamt_q[W-1:SW];
    wire left_large = (~&shamt_q[W-1:SW]) | (~|shamt_narrow);
    wire signed [W-1:0] shifted_right = x_q >>> shamt_narrow;
    wire signed [W-1:0] shifted_left = x_q << left_amount;

    wire unshifted = shamt_q[W-1] ? left_large : right_large;
    wire fill = ~shamt_q[W-1] & x_q[W-1];

    always @(posedge clk) begin
        x_q <= x;
        shamt_q <= shamt;
        casez ({unshifted, shamt_q[W-1]})
            2'b1?: shft <= {W{fill}};
            2'b00: shft <= shifted_right;
            2'b01: shft <= shifted_left;
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

// Population count of the magnitude, as Python's int.bit_count(), so a negative operand counts the ones of -x.
// The negation that overflows a signed word is exactly the magnitude 2**(W-1) read unsigned, so unlike holoso_iabss
// nothing saturates here and the count never reaches W. WY rides the parameters rather than the port range so that
// a caller disagreeing about the width fails to elaborate instead of truncating.
//
// Negating first would put a W-bit carry chain in series with the reduction and misses timing by a wide margin, so
// the sign is folded into the operand instead: -x = ~x + 1 agrees with ~x everywhere above the lowest set bit of x,
// and below it the increment clears a run of tz(x) ones and sets one, hence popcount(-x) = popcount(~x) + 1 - tz(x).
// Counting x ^ {W{sign}} costs nothing (the sign is just another item), the +1 rides in as a constant item, and the
// trailing-zero count comes from its own shallow priority tree that runs beside the reduction.
//
// The reduction is written as explicit counters and carry-save levels rather than as an addition tree because a
// synthesizer flattens a chain of additions into one carry-save tree that consumes every operand in its first
// compression level. That schedule ignores arrival times and would stack the whole reduction underneath the
// trailing-zero cone. Here the cone lands beside a carry-save pair instead, and only the closing addition sees it.
// A level counts up to seven values per column into three weighted results, and three values collapse to a
// carry-save pair, so the value count runs 7 -> 3 -> 2 at common widths: one logic level each.
// None of that is paid for in fabric, which is the counterintuitive part: measured against the addition tree it
// replaces, this network holds the flip-flop count exactly and spends fewer 4-LUTs at common widths,
// because explicit counters pack into 4-LUTs where a generic multiply-accumulate expansion does not.
// A synthesizer that retimes will still grow its own registers here, but that is the speed being taken,
// not the structure costing it.
module holoso_ipopcnt#(parameter W = 44, parameter integer WY = 6, parameter integer LATENCY = 0) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire signed [W-1:0] x,
    output reg out_valid,
    output reg [WY-1:0] y
);
    localparam integer LATENCY_REF = 2;
    localparam integer WY_REF = $clog2(W);
    // Items to count: the W-1 magnitude bits below the sign (the topmost one is zero after folding), the sign, and
    // the constant that completes the two's complement of the inverted trailing-zero count.
    localparam integer NI = W + 1;
    localparam integer NL = (NI + 6) / 7;   // seven items per counter is the widest cut a LUT holds in one level
    localparam integer NT = (W + 3) / 4;    // trailing-zero nibbles
    localparam integer TL = ($clog2(NT) + 1) / 2;
    localparam integer NTP = 1 << (2 * TL);  // nibbles padded to a power of four: the priority tree is four-way
    generate
        if ((LATENCY != 0) && (LATENCY != LATENCY_REF)) begin : g_invalid_latency
            _holoso_invalid_integer_latency u_invalid();
        end
        if (WY != WY_REF) begin : g_invalid_result_width
            _holoso_invalid_ipopcnt_result_width u_invalid();
        end
    endgenerate

    function integer reduced;  // values leaving a compression level that receives the given number of values
        input integer n;
        begin
            if (n > 3) reduced = 3 * ((n + 6) / 7);  // seven-input column counters, three weighted results each
            else if (n == 3) reduced = 2;            // a carry-save adder
            else reduced = n;
        end
    endfunction

    function integer values_at;  // values entering the given compression level
        input integer level;
        integer n, i;
        begin
            n = NL;
            for (i = 0; i < level; i = i + 1) n = reduced(n);
            values_at = n;
        end
    endfunction

    function integer value_base;  // flat index of the first value of the given compression level
        input integer level;
        integer b, i;
        begin
            b = 0;
            for (i = 0; i < level; i = i + 1) b = b + values_at(i);
            value_base = b;
        end
    endfunction

    function integer depth;  // compression levels needed to reach a carry-save pair
        input integer start;
        integer n;
        begin
            n = start;
            depth = 0;
            while (n > 2) begin
                n = reduced(n);
                depth = depth + 1;
            end
        end
    endfunction

    function integer tz_base;  // flat index of the first node of the given priority-tree level, zero being nibbles
        input integer level;
        begin
            tz_base = (4 * NTP - 4 * (NTP >> (2 * level))) / 3;
        end
    endfunction

    function [2:0] count7;  // seven-input counter, written as gates so it stays one logic level rather than a chain
        input [6:0] v;
        reg s0, k0, s1, k1, s2, k2;
        begin
            s0 = v[0] ^ v[1] ^ v[2];
            k0 = (v[0] & v[1]) | (v[0] & v[2]) | (v[1] & v[2]);
            s1 = v[3] ^ v[4] ^ v[5];
            k1 = (v[3] & v[4]) | (v[3] & v[5]) | (v[4] & v[5]);
            s2 = s0 ^ s1 ^ v[6];
            k2 = (s0 & s1) | (s0 & v[6]) | (s1 & v[6]);
            count7 = {(k0 & k1) | ((k0 ^ k1) & k2), k0 ^ k1 ^ k2, s2};
        end
    endfunction

    function [WY-1:0] nibble_tz;  // trailing zeros of a nonzero nibble; a zero nibble is resolved one level up
        input [3:0] v;
        begin
            nibble_tz = v[0] ? 0 : v[1] ? 1 : v[2] ? 2 : 3;
        end
    endfunction

    localparam integer RL = depth(NL);
    localparam integer NV = value_base(RL) + values_at(RL);
    localparam integer NTV = tz_base(TL) + 1;

    reg signed [W-1:0] x_q;
    reg input_valid_q;
    wire sign = x_q[W-1];

    // Seven constant zeros rather than a (7*NL-NI)-wide replication, which vanishes wherever seven divides the item
    // count. The surplus slots sit inside the group the last counter already reads.
    wire [7*NL+6:0] items = {7'b0, 1'b1, sign, x_q[W-2:0] ^ {(W-1){sign}}};
    // Four constant zeros rather than a (NTP*4-W)-wide replication, which vanishes wherever four divides the width.
    wire [W+3:0] nibbles = {4'b0, x_q};

    wire [WY-1:0] value [0:NV-1];
    wire [WY-1:0] tz_val [0:NTV-1];
    wire tz_zero [0:NTV-1];
    wire [WY-1:0] pair_hi;

    genvar i, l, p, c, b, r, j;
    generate
        for (i = 0; i < NL; i = i + 1) begin : g_leaf
            assign value[i] = count7(items[7*i +: 7]);
        end
        for (l = 0; l < RL; l = l + 1) begin : g_level
            if (values_at(l) > 3) begin : g_count
                for (c = 0; c < (values_at(l) + 6) / 7; c = c + 1) begin : g_chunk
                    wire [2:0] cnt [0:WY-1];
                    for (b = 0; b < WY; b = b + 1) begin : g_column
                        wire [6:0] slice;
                        for (j = 0; j < 7; j = j + 1) begin : g_slice
                            if (7 * c + j < values_at(l)) begin : g_take
                                assign slice[j] = value[value_base(l) + 7 * c + j][b];
                            end else begin : g_pad
                                assign slice[j] = 1'b0;
                            end
                        end
                        assign cnt[b] = count7(slice);
                    end
                    for (r = 0; r < 3; r = r + 1) begin : g_weight
                        for (b = 0; b < WY; b = b + 1) begin : g_bit
                            if (b >= r) begin : g_shifted
                                assign value[value_base(l+1) + 3 * c + r][b] = cnt[b-r][r];
                            end else begin : g_below
                                assign value[value_base(l+1) + 3 * c + r][b] = 1'b0;
                            end
                        end
                    end
                end
            end else begin : g_carry_save
                assign value[value_base(l+1)] =
                    value[value_base(l)] ^ value[value_base(l)+1] ^ value[value_base(l)+2];
                assign value[value_base(l+1)+1] = ((value[value_base(l)] & value[value_base(l)+1])
                    | (value[value_base(l)] & value[value_base(l)+2])
                    | (value[value_base(l)+1] & value[value_base(l)+2])) << 1;
            end
        end
        if (values_at(RL) > 1) begin : g_pair
            assign pair_hi = value[value_base(RL)+1];
        end else begin : g_single
            assign pair_hi = {WY{1'b0}};
        end

        for (i = 0; i < NTP; i = i + 1) begin : g_tz_leaf
            if (i < NT) begin : g_nibble
                assign tz_zero[i] = ~|nibbles[4*i +: 4];
                assign tz_val[i] = nibble_tz(nibbles[4*i +: 4]);
            end else begin : g_pad
                assign tz_zero[i] = 1'b1;
                assign tz_val[i] = {WY{1'b0}};
            end
        end
        for (l = 1; l <= TL; l = l + 1) begin : g_tz_level
            for (p = 0; p < (NTP >> (2 * l)); p = p + 1) begin : g_tz_node
                localparam integer ME = tz_base(l) + p;
                localparam integer CH = tz_base(l-1) + 4 * p;
                localparam integer WT = 1 << (2 * l);  // bits spanned by one child, so a digit is a set bit
                assign tz_zero[ME] = tz_zero[CH] & tz_zero[CH+1] & tz_zero[CH+2] & tz_zero[CH+3];
                assign tz_val[ME] =
                    !tz_zero[CH]   ? tz_val[CH] :
                    !tz_zero[CH+1] ? (tz_val[CH+1] | WT) :
                    !tz_zero[CH+2] ? (tz_val[CH+2] | (2 * WT)) : (tz_val[CH+3] | (3 * WT));
            end
        end
    endgenerate

    always @(posedge clk) begin
        x_q <= x;
        y <= value[value_base(RL)] + pair_hi + ~(tz_val[NTV-1] & {WY{sign}});
        if (rst) begin
            input_valid_q <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            input_valid_q <= in_valid;
            out_valid <= input_valid_q;
        end
    end
endmodule

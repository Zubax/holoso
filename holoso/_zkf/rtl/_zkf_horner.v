/// Fixed-point Horner polynomial evaluator for the transcendental table+polynomial cores.
/// Register stages: D*(2+STAGE_PRODUCT).
/// Zero-bubble, throughput-1. A generic sideband (sb_in -> sb_out), the valid flag, and the reduced argument
/// (w -> w_out) are pipelined alongside the accumulator so the instantiating module need not know D.
///
/// Computes acc = c[D]; then for j = D-1 .. 0: acc = c[j] + floor(acc * w / 2^WRARG), i.e. Horner in the segment-local
/// argument wn = w / 2^WRARG in [0,1). Coefficients share the fractional scale 2^-CF: c[j] is a signed WCOEF-bit
/// value at bit offset j*WCOEF of the flat `coeffs` bus. The arithmetic right shift `>>> WRARG` floors toward minus
/// infinity, matching the Python reference model's truncating integer Horner exactly.
///
/// Each degree step is one shared _zkf_pmul multiply (acc*w, w always unsigned -- latency 1+STAGE_PRODUCT) plus
/// one coefficient-add register stage, so the per-degree depth is 2+STAGE_PRODUCT. The non-arithmetic payload
/// (live coefficients, w, caller sideband) is delayed in a plain pipe next to the multiplier instead of riding through
/// _zkf_pmul's internal sideband registers. This is because the payload is wide and passing it through the pmul
/// sideband hurts placement and timings.
///
/// ACC_SIGNED selects the accumulator multiply signedness forwarded to _zkf_pmul.A_SIGNED. log2 (default, =1) keeps a
/// signed accumulator because its Chebyshev coefficients alternate sign, so intermediate `acc` can go negative. exp2
/// (=0) has an all-non-negative coefficient set and the recurrence acc = c[j] + (acc*w)>>WRARG preserves
/// non-negativity for every w, so acc stays >= 0; the fully-unsigned slice grid then packs whole WMULTIPLIER-bit DSP
/// tiles (no sign bit), roughly a third fewer tiles, with bit-identical results (the product is non-negative either
/// way, so its raw bits and the floor shift below are unchanged).
///
/// WMULTIPLIER and STAGE_PRODUCT are forwarded to _zkf_pmul.
///
/// WACC must hold every intermediate `acc` without wrap (the generator sizes it from the actual coefficient set).
/// Reset clears only the valid pipeline; the datapath registers free-run (project reset strategy).

`default_nettype none

module _zkf_horner #(
    parameter integer D             = 2,  // polynomial degree (D+1 coefficients), >= 1
    parameter integer WCOEF         = 32, // signed coefficient width
    parameter integer WRARG         = 8,  // reduced-argument width; wn = w / 2^WRARG
    parameter integer WACC          = 40, // signed accumulator width (>= every intermediate, sized by the generator)
    parameter integer WSB           = 1,  // sideband width carried alongside the pipeline
    parameter integer ACC_SIGNED    = 1,  // accumulator multiply signedness (A_SIGNED to _zkf_pmul); 0 only for exp2
    parameter integer WMULTIPLIER   = 0,  // forwarded to _zkf_pmul
    parameter integer STAGE_PRODUCT = 0   // forwarded to _zkf_pmul
) (
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    in_valid,
    input  wire        [WSB-1:0]   sb_in,
    // coeffs and acc are fixed-point carriers: every coefficient is a small signed value padded to WCOEF (sign +
    // margin), and acc holds 2**f in [1,2) (exp2) or P(t) in [1,1/ln2] (log2) at scale 2**CF.
    input  wire [(D+1)*WCOEF-1:0]  coeffs,   // c[j] (signed) at bits [j*WCOEF +: WCOEF], j = 0..D
    input  wire        [WRARG-1:0] w,        // reduced argument, unsigned, in [0, 2^WRARG)
    output wire                    out_valid,
    output wire        [WSB-1:0]   sb_out,
    output wire        [WRARG-1:0] w_out,
    output wire signed [WACC-1:0]  acc       // Horner result, signed, scale 2^-CF
);
    generate
        if (STAGE_PRODUCT < 0) begin : g_invalid_stage_product
            _zkf_invalid_stage_product_out_of_range u_invalid();
        end
    endgenerate

    // a_*[s] is the state entering degree step s (s = 0..D).
    wire signed [WACC-1:0]    a_acc [0:D];
    wire [(D+1)*WCOEF-1:0]    a_co  [0:D];
    wire [WRARG-1:0]          a_w   [0:D];
    wire                      a_val [0:D];
    wire [WSB-1:0]            a_sb  [0:D];

    assign a_acc[0] = $signed(coeffs[D*WCOEF +: WCOEF]);  // acc starts at the top coefficient c[D]
    assign a_co[0]  = coeffs;
    assign a_w[0]   = w;
    assign a_val[0] = in_valid;
    assign a_sb[0]  = sb_in;

    genvar s;
    generate
        for (s = 0; s < D; s = s + 1) begin : g_step
            localparam integer J     = D - 1 - s;          // coefficient index resolved at this step
            // Coefficients are consumed in strictly descending index order (J = D-1 .. 0), so entering step s only
            // c[0..J] are still live: carry just the low (J+1)*WCOEF bits, MSB-aligned to keep the `[J*WCOEF +: WCOEF]`
            // read at the top of the slice. c[D] already seeded a_acc[0] and is never carried. a_co[s+1] zero-extends
            // the narrow forward into the uniform wiring array, and the next step slices off what it needs.
            localparam integer COW   = (J + 1) * WCOEF;
            localparam integer WSB_H = COW + WRARG + WSB;  // delayed payload: {live coeffs, w, module sideband}

            // Multiply stage: acc*w via the shared pipelined multiplier (acc per ACC_SIGNED, w unsigned). The live
            // coefficient bus, w (forwarded unchanged to the next degree), and the module sideband are delayed in a
            // separate pipe so the multiplier does not carry a wide non-arithmetic sideband through its DSP-adjacent
            // registers.
            wire                  prod_v;
            wire                  prod_sb_valid;
            wire [WSB_H-1:0]      prod_sb;
            wire [WACC+WRARG-1:0] prod_p;  // raw two's-complement acc*w (signedness assigned by the caller below)
            _zkf_pmul #(
                .WA(WACC), .WB(WRARG), .A_SIGNED(ACC_SIGNED), .B_SIGNED(0),
                .WSB(1), .WMULTIPLIER(WMULTIPLIER), .STAGE_PRODUCT(STAGE_PRODUCT)
            ) u_pmul (
                .clk(clk), .rst(rst), .in_valid(a_val[s]), .sb_in(1'b0),
                .a(a_acc[s]), .b(a_w[s]),
                .out_valid(prod_v), .sb_out(), .p(prod_p)
            );
            zkf_pipe #(.W(WSB_H), .N(1 + STAGE_PRODUCT)) u_payload_delay (
                .clk(clk), .rst(rst), .in_valid(a_val[s]), .in({a_co[s][COW-1:0], a_w[s], a_sb[s]}),
                .out_valid(prod_sb_valid), .out(prod_sb)
            );
            wire _unused_prod_sb_valid = &{1'b0, prod_sb_valid, 1'b0};
            wire [COW-1:0]   p_co = prod_sb[WSB_H-1 -: COW];
            wire [WRARG-1:0] p_w  = prod_sb[WSB+WRARG-1 -: WRARG];
            wire [WSB-1:0]   p_sb = prod_sb[WSB-1:0];

            // Coefficient-add stage (the surviving +1 register): acc = c[J] + floor(acc*w / 2^WRARG). The arithmetic
            // right shift floors toward minus infinity, matching the truncating-Horner reference exactly. With
            // ACC_SIGNED=0 (exp2) prod_p is a non-negative unsigned product whose top bits are structurally 0, so the
            // arithmetic `>>>` behaves identically to a logical shift -- keep it as-is; do not specialize on ACC_SIGNED.
            wire signed [WACC-1:0] next_acc = $signed(p_co[J*WCOEF +: WCOEF]) + $signed($signed(prod_p) >>> WRARG);
            reg signed [WACC-1:0] r_acc;
            reg [COW-1:0]         r_co;
            reg [WRARG-1:0]       r_w;
            reg                   r_val;
            reg [WSB-1:0]         r_sb;
            always @(posedge clk) begin
                if (rst) r_val <= 1'b0;
                else     r_val <= prod_v;
                r_acc <= next_acc;
                r_co  <= p_co;
                r_w   <= p_w;
                r_sb  <= p_sb;
            end
            assign a_acc[s + 1] = r_acc;
            assign a_co[s + 1]  = r_co;
            assign a_w[s + 1]   = r_w;
            assign a_val[s + 1] = r_val;
            assign a_sb[s + 1]  = r_sb;
        end
    endgenerate

    assign acc       = a_acc[D];
    assign out_valid = a_val[D];
    assign sb_out    = a_sb[D];
    assign w_out     = a_w[D];
endmodule

`default_nettype wire

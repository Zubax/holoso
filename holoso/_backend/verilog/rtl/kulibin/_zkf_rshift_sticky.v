/// Sticky-folded right-shift barrel: y = x >> shamt, with y[0] OR-collecting every bit dropped by the shift plus
/// the bit that ends up at position 0. Output y[W-1:1] is the plain shifted value.
/// Used by the adder, to int converter, etc.
///
/// Saturation: when shamt >= W, the cascade naturally produces zero magnitude with sticky = |x.
/// Callers may either rely on that and clamp shamt to W (zkf_to_int does this), or pass a WSHIFT wider than $clog2(W)
/// bits and let the module's own saturation check kick in for over-range values (zkf_add does this).
///
/// Implementation: radix-4 cascade. Each stage selects one of four shifts (0, 4^i, 2*4^i, 3*4^i) with a 4:1 mux;
/// slices commonly pack a 4:1 mux, so the cascade depth is half of the equivalent radix-2 version.
///
/// STAGE_SPLIT=0: Pure combinational, single-cycle cascade (no clk used).
///
/// STAGE_SPLIT=1: Insert one register stage in the middle of the radix-4 cascade. y is delayed by 1 cycle and
/// the consumer must add a matching cycle to the surrounding pipeline.

`default_nettype none

module _zkf_rshift_sticky #(
    parameter W           = 16,
    parameter WSHIFT      = $clog2(W) + 1,
    parameter STAGE_SPLIT = 0
) (
    input  wire              clk,    // unused when STAGE_SPLIT == 0
    input  wire      [W-1:0] x,
    input  wire [WSHIFT-1:0] shamt,
    output wire      [W-1:0] y
);
    generate
        if ((STAGE_SPLIT != 0) && (STAGE_SPLIT != 1)) begin : g_invalid_stage_split
            _zkf_invalid_stage_split u_invalid();
        end
    endgenerate

    localparam WLOCAL = $clog2(W);
    localparam NSTAGE = (WLOCAL + 1) / 2;        // radix-4 stages
    localparam WPAIR  = NSTAGE * 2;              // shamt bits the cascade would consume
    // Register barrier sits after stage SPLIT_AFTER (only used when STAGE_SPLIT != 0). NSTAGE/2 is
    // a natural midpoint: for the typical NSTAGE=3 case it leaves stages 0..1 before and stage 2
    // after, balancing mux levels across the register.
    localparam SPLIT_AFTER = NSTAGE / 2;

    // Cascade state: data[i] and sticky[i] are the *input* to stage i. Each stage produces a pair of combinational
    // outputs data_pre[i+1]/sticky_pre[i+1], which then feed data[i+1]/sticky[i+1] either directly (combinational)
    // or through a register barrier (when STAGE_SPLIT != 0 and i == SPLIT_AFTER).
    wire [W-1:0] data       [0:NSTAGE];
    wire         sticky     [0:NSTAGE];
    wire [W-1:0] data_pre   [0:NSTAGE];   // index 0 unused
    wire         sticky_pre [0:NSTAGE];   // index 0 unused
    assign data[0]   = x;
    assign sticky[0] = 1'b0;

    // When STAGE_SPLIT != 0, the late half of the cascade fires one cycle after `shamt` was applied, so those stages
    // must read a registered copy. shamt_late is that copy (combinational alias of shamt when STAGE_SPLIT == 0).
    // Early stages always read the live shamt directly.
    wire [WSHIFT-1:0] shamt_late;
    generate
        if (STAGE_SPLIT == 0) begin : g_shamt_pass
            assign shamt_late = shamt;
        end else begin : g_shamt_register
            reg [WSHIFT-1:0] shamt_late_r;
            always @(posedge clk) shamt_late_r <= shamt;
            assign shamt_late = shamt_late_r;
        end
    endgenerate

    genvar i;
    generate
        for (i = 0; i < NSTAGE; i = i + 1) begin : g_stage
            localparam integer S1 = 1 << (2 * i);     // sel=1
            localparam integer S2 = 1 << (2 * i + 1); // sel=2
            localparam integer S3 = S1 + S2;          // sel=3

            // Pick the shamt source: early stages see the live input; late stages see the registered
            // copy so their data and shamt stay aligned across the cascade-internal register barrier.
            wire [WSHIFT-1:0] shamt_use;
            if ((STAGE_SPLIT != 0) && (i > SPLIT_AFTER)) begin : g_shamt_use_late
                assign shamt_use = shamt_late;
            end else begin : g_shamt_use_early
                assign shamt_use = shamt;
            end

            // Each stage's 2-bit selector comes from shamt_use[2*i +: 2] when both bits are in range, from
            // {0, shamt_use[2*i]} when only the low bit is in range, or constant 0 when the stage sits above
            // shamt's MSB. Slicing per-stage avoids an intermediate padded `shamt_ext` wire, which empirically
            // confuses Yosys's flatten+ABC pipeline on wide configurations.
            wire [1:0] sel;
            if (2*i + 1 < WSHIFT) begin : g_sel_full
                assign sel = shamt_use[2*i +: 2];
            end else if (2*i < WSHIFT) begin : g_sel_partial
                assign sel = {1'b0, shamt_use[2*i]};
            end else begin : g_sel_zero
                assign sel = 2'b00;
            end

            wire [W-1:0] d0 = data[i];
            wire [W-1:0] d1;
            wire [W-1:0] d2;
            wire [W-1:0] d3;
            wire         l1;
            wire         l2;
            wire         l3;

            if (S1 < W) begin : g_s1
                assign d1 = {{S1{1'b0}}, data[i][W-1:S1]};
                assign l1 = |data[i][S1-1:0];
            end else begin : g_s1_sat
                assign d1 = {W{1'b0}};
                assign l1 = |data[i];
            end
            if (S2 < W) begin : g_s2
                assign d2 = {{S2{1'b0}}, data[i][W-1:S2]};
                assign l2 = |data[i][S2-1:0];
            end else begin : g_s2_sat
                assign d2 = {W{1'b0}};
                assign l2 = |data[i];
            end
            if (S3 < W) begin : g_s3
                assign d3 = {{S3{1'b0}}, data[i][W-1:S3]};
                assign l3 = |data[i][S3-1:0];
            end else begin : g_s3_sat
                assign d3 = {W{1'b0}};
                assign l3 = |data[i];
            end

            // 4:1 mux. Ternary chain in this exact form maps to a mux-friendly pattern.
            assign data_pre[i+1] = (sel == 2'd0) ? d0
                                 : (sel == 2'd1) ? d1
                                 : (sel == 2'd2) ? d2
                                 :                 d3;
            assign sticky_pre[i+1] = sticky[i]
                                   | ((sel == 2'd1) & l1)
                                   | ((sel == 2'd2) & l2)
                                   | ((sel == 2'd3) & l3);
        end
    endgenerate

    // Wire data[i]/sticky[i] from the per-stage combinational outputs. When STAGE_SPLIT != 0 the
    // boundary at index SPLIT_AFTER+1 is registered, breaking the cascade into two clock periods.
    genvar j;
    generate
        for (j = 1; j <= NSTAGE; j = j + 1) begin : g_register
            if ((STAGE_SPLIT != 0) && (j == SPLIT_AFTER + 1)) begin : g_split_register
                reg [W-1:0] data_r;
                reg         sticky_r;
                always @(posedge clk) begin
                    data_r   <= data_pre[j];
                    sticky_r <= sticky_pre[j];
                end
                assign data[j]   = data_r;
                assign sticky[j] = sticky_r;
            end else begin : g_passthrough
                assign data[j]   = data_pre[j];
                assign sticky[j] = sticky_pre[j];
            end
        end
    endgenerate

    // Top-of-range saturation: if shamt has bits set above what the cascade consumes, collapse to {0, |x}. The check
    // and its |x companion must come from the same cycle as the data the cascade produced, so when STAGE_SPLIT != 0
    // we register both alongside the cascade's data/sticky barrier. With WSHIFT <= WPAIR the check resolves to a
    // constant 1'b0 at elaboration.
    wire shamt_ge_w_use;
    wire x_or_use;
    generate
        if (WSHIFT > WPAIR) begin : g_sat_check
            wire shamt_ge_w_now = |shamt[WSHIFT-1:WPAIR];
            wire x_or_now       = |x;
            if (STAGE_SPLIT == 0) begin : g_sat_pass
                assign shamt_ge_w_use = shamt_ge_w_now;
                assign x_or_use       = x_or_now;
            end else begin : g_sat_register
                reg shamt_ge_w_r;
                reg x_or_r;
                always @(posedge clk) begin
                    shamt_ge_w_r <= shamt_ge_w_now;
                    x_or_r       <= x_or_now;
                end
                assign shamt_ge_w_use = shamt_ge_w_r;
                assign x_or_use       = x_or_r;
            end
        end else begin : g_no_sat_check
            assign shamt_ge_w_use = 1'b0;
            assign x_or_use       = 1'b0;
        end
    endgenerate

    wire [W-1:0] final_data;
    wire         final_sticky;
    assign final_data   = shamt_ge_w_use ? {W{1'b0}} : data[NSTAGE];
    assign final_sticky = shamt_ge_w_use ? x_or_use  : sticky[NSTAGE];

    assign y[W-1:1] = final_data[W-1:1];
    assign y[0]     = final_data[0] | final_sticky;
endmodule

`default_nettype wire

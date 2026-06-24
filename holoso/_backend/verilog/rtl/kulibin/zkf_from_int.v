/// Streamed cast from signed two's-complement integer to Zubax Kulibin float.
///
/// STAGE_INPUT=0: input combinational paths are exposed.
/// STAGE_INPUT=1: inputs are latched, the external module sees registers at the input (+1 cycle).
///
/// STAGE_NORMALIZE=0/1/2: number of internal register stages inside normshift (forward to _zkf_normshift.STAGE_SPLIT).
///
/// STAGE_PACK=0: pack inputs are combinational (default).
/// STAGE_PACK=1: register pack inputs (forwarded to _zkf_pack.STAGE_INPUT) (+1 cycle).
///
/// STAGE_OUTPUT=0: outputs are combinational (default).
/// STAGE_OUTPUT=1: registered (+1 cycle).

`default_nettype none

`define ZKF_FROM_INT_LATENCY (1 + STAGE_INPUT + STAGE_NORMALIZE + STAGE_PACK + STAGE_OUTPUT)

module zkf_from_int #(
    parameter WEXP            = 6,
    parameter WMAN            = 18,
    parameter WINT            = 32,
    parameter STAGE_INPUT     = 0,
    parameter STAGE_NORMALIZE = 0,
    parameter STAGE_PACK      = 0,
    parameter STAGE_OUTPUT    = 0,
    parameter LATENCY         = `ZKF_FROM_INT_LATENCY   // must equal the register-stage count; checked below
) (
    input wire clk,
    input wire rst,

    input wire                   in_valid,
    input wire signed [WINT-1:0] a,

    output wire                  out_valid,
    output wire [WEXP+WMAN-1:0]  y
);
    generate
        if ((WEXP < 2) || (WMAN < 4) || (WINT < 2)) begin : g_invalid
            _zkf_invalid_wexp_or_wman u_invalid();
        end
        // Shift by WEXP >= 32 would overflow Verilog's integer constant arithmetic and yield tool-dependent values.
        if (WEXP >= 32) begin : g_invalid_wexp_too_wide
            _zkf_invalid_from_int_wexp_too_wide_unportable u_invalid();
        end
        if ((STAGE_INPUT != 0) && (STAGE_INPUT != 1)) begin : g_invalid_stage_input
            _zkf_invalid_stage_input u_invalid();
        end
        if (LATENCY != `ZKF_FROM_INT_LATENCY) begin : g_invalid_latency
            _zkf_invalid_latency_mismatch u_invalid();
        end
    endgenerate

    // Magnitude container width: must be at least WINT (to hold |a|, including |INT_MIN| = 2^(WINT-1))
    // and at least WMAN+3 so a static slice of [WX-WMAN-3:0] always provides at least one sticky bit.
    localparam WX    = (WINT > (WMAN + 3)) ? WINT : (WMAN + 3);
    // The biased exponent fed to _zkf_pack is the leading-one position plus BIAS, maxing at (WX-1)+BIAS. WEU must
    // hold that as a non-negative signed value so the packer reads it positive (and its overflow detector fires)
    // for the widest operands, and must also meet the packer's intrinsic minimum of WEXP+2 signed bits. Sizing
    // from the bare position (clog2(WX)+1) under-counts by BIAS and silently wraps wide-WINT operands to a
    // spurious negative (underflow).
    localparam EXP_BIASED_MAX = (WX - 1) + ((1 << (WEXP - 1)) - 1);
    localparam WEU_LOD        = $clog2(EXP_BIASED_MAX + 1) + 1;
    localparam WEU            = (WEU_LOD > (WEXP + 2)) ? WEU_LOD : (WEXP + 2);

    // Optional input register stage.
    wire             in_valid_q;
    wire [WINT-1:0]  a_q;
    zkf_pipe #(.W(WINT), .N(STAGE_INPUT ? 1 : 0)) u_input_pipe (
        .clk(clk), .rst(rst), .in_valid(in_valid), .in(a), .out_valid(in_valid_q), .out(a_q)
    );

    // Stage-1 cone: form |a| via XOR-and-increment so the carry chain handles the negation; this also handles
    // INT_MIN correctly because the resulting unsigned magnitude 2^(WINT-1) fits in WINT bits.
    wire            sign_in    = a_q[WINT-1];
    wire [WINT-1:0] inv_in     = a_q ^ {WINT{sign_in}};
    wire [WINT-1:0] mag_in     = inv_in + {{(WINT-1){1'b0}}, sign_in};
    wire [WX-1:0]   mag_ext_in = {{(WX-WINT){1'b0}}, mag_in};

    // Stage 1: register sign and magnitude. Reset only validity; payload free-runs.
    reg            s1_valid;
    reg            s1_sign;
    // WX-wide magnitude reg.
    reg [WX-1:0]   s1_mag_ext;

    always @(posedge clk) begin
        if (rst) begin
            s1_valid <= 1'b0;
        end else begin
            s1_valid <= in_valid_q;
        end
        s1_sign    <= sign_in;
        s1_mag_ext <= mag_ext_in;
    end

    // -- Normalize-and-pack. _zkf_fixed_to_float owns the _zkf_normshift instance, the combinational significand /
    // G / R / sticky extraction, the exp_unbiased = EXP_BIASED_TOP - shamt arithmetic (forwarded to _zkf_pack with
    // EXP_IS_BIASED=1 so the packer skips the bias add), and the _zkf_pack output stage. STAGE_NORMALIZE and
    // STAGE_PACK forward directly to the helper. WSB=1 is unused; sb_in is tied to 1'b0 and sb_out is discarded.
    localparam integer EXP_BIASED_TOP = EXP_BIASED_MAX;
    wire sb_out_unused;
    localparam [WEU-1:0] EXP_BIASED_TOP_EXT = EXP_BIASED_TOP[WEU-1:0];
    _zkf_fixed_to_float #(
        .WEXP(WEXP), .WMAN(WMAN),
        .WMAG(WX), .WEU(WEU),
        .EXP_IS_BIASED(1),
        .WSB(1),
        .STAGE_NORMALIZE(STAGE_NORMALIZE),
        .STAGE_PACK(STAGE_PACK),
        .STAGE_OUTPUT(STAGE_OUTPUT)
    ) u_fixed_to_float (
        .clk(clk), .rst(rst),
        .in_valid(s1_valid),
        .sign(s1_sign),
        .force_zero(1'b0),
        .force_inf(1'b0),
        .exp_offset(EXP_BIASED_TOP_EXT),
        .mag(s1_mag_ext),
        .sb_in(1'b0),
        .out_valid(out_valid),
        .y(y),
        .sb_out(sb_out_unused)
    );
endmodule

`undef ZKF_FROM_INT_LATENCY

`default_nettype wire

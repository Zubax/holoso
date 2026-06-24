/// Folded (iterative) CORDIC engine shared by the ZKF trigonometric operators. One (x, y, z) datapath is reused over
/// several cycles, a configurable number of iterations unrolled per cycle, instead of an N-stage pipeline -- so the
/// area is a single datapath at the cost of an initiation interval equal to the latency.
///
/// UNROLL100 is the latency knob (iterations per cycle x100); pick the largest that closes timings:
///     50 = one iteration per two cycles (split shift/add to halve the per-iteration combinational path, at 2N cycles);
///     100 = one iteration per cycle;
///     200/300/400 = 2/3/4 iterations per cycle (fewer cycles, longer path).
///
/// MODE selects the trajectory; the x/y/z update is otherwise identical:
///
///   MODE = 0 (ROTATION):  sigma_i = (z_i >= 0) ? +1 : -1   -- drives the angle z to 0; (x, y) rotates by z0.
///                         Used by zkf_sincos with (x0, y0) = (1/gain, 0) so (xn, yn) ~ (cos z0, sin z0) and
///                         zn is the small residual the wrapper finishes with one linear rotation.
///
///   MODE = 1 (VECTORING): sigma_i = (y_i >= 0) ? -1 : +1   -- drives y to 0; zn = z0 + atan2(y0, x0), xn ~ |(x0,y0)|.
///                         Used by zkf_atan2.
///
/// Each iteration: x' = x -/+ (y >>> i); y' = y +/- (x >>> i); z' = z -/+ L[i]. The shift `>>> i` truncates toward
/// -inf (matches the Python model's `>> i`). In the fold the shift amount i is the running iteration index, so it is a
/// variable (barrel) shift and L[i] is a variable index into the flat LUT bus -- unlike the pipelined CORDIC's
/// per-stage constant shifts. Each update is one controlled add/sub (a + (b ^ {W{sub}}) + sub -> one carry/adder chain).
///
/// Structure: a single x/y rotator (fast = U iters/cycle, or pipe = one iter / two cycles) that consumes a sigma
/// stream, plus -- only when PARALLEL is set in rotation mode -- a separate z-engine that produces that stream ahead of
/// time. The sigma sequence is identical either way; "coupled" (lock-step) and "decoupled" differ only in HOW sigma
/// reaches the rotator: an inline combinational tap off the in-step z-chain / y (coupled), or a registered read from
/// sig_mem fed by the ahead-running z-engine (decoupled). So the rotator is written once; the z handling is the only
/// thing that varies. Decoupling lets the sincos wrapper launch the residual-angle correction during the CORDIC.
///
/// PARALLEL only helps -- and is only legal -- with the half-rate (pipe) rotator: the z-recurrence is one narrow add,
/// so its fast rate is one iteration/cycle, which laps a half-rate x/y but merely ties a full-rate one. So a full-rate
/// rotator stays lock-step; the natural default is PARALLEL = (UNROLL100 < 100) that should not be changed except
/// for testing.
///
/// Handshake: assert `start` for one cycle with x0/y0/z0 valid; `busy` is high while iterating; `done` pulses for one
/// cycle with xn/yn/zn (and the registered sideband sb_out) valid. `start` is ignored while busy. Reset clears the FSM.

`default_nettype none

module _zkf_cordic #(
    parameter integer N           = 14,  // iterations
    parameter integer UNROLL100   = 100, // iters/cycle x100: 50=half-rate, 100/200/300/400=1/2/3/4 per cycle
    parameter integer PARALLEL    = (UNROLL100 < 100) ? 1 : 0,  // MODE=0: run the z-path ahead
    parameter integer WX          = 32,  // signed x/y width
    parameter integer WZ          = 32,  // signed angle width
    parameter integer MODE        = 0,   // 0 = rotation, 1 = vectoring
    parameter integer WSB         = 1    // sideband carried from start to done
) (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,
    input  wire       [WSB-1:0] sb_in,
    input  wire signed [WX-1:0] x0,
    input  wire signed [WX-1:0] y0,
    input  wire signed [WZ-1:0] z0,
    input  wire [(((N > 0) ? N : 1)*WZ)-1:0] lut,  // L[i] (unsigned) at bits [i*WZ +: WZ]
    output wire                 busy,
    output wire                 done,        // pulses with xn/yn (and sb_out) valid
    output wire                 z_done,      // MODE=0 decoupled: pulses with zn valid, ahead of `done` (else == done)
    output wire       [WSB-1:0] sb_out,
    output wire signed [WX-1:0] xn,
    output wire signed [WX-1:0] yn,
    output wire signed [WZ-1:0] zn
);
    localparam integer N_EFF = (N > 0) ? N : 1;  // Keeps invalid N structurally well-formed until validation fires.
    localparam integer U    = (UNROLL100 < 100) ? 1 : (UNROLL100 / 100);
    localparam integer PIPE = (UNROLL100 < 100) ? 1 : 0;
    localparam integer DECOUPLE = ((MODE == 0) && (PARALLEL != 0)) ? 1 : 0;
    localparam integer WI   = $clog2(N_EFF + U + 1);  // index width (holds i_r + U / zi_r + 1, no wrap)

    generate
        if ((UNROLL100 != 50) && ((UNROLL100 < 100) || ((UNROLL100 % 100) != 0))) begin : g_invalid_unroll100
            _zkf_invalid_unroll100 u_invalid();
        end
        if (N <= 0) begin : g_invalid_n
            _zkf_invalid_cordic_n_must_be_positive u_invalid();
        end
        if ((MODE != 0) && (MODE != 1)) begin : g_invalid_mode
            _zkf_invalid_cordic_mode u_invalid();
        end
        if (DECOUPLE && (PIPE == 0)) begin : g_invalid_decouple
            _zkf_decouple_needs_half_rate u_invalid();
        end
    endgenerate

    reg signed [WX-1:0] x_r;
    reg signed [WX-1:0] y_r;                   // y datapath (feeds yn, covered end-to-end)
    reg signed [WZ-1:0] z_r;
    reg        [WI-1:0] i_r;                   // base iteration index for this cycle
    reg                 run_r, done_r;
    reg       [WSB-1:0] sb_r;
    reg     [N_EFF-1:0] sig_mem; // sigma replay: written by the decoupled z-engine, read by the rotator.
    wire                z_done_int;

    // Unpack the flat L[] bus into an indexable array. Reading lut[idx*WZ +: WZ] with a runtime idx makes the synthesis
    // tool materialize the idx*WZ bit-offset multiply (when WZ is not a power of two, a DSP/soft multiplier ahead of the
    // angle mux -- the Diamond/LSE critical path on the wide vectoring engine, where it lands a multiplier + a huge
    // fanout net on the per-iteration L read). Slicing the bus at the COMPILE-TIME offsets i*WZ into lut_a[] and reading
    // lut_a[idx] instead leaves a plain N:1 word mux with no index arithmetic. Bit-identical to the flat-slice reads.
    // The array carries zero sentinel entries past index N-1: the fast rotator reads lut_a[i_r + u] for u in 0..U-1,
    // where en=0 makes any read past N-1 a don't-care/unused. i_r advances by U per active cycle, so after the final
    // group it can idle at ceil(N/U)*U (<= N+U-1 when N is not a multiple of U); the widest read is then i_r + (U-1) <=
    // N + 2U - 2. The decoupled prefetch reads lut_a[zi_nxt] up to N. Sizing to N + 2U - 1 keeps EVERY read (including
    // the post-run idle reads for U > 1) in bounds with a one-entry margin, without an index clamp on the LUT read.
    // (For U == 1 this is N + 1, i.e. the single original sentinel plus the prefetch slot -- unchanged behaviour.)
    localparam integer LUT_HI = N_EFF + 2*U - 1;
    wire [WZ-1:0] lut_a [0:LUT_HI];
    genvar gl;
    generate
        for (gl = 0; gl < N_EFF; gl = gl + 1) begin : g_lut_unpack
            assign lut_a[gl] = lut[gl*WZ +: WZ];   // constant (elaboration-time) offset -- no runtime multiply
        end
        for (gl = N_EFF; gl <= LUT_HI; gl = gl + 1) begin : g_lut_sentinel
            assign lut_a[gl] = {WZ{1'b0}};         // don't-care sentinels for terminal (en=0 / prefetch) indices
        end
    endgenerate

    generate
        // ============================================================================================================
        // Decoupled z-engine (MODE=0, PARALLEL): runs the sigma recurrence at full rate -- one z = z -/+ L[i] add per
        // cycle -- AHEAD of the half-rate rotator, buffering sigma into sig_mem and exposing the residual (z_r) +
        // z_done early. Absent (and z handled inline by the rotator) otherwise.
        // ============================================================================================================
        if (DECOUPLE) begin : g_sigma
            // z_r is the residual accumulator (zn = z_r); each cycle's sigma bit is written into sig_mem for the rotator
            // to replay. sig_mem[0] = sign(z0) is preloaded at start so the rotator reads sigma_0 on its first iteration
            // without a head-start. The z-index advances deterministically, so L[zi_r] is pre-fetched one cycle ahead
            // into li_r -- lifting the wide lut[] index mux out of the WZ-wide add cone, leaving only the controlled add
            // (and the MSB sign tap) on the recurrence's critical path.
            reg        [WI-1:0]  zi_r;              // z-iteration index for this cycle
            reg                  z_run_r, z_dn_r;
            reg        [WZ-1:0]  li_r;              // pre-fetched L[zi_r]
            wire                 zneg = z_r[WZ-1];  // true => sigma = -1
            wire                 zsub = ~zneg;      // z subtracts L[i] when sigma = +1
            wire signed [WZ-1:0] gnz  = z_r + ($signed({1'b0, li_r}) ^ {WZ{zsub}}) + {{(WZ-1){1'b0}}, zsub};
            wire        [WI-1:0] zi_nxt = zi_r + 1'b1;
            wire                 z_last = (zi_nxt >= N_EFF[WI-1:0]);
            always @(posedge clk) begin
                if (rst) begin
                    z_run_r <= 1'b0;
                    z_dn_r  <= 1'b0;
                end else begin
                    z_dn_r <= 1'b0;
                    if (!z_run_r) begin
                        if (start && !run_r) begin  // Premature start while busy breaks correctness.
                            z_r        <= z0;
                            zi_r       <= {WI{1'b0}};
                            sig_mem[0] <= z0[WZ-1];     // sigma_0 = sign(z0), read by the rotator at iteration 0
                            li_r       <= lut_a[0];     // pre-fetch L[0]
                            z_run_r    <= 1'b1;
                            z_dn_r     <= 1'b0;
                        end
                    end else begin
                        z_r           <= gnz;
                        zi_r          <= zi_nxt;
                        sig_mem[zi_r] <= zneg;                  // sigma_(zi_r) = sign(z_(zi_r))
                        li_r          <= lut_a[zi_nxt];         // pre-fetch next L (don't-care past N, then unused)
                        if (z_last) begin
                            z_run_r <= 1'b0;
                            z_dn_r  <= 1'b1;
                        end
                    end
                end
            end
            assign z_done_int = z_dn_r;
        end

        // ============================================================================================================
        // Rotator: the x/y datapath, shared by every mode. Per lane the sigma sign comes from sig_mem (decoupled) or,
        // in lock-step, the inline z-chain (rotation) / y (vectoring); the inline z-chain advances z_r in g_zadv. The
        // x/y add/sub, the FSM, and the handshake below the sigma source are identical across modes.
        // ============================================================================================================
        if (PIPE == 0) begin : g_fast
            // U iterations per cycle. Combinational chain from the registered state, starting at index i_r. Iterations
            // whose index reaches N pass through unchanged (the last cycle may be a partial group if N % U != 0).
            wire signed [WX-1:0] cx [0:U];
            wire signed [WX-1:0] cy [0:U];
            wire signed [WZ-1:0] cz [0:U];     // inline z-chain (the full-rate rotator is always lock-step)
            assign cx[0] = x_r;
            assign cy[0] = y_r;
            assign cz[0] = z_r;

            genvar u;
            for (u = 0; u < U; u = u + 1) begin : g_unroll
                wire [WI-1:0]        idx   = i_r + u[WI-1:0];
                wire                 en    = (idx < N_EFF[WI-1:0]);
                wire signed [WX-1:0] ysh   = cy[u] >>> idx;
                wire signed [WX-1:0] xsh   = cx[u] >>> idx;
                wire [WZ-1:0]        li    = lut_a[idx];
                wire                 neg   = (MODE == 0) ? cz[u][WZ-1] : ~cy[u][WX-1];  // true => sigma = -1
                wire                 sub_x = ~neg;      // x subtracts ysh when sigma = +1
                wire                 sub_y =  neg;      // y subtracts xsh when sigma = -1
                wire                 sub_z = ~neg;      // z subtracts li  when sigma = +1
                wire signed [WX-1:0] nx    = cx[u] + (ysh ^ {WX{sub_x}}) + {{(WX-1){1'b0}}, sub_x};
                wire signed [WX-1:0] ny    = cy[u] + (xsh ^ {WX{sub_y}}) + {{(WX-1){1'b0}}, sub_y};
                wire signed [WZ-1:0] nz    = cz[u] + ($signed({1'b0, li}) ^ {WZ{sub_z}}) + {{(WZ-1){1'b0}}, sub_z};
                assign cx[u+1] = en ? nx : cx[u];
                assign cy[u+1] = en ? ny : cy[u];
                assign cz[u+1] = en ? nz : cz[u];
            end

            // i_r advances unconditionally so latency stays constant data-independent.
            wire last = (i_r + U[WI-1:0]) >= N_EFF[WI-1:0];

            always @(posedge clk) begin
                if (rst) begin
                    run_r  <= 1'b0;
                    done_r <= 1'b0;
                end else begin
                    done_r <= 1'b0;
                    if (!run_r) begin
                        if (start) begin
                            x_r <= x0; y_r <= y0; z_r <= z0; i_r <= {WI{1'b0}};
                            sb_r <= sb_in;
                            run_r  <= 1'b1;
                            done_r <= 1'b0;
                        end
                    end else begin
                        x_r <= cx[U]; y_r <= cy[U]; z_r <= cz[U];
                        i_r <= i_r + U[WI-1:0];
                        if (last) begin
                            run_r  <= 1'b0;
                            done_r <= 1'b1;
                        end
                    end
                end
            end
            assign z_done_int = done_r;        // lock-step: zn lands coincident with done
        end else begin : g_pipe
            // One iteration per 2 cycles for wide datapaths: phase 0 registers the shifted operands and the sampled
            // sigma sign; phase 1 applies the add/sub. Splitting the long shift -> wide-add cone across a register
            // closes timing, at 2*N cycles.
            reg                 phase_r;           // 0 = shift/sample, 1 = add/advance
            reg signed [WX-1:0] xsh_r;             // x>>>i sampled in phase 0
            reg signed [WX-1:0] ysh_r;             // y>>>i sampled in phase 0
            reg                 neg_r;             // sigma sign sampled in phase 0 (true => sigma = -1)
            wire [WI-1:0]        idx = i_r;
            wire signed [WX-1:0] xsh = x_r >>> idx;
            wire signed [WX-1:0] ysh = y_r >>> idx;
            wire                 neg;
            wire                 sub_x = ~neg_r;
            wire                 sub_y =  neg_r;
            wire signed [WX-1:0] nx = x_r + (ysh_r ^ {WX{sub_x}}) + {{(WX-1){1'b0}}, sub_x};
            wire signed [WX-1:0] ny = y_r + (xsh_r ^ {WX{sub_y}}) + {{(WX-1){1'b0}}, sub_y};
            wire last = (i_r + 1'b1) >= N_EFF[WI-1:0];
            if (DECOUPLE) begin : g_sig
                assign neg = sig_mem[idx];                  // sigma replayed from the ahead-running z-engine
            end else begin : g_sig
                // MODE is elaboration-fixed; only one ternary arm is live per build (the coupled engine runs MODE==0
                // sincos; the MODE!=0 vectoring arm is never elaborated here).
                // verilator coverage_off
                assign neg = (MODE == 0) ? z_r[WZ-1] : ~y_r[WX-1];
                // verilator coverage_on
            end
            always @(posedge clk) begin
                if (rst) begin
                    run_r   <= 1'b0;
                    done_r  <= 1'b0;
                    phase_r <= 1'b0;
                end else begin
                    done_r <= 1'b0;
                    if (!run_r) begin
                        if (start) begin
                            x_r <= x0; y_r <= y0; i_r <= {WI{1'b0}};
                            sb_r <= sb_in; phase_r <= 1'b0;
                            run_r  <= 1'b1;
                            done_r <= 1'b0;
                        end
                    end else if (phase_r == 1'b0) begin
                        xsh_r <= xsh; ysh_r <= ysh; neg_r <= neg;
                        phase_r <= 1'b1;
                    end else begin
                        x_r <= nx; y_r <= ny;
                        i_r <= i_r + 1'b1;
                        phase_r <= 1'b0;
                        if (last) begin
                            run_r  <= 1'b0;
                            done_r <= 1'b1;
                        end
                    end
                end
            end

            if (!DECOUPLE) begin : g_zadv                  // lock-step z-path: sample L (phase 0), add to z_r (phase 1)
                reg        [WZ-1:0] li_r;
                wire       [WZ-1:0] li    = lut_a[idx];
                wire                sub_z = ~neg_r;
                wire signed [WZ-1:0] nz   = z_r + ($signed({1'b0, li_r}) ^ {WZ{sub_z}}) + {{(WZ-1){1'b0}}, sub_z};
                always @(posedge clk) begin
                    if (!run_r) begin
                        if (start) z_r <= z0;
                    end else if (phase_r == 1'b0) begin
                        li_r <= li;
                    end else begin
                        z_r <= nz;
                    end
                end
                assign z_done_int = done_r;
            end
        end
    endgenerate

    assign busy   = run_r;
    assign done   = done_r;
    assign z_done = z_done_int;
    assign xn     = x_r;
    assign yn     = y_r;
    assign zn     = z_r;
    assign sb_out = sb_r;
endmodule

`default_nettype wire

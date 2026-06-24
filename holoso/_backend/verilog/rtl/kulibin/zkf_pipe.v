/// A delay line of N register stages, each W bits wide. Latency is exactly N cycles; if N=0 (default) it is a no-op
/// passthrough with clk/rst unused/ignored. Reset clears only the valid flag (stream control); the W-bit payload
/// registers free-run per the project reset policy, so the payload is meaningful only while out_valid is asserted.
///
/// Public utility for aligning a consumer's own control/sideband signals with the output of a Kulibin float operator:
/// compute the operator's LATENCY locally, pass it to the operator (which checks it) and to a zkf_pipe #(.N(LATENCY))
/// that delays the sideband by the same number of cycles. This avoids threading sideband ports through the operators.

`default_nettype none

module zkf_pipe #(parameter W = 1, parameter N = 0) (
    input wire clk,
    input wire rst,

    input wire         in_valid,
    input wire [W-1:0] in,

    output wire         out_valid,
    output wire [W-1:0] out
);
    generate
        if (N) begin : g_registered
            reg [N-1:0] valid_pipe;
            reg [W-1:0] data_pipe [0:N-1];

            integer i;
            always @(posedge clk) begin
                // Reset only stream validity.
                if (rst) begin
                    valid_pipe <= {N{1'b0}};
                end else begin
                    valid_pipe[0] <= in_valid;
                    for (i = 1; i < N; i = i + 1) begin
                        valid_pipe[i] <= valid_pipe[i-1];
                    end
                end

                // Payload registers intentionally free-run so reset is not on the datapath.
                data_pipe[0] <= in;
                for (i = 1; i < N; i = i + 1) begin
                    data_pipe[i] <= data_pipe[i-1];
                end
            end
            assign out_valid = valid_pipe[N-1];
            assign out       = data_pipe[N-1];

        end else begin : g_passthrough
            assign out_valid = in_valid;
            assign out       = in;
        end
    endgenerate
endmodule

`default_nettype wire

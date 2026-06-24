/// Combinational absolute value: zero the sign bit.

`default_nettype none

module zkf_abs #(parameter WEXP = 6, parameter WMAN = 18) (
    input  wire [WEXP+WMAN-1:0] x,
    output wire [WEXP+WMAN-1:0] y
);
    assign y = {1'b0, x[WEXP+WMAN-2:0]};
endmodule

`default_nettype wire

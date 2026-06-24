/// Combinational predicate: y = 1 iff x is finite (exponent field is not all-ones).

`default_nettype none

module zkf_is_finite #(parameter WEXP = 6, parameter WMAN = 18) (input  wire [WEXP+WMAN-1:0] x, output wire y);
    localparam WFRAC = WMAN - 1;
    localparam WFULL = WEXP + WMAN;
    assign y = ~&x[WFULL-2:WFRAC];
endmodule

`default_nettype wire

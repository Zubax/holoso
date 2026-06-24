/// Combinational negation: flip the sign bit.
/// Applied to canonical +0 this produces a -0 bit pattern which the format still decodes as +0;
/// sequential modules downstream canonicalize their outputs.

`default_nettype none

module zkf_neg #(parameter WEXP = 6, parameter WMAN = 18) (
    input  wire [WEXP+WMAN-1:0] x,
    output wire [WEXP+WMAN-1:0] y
);
    assign y = {~x[WEXP+WMAN-1], x[WEXP+WMAN-2:0]};
endmodule

`default_nettype wire

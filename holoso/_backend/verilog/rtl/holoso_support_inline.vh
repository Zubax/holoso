// BEGIN holoso_support_inline.vh: the file is spliced into each generated module.

// Combinational mapping from float to boolean: a zero or a subnormal (if supported) float is false, otherwise true.
// E.g., if IEEE 754 binary32 is used (with subnormals), values with magnitude under ~1e-38 are mapped to falsity.
function holoso_ftobool;
    input [W-1:0] x;
    holoso_ftobool = |x[W-2:WMAN-1];
endfunction

// Combinational mapping from boolean to float: falsity is zero, truth is one.
function [W-1:0] holoso_ffrombool;
    input b;
    holoso_ffrombool = b ? {2'b00, {(WEXP - 1) {1'b1}}, {(WMAN - 1) {1'b0}}} : {W{1'b0}};
endfunction

// Combinational predicate: y=1 iff x is finite (i.e., x is not an infinity).
function holoso_fisfinite;
    input [W-1:0] x;
    holoso_fisfinite = ~&x[W-2:WMAN-1];
endfunction

// Combinational saturator: replaces infinity with the largest finite value of the same sign; finite pass through.
function [W-1:0] holoso_fsaturate;
    input [W-1:0] x;
    holoso_fsaturate = (&x[W-2:WMAN-1]) ? {x[W-1], {(WEXP - 1) {1'b1}}, 1'b0, {(WMAN - 1) {1'b1}}} : x;
endfunction

// END of holoso_support_inline.vh

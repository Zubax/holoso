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

function holoso_fisposinf;
    input [W-1:0] x;
    holoso_fisposinf = ~holoso_fisfinite(x) & ~x[W-1];
endfunction

function holoso_fisneginf;
    input [W-1:0] x;
    holoso_fisneginf = ~holoso_fisfinite(x) & x[W-1];
endfunction

// Combinational saturator: replaces infinity with the largest finite value of the same sign; finite pass through.
function [W-1:0] holoso_fsaturate;
    input [W-1:0] x;
    holoso_fsaturate = (&x[W-2:WMAN-1]) ? {x[W-1], {(WEXP - 1) {1'b1}}, 1'b0, {(WMAN - 1) {1'b1}}} : x;
endfunction

// Combinational floating-point sign conditioner (absolute first, then optional negate): op[0]=negate, op[1]=absolute.
//      op=0: +x        op=1: -x        op=2: +|x|      op=3: -|x|
function [W-1:0] holoso_fsgnop;
    input [W-1:0] x;
    input [1:0]   op;
    holoso_fsgnop = {(x[W-1] & ~op[1]) ^ op[0], x[W-2:0]};
endfunction

// END of holoso_support_inline.vh

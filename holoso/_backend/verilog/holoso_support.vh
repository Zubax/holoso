// This is a support header to go along holoso_support.v. Refer there for details.

`ifndef HOLOSO_SUPPORT_VH
`define HOLOSO_SUPPORT_VH

// Sign operator for holoso_fsgnop.
`define HOLOSO_FSGNOP_NONE      0   //     +x
`define HOLOSO_FSGNOP_NEG       1   //     -x
`define HOLOSO_FSGNOP_ABS       2   // +abs(x)
`define HOLOSO_FSGNOP_ABS_NEG   3   // -abs(x)

`endif

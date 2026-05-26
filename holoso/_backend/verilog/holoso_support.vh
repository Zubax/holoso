// This is a support header to go along holoso_support.v. Refer there for details.

`ifndef HOLOSO_REGFILE_VH
`define HOLOSO_REGFILE_VH

// Lane selector for flattened holoso_regfile buses.
// PORT is zero-based. WIDTH is the width of one lane on the selected flattened bus.
// Use it inside an indexed part-select:
//
//     rd_addr[`HOLOSO_REGFILE_LANE(WADDR, 0)]
//     rd_data[`HOLOSO_REGFILE_LANE(W,     0)]
//     wr_addr[`HOLOSO_REGFILE_LANE(WADDR, 1)]
//     wr_data[`HOLOSO_REGFILE_LANE(W,     1)]
//     wr_en[1]
//     load_data[`HOLOSO_REGFILE_LANE(W,  0)]   // lane i drives register i (only for i < NLOAD)
//     view[`HOLOSO_REGFILE_LANE(W,       0)]   // lane i mirrors register i (all NREG registers)
`define HOLOSO_REGFILE_LANE(WIDTH, PORT) ((PORT) * (WIDTH)) +: (WIDTH)


// Sign operator for holoso_fsgnop.
`define HOLOSO_FSGNOP_NONE      0   //     +x
`define HOLOSO_FSGNOP_NEG       1   //     -x
`define HOLOSO_FSGNOP_ABS       2   // +abs(x)
`define HOLOSO_FSGNOP_ABS_NEG   3   // -abs(x)


`endif

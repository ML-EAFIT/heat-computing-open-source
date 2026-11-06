// Tang Nano 20K board shell.
// The 20K's S1/S2 keys are active high (the 9K's are active low), so they are
// inverted here; rtl/top.v, the generated case/sparse/pmode wrappers and every
// testbench stay unchanged.
//   default            : wraps rtl/top.v, N / HIST_BITS passed through (-chparam)
//   -DINNER=case_top   : wraps a generated wrapper (case_top, sparse_top, pmode_wrap)
`default_nettype none
module tn20k_top #(
    parameter N         = 16,
    parameter HIST_BITS = 12
) (
    input  wire       clk,
    input  wire       btn_rst,
    input  wire       btn_run,
    input  wire       uart_rxd,
    output wire       uart_txd,
    output wire [5:0] led
);
`ifdef INNER
    `INNER u_inner (
`else
    top #(.N(N), .HIST_BITS(HIST_BITS)) u_inner (
`endif
        .clk       (clk),
        .btn_rst_n (~btn_rst),
        .btn_run_n (~btn_run),
        .uart_rxd  (uart_rxd),
        .uart_txd  (uart_txd),
        .led       (led)
    );
endmodule
`default_nettype wire

// -----------------------------------------------------------------------------
// File   : tb_top_script.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-30
// -----------------------------------------------------------------------------
// Description:
//   Generic end-to-end test of top.v through its UART only (8 clocks/bit).
//   Plays <SCRIPT_DIR>/script.hex, written by model/uart_script.Script:
//     00_0000xx  send byte xx
//     01_nnnnnn  wait until n more reply bytes have arrived
//     ff_000000  end
//   The top.v parameters come from <SCRIPT_DIR>/script.vh (compile with
//   -I <SCRIPT_DIR>). Every received byte goes to +dump=<file>; the caller
//   compares it with <SCRIPT_DIR>/expected.hex (model/uart_script.TopModel).
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-30  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module tb_top_script;

parameter SCRIPT_DIR = "build/script/";
parameter VEC        = "../model/vectors/";
localparam CPB = 8;

`include "script.vh"

reg        clk = 1'b0;
reg        btn_rst_n = 1'b1;
reg        btn_run_n = 1'b1;
reg        rxd = 1'b1;
wire       txd;
wire [5:0] led;
wire [7:0] rx_byte;
wire       rx_vld;

reg [31:0] script [0:SCRIPT_LEN-1];
integer    pc, fd, nrx, nexp, nsent;
reg [1023:0] dump_file;

always #5 clk = ~clk;

top #(
    .N(TOP_N), .IW(TOP_IW), .WFRAC(TOP_WFRAC), .AW(TOP_AW), .HIST_BITS(TOP_HIST_BITS), .USE_PWL(0),
    .CLK_HZ(8000000), .BAUD(1000000), .DEB_BITS(4), .DEMO_SWEEPS(16), .SEED(TOP_SEED),
    .W_FILE(TOP_W_FILE), .H_FILE(TOP_H_FILE), .O_FILE(TOP_O_FILE), .T_FILE(TOP_T_FILE),
    .LUT_FILE({VEC, "sigmoid_lut.hex"}),
    .DEF_MODE(TOP_DEF_MODE), .DEF_BETA0(TOP_DEF_BETA0), .DEF_BETA1(TOP_DEF_BETA1),
    .DEF_DELTA(TOP_DEF_DELTA), .DEF_SPS(TOP_DEF_SPS),
    .SPARSE(TOP_SPARSE), .P(TOP_P), .D(TOP_D), .DW(TOP_DW), .LW(TOP_LW), .ROUNDS(TOP_ROUNDS),
    .NODE_PREFIX(""), .NBR_PREFIX(""), .UP_RAM(TOP_UP_RAM),
    .NOISE_CTL(TOP_NOISE_CTL), .RO_JITTER(0), .TEMPER_K(TOP_TEMPER_K)
) dut (
    .clk(clk), .btn_rst_n(btn_rst_n), .btn_run_n(btn_run_n),
    .uart_rxd(rxd), .uart_txd(txd), .led(led)
);

// host-side receiver
uart_rx #(.CLKS_PER_BIT(CPB)) host_rx (.clk(clk), .rst(1'b0), .rxd(txd), .out_data(rx_byte), .out_valid(rx_vld));

always @(posedge clk) begin
    if (rx_vld) begin
        $fdisplay(fd, "%02x", rx_byte);
        nrx = nrx + 1;
    end
end

task send;
    input [7:0] b;
    integer k;
    begin
        rxd = 1'b0;
        repeat (CPB) @(posedge clk);
        for (k = 0; k < 8; k = k + 1) begin
            rxd = b[k];
            repeat (CPB) @(posedge clk);
        end
        rxd = 1'b1;
        repeat (CPB + 2) @(posedge clk);
    end
endtask

initial begin
    $readmemh({SCRIPT_DIR, "script.hex"}, script);
    if (!$value$plusargs("dump=%s", dump_file)) dump_file = {SCRIPT_DIR, "dump.hex"};
    fd    = $fopen(dump_file, "w");
    nrx   = 0;
    nexp  = 0;
    nsent = 0;
    repeat (40) @(posedge clk);
    pc = 0;
    while (script[pc][31:24] != 8'hFF) begin
        if (script[pc][31:24] == 8'h00) begin
            send(script[pc][7:0]);
            nsent = nsent + 1;
        end else begin
            nexp = nexp + script[pc][23:0];
            while (nrx < nexp) @(posedge clk);
        end
        pc = pc + 1;
    end
    repeat (50) @(posedge clk);
    $fclose(fd);
    $display("tb_top_script[%0s]: sent %0d bytes, received %0d bytes", SCRIPT_DIR, nsent, nrx);
    $finish;
end

endmodule

`resetall

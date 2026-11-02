// -----------------------------------------------------------------------------
// File   : tb_readout_counter.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   Feeds 5000 pseudo-random 8-spin samples (with repeats) into
//   readout_counter and checks the histogram, the up counters and the sample
//   count against a reference model in the testbench, then checks clear.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module tb_readout_counter;

parameter UP_RAM = 0;

reg         clk = 1'b0;
reg         rst = 1'b1;
reg         clear = 1'b0;
reg         sample = 1'b0;
reg  [7:0]  spins = 8'd0;
reg  [7:0]  haddr = 8'd0;
reg  [2:0]  usel = 3'd0;
wire [31:0] hq, uq, ns;
wire        busy;

integer ref_hist [0:255];
integer ref_up [0:7];
integer i, k, errors;
reg [31:0] x;

always #5 clk = ~clk;

readout_counter #(.N(8), .IW(3), .HIST_BITS(8), .CW(32), .UP_RAM(UP_RAM)) dut (
    .clk(clk), .rst(rst), .clear(clear), .sample(sample), .spins(spins),
    .hist_addr(haddr), .hist_q(hq), .up_sel(usel), .up_q(uq), .n_samples(ns), .busy(busy)
);

task check_all;
    begin
        for (k = 0; k < 256; k = k + 1) begin
            haddr <= k;
            @(posedge clk); @(posedge clk); #1;
            if (hq !== ref_hist[k]) errors = errors + 1;
        end
        for (k = 0; k < 8; k = k + 1) begin
            usel <= k;
            @(posedge clk); @(posedge clk); #1;
            if (uq !== ref_up[k]) errors = errors + 1;
        end
    end
endtask

initial begin
    errors = 0;
    for (k = 0; k < 256; k = k + 1) ref_hist[k] = 0;
    for (k = 0; k < 8; k = k + 1) ref_up[k] = 0;
    x = 32'h12345678;
    repeat (2) @(posedge clk);
    rst <= 1'b0;
    @(posedge clk);
    for (i = 0; i < 5000; i = i + 1) begin
        x = x ^ (x << 13); x = x ^ (x >> 17); x = x ^ (x << 5);
        spins  <= (x[3:0] == 4'd0) ? 8'hA5 : x[15:8];   // frequent repeats
        sample <= 1'b1;
        @(posedge clk);
        #1;
        ref_hist[spins] = ref_hist[spins] + 1;
        for (k = 0; k < 8; k = k + 1) ref_up[k] = ref_up[k] + spins[k];
        sample <= 1'b0;
        while (busy) @(posedge clk);
    end
    @(posedge clk);
    if (ns !== 5000) errors = errors + 1;
    check_all;
    $display("tb_readout_counter: 5000 samples, n_samples=%0d, hist[0xA5]=%0d, errors after counting=%0d", ns, ref_hist[8'hA5], errors);
    clear <= 1'b1;
    @(posedge clk);
    clear <= 1'b0;
    @(posedge clk);
    while (busy) @(posedge clk);
    for (k = 0; k < 256; k = k + 1) ref_hist[k] = 0;
    for (k = 0; k < 8; k = k + 1) ref_up[k] = 0;
    check_all;
    if (ns !== 0) errors = errors + 1;
    if (errors == 0) $display("tb_readout_counter: PASS");
    else             $display("tb_readout_counter: FAIL (%0d)", errors);
    $finish;
end

endmodule

`resetall

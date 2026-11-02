// -----------------------------------------------------------------------------
// File   : tb_noise_bank.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   Checks every noise_bank lane (and therefore lfsr.v) against the first 256
//   draws exported by model/export_vectors.py (model/fixed.lfsr_draw16).
//   Also checks reseeding and the lane-to-lane correlation of bit 0.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module tb_noise_bank;

parameter VEC = "../model/vectors/";

reg         clk = 1'b0;
reg         rst = 1'b1;
reg  [3:0]  en  = 4'd0;
reg         seed_load = 1'b0;
reg  [31:0] seed_in   = 32'd0;
wire [63:0] rnd;

reg [15:0] exp0 [0:255];
reg [15:0] exp1 [0:255];
reg [15:0] exp2 [0:255];
reg [15:0] exp3 [0:255];

integer i, errors, agree;
reg [15:0] first0;

always #5 clk = ~clk;

noise_bank #(.LANES(4), .SEED(32'hACE1F00D), .RO_JITTER(0)) dut (
    .clk(clk), .rst(rst), .en(en), .seed_load(seed_load), .seed_in(seed_in), .rnd(rnd)
);

initial begin
    $readmemh({VEC, "lfsr_lane0.hex"}, exp0);
    $readmemh({VEC, "lfsr_lane1.hex"}, exp1);
    $readmemh({VEC, "lfsr_lane2.hex"}, exp2);
    $readmemh({VEC, "lfsr_lane3.hex"}, exp3);
    errors = 0;
    agree  = 0;
    repeat (2) @(posedge clk);
    rst <= 1'b0;
    @(posedge clk);
    for (i = 0; i < 256; i = i + 1) begin
        #1;
        if (rnd[15:0]  !== exp0[i]) errors = errors + 1;
        if (rnd[31:16] !== exp1[i]) errors = errors + 1;
        if (rnd[47:32] !== exp2[i]) errors = errors + 1;
        if (rnd[63:48] !== exp3[i]) errors = errors + 1;
        if (rnd[0] == rnd[16]) agree = agree + 1;
        en <= 4'hF;
        @(posedge clk);
        en <= 4'h0;
    end
    // reseed with the default seed must replay the same sequence
    seed_load <= 1'b1;
    seed_in   <= 32'hACE1F00D;
    @(posedge clk);
    seed_load <= 1'b0;
    #1;
    first0 = rnd[15:0];
    if (first0 !== exp0[0]) errors = errors + 1;
    $display("tb_noise_bank: 4 lanes x 256 draws checked, reseed replay %s, lane0/lane1 bit0 agreement %0d/256",
             (first0 === exp0[0]) ? "ok" : "BAD", agree);
    if (errors == 0) $display("tb_noise_bank: PASS");
    else             $display("tb_noise_bank: FAIL (%0d mismatches)", errors);
    $finish;
end

endmodule

`resetall

// -----------------------------------------------------------------------------
// File   : tb_sigmoid.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   Sweeps all 1024 logits through sigmoid_lut.v and sigmoid_pwl.v. The LUT
//   must equal the table file; the PWL must equal model/fixed.sigmoid_pwl.
//   Reports monotonicity and the max |PWL - LUT| error.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module tb_sigmoid;

parameter VEC = "../model/vectors/";

reg               clk = 1'b0;
reg               en  = 1'b0;
reg  signed [9:0] logit = 10'sd0;
wire [15:0]       p_lut, p_pwl;

reg [15:0] lut [0:1023];
reg [31:0] pwl [0:1023];
integer k, errors, worst, d, prev, nonmono;

always #5 clk = ~clk;

sigmoid_lut #(.AW(10), .PW(16), .INIT_FILE({VEC, "sigmoid_lut.hex"})) lut_inst (
    .clk(clk), .en(en), .logit(logit), .p(p_lut));
sigmoid_pwl pwl_inst (.clk(clk), .en(en), .logit(logit), .p(p_pwl));

initial begin
    $readmemh({VEC, "sigmoid_lut.hex"}, lut);
    $readmemh({VEC, "pwl_vectors.hex"}, pwl);
    errors = 0; worst = 0; prev = 0; nonmono = 0;
    for (k = 0; k < 1024; k = k + 1) begin
        logit <= k - 512;
        en    <= 1'b1;
        @(posedge clk);
        #1;
        if (p_lut !== lut[k]) errors = errors + 1;
        if (p_pwl !== pwl[k][15:0]) errors = errors + 1;
        d = p_pwl - p_lut;
        if (d < 0) d = -d;
        if (d > worst) worst = d;
        if (p_lut < prev) nonmono = nonmono + 1;
        prev = p_lut;
    end
    $display("tb_sigmoid: LUT[0]=%0d LUT[512]=%0d LUT[1023]=%0d, non-monotone steps=%0d, max|PWL-LUT|=%0d/65536",
             lut[0], lut[512], lut[1023], nonmono, worst);
    if (errors == 0 && nonmono == 0) $display("tb_sigmoid: PASS");
    else                             $display("tb_sigmoid: FAIL (%0d mismatches)", errors);
    $finish;
end

endmodule

`resetall

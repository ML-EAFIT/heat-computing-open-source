// -----------------------------------------------------------------------------
// File   : tb_pbit.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   4096 random (field, beta, rnd) triples from model/vectors/pbit_vectors.hex
//   are applied to pbit.v (LUT variant). Logit, probability and output bit
//   must equal model/fixed.py exactly.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module tb_pbit;

parameter VEC = "../model/vectors/";

reg         clk = 1'b0;
reg         rst = 1'b1;
reg         start = 1'b0;
reg  signed [13:0] field = 14'sd0;
reg  [7:0]  beta = 8'd0;
reg  [15:0] rnd = 16'd0;
wire        bit_out, done, take;
wire signed [9:0] logit;
wire [15:0] p;

reg [75:0] vec [0:4095];
integer i, errors, ones, cycles;

always #5 clk = ~clk;

pbit #(.AW(14), .USE_PWL(0), .LUT_FILE({VEC, "sigmoid_lut.hex"})) dut (
    .clk(clk), .rst(rst), .start(start), .field(field), .beta(beta), .rnd(rnd),
    .bit_out(bit_out), .done(done), .rnd_take(take), .logit_dbg(logit), .p_dbg(p)
);

initial begin
    $readmemh({VEC, "pbit_vectors.hex"}, vec);
    errors = 0;
    ones   = 0;
    repeat (2) @(posedge clk);
    rst <= 1'b0;
    for (i = 0; i < 4096; i = i + 1) begin
        field <= vec[i][73:60];          // acc is 16 bit in the file, fits in 14
        beta  <= vec[i][59:52];
        rnd   <= vec[i][51:36];
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        cycles = 1;
        #1;
        while (!done) begin
            @(posedge clk);
            #1;
            cycles = cycles + 1;
        end
        if (logit !== vec[i][29:20] || p !== vec[i][19:4] || bit_out !== vec[i][0] || take !== 1'b1) begin
            errors = errors + 1;
            if (errors < 5)
                $display("  mismatch %0d: logit %0d/%0d p %0d/%0d bit %0d/%0d", i, logit,
                         $signed(vec[i][29:20]), p, vec[i][19:4], bit_out, vec[i][0]);
        end
        ones = ones + bit_out;
    end
    $display("tb_pbit: 4096 updates, start->done = %0d cycles, ones = %0d", cycles, ones);
    if (errors == 0) $display("tb_pbit: PASS");
    else             $display("tb_pbit: FAIL (%0d mismatches)", errors);
    $finish;
end

endmodule

`resetall

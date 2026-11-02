// -----------------------------------------------------------------------------
// File   : tb_denoise.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   Runs 8 denoising chains (T = 4 steps, K = 16 sweeps each) through
//   denoise_demo.v and checks x^{t-1} after every step against
//   model/denoise.py (vectors/denoise/runs.hex) bit for bit. Prints x^0 as a
//   4 x 4 image.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module tb_denoise;

parameter VEC = "../model/vectors/";

reg         clk = 1'b0;
reg         rst = 1'b1;
reg         start = 1'b0;
reg  [15:0] xT = 16'd0;
wire        sd, done;
wire [15:0] xo;

reg [79:0] runs [0:7];
integer r, s, errors, row, col;
reg [15:0] expv;

always #5 clk = ~clk;

denoise_demo #(
    .W_FILE({VEC, "denoise/weights.hex"}), .H_FILE({VEC, "denoise/bias.hex"}),
    .O_FILE({VEC, "denoise/order.hex"}), .G_FILE({VEC, "denoise/gamma.hex"}),
    .LUT_FILE({VEC, "sigmoid_lut.hex"})
) dut (
    .clk(clk), .rst(rst), .start(start), .x_t(xT), .step_done(sd), .done(done), .x_out(xo)
);

initial begin
    $readmemh({VEC, "denoise/runs.hex"}, runs);
    errors = 0;
    repeat (3) @(posedge clk);
    rst <= 1'b0;
    for (r = 0; r < 8; r = r + 1) begin
        xT    <= runs[r][79:64];
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        for (s = 0; s < 4; s = s + 1) begin
            @(posedge clk); #1;
            while (!sd) begin @(posedge clk); #1; end
            expv = runs[r][63 - 16*s -: 16];
            if (xo !== expv) begin
                errors = errors + 1;
                $display("  run %0d step %0d: rtl %h expected %h", r, s, xo, expv);
            end
        end
        $write("run %0d x^T=%h x^0=%h :", r, runs[r][79:64], xo);
        for (row = 0; row < 4; row = row + 1) begin
            $write(" ");
            for (col = 0; col < 4; col = col + 1) $write("%s", xo[row*4+col] ? "#" : ".");
        end
        $write("\n");
    end
    if (errors == 0) $display("tb_denoise: PASS (8 chains x 4 steps bit-exact)");
    else             $display("tb_denoise: FAIL (%0d)", errors);
    $finish;
end

endmodule

`resetall

// -----------------------------------------------------------------------------
// File   : tb_pmode.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-30
// -----------------------------------------------------------------------------
// Description:
//   Bit-exact check of pmode_array against model/pmode.PmodeArray. The case
//   (couplings, fields, initial state, constants) is chosen at compile time:
//     iverilog -I ../model/vectors/pm_<case> -P tb_pmode.CASE_DIR=...
//   For the first PM_STEPS steps every site after every step must equal
//   <case>/trace.hex. With +steps=<n> +trace=<file> the testbench runs n
//   steps and writes every post-step state for tb/pmode_study.py.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-30  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module tb_pmode;

parameter CASE_DIR = "../model/vectors/pm_chain4/";
parameter VEC      = "../model/vectors/";

`include "pm.vh"

localparam M  = PM_M;
localparam MW = PM_MW;

reg            clk = 1'b0;
reg            rst = 1'b1;
reg            run = 1'b0;
reg  [31:0]    nst = 32'd0;
wire           busy, sd;
wire [16*M-1:0] x_flat;

reg [16*M-1:0] expect_mem [0:PM_STEPS-1];
integer steps, t, errors, fd, cyc, cyc1;
reg [1023:0] trace_file;

always #5 clk = ~clk;
always @(posedge clk) cyc = cyc + 1;

pmode_array #(
    .M(M), .MW(MW), .SEED(PM_SEED),
    .A_FILE({CASE_DIR, "a.hex"}), .H_FILE({CASE_DIR, "h.hex"}), .X_FILE({CASE_DIR, "x0.hex"}),
    .LUT_FILE({VEC, "sigmoid_lut.hex"}), .GAUSS_FILE({VEC, "gauss_lut.hex"})
) dut (
    .clk(clk), .rst(rst),
    .a_we(1'b0), .a_addr({(2*MW){1'b0}}), .a_data(8'd0),
    .h_we(1'b0), .h_addr({MW{1'b0}}), .h_data(8'd0),
    .x_we(1'b0), .x_addr({MW{1'b0}}), .x_data(16'd0),
    .lam(PM_LAM[7:0]), .kshift(PM_K[3:0]), .sig(PM_SIG[15:0]),
    .run(run), .n_steps(nst), .busy(busy), .step_done(sd), .x_flat(x_flat)
);

initial begin
    $readmemh({CASE_DIR, "trace.hex"}, expect_mem);
    if (!$value$plusargs("steps=%d", steps)) steps = PM_STEPS;
    fd = 0;
    if ($value$plusargs("trace=%s", trace_file)) fd = $fopen(trace_file, "w");
    errors = 0;
    t = 0;
    cyc = 0;
    repeat (3) @(posedge clk);
    rst <= 1'b0;
    @(posedge clk);
    nst <= steps;
    run <= 1'b1;
    @(posedge clk);
    run <= 1'b0;
    while (t < steps) begin
        @(posedge clk);
        #1;
        if (sd) begin
            if (t < PM_STEPS && x_flat !== expect_mem[t]) begin
                errors = errors + 1;
                if (errors < 5) $display("  step %0d: rtl %h expected %h", t, x_flat, expect_mem[t]);
            end
            if (fd != 0) $fdisplay(fd, "%h", x_flat);
            if (t == 0) cyc1 = cyc;
            t = t + 1;
        end
    end
    $display("tb_pmode[%0s]: M=%0d, %0d steps, %0d cycles/step, bit-exact over first %0d steps: %0d mismatches",
             CASE_DIR, M, steps, (cyc - cyc1) / ((steps > 1) ? (steps - 1) : 1),
             (steps < PM_STEPS) ? steps : PM_STEPS, errors);
    if (fd != 0) $fclose(fd);
    if (errors == 0) $display("tb_pmode: PASS");
    else             $display("tb_pmode: FAIL");
    $finish;
end

endmodule

`resetall

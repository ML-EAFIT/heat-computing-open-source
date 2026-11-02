// -----------------------------------------------------------------------------
// File   : tb_pbit_array.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   Bit-exact check of pbit_array + schedule_ctrl against model/fixed.py.
//   The case (weights, bias, order, beta schedule) is chosen at compile time:
//     iverilog -I ../model/vectors/<case> -P tb_pbit_array.CASE_DIR=...
//   For the first CASE_SWEEPS sweeps the spin state after every sweep must
//   equal <case>/trace.hex. With +sweeps=<n> +trace=<file> the testbench runs
//   n sweeps and writes every post-sweep state for tb/compare.py
//   (histograms, TV/KL against exact Boltzmann and the golden model).
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module tb_pbit_array;

parameter CASE_DIR = "../model/vectors/ring8/";
parameter VEC      = "../model/vectors/";
parameter USE_PWL  = 0;
// expected trace file; use "trace_pwl.hex" together with USE_PWL = 1
parameter TRACE    = "trace.hex";

`include "case.vh"

function integer clog2;
    input integer v;
    integer r;
    begin
        r = 0;
        while ((1 << r) < v) r = r + 1;
        clog2 = (r < 1) ? 1 : r;
    end
endfunction

localparam N  = CASE_N;
localparam IW = clog2(N);
localparam AW = 8 + clog2(N + 1) + 1;

reg          clk = 1'b0;
reg          rst = 1'b1;
reg          run = 1'b0;
reg  [31:0]  nsw = 32'd0;
wire [7:0]   beta;
wire         busy, swd;
wire [N-1:0] spins;

reg [N-1:0] expect_mem [0:CASE_SWEEPS-1];
integer sweeps, t, errors, fd, cyc0, cyc1, cyc;
reg [1023:0] trace_file;

always #5 clk = ~clk;
always @(posedge clk) cyc = cyc + 1;

schedule_ctrl #(.TABLE_FILE({CASE_DIR, "beta_table.hex"})) sched_inst (
    .clk(clk), .rst(rst), .restart(1'b0), .sweep_done(swd),
    .cfg_mode(CASE_MODE[1:0]), .cfg_beta0(CASE_BETA0[7:0]), .cfg_beta1(CASE_BETA1[7:0]),
    .cfg_delta(CASE_DELTA[7:0]), .cfg_sps(CASE_SPS[15:0]),
    .tbl_we(1'b0), .tbl_addr(4'd0), .tbl_data(8'd0), .beta(beta)
);

pbit_array #(
    .N(N), .IW(IW), .WB(8), .WFRAC(CASE_WFRAC), .AW(AW), .USE_PWL(USE_PWL), .SEED(CASE_SEED),
    .W_FILE({CASE_DIR, "weights.hex"}), .H_FILE({CASE_DIR, "bias.hex"}),
    .O_FILE({CASE_DIR, "order.hex"}), .LUT_FILE({VEC, "sigmoid_lut.hex"})
) dut (
    .clk(clk), .rst(rst),
    .w_we(1'b0), .w_addr({2*IW{1'b0}}), .w_data(8'd0),
    .h_we(1'b0), .h_addr({IW{1'b0}}), .h_data(8'd0),
    .o_we(1'b0), .o_addr({IW{1'b0}}), .o_data({IW{1'b0}}),
    .spin_we(1'b0), .spin_data({N{1'b0}}),
    .seed_load(1'b0), .seed_in(32'd0),
    .run(run), .n_sweeps(nsw), .hold(1'b0), .stop(1'b0), .beta(beta),
    .busy(busy), .sweep_done(swd), .spins(spins)
);

initial begin
    $readmemh({CASE_DIR, TRACE}, expect_mem);
    if (!$value$plusargs("sweeps=%d", sweeps)) sweeps = CASE_SWEEPS;
    fd = 0;
    if ($value$plusargs("trace=%s", trace_file)) fd = $fopen(trace_file, "w");
    errors = 0;
    t = 0;
    cyc = 0;
    repeat (3) @(posedge clk);
    rst <= 1'b0;
    @(posedge clk);
    nsw <= sweeps;
    run <= 1'b1;
    @(posedge clk);
    run <= 1'b0;
    cyc0 = cyc;
    while (t < sweeps) begin
        @(posedge clk);
        #1;
        if (swd) begin
            if (t < CASE_SWEEPS && spins !== expect_mem[t]) begin
                errors = errors + 1;
                if (errors < 5) $display("  sweep %0d: rtl %h expected %h", t, spins, expect_mem[t]);
            end
            if (fd != 0) $fdisplay(fd, "%h", spins);
            if (t == 0) cyc1 = cyc;
            t = t + 1;
        end
    end
    $display("tb_pbit_array[%0s%0s]: N=%0d, %0d sweeps, %0d cycles/sweep, bit-exact over first %0d sweeps: %0d mismatches",
             CASE_DIR, (USE_PWL != 0) ? " PWL" : "", N, sweeps, (cyc - cyc1) / ((sweeps > 1) ? (sweeps - 1) : 1),
             (sweeps < CASE_SWEEPS) ? sweeps : CASE_SWEEPS, errors);
    if (fd != 0) $fclose(fd);
    if (errors == 0) $display("tb_pbit_array: PASS");
    else             $display("tb_pbit_array: FAIL");
    $finish;
end

endmodule

`resetall

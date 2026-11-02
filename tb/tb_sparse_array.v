// -----------------------------------------------------------------------------
// File   : tb_sparse_array.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-29
// -----------------------------------------------------------------------------
// Description:
//   Bit-exact check of sparse_array + schedule_ctrl against
//   model/fixed.SparseParallelSampler. The case (lane RAM images, rounds,
//   beta schedule) is chosen at compile time:
//     iverilog -I ../model/vectors/sp_<case>_p<P> -P tb_sparse_array.CASE_DIR=...
//   For the first CASE_SWEEPS sweeps the spin state after every sweep must
//   equal <case>/trace.hex. With +sweeps=<n> +trace=<file> the testbench runs
//   n sweeps and writes every post-sweep state for tb/compare.py.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-29  jero98772  v1.0  Initial version
//   2026-10-30  jero98772  v1.1  noise options (CASE_NZ_*), NOISE_CTL = 1
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module tb_sparse_array;

parameter CASE_DIR = "../model/vectors/sp_ring8_p2/";
parameter VEC      = "../model/vectors/";

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
localparam AW = 8 + clog2(CASE_D + 1) + 1;

reg          clk = 1'b0;
reg          rst = 1'b1;
reg          run = 1'b0;
reg  [31:0]  nsw = 32'd0;
wire [7:0]   beta;
wire         busy, swd;
wire [N-1:0] spins;

reg [N-1:0] expect_mem [0:CASE_SWEEPS-1];
integer sweeps, t, errors, fd, cyc1, cyc;
reg [1023:0] trace_file;

always #5 clk = ~clk;
always @(posedge clk) cyc = cyc + 1;

schedule_ctrl #(.TABLE_FILE({CASE_DIR, "beta_table.hex"})) sched_inst (
    .clk(clk), .rst(rst), .restart(1'b0), .sweep_done(swd),
    .cfg_mode(CASE_MODE[1:0]), .cfg_beta0(CASE_BETA0[7:0]), .cfg_beta1(CASE_BETA1[7:0]),
    .cfg_delta(CASE_DELTA[7:0]), .cfg_sps(CASE_SPS[15:0]),
    .tbl_we(1'b0), .tbl_addr(4'd0), .tbl_data(8'd0), .beta(beta)
);

sparse_array #(
    .N(N), .IW(IW), .P(CASE_P), .D(CASE_D), .DW(CASE_DW), .LW(CASE_LW), .ROUNDS(CASE_ROUNDS),
    .WB(8), .WFRAC(CASE_WFRAC), .AW(AW), .USE_PWL(0), .SEED(CASE_SEED),
    .NODE_PREFIX({CASE_DIR, "node_e"}), .NBR_PREFIX({CASE_DIR, "nbr_e"}),
    .LUT_FILE({VEC, "sigmoid_lut.hex"}), .NOISE_CTL(1)
) dut (
    .clk(clk), .rst(rst),
    .m_we(1'b0), .m_lane(2'd0), .m_addr({CASE_LW{1'b0}}), .m_data({(IW+9){1'b0}}),
    .n_we(1'b0), .n_lane(2'd0), .n_addr({(CASE_LW+CASE_DW){1'b0}}), .n_data({(IW+8){1'b0}}),
    .r_we(1'b0), .r_data({(CASE_LW+1){1'b0}}),
    .spin_we(1'b0), .spin_data({N{1'b0}}),
    .seed_load(1'b0), .seed_in(32'd0),
    .run(run), .n_sweeps(nsw), .hold(1'b0), .stop(1'b0), .beta(beta),
    .nz_tau(CASE_NZ_TAU != 0), .nz_share(CASE_NZ_SHARE != 0), .nz_q(CASE_NZ_Q[15:0]),
    .busy(busy), .sweep_done(swd), .spins(spins)
);

initial begin
    $readmemh({CASE_DIR, "trace.hex"}, expect_mem);
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
    $display("tb_sparse_array[%0s]: N=%0d P=%0d D=%0d rounds=%0d, %0d sweeps, %0d cycles/sweep (%0d/round), bit-exact over first %0d sweeps: %0d mismatches",
             CASE_DIR, N, CASE_P, CASE_D, CASE_ROUNDS, sweeps,
             (cyc - cyc1) / ((sweeps > 1) ? (sweeps - 1) : 1),
             (cyc - cyc1) / ((sweeps > 1) ? (sweeps - 1) : 1) / CASE_ROUNDS,
             (sweeps < CASE_SWEEPS) ? sweeps : CASE_SWEEPS, errors);
    if (fd != 0) $fclose(fd);
    if (errors == 0) $display("tb_sparse_array: PASS");
    else             $display("tb_sparse_array: FAIL");
    $finish;
end

endmodule

`resetall

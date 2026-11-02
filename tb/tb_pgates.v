// -----------------------------------------------------------------------------
// File   : tb_pgates.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-25
// -----------------------------------------------------------------------------
// Description:
//   Drives pgates.v through the clamp script of one p-gate case
//   (vectors/<case>/pg_script.hex: free run, forward clamps, backward clamps,
//   or factoring products for factor6). Checks the spins after every sweep
//   against model/fixed.py (pg_trace.hex) bit for bit and writes them to
//   +dump=<file> for the truth-table statistics in compare.py --pgates.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-25  jero98772  v1.0  Initial version
//   2026-10-30  jero98772  v1.1  PG_K / PG_LADDER: tempered cases (dump per round)
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module tb_pgates;

parameter CASE_DIR = "../model/vectors/pgate_and/";
parameter VEC      = "../model/vectors/";

`include "case.vh"
`include "pg.vh"

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
reg          start = 1'b0;
reg  [N-1:0] cmask = {N{1'b0}};
reg  [N-1:0] cval  = {N{1'b0}};
reg  [31:0]  nsw   = 32'd0;
wire         busy, swd, done;
wire [N-1:0] spins;

reg [63:0]  script [0:PG_SCENARIOS-1];
reg [N-1:0] expect_mem [0:PG_SWEEPS-1];
integer k, t, errors, fd;
reg [1023:0] dump_file;

always #5 clk = ~clk;

pgate #(
    .N(N), .IW(IW), .AW(AW), .SEED(CASE_SEED),
    .W_FILE({CASE_DIR, "weights.hex"}), .H_FILE({CASE_DIR, "bias.hex"}),
    .O_FILE({CASE_DIR, "order.hex"}), .LUT_FILE({VEC, "sigmoid_lut.hex"}),
    .K(PG_K), .LADDER(PG_LADDER)
) dut (
    .clk(clk), .rst(rst), .start(start), .clamp_mask(cmask), .clamp_val(cval),
    .n_sweeps(nsw), .beta(PG_BETA[7:0]),
    .busy(busy), .sweep_done(swd), .done(done), .spins(spins)
);

// every sweep of every scenario, in order
always @(posedge clk) begin
    if (swd) begin
        if (t < PG_SWEEPS && spins !== expect_mem[t]) begin
            errors = errors + 1;
            if (errors < 5) $display("  sweep %0d: rtl %h expected %h", t, spins, expect_mem[t]);
        end
        if (fd != 0) $fdisplay(fd, "%h", spins);
        t = t + 1;
    end
end

initial begin
    $readmemh({CASE_DIR, "pg_script.hex"}, script);
    $readmemh({CASE_DIR, "pg_trace.hex"}, expect_mem);
    fd = 0;
    if ($value$plusargs("dump=%s", dump_file)) fd = $fopen(dump_file, "w");
    errors = 0;
    t = 0;
    repeat (3) @(posedge clk);
    rst <= 1'b0;
    @(posedge clk);
    for (k = 0; k < PG_SCENARIOS; k = k + 1) begin
        cmask <= script[k][63:48];
        cval  <= script[k][47:32];
        nsw   <= script[k][31:0];
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        @(posedge clk);
        while (busy) @(posedge clk);
    end
    repeat (2) @(posedge clk);
    $display("tb_pgates[%0s]: N=%0d, %0d scenarios, %0d sweeps, bit-exact vs fixed.py: %0d mismatches",
             CASE_DIR, N, PG_SCENARIOS, t, errors);
    if (fd != 0) $fclose(fd);
    if (errors == 0 && t == PG_SWEEPS) $display("tb_pgates: PASS");
    else                               $display("tb_pgates: FAIL");
    $finish;
end

endmodule

`resetall

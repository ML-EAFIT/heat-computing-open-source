// -----------------------------------------------------------------------------
// File   : tb_schedule_ctrl.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   Runs the three schedule modes (constant, linear, table) for 64 sweeps
//   each and compares beta with model/fixed.Schedule (sched_vectors.hex).
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module tb_schedule_ctrl;

parameter VEC = "../model/vectors/";

reg        clk = 1'b0;
reg        rst = 1'b1;
reg        restart = 1'b0;
reg        sweep_done = 1'b0;
reg [1:0]  mode = 2'd0;
reg [7:0]  b0 = 8'd0, b1 = 8'd0, dl = 8'd0;
reg [15:0] sps = 16'd1;
reg        tbl_we = 1'b0;
reg [3:0]  tbl_addr = 4'd0;
reg [7:0]  tbl_data = 8'd0;
wire [7:0] beta;

reg [7:0] expv [0:191];
integer m, s, errors;

always #5 clk = ~clk;

schedule_ctrl #(.TABLE_FILE("")) dut (
    .clk(clk), .rst(rst), .restart(restart), .sweep_done(sweep_done),
    .cfg_mode(mode), .cfg_beta0(b0), .cfg_beta1(b1), .cfg_delta(dl), .cfg_sps(sps),
    .tbl_we(tbl_we), .tbl_addr(tbl_addr), .tbl_data(tbl_data), .beta(beta)
);

task configure;
    input [1:0] md; input [7:0] a; input [7:0] b; input [7:0] d; input [15:0] n;
    begin
        mode <= md; b0 <= a; b1 <= b; dl <= d; sps <= n;
        @(posedge clk);
        restart <= 1'b1;
        @(posedge clk);
        restart <= 1'b0;
        @(posedge clk);
    end
endtask

initial begin
    $readmemh({VEC, "sched_vectors.hex"}, expv);
    errors = 0;
    repeat (2) @(posedge clk);
    rst <= 1'b0;
    for (s = 0; s < 16; s = s + 1) begin          // table = 2 + 9k
        tbl_we <= 1'b1; tbl_addr <= s; tbl_data <= 2 + 9 * s;
        @(posedge clk);
    end
    tbl_we <= 1'b0;
    for (m = 0; m < 3; m = m + 1) begin
        case (m)
            0: configure(2'd0, 8'd20, 8'd20, 8'd0, 16'd1);
            1: configure(2'd1, 8'd4,  8'd60, 8'd3, 16'd5);
            default: configure(2'd2, 8'd0, 8'd0, 8'd0, 16'd3);
        endcase
        for (s = 0; s < 64; s = s + 1) begin
            #1;
            if (beta !== expv[m*64 + s]) begin
                errors = errors + 1;
                if (errors < 5) $display("  mode %0d sweep %0d: beta %0d exp %0d", m, s, beta, expv[m*64+s]);
            end
            sweep_done <= 1'b1;
            @(posedge clk);
            sweep_done <= 1'b0;
            @(posedge clk);
        end
    end
    $display("tb_schedule_ctrl: 3 modes x 64 sweeps, final linear beta=%0d table beta=%0d", expv[127], beta);
    if (errors == 0) $display("tb_schedule_ctrl: PASS");
    else             $display("tb_schedule_ctrl: FAIL (%0d mismatches)", errors);
    $finish;
end

endmodule

`resetall

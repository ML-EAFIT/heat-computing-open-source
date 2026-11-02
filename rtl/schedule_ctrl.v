// -----------------------------------------------------------------------------
// File   : schedule_ctrl.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   Inverse-temperature (beta) annealing controller, unsigned Q4.4.
//     mode 0 : constant beta0
//     mode 1 : linear, beta += delta every SPS sweeps, saturating at beta1
//     mode 2 : table, beta = table[k], k advances every SPS sweeps and holds
//              at entry 15 (16-entry table, writable, init from file)
//   restart (or rst) reloads beta and clears the counters. SPS = 0 acts as 1.
//
//   Idea source: simulated annealing "follow a schedule of increasing beta
//   values. At each beta we run a handful of Gibbs sweeps"; raising beta =
//   scaling all weights by beta [thrml/03_codon_optimization.py:L309-311,
//   L322-344]. [FPGA-APPROX] Q4.4 and the three modes.
//   Mirror: model/fixed.Schedule.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module schedule_ctrl #
(
    // 16-entry beta table file ("" = all zero)
    parameter TABLE_FILE = ""
)
(
    input  wire        clk,
    input  wire        rst,
    input  wire        restart,
    input  wire        sweep_done,
    input  wire [1:0]  cfg_mode,
    input  wire [7:0]  cfg_beta0,
    input  wire [7:0]  cfg_beta1,
    input  wire [7:0]  cfg_delta,
    input  wire [15:0] cfg_sps,
    input  wire        tbl_we,
    input  wire [3:0]  tbl_addr,
    input  wire [7:0]  tbl_data,
    output wire [7:0]  beta
);

reg [7:0] table_mem [15:0];

integer i;
initial begin
    for (i = 0; i < 16; i = i + 1) begin
        table_mem[i] = 8'd0;
    end
    if (TABLE_FILE != "") begin
        $readmemh(TABLE_FILE, table_mem, 0, 15);
    end
end

always @(posedge clk) begin
    if (tbl_we) begin
        table_mem[tbl_addr] <= tbl_data;
    end
end

reg [7:0]  beta_reg = 8'd0,  beta_next;
reg [15:0] cnt_reg  = 16'd0, cnt_next;
reg [3:0]  k_reg    = 4'd0,  k_next;
reg [8:0]  lin;

wire [15:0] sps_eff = (cfg_sps == 16'd0) ? 16'd1 : cfg_sps;
wire [7:0]  beta_init = (cfg_mode == 2'd2) ? table_mem[0] : cfg_beta0;

always @* begin
    beta_next = beta_reg;
    cnt_next  = cnt_reg;
    k_next    = k_reg;
    lin       = {1'b0, beta_reg} + {1'b0, cfg_delta};

    if (restart) begin
        beta_next = beta_init;
        cnt_next  = 16'd0;
        k_next    = 4'd0;
    end else if (sweep_done) begin
        if (cnt_reg + 16'd1 < sps_eff) begin
            cnt_next = cnt_reg + 16'd1;
        end else begin
            cnt_next = 16'd0;
            case (cfg_mode)
                2'd1: begin
                    beta_next = (lin > {1'b0, cfg_beta1}) ? cfg_beta1 : lin[7:0];
                end
                2'd2: begin
                    k_next    = (k_reg == 4'd15) ? 4'd15 : (k_reg + 4'd1);
                    beta_next = table_mem[k_next];
                end
                default: begin
                    beta_next = beta_reg;
                end
            endcase
        end
    end
end

always @(posedge clk) begin
    beta_reg <= beta_next;
    cnt_reg  <= cnt_next;
    k_reg    <= k_next;

    if (rst) begin
        beta_reg <= beta_init;
        cnt_reg  <= 16'd0;
        k_reg    <= 4'd0;
    end
end

assign beta = beta_reg;

endmodule

`resetall

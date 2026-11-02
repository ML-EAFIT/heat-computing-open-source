// -----------------------------------------------------------------------------
// File   : coupling_row.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   Computes the local field of node `row`:
//     field = h_row + sum_{j<N} (s_j ? J_row,j : -J_row,j)
//   by streaming row `row` of the weight RAM (address {row, j}, 1-cycle read
//   latency) through weight_mac. Takes N + 2 cycles from start to done.
//   The diagonal J_row,row must be stored as 0.
//
//   Idea source: neighbour states are combined with the model weights by a
//   linear circuit [data/An efficient ... 2510.23972v2.pdf, p.25 App. E;
//   p.26 Eq. E7]. [FPGA-APPROX] one physical resistor network per cell is
//   replaced by one serial MAC shared by all cells (time multiplexing).
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module coupling_row #
(
    // number of p-bits
    parameter N  = 16,
    // node index width = clog2(N)
    parameter IW = 4,
    // weight width
    parameter WB = 8,
    // accumulator width
    parameter AW = 14
)
(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,
    input  wire [IW-1:0]        row,
    input  wire signed [WB-1:0] h_in,
    input  wire [N-1:0]         spins,
    output wire [2*IW-1:0]      w_addr,
    input  wire signed [WB-1:0] w_data,
    output wire signed [AW-1:0] field,
    output wire                 done
);

initial begin
    if ((1 << IW) < N) begin
        $error("coupling_row: IW too small for N");
        $finish;
    end
end

localparam integer   LASTI = N - 1;
localparam integer   NI    = N;
localparam [IW:0]    NCOL  = NI[IW:0];
localparam [IW-1:0] LAST  = LASTI[IW-1:0];

reg          run_reg  = 1'b0,          run_next;
reg [IW:0]   j_reg    = {(IW+1){1'b0}}, j_next;      // next column to issue
reg          vld_reg  = 1'b0,          vld_next;     // RAM data valid this cycle
reg [IW-1:0] jd_reg   = {IW{1'b0}},    jd_next;      // column of the RAM data
reg          done_reg = 1'b0,          done_next;
reg [IW-1:0] row_reg  = {IW{1'b0}},    row_next;

always @* begin
    run_next  = run_reg;
    j_next    = j_reg;
    vld_next  = 1'b0;
    jd_next   = jd_reg;
    done_next = 1'b0;
    row_next  = row_reg;

    if (start) begin
        run_next = 1'b1;
        j_next   = {(IW+1){1'b0}};
        row_next = row;
    end else if (run_reg) begin
        if (j_reg < NCOL) begin
            vld_next = 1'b1;
            jd_next  = j_reg[IW-1:0];
            j_next   = j_reg + 1'b1;
        end else begin
            run_next = 1'b0;
        end
    end

    // the last RAM word is accumulated in the cycle after the final issue
    if (vld_reg && (jd_reg == LAST)) begin
        done_next = 1'b1;
    end
end

always @(posedge clk) begin
    run_reg  <= run_next;
    j_reg    <= j_next;
    vld_reg  <= vld_next;
    jd_reg   <= jd_next;
    done_reg <= done_next;
    row_reg  <= row_next;

    if (rst) begin
        run_reg  <= 1'b0;
        vld_reg  <= 1'b0;
        done_reg <= 1'b0;
    end
end

assign w_addr = {row_reg, j_reg[IW-1:0]};
assign done   = done_reg;

weight_mac #(
    .WB (WB),
    .AW (AW)
) weight_mac_inst (
    .clk    (clk),
    .rst    (rst),
    .load   (start),
    .h      (h_in),
    .add_en (vld_reg),
    .w      (w_data),
    .s      (spins[jd_reg]),
    .acc    (field)
);

endmodule

`resetall

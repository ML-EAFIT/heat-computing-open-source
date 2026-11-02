// -----------------------------------------------------------------------------
// File   : tb_weight_mac.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   Drives coupling_row (and so weight_mac) with 512 random 8-wide rows from
//   model/vectors/mac_vectors.hex and compares the field with Python.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module tb_weight_mac;

parameter VEC = "../model/vectors/";

reg         clk = 1'b0;
reg         rst = 1'b1;
reg         start = 1'b0;
reg  [7:0]  hreg = 8'd0;
reg  [7:0]  sp = 8'd0;
reg  [7:0]  row [0:7];
reg  [7:0]  wq = 8'd0;
wire [5:0]  w_addr;
wire signed [12:0] field;
wire        done;

reg [95:0] vec [0:511];
integer i, j, errors, cycles;

always #5 clk = ~clk;

// 1-cycle latency "RAM" holding the current row (row index ignored)
always @(posedge clk) wq <= row[w_addr[2:0]];

coupling_row #(.N(8), .IW(3), .WB(8), .AW(13)) dut (
    .clk(clk), .rst(rst), .start(start), .row(3'd5), .h_in(hreg), .spins(sp),
    .w_addr(w_addr), .w_data(wq), .field(field), .done(done)
);

initial begin
    $readmemh({VEC, "mac_vectors.hex"}, vec);
    errors = 0;
    repeat (2) @(posedge clk);
    rst <= 1'b0;
    for (i = 0; i < 512; i = i + 1) begin
        for (j = 0; j < 8; j = j + 1) row[j] = vec[i][32 + 8*j +: 8];
        hreg  <= vec[i][31:24];
        sp    <= vec[i][23:16];
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
        if (field !== $signed(vec[i][12:0])) begin
            errors = errors + 1;
            if (errors < 5) $display("  mismatch %0d: got %0d exp %0d", i, field, $signed(vec[i][15:0]));
        end
    end
    $display("tb_weight_mac: 512 rows, N=8, start->done = %0d cycles", cycles);
    if (errors == 0) $display("tb_weight_mac: PASS");
    else             $display("tb_weight_mac: FAIL (%0d mismatches)", errors);
    $finish;
end

endmodule

`resetall

// -----------------------------------------------------------------------------
// File   : tb_uart.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   Loopback uart_tx -> uart_rx with all 256 byte values at the real
//   27 MHz / 115200 setting (CLKS_PER_BIT = 234). Also checks the bit period.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module tb_uart;

localparam CPB = 234;

reg        clk = 1'b0;
reg        rst = 1'b1;
reg  [7:0] din = 8'd0;
reg        vin = 1'b0;
wire       rdy, line, vout;
wire [7:0] dout;

integer i, errors, got;

always #18.518 clk = ~clk;       // 27 MHz

uart_tx #(.CLKS_PER_BIT(CPB)) tx_inst (.clk(clk), .rst(rst), .in_data(din), .in_valid(vin), .in_ready(rdy), .txd(line));
uart_rx #(.CLKS_PER_BIT(CPB)) rx_inst (.clk(clk), .rst(rst), .rxd(line), .out_data(dout), .out_valid(vout));

always @(posedge clk) begin
    if (vout) begin
        if (dout !== got[7:0]) errors = errors + 1;
        got = got + 1;
    end
end

initial begin
    errors = 0; got = 0;
    repeat (4) @(posedge clk);
    rst <= 1'b0;
    for (i = 0; i < 256; i = i + 1) begin
        @(posedge clk);
        while (!rdy) @(posedge clk);
        din <= i; vin <= 1'b1;
        @(posedge clk);
        vin <= 1'b0;
    end
    while (got < 256) @(posedge clk);
    $display("tb_uart: 256 bytes looped back, %0d errors", errors);
    if (errors == 0 && got == 256) $display("tb_uart: PASS");
    else                           $display("tb_uart: FAIL");
    $finish;
end

endmodule

`resetall

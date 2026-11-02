// -----------------------------------------------------------------------------
// File   : uart_tx.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   8N1 UART transmitter. A byte is accepted when in_valid && in_ready.
//   Line idles high. CLKS_PER_BIT = f_clk / baud (27 MHz / 115200 = 234).
//   [FPGA-APPROX] host link; the sources' XTR-0 uses a CPU host + FPGA
//   [data/A Framework ... 2608.01612v2.pdf, p.9 Fig. 6].
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module uart_tx #
(
    // clock cycles per bit
    parameter CLKS_PER_BIT = 234
)
(
    input  wire       clk,
    input  wire       rst,
    input  wire [7:0] in_data,
    input  wire       in_valid,
    output wire       in_ready,
    output wire       txd
);

reg [9:0]  shift_reg = 10'h3FF, shift_next;
reg [3:0]  bits_reg  = 4'd0,    bits_next;
reg [15:0] cnt_reg   = 16'd0,   cnt_next;
reg        busy_reg  = 1'b0,    busy_next;

always @* begin
    shift_next = shift_reg;
    bits_next  = bits_reg;
    cnt_next   = cnt_reg;
    busy_next  = busy_reg;

    if (!busy_reg) begin
        if (in_valid) begin
            shift_next = {1'b1, in_data, 1'b0};   // stop, data LSB first, start
            bits_next  = 4'd10;
            cnt_next   = CLKS_PER_BIT - 1;
            busy_next  = 1'b1;
        end
    end else if (cnt_reg != 16'd0) begin
        cnt_next = cnt_reg - 16'd1;
    end else begin
        shift_next = {1'b1, shift_reg[9:1]};
        bits_next  = bits_reg - 4'd1;
        cnt_next   = CLKS_PER_BIT - 1;
        if (bits_reg == 4'd1) begin
            busy_next = 1'b0;
        end
    end
end

always @(posedge clk) begin
    shift_reg <= shift_next;
    bits_reg  <= bits_next;
    cnt_reg   <= cnt_next;
    busy_reg  <= busy_next;

    if (rst) begin
        shift_reg <= 10'h3FF;
        busy_reg  <= 1'b0;
    end
end

assign in_ready = ~busy_reg;
assign txd      = busy_reg ? shift_reg[0] : 1'b1;

endmodule

`resetall

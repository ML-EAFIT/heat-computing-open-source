// -----------------------------------------------------------------------------
// File   : uart_rx.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   8N1 UART receiver with a two-flop input synchronizer and mid-bit
//   sampling. out_valid pulses for one cycle per received byte (no
//   backpressure). Frames with a bad stop bit are dropped.
//   [FPGA-APPROX] host link (see uart_tx.v).
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module uart_rx #
(
    // clock cycles per bit
    parameter CLKS_PER_BIT = 234
)
(
    input  wire       clk,
    input  wire       rst,
    input  wire       rxd,
    output wire [7:0] out_data,
    output wire       out_valid
);

// state encoding: binary, 2 bits
localparam [1:0] S_IDLE  = 2'd0,
                 S_START = 2'd1,
                 S_DATA  = 2'd2,
                 S_STOP  = 2'd3;

// CDC: asynchronous rxd -> clk
reg        s1_reg    = 1'b1;
reg        s2_reg    = 1'b1;
reg [1:0]  state_reg = S_IDLE, state_next;
reg [15:0] cnt_reg   = 16'd0,  cnt_next;
reg [2:0]  bit_reg   = 3'd0,   bit_next;
reg [7:0]  data_reg  = 8'd0,   data_next;
reg        vld_reg   = 1'b0,   vld_next;

always @* begin
    state_next = state_reg;
    cnt_next   = cnt_reg;
    bit_next   = bit_reg;
    data_next  = data_reg;
    vld_next   = 1'b0;

    case (state_reg)
        S_IDLE: begin
            if (!s2_reg) begin
                cnt_next   = (CLKS_PER_BIT / 2) - 1;
                state_next = S_START;
            end
        end
        S_START: begin
            if (cnt_reg != 16'd0) begin
                cnt_next = cnt_reg - 16'd1;
            end else if (!s2_reg) begin
                cnt_next   = CLKS_PER_BIT - 1;
                bit_next   = 3'd0;
                state_next = S_DATA;
            end else begin
                state_next = S_IDLE;              // glitch
            end
        end
        S_DATA: begin
            if (cnt_reg != 16'd0) begin
                cnt_next = cnt_reg - 16'd1;
            end else begin
                data_next = {s2_reg, data_reg[7:1]};
                cnt_next  = CLKS_PER_BIT - 1;
                bit_next  = bit_reg + 3'd1;
                if (bit_reg == 3'd7) begin
                    state_next = S_STOP;
                end
            end
        end
        S_STOP: begin
            if (cnt_reg != 16'd0) begin
                cnt_next = cnt_reg - 16'd1;
            end else begin
                vld_next   = s2_reg;
                state_next = S_IDLE;
            end
        end
        default: begin
            state_next = S_IDLE;
        end
    endcase
end

always @(posedge clk) begin
    s1_reg    <= rxd;
    s2_reg    <= s1_reg;
    state_reg <= state_next;
    cnt_reg   <= cnt_next;
    bit_reg   <= bit_next;
    data_reg  <= data_next;
    vld_reg   <= vld_next;

    if (rst) begin
        state_reg <= S_IDLE;
        vld_reg   <= 1'b0;
    end
end

assign out_data  = data_reg;
assign out_valid = vld_reg;

endmodule

`resetall

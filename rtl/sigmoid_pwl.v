// -----------------------------------------------------------------------------
// File   : sigmoid_pwl.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   Piecewise-linear sigmoid using shifts and adds only (no BSRAM, no DSP).
//   Same interface and one-cycle latency as sigmoid_lut.v. Segments in real
//   units of |x| (x = logit/32):
//     |x| >= 5      : 1
//     [2.375, 5)    : |x|/32 + 0.84375
//     [1, 2.375)    : |x|/8  + 0.625
//     [0, 1)        : |x|/4  + 0.5
//   and p(-x) = 1 - p(x). Max abs error vs the LUT is 0.019 (model/fixed.py).
//
//   Idea source: sigmoidal RNG bias
//   [data/An efficient ... 2510.23972v2.pdf, p.5 Eq. 11].
//   [FPGA-APPROX] the segment constants are a standard approximation, not
//   taken from the sources. Mirror: model/fixed.sigmoid_pwl.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module sigmoid_pwl
(
    input  wire              clk,
    input  wire              en,
    input  wire signed [9:0] logit,
    output wire        [15:0] p
);

reg  [15:0] p_reg = 16'd0, p_next;
reg  [9:0]  a;
reg  [16:0] y;

always @* begin
    p_next = p_reg;
    a      = logit[9] ? (~logit + 10'd1) : logit;   // |x| in Q.5 (512 -> 0x200 ok)
    if (a >= 10'd160) begin
        y = 17'd65535;
    end else if (a >= 10'd76) begin
        y = {1'b0, a, 6'd0} + 17'd55296;
    end else if (a >= 10'd32) begin
        y = {a[8:0], 8'd0} + 17'd40960;
    end else begin
        y = {a[7:0], 9'd0} + 17'd32768;
    end
    if (y > 17'd65535) begin
        y = 17'd65535;
    end
    if (en) begin
        p_next = logit[9] ? (17'd65536 - y) : y[15:0];
    end
end

always @(posedge clk) begin
    p_reg <= p_next;
end

assign p = p_reg;

endmodule

`resetall

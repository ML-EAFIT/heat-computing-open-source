// -----------------------------------------------------------------------------
// File   : weight_mac.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   Multiplier-free synapse accumulator for spin inputs s in {-1,+1}:
//     load   : acc <= sign_extend(h)
//     add_en : acc <= acc + (s ? +w : -w)
//   Because s is +-1, the "multiply" is just a conditional negate. This is the
//   digital form of the sign-corrected input y = x XOR s feeding the resistor
//   network of the sampling cell.
//
//   Idea source: [data/An efficient ... 2510.23972v2.pdf, p.25 Fig. 10;
//   p.26 Eq. E7]; MAC description in [thrml/02_spin_models.py:L27].
//   [FPGA-APPROX] AW must satisfy AW >= WB + clog2(N+1) + 1 (no overflow).
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module weight_mac #
(
    // weight / bias width (signed)
    parameter WB = 8,
    // accumulator width (signed)
    parameter AW = 14
)
(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 load,
    input  wire signed [WB-1:0] h,
    input  wire                 add_en,
    input  wire signed [WB-1:0] w,
    input  wire                 s,
    output wire signed [AW-1:0] acc
);

reg signed [AW-1:0] acc_reg = {AW{1'b0}}, acc_next;

wire signed [AW-1:0] h_ext = {{(AW-WB){h[WB-1]}}, h};
wire signed [AW-1:0] w_ext = {{(AW-WB){w[WB-1]}}, w};

always @* begin
    acc_next = acc_reg;
    if (load) begin
        acc_next = h_ext;
    end else if (add_en) begin
        acc_next = s ? (acc_reg + w_ext) : (acc_reg - w_ext);
    end
end

always @(posedge clk) begin
    acc_reg <= acc_next;

    if (rst) begin
        acc_reg <= {AW{1'b0}};
    end
end

assign acc = acc_reg;

endmodule

`resetall

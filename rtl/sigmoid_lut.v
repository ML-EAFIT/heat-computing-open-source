// -----------------------------------------------------------------------------
// File   : sigmoid_lut.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   Sigmoid ROM: p = round(65536 * sigma(logit/32)), clipped to 65535.
//   Input logit is signed Q5.5 (range [-16, 16)). Address = logit + 512,
//   i.e. offset binary. One-cycle synchronous read so it maps onto one
//   Gowin BSRAM block (1024 x 16 = 16 kbit). Table file is produced by
//   model/export_vectors.py (mirror: model/fixed.sigmoid_lut).
//
//   Idea source: the RNG bias is a sigmoidal function of the control voltage
//   [data/An efficient ... 2510.23972v2.pdf, p.6 Fig. 4a; p.5 Eq. 11].
//   [FPGA-APPROX] quantized table instead of device physics.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module sigmoid_lut #
(
    // address width = logit width (Q5.5 -> 10 bits)
    parameter AW        = 10,
    // probability width
    parameter PW        = 16,
    // $readmemh table file
    parameter INIT_FILE = "../model/vectors/sigmoid_lut.hex"
)
(
    input  wire                 clk,
    input  wire                 en,
    input  wire signed [AW-1:0] logit,
    output wire        [PW-1:0] p
);

(* ram_style = "block" *)
reg [PW-1:0] rom [(2**AW)-1:0];

initial begin
    $readmemh(INIT_FILE, rom, 0, (2**AW)-1);
end

reg [PW-1:0] p_reg = {PW{1'b0}};

wire [AW-1:0] addr = {~logit[AW-1], logit[AW-2:0]};

always @(posedge clk) begin
    if (en) begin
        p_reg <= rom[addr];
    end
end

assign p = p_reg;

endmodule

`resetall

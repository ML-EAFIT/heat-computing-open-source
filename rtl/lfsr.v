// -----------------------------------------------------------------------------
// File   : lfsr.v
// Author : jero98772 (generated with Claude Code)
// Date   : 2026-10-23
// -----------------------------------------------------------------------------
// Description:
//   32-bit Galois (right-shift) LFSR that leaps OUT_BITS steps per draw.
//   rnd is the next OUT_BITS-bit draw, computed combinationally from the
//   current state (bit k = LFSR output of step k). Asserting en consumes the
//   draw and advances the state by OUT_BITS steps. seed_load reloads state.
//
//   Idea source: replaces the all-transistor RNG ("digitizing comparator fed
//   by a source of Gaussian noise") of
//   [data/An efficient ... 2510.23972v2.pdf, p.37 App. K; p.38 Fig. 15a].
//   [FPGA-APPROX] pseudo-random, period 2^32-1 (mask verified in
//   model/fixed.py: lfsr_is_maximal). Bit-exact mirror: fixed.lfsr_draw16.
// -----------------------------------------------------------------------------
// Change Log:
//   2026-10-23  jero98772  v1.0  Initial version
// -----------------------------------------------------------------------------

`resetall
`timescale 1ns / 1ps
`default_nettype none

module lfsr #
(
    // Galois feedback mask (must be maximal-length, see model/fixed.py)
    parameter [31:0] MASK     = 32'hB4BCD35C,
    // reset seed (non-zero)
    parameter [31:0] SEED     = 32'hACE1F00D,
    // bits produced per draw
    parameter        OUT_BITS = 16
)
(
    input  wire                clk,
    input  wire                rst,
    input  wire                en,
    input  wire                seed_load,
    input  wire [31:0]         seed_in,
    output wire [OUT_BITS-1:0] rnd
);

initial begin
    if (SEED == 32'd0) begin
        $error("lfsr: SEED must be non-zero");
        $finish;
    end
end

reg [31:0] state_reg = SEED, state_next;

// OUT_BITS Galois steps; returns {output bits, new state}
function [OUT_BITS+31:0] leap;
    input [31:0] s;
    integer      k;
    reg   [31:0] t;
    reg   [OUT_BITS-1:0] r;
    begin
        t = s;
        r = {OUT_BITS{1'b0}};
        for (k = 0; k < OUT_BITS; k = k + 1) begin
            r[k] = t[0];
            t    = t[0] ? ((t >> 1) ^ MASK) : (t >> 1);
        end
        leap = {r, t};
    end
endfunction

wire [OUT_BITS+31:0] leap_w = leap(state_reg);

assign rnd = leap_w[OUT_BITS+31:32];

always @* begin
    state_next = state_reg;
    if (seed_load) begin
        state_next = (seed_in == 32'd0) ? 32'd1 : seed_in;
    end else if (en) begin
        state_next = leap_w[31:0];
    end
end

always @(posedge clk) begin
    state_reg <= state_next;

    if (rst) begin
        state_reg <= SEED;
    end
end

endmodule

`resetall
